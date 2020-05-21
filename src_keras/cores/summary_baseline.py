import numpy as np
import pandas as pd
from tqdm import tqdm
from bert4keras.models import BERT
from bert4keras.tokenizers import Tokenizer, load_vocab
from keras import backend as K
from bert4keras.snippets import parallel_apply
from keras.optimizers import Adam
from rouge import Rouge
import keras
import math
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import argparse


def boolean_string(s: str):
    if s not in {'False', 'True', 'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True' or s == 'true'


parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, required=True, help='BERT配置文件路径')
parser.add_argument('--checkpoint_path', type=str, required=True, help='BERT权重路径')
parser.add_argument('--dict_path', type=str, required=True, help='词表路径')
parser.add_argument('--albert', default=False, type=boolean_string, required=False, help='是否使用Albert')

parser.add_argument('--train_data_path', type=str, required=True, help='训练集路径')
parser.add_argument('--val_data_path', type=str, required=True, help='验证集路径')
parser.add_argument('--sample_path', type=str, required=False, help='语料样例路径')

parser.add_argument('--epochs', default=5, type=int, required=False, help='训练循环')
parser.add_argument('--batch_size', default=8, type=int, required=False, help='训练batch_size')
parser.add_argument('--lr', default=1e-5, type=float, required=False, help='学习率')
parser.add_argument('--topk', default=2, type=int, required=False, help='解码TopK')
parser.add_argument('--max_input_len', default=256, type=int, required=False, help='最大输入长度')
parser.add_argument('--max_output_len', default=32, type=int, required=False, help='最大输出长度')

args = parser.parse_args()
print('args:\n' + args.__repr__())


def padding(x):
    """padding至batch内的最大长度
    """
    ml = max([len(i) for i in x])
    return np.array([i + [0] * (ml - len(i)) for i in x])


class DataGenerator(keras.utils.Sequence):

    # 对于所有数据输入，每个 epoch 取 dataSize 个数据
    # data 为 pandas iterator
    def __init__(self, data_path, batch_size=8):
        print("init")
        self.data_path = data_path
        data = pd.read_csv(data_path,
                           sep='\t',
                           header=None,
                           )
        self.batch_size = batch_size
        self.dataItor = data
        self.data = data.dropna().sample(frac=1)

    def __len__(self):
        # 计算每一个epoch的迭代次数
        return math.floor(len(self.data) / (self.batch_size)) - 1

    def __getitem__(self, index):
        # 生成每个batch数据
        batch = self.data[index * self.batch_size:(index + 1) * self.batch_size]

        # 生成数据
        x, y = self.data_generation(batch, index, len(self.data))
        return [x, y], None

    def on_epoch_end(self):
        # 在每一次epoch结束进行一次随机

        self.data = self.data.sample(frac=1)

    def data_generation(self, batch, index, lenth):
        batch_x = []
        batch_y = []
        for a, b in batch.iterrows():
            content_len = len(b[1])
            title_len = len(b[0])
            if (content_len + title_len > max_input_len):
                content = b[1][:max_input_len - title_len]
            else:
                content = b[1]
            x, s = tokenizer.encode(content, b[0])
            batch_x.append(x)
            batch_y.append(s)
        return padding(batch_x), padding(batch_y)


def get_model(config_path, checkpoint_path, keep_words, albert=False, lr=1e-5):
    if albert == True:
        print("Using Albert!")

    model = BERT(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        application='seq2seq',
        keep_words=keep_words,
        albert=albert
    )

    y_in = model.input[0][:, 1:]  # 目标tokens
    y_mask = model.input[1][:, 1:]
    y = model.output[:, :-1]  # 预测tokens，预测与目标错开一位

    # 交叉熵作为loss，并mask掉输入部分的预测
    cross_entropy = K.sparse_categorical_crossentropy(y_in, y)
    cross_entropy = K.sum(cross_entropy * y_mask) / K.sum(y_mask)

    model.add_loss(cross_entropy)
    model.compile(optimizer=Adam(lr))
    return model


def gen_sent(s, topk=2):
    """beam search解码
    每次只保留topk个最优候选结果；如果topk=1，那么就是贪心搜索
    """
    content_len = max_input_len - max_output_len
    token_ids, segment_ids = tokenizer.encode(s[:content_len])
    target_ids = [[] for _ in range(topk)]  # 候选答案id
    target_scores = [0] * topk  # 候选答案分数
    for i in range(max_output_len):  # 强制要求输出不超过max_output_len字
        _target_ids = [token_ids + t for t in target_ids]
        _segment_ids = [segment_ids + [1] * len(t) for t in target_ids]
        _probas = model.predict([_target_ids, _segment_ids
                                 ])[:, -1, 3:]  # 直接忽略[PAD], [UNK], [CLS]
        _log_probas = np.log(_probas + 1e-6)  # 取对数，方便计算
        _topk_arg = _log_probas.argsort(axis=1)[:, -topk:]  # 每一项选出topk
        _candidate_ids, _candidate_scores = [], []
        for j, (ids, sco) in enumerate(zip(target_ids, target_scores)):
            # 预测第一个字的时候，输入的topk事实上都是同一个，
            # 所以只需要看第一个，不需要遍历后面的。
            if i == 0 and j > 0:
                continue
            for k in _topk_arg[j]:
                _candidate_ids.append(ids + [k + 3])
                _candidate_scores.append(sco + _log_probas[j][k])
        _topk_arg = np.argsort(_candidate_scores)[-topk:]  # 从中选出新的topk
        target_ids = [_candidate_ids[k] for k in _topk_arg]
        target_scores = [_candidate_scores[k] for k in _topk_arg]
        best_one = np.argmax(target_scores)
        if target_ids[best_one][-1] == 3:
            return tokenizer.decode(target_ids[best_one])
    # 如果max_output_len字都找不到结束符，直接返回
    return tokenizer.decode(target_ids[np.argmax(target_scores)])


def just_show():
    if sample_path == None:
        return
    with open(sample_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            content = line.split('\t')[1].strip('\n')
            print(u'生成标题:', gen_sent(content))


class Evaluate(keras.callbacks.Callback):
    def __init__(self, val_data_path, topk):
        self.rouge = Rouge()
        self.smooth = SmoothingFunction().method1
        self.data = pd.read_csv(val_data_path, sep='\t', header=None, )
        self.lowest = 1e10
        self.topk = topk

    def on_epoch_end(self, epoch, logs=None):
        just_show()

        total = 0
        rouge_1, rouge_2, rouge_l, bleu = 0, 0, 0, 0

        for a, b in self.data.iterrows():
            total += 1
            generated_title = gen_sent(b[1], self.topk)
            real_title = b[0]
            real_title = " ".join(real_title)
            generated_title = " ".join(generated_title)
            scores = self.rouge.get_scores(generated_title, real_title)
            rouge_1 += scores[0]['rouge-1']['f']
            rouge_2 += scores[0]['rouge-2']['f']
            rouge_l += scores[0]['rouge-l']['f']
            bleu += sentence_bleu(references=[real_title.split(' ')],
                                  hypothesis=generated_title.split(' '),
                                  smoothing_function=self.smooth)

        rouge_1 /= total
        rouge_2 /= total
        rouge_l /= total
        bleu /= total
        output = {
            'rouge-1': rouge_1,
            'rouge-2': rouge_2,
            'rouge-l': rouge_l,
            'bleu': bleu,
        }
        print(output)


config_path = args.config_path
checkpoint_path = args.checkpoint_path
dict_path = args.dict_path
sample_path = args.sample_path

min_count = 0
max_input_len = args.max_input_len
max_output_len = args.max_output_len
batch_size = args.batch_size
epochs = args.epochs
topk = args.topk

train_data_path = args.train_data_path
val_data_path = args.val_data_path

_token_dict = load_vocab(dict_path)  # 读取词典
_tokenizer = Tokenizer(_token_dict, do_lower_case=True)  # 建立临时分词器


def read_texts():
    txts = [train_data_path, val_data_path]
    for txt in txts:
        lines = open(txt).readlines()
        for line in lines:
            d = line.split('\t')
            yield d[1][:max_input_len], d[0]


def _batch_texts():
    texts = []
    for text in read_texts():
        texts.extend(text)
        if len(texts) >= 1000:
            yield texts
            texts = []
    if texts:
        yield texts


def _tokenize_and_count(texts):
    _tokens = {}
    for text in texts:
        for token in _tokenizer.tokenize(text):
            _tokens[token] = _tokens.get(token, 0) + 1
    return _tokens


tokens = {}


def _total_count(result):
    for k, v in result.items():
        tokens[k] = tokens.get(k, 0) + v


# 词频统计
parallel_apply(
    func=_tokenize_and_count,
    iterable=tqdm(_batch_texts(), desc=u'构建词汇表中'),
    workers=10,
    max_queue_size=500,
    callback=_total_count,
)

tokens = [(i, j) for i, j in tokens.items() if j >= min_count]
tokens = sorted(tokens, key=lambda t: -t[1])
tokens = [t[0] for t in tokens]

token_dict, keep_words = {}, []  # keep_words是在bert中保留的字表

for t in ['[PAD]', '[UNK]', '[CLS]', '[SEP]']:
    token_dict[t] = len(token_dict)
    keep_words.append(_token_dict[t])

for t in tokens:
    if t in _token_dict and t not in token_dict:
        token_dict[t] = len(token_dict)
        keep_words.append(_token_dict[t])

tokenizer = Tokenizer(token_dict, do_lower_case=True)  # 建立分词器

rouge = Rouge()
model = get_model(config_path, checkpoint_path, keep_words, args.albert, args.lr)

evaluator = Evaluate(val_data_path, topk)

model.fit_generator(
    DataGenerator(train_data_path, batch_size),
    epochs=epochs,
    callbacks=[evaluator],
    verbose=1
)

import numpy as np
import pandas as pd
import json
import keras
from keras.layers import *
from keras_layer_normalization import LayerNormalization
from keras.models import Model
from rouge import Rouge
from keras import backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam
import argparse
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

parser = argparse.ArgumentParser()
parser.add_argument('--train_data_path', type=str, required=True, help='训练集路径')
parser.add_argument('--val_data_path', type=str, required=True, help='验证集路径')
parser.add_argument('--sample_path', type=str, required=False, help='语料样例路径')

parser.add_argument('--epochs', default=10, type=int, required=False, help='训练循环')
parser.add_argument('--batch_size', default=64, type=int, required=False, help='训练batch_size')
parser.add_argument('--lr', default=1e-3, type=float, required=False, help='学习率')
parser.add_argument('--topk', default=3, type=int, required=False, help='解码TopK')
parser.add_argument('--max_input_len', default=128, type=int, required=False, help='最大输入长度')
parser.add_argument('--max_output_len', default=64, type=int, required=False, help='最大输出长度')
args = parser.parse_args()

lr = args.lr
topk = args.topk

TRAIN_PATH = args.train_data_path
TEST_PATH = args.val_data_path
Sample_PATH = args.sample_path

maxlen = args.max_input_len
batch_size = args.batch_size
epochs = args.epochs

db = pd.read_csv(
    TRAIN_PATH, sep="\t", names=['title', 'content']
)

test = pd.read_csv(
    TEST_PATH, sep="\t", names=['title', 'content']
)

sample1 = pd.read_csv(
    Sample_PATH, sep="\t", names=['title', 'content']
)

min_count = 32
char_size = 128
z_dim = 128


def add_one(x):
    return x + 1


chars = {}
for a in db['title'].items():
    for w in a[1:2]:
        for q in w:
            chars[q] = chars.get(q, 0) + 1
            # print(chars[q])
for b in db['content'].items():
    for w in b[1:2]:
        for q in w:
            chars[q] = chars.get(q, 0) + 1

chars = {i: j for i, j in chars.items() if j >= min_count}
# 0: mask
# 1: unk
# 2: start
# 3: end
id2char = {i + 4: j for i, j in enumerate(chars)}
char2id = {j: i for i, j in id2char.items()}
json.dump([chars, id2char, char2id], open('seq2seq_config.json', 'w'))


def str2id(s, start_end=False):
    # 文字转整数id
    if start_end:  # 补上<start>和<end>标记
        ids = [char2id.get(c, 1) for c in s[:maxlen - 2]]
        ids = [2] + ids + [3]
    else:  # 普通转化
        ids = [char2id.get(c, 1) for c in s[:maxlen]]
    return ids


def id2str(ids):
    # id转文字，找不到的用空字符代替
    return ''.join([id2char.get(i, '') for i in ids])


def padding(x):
    # padding至batch内的最大长度
    ml = max([len(i) for i in x])
    return [i + [0] * (ml - len(i)) for i in x]


def data_generator():
    # 数据生成器
    X, Y = [], []
    while True:
        for a, b in zip(db['content'].items(), db['title'].items()):
            # print(str2id(a['交换超立方体网络容错路由研究'], start_end=True))
            X.append(str2id(a[1]))
            Y.append(str2id(b[1], start_end=True))
            if len(X) == batch_size:
                X = np.array(padding(X))
                Y = np.array(padding(Y))
                yield [X, Y], None
                X, Y = [], []


def to_one_hot(x):
    """输出一个词表大小的向量，来标记该词是否在文章出现过
    """
    x, x_mask = x
    x = K.cast(x, 'int32')
    x = K.one_hot(x, len(chars) + 4)
    x = K.sum(x_mask * x, 1, keepdims=True)
    x = K.cast(K.greater(x, 0.5), 'float32')
    return x


chars1 = {}
for a in test['title'].items():
    for w in a[1]:
        chars1[w] = chars1.get(w, 0) + 1
for b in test['content'].items():
    for w in b[1]:
        chars1[w] = chars1.get(w, 0) + 1
chars1 = {i: j for i, j in chars1.items() if j >= min_count}
# 0: mask
# 1: unk
# 2: start
# 3: end
id2char1 = {i + 4: j for i, j in enumerate(chars1)}
char2id1 = {j: i for i, j in id2char1.items()}


def str2id1(s, start_end=False):
    # 文字转整数id
    if start_end:  # 补上<start>和<end>标记
        ids = [char2id1.get(c, 1) for c in s[:maxlen - 2]]
        ids = [2] + ids + [3]
    else:  # 普通转化
        ids = [char2id1.get(c, 1) for c in s[:maxlen]]
    return ids


def id2str1(ids):
    # id转文字，找不到的用空字符代替
    return ''.join([id2char1.get(i, '') for i in ids])


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


def fk():
    testX, testY = [], []
    while True:
        for a, b in zip(test['content'].items(), test['title'].items()):
            testX.append(str2id1(a[1]))
            testY.append(str2id1(b[1], start_end=True))
            if len(testX) == batch_size:
                testX = np.array(padding(testX))
                testY = np.array(padding(testY))
                yield [testX, testY], None
                testX, testY = [], []


class ScaleShift(Layer):
    """缩放平移变换层（Scale and shift）
    """

    def __init__(self, **kwargs):
        super(ScaleShift, self).__init__(**kwargs)

    def build(self, input_shape):
        kernel_shape = (1,) * (len(input_shape) - 1) + (input_shape[-1],)
        self.log_scale = self.add_weight(name='log_scale',
                                         shape=kernel_shape,
                                         initializer='zeros')
        self.shift = self.add_weight(name='shift',
                                     shape=kernel_shape,
                                     initializer='zeros')

    def call(self, inputs):
        x_outs = K.exp(self.log_scale) * inputs + self.shift
        return x_outs


class OurLayer(Layer):
    """定义新的Layer，增加reuse方法，允许在定义Layer时调用现成的层
    """

    def reuse(self, layer, *args, **kwargs):
        if not layer.built:
            if len(args) > 0:
                inputs = args[0]
            else:
                inputs = kwargs['inputs']
            if isinstance(inputs, list):
                input_shape = [K.int_shape(x) for x in inputs]
            else:
                input_shape = K.int_shape(inputs)
            layer.build(input_shape)
        outputs = layer.call(*args, **kwargs)
        if not keras.__version__.startswith('2.3.'):
            for w in layer.trainable_weights:
                if w not in self._trainable_weights:
                    self._trainable_weights.append(w)
            for w in layer.non_trainable_weights:
                if w not in self._non_trainable_weights:
                    self._non_trainable_weights.append(w)
            for u in layer.updates:
                if not hasattr(self, '_updates'):
                    self._updates = []
                if u not in self._updates:
                    self._updates.append(u)
        return outputs


class OurBidirectional(OurLayer):
    """自己封装双向RNN，允许传入mask，保证对齐
    """

    def __init__(self, layer, **args):
        super(OurBidirectional, self).__init__(**args)
        self.forward_layer = layer.__class__.from_config(layer.get_config())
        self.backward_layer = layer.__class__.from_config(layer.get_config())
        self.forward_layer.name = 'forward_' + self.forward_layer.name
        self.backward_layer.name = 'backward_' + self.backward_layer.name

    def reverse_sequence(self, x, mask):
        """这里的mask.shape是[batch_size, seq_len, 1]
        """
        seq_len = K.round(K.sum(mask, 1)[:, 0])
        seq_len = K.cast(seq_len, 'int32')
        return tf.reverse_sequence(x, seq_len, seq_dim=1)

    def call(self, inputs):
        x, mask = inputs
        x_forward = self.reuse(self.forward_layer, x)
        x_backward = self.reverse_sequence(x, mask)
        x_backward = self.reuse(self.backward_layer, x_backward)
        x_backward = self.reverse_sequence(x_backward, mask)
        x = K.concatenate([x_forward, x_backward], -1)
        if K.ndim(x) == 3:
            return x * mask
        else:
            return x

    def compute_output_shape(self, input_shape):
        return input_shape[0][:-1] + (self.forward_layer.units * 2,)


import tensorflow as tf

tf.reverse_sequence


def seq_avgpool(x):
    """seq是[None, seq_len, s_size]的格式，
    mask是[None, seq_len, 1]的格式，先除去mask部分，
    然后再做avgpooling。
    """
    seq, mask = x
    return K.sum(seq * mask, 1) / (K.sum(mask, 1) + 1e-6)


def seq_maxpool(x):
    """seq是[None, seq_len, s_size]的格式，
    mask是[None, seq_len, 1]的格式，先除去mask部分，
    然后再做maxpooling。
    """
    seq, mask = x
    seq -= (1 - mask) * 1e10
    return K.max(seq, 1)


class SelfModulatedLayerNormalization(OurLayer):
    """模仿Self-Modulated Batch Normalization，
    只不过将Batch Normalization改为Layer Normalization
    """

    def __init__(self, num_hidden, **kwargs):
        super(SelfModulatedLayerNormalization, self).__init__(**kwargs)
        self.num_hidden = num_hidden

    def build(self, input_shape):
        super(SelfModulatedLayerNormalization, self).build(input_shape)
        output_dim = input_shape[0][-1]
        self.layernorm = LayerNormalization(center=False, scale=False)
        self.beta_dense_1 = Dense(self.num_hidden, activation='relu')
        self.beta_dense_2 = Dense(output_dim)
        self.gamma_dense_1 = Dense(self.num_hidden, activation='relu')
        self.gamma_dense_2 = Dense(output_dim)

    def call(self, inputs):
        inputs, cond = inputs
        inputs = self.reuse(self.layernorm, inputs)
        beta = self.reuse(self.beta_dense_1, cond)
        beta = self.reuse(self.beta_dense_2, beta)
        gamma = self.reuse(self.gamma_dense_1, cond)
        gamma = self.reuse(self.gamma_dense_2, gamma)
        for _ in range(K.ndim(inputs) - K.ndim(cond)):
            beta = K.expand_dims(beta, 1)
            gamma = K.expand_dims(gamma, 1)
        return inputs * (gamma + 1) + beta

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class Attention(OurLayer):
    """多头注意力机制
    """

    def __init__(self, heads, size_per_head, key_size=None,
                 mask_right=False, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.heads = heads
        self.size_per_head = size_per_head
        self.out_dim = heads * size_per_head
        self.key_size = key_size if key_size else size_per_head
        self.mask_right = mask_right

    def build(self, input_shape):
        super(Attention, self).build(input_shape)
        self.q_dense = Dense(self.key_size * self.heads, use_bias=False)
        self.k_dense = Dense(self.key_size * self.heads, use_bias=False)
        self.v_dense = Dense(self.out_dim, use_bias=False)

    def mask(self, x, mask, mode='mul'):
        if mask is None:
            return x
        else:
            for _ in range(K.ndim(x) - K.ndim(mask)):
                mask = K.expand_dims(mask, K.ndim(mask))
            if mode == 'mul':
                return x * mask
            else:
                return x - (1 - mask) * 1e10

    def call(self, inputs):
        q, k, v = inputs[:3]
        v_mask, q_mask = None, None
        if len(inputs) > 3:
            v_mask = inputs[3]
            if len(inputs) > 4:
                q_mask = inputs[4]
        # 线性变换
        qw = self.reuse(self.q_dense, q)
        kw = self.reuse(self.k_dense, k)
        vw = self.reuse(self.v_dense, v)
        # 形状变换
        qw = K.reshape(qw, (-1, K.shape(qw)[1], self.heads, self.key_size))
        kw = K.reshape(kw, (-1, K.shape(kw)[1], self.heads, self.key_size))
        vw = K.reshape(vw, (-1, K.shape(vw)[1], self.heads, self.size_per_head))
        # 维度置换
        qw = K.permute_dimensions(qw, (0, 2, 1, 3))
        kw = K.permute_dimensions(kw, (0, 2, 1, 3))
        vw = K.permute_dimensions(vw, (0, 2, 1, 3))
        # Attention
        a = tf.einsum('ijkl,ijml->ijkm', qw, kw) / self.key_size ** 0.5
        a = K.permute_dimensions(a, (0, 3, 2, 1))
        a = self.mask(a, v_mask, 'add')
        a = K.permute_dimensions(a, (0, 3, 2, 1))
        if self.mask_right:
            ones = K.ones_like(a[:1, :1])
            mask = (ones - K.tf.matrix_band_part(ones, -1, 0)) * 1e10
            a = a - mask
        a = K.softmax(a)
        # 完成输出
        o = tf.einsum('ijkl,ijlm->ijkm', a, vw)
        o = K.permute_dimensions(o, (0, 2, 1, 3))
        o = K.reshape(o, (-1, K.shape(o)[1], self.out_dim))
        o = self.mask(o, q_mask, 'mul')
        return o

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.out_dim)


# 搭建seq2seq模型

x_in = Input(shape=(None,))
y_in = Input(shape=(None,))
x, y = x_in, y_in

x_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(x)
y_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(y)

x_one_hot = Lambda(to_one_hot)([x, x_mask])
x_prior = ScaleShift()(x_one_hot)  # 学习输出的先验分布（标题的字词很可能在文章出现过）

embedding = Embedding(len(chars) + 4, char_size)
x = embedding(x)
y = embedding(y)

# encoder，双层双向LSTM
x = LayerNormalization()(x)
x = OurBidirectional(CuDNNLSTM(z_dim // 2, return_sequences=True))([x, x_mask])
x = LayerNormalization()(x)
x = OurBidirectional(CuDNNLSTM(z_dim // 2, return_sequences=True))([x, x_mask])
x_max = Lambda(seq_maxpool)([x, x_mask])

# decoder，双层单向LSTM
y = SelfModulatedLayerNormalization(z_dim // 4)([y, x_max])
y = CuDNNLSTM(z_dim, return_sequences=True)(y)
y = SelfModulatedLayerNormalization(z_dim // 4)([y, x_max])
y = CuDNNLSTM(z_dim, return_sequences=True)(y)
y = SelfModulatedLayerNormalization(z_dim // 4)([y, x_max])

# attention交互
xy = Attention(8, 16)([y, x, x, x_mask])
xy = Concatenate()([y, xy])

# 输出分类
xy = Dense(char_size)(xy)
xy = LeakyReLU(0.2)(xy)
xy = Dense(len(chars) + 4)(xy)
xy = Lambda(lambda x: (x[0] + x[1]) / 2)([xy, x_prior])  # 与先验结果平均
xy = Activation('softmax')(xy)

# 交叉熵作为loss，但mask掉padding部分
cross_entropy = K.sparse_categorical_crossentropy(y_in[:, 1:], xy[:, :-1])
cross_entropy = K.sum(cross_entropy * y_mask[:, 1:, 0]) / K.sum(y_mask[:, 1:, 0])

model = Model([x_in, y_in], xy)
model.add_loss(cross_entropy)
model.compile(optimizer=Adam(1e-3))


def gen_sent(s, topk=3, maxlen=64):
    """beam search解码
    每次只保留topk个最优候选结果；如果topk=1，那么就是贪心搜索
    """
    xid = np.array([str2id(s)] * topk)  # 输入转id
    yid = np.array([[2]] * topk)  # 解码均以<start>开头，这里<start>的id为2
    scores = [0] * topk  # 候选答案分数
    for i in range(maxlen):  # 强制要求输出不超过maxlen字
        proba = model.predict([xid, yid])[:, i, 3:]  # 直接忽略<padding>、<unk>、<start>
        log_proba = np.log(proba + 1e-6)  # 取对数，方便计算
        arg_topk = log_proba.argsort(axis=1)[:, -topk:]  # 每一项选出topk
        _yid = []  # 暂存的候选目标序列
        _scores = []  # 暂存的候选目标序列得分
        if i == 0:
            for j in range(topk):
                _yid.append(list(yid[j]) + [arg_topk[0][j] + 3])
                _scores.append(scores[j] + log_proba[0][arg_topk[0][j]])
        else:
            for j in range(topk):
                for k in range(topk):  # 遍历topk*topk的组合
                    _yid.append(list(yid[j]) + [arg_topk[j][k] + 3])
                    _scores.append(scores[j] + log_proba[j][arg_topk[j][k]])
            _arg_topk = np.argsort(_scores)[-topk:]  # 从中选出新的topk
            _yid = [_yid[k] for k in _arg_topk]
            _scores = [_scores[k] for k in _arg_topk]
        yid = np.array(_yid)
        scores = np.array(_scores)
        best_one = np.argmax(scores)
        if yid[best_one][-1] == 3:
            return id2str(yid[best_one])
    # 如果maxlen字都找不到<end>，直接返回
    return id2str(yid[np.argmax(scores)])


smooth = SmoothingFunction()


# s1 = u'夏天来临，皮肤在强烈紫外线的照射下，晒伤不可避免，因此，晒后及时修复显得尤为重要，否则可能会造成长期伤害。专家表示，选择晒后护肤品要慎重，芦荟凝胶是最安全，有效的一种选择，晒伤严重者，还请及时就医 。'
# s2 = u'8月28日，网络爆料称，华住集团旗下连锁酒店用户数据疑似发生泄露。从卖家发布的内容看，数据包含华住旗下汉庭、禧玥、桔子、宜必思等10余个品牌酒店的住客信息。泄露的信息包括华住官网注册资料、酒店入住登记的身份信息及酒店开房记录，住客姓名、手机号、邮箱、身份证号、登录账号密码等。卖家对这个约5亿条数据打包出售。第三方安全平台威胁猎人对信息出售者提供的三万条数据进行验证，认为数据真实性非常高。当天下午，华住集 团发声明称，已在内部迅速开展核查，并第一时间报警。当晚，上海警方消息称，接到华住集团报案，警方已经介入调查。'
# sp = u'针对现有的软件众包工人选择机制对工人间协同开发考虑不足的问题,在竞标模式的基础上提出一种基于活跃时间分组的软件众包工人选择机制。首先,基于活跃时间将众包工人划分为多个协同开发组;然后,根据组内工人开发能力和协同因子计算协同工作组权重;最后,选定权重最大的协同工作组为最优工作组,并根据模块复杂度为每个任务模块从该组内选择最适合的工人。实验结果表明,该机制相比能力优先选择方法在工人平均能力上仅有0. 57%的差距,同时因为保证了工人间的协同而使项目风险平均降低了32%,能有效指导需多人协同进行的众包软件任务的工人选择。'

class Evaluate(Callback):
    def __init__(self):
        self.data = pd.read_csv(TEST_PATH, sep='\t', header=None, )
        self.lowest = 1e10
        self.lowest1 = 1e10

    def on_epoch_end(self, epoch, logs=None):
        # 训练过程中观察一两个例子，显示标题质量提高的过程
        for a, b in sample1.iterrows():
            print('生成标题: ', gen_sent(b['title']))
        # 保存最优结果
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            model.save_weights('./best_model.weights')
        if logs['val_loss'] <= self.lowest1:
            self.lowest1 = logs['val_loss']
        rouge_scoresl = []
        rouge_scores1 = []
        rouge_scores2 = []
        bleu_scores = []
        for a, b in self.data.iterrows():
            generated_title = gen_sent(b[1], 3)
            real_title = b[0]

            token_title = " ".join(str(c) for c in real_title[:maxlen])
            token_gen_title = " ".join(str(c) for c in generated_title[:maxlen])
            rouge_score = rouge.get_scores(token_gen_title, token_title)
            rouge_scoresl.append(rouge_score[0]['rouge-l']['f'])
            rouge_scores1.append(rouge_score[0]['rouge-1']['f'])
            rouge_scores2.append(rouge_score[0]['rouge-2']['f'])
            bleu_scores.append(sentence_bleu(references=[token_title.split(' ')],
                                             hypothesis=token_gen_title.split(' '),
                                             smoothing_function=smooth.method1))
        print("rouge-l scores: ", np.mean(rouge_scoresl))
        print("rouge-1 scores: ", np.mean(rouge_scores1))
        print("rouge-2 scores: ", np.mean(rouge_scores2))
        print(" bleu   scores: ", np.mean(bleu_scores))


evaluator = Evaluate()
rouge = Rouge()
history = model.fit_generator(data_generator(),
                              steps_per_epoch=1000,
                              epochs=epochs,
                              validation_data=fk(),
                              validation_steps=20,
                              callbacks=[evaluator]
                              )

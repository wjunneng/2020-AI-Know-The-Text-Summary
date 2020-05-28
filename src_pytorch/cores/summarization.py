#! -*- coding: utf-8 -*-
import os
import sys

sys.path.append('/home/wjunneng/Ubuntu/2020-AI-Know-The-Text-Summary')
os.chdir(sys.path[0])

import time
import torch
from torch.utils.tensorboard import SummaryWriter

import pandas as pd
import os, json, codecs, logging, random
from tqdm import trange, tqdm
import numpy as np
import argparse
from src_pytorch.cores.UniLM import UniLMModel
from src_pytorch.cores.tokenizer import SimpleTokenizer, load_vocab
from src_pytorch.cores.configuration_bert import BertConfig

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


def read_text(max_input_len, max_output_len):
    df = pd.read_csv('../../data/input/train.csv')
    text = df['article'].values
    summarization = df['summarization'].values

    for t, s in zip(text, summarization):
        if len(s) <= max_output_len:
            yield t[:max_input_len], s


def padding(x):
    """padding至batch内的最大长度
    """
    ml = max([len(i) for i in x])
    return np.array([i + [0] * (ml - len(i)) for i in x])


def data_generator(tokenizer, batch_size=4, max_output_len=32, max_input_len=460):
    while True:
        X, S = [], []
        for a, b in read_text(max_input_len, max_output_len):
            # x为text和summaryzation融合信息，s为融合后的segment ids
            x, s = tokenizer.encode(a, b)
            X.append(x)
            S.append(s)
            if len(X) == batch_size:
                X = padding(X)
                S = padding(S)
                yield [X, S], None
                X, S = [], []


def gen_sent(s, tokenizer, model, args):
    """beam search解码
    每次只保留topk个最优候选结果；如果topk=1，那么就是贪心搜索
    """
    topk = args.topk
    token_ids, segment_ids = tokenizer.encode(s[:args.max_input_len])
    # 候选答案id
    target_ids = [[] for _ in range(topk)]
    # 候选答案分数
    target_scores = [0] * topk
    # 强制要求输出不超过max_output_len字
    model.eval()

    with torch.no_grad():
        for i in range(args.max_output_len):
            _target_ids = [token_ids + t for t in target_ids]
            _segment_ids = [segment_ids + [1] * len(t) for t in target_ids]

            input_ids = torch.tensor(_target_ids, dtype=torch.long).to(args.device)
            token_type_ids = torch.tensor(_segment_ids, dtype=torch.long).to(args.device)

            outputs = model(input_ids, catcu_lss=False, token_type_ids=token_type_ids)

            _probas = outputs[0][:, -1, :]
            # 取对数，方便计算
            # _log_probas = np.log(_probas + 1e-6)
            _log_probas = _probas.cpu().numpy()
            # 每一项选出topk
            _topk_arg = _log_probas.argsort(axis=1)[:, -topk:]
            _candidate_ids, _candidate_scores = [], []
            for j, (ids, sco) in enumerate(zip(target_ids, target_scores)):
                # 预测第一个字的时候，输入的topk事实上都是同一个，
                # 所以只需要看第一个，不需要遍历后面的。
                if i == 0 and j > 0:
                    continue
                for k in _topk_arg[j]:
                    _candidate_ids.append(ids + [k])
                    _candidate_scores.append(sco + _log_probas[j][k])
            _topk_arg = np.argsort(_candidate_scores)[-topk:]
            for j, k in enumerate(_topk_arg):
                # target_ids[j].append(_candidate_ids[k][-1])
                target_ids[j] = _candidate_ids[k]
                target_scores[j] = _candidate_scores[k]
            ends = [j for j, k in enumerate(target_ids) if k[-1] == 200]
            if len(ends) > 0:
                k = np.argmax([target_scores[j] for j in ends])
                return tokenizer.decode(target_ids[ends[k]])

    # 如果max_output_len字都找不到结束符，直接返回
    return tokenizer.decode(target_ids[np.argmax(target_scores)])


def main():
    parser = argparse.ArgumentParser()

    # parameters
    parser.add_argument("--data_dir", default='../../data/input/train.csv', type=str, required=False, )
    parser.add_argument("--model_name_or_path", default='../../data/chinese_wwm_pytorch', type=str, required=False, )
    parser.add_argument("--output_dir", default='../../data/output', type=str, required=False, )

    # Other parameters
    parser.add_argument("--cache_dir", default="", type=str, help="cache dir")
    parser.add_argument("--max_input_len", default=460, type=int, help="文本最长输入长度")
    parser.add_argument("--max_output_len", default=32, type=int, help="最长输出摘要长度")
    parser.add_argument("--cut_vocab", default=True, action="store_true", help="是否精简原字表")
    parser.add_argument("--min_count", default=30, type=int, help="精简掉出现频率少于此的word")
    parser.add_argument("--topk", default=1, type=int, help="beam search参数")
    parser.add_argument("--topp", default=0., type=float, help="核采样参数")
    parser.add_argument("--do_train", default=True, action="store_true", help="是否fine tuning")
    parser.add_argument("--do_show", default=True, action="store_true", help="是否进行预测")
    parser.add_argument("--batch_size", default=2, type=int, help="批量大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, )
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="学习率")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="学习率衰减")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="衰减率")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="梯度裁减值")
    parser.add_argument("--num_train_epochs", default=3, type=int, help="训练epochs次数", )
    parser.add_argument("--warmup_steps", default=0, type=int, help="学习率线性预热步数")
    parser.add_argument("--logging_steps", type=int, default=1000, help="每多少步打印日志")
    parser.add_argument("--seed", type=int, default=42, help="初始化随机种子")
    parser.add_argument("--max_steps", default=100000, type=int, help="训练的总步数", )
    parser.add_argument("--save_steps", default=50000, type=int, help="保存的间隔steps", )

    args = parser.parse_args()
    args.do_train = True
    args.do_show = True
    args.overwrite_output_dir = args.output_dir

    if (
            os.path.exists(args.output_dir) and os.listdir(args.output_dir)
            and args.do_train and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir)
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Set seed
    set_seed(args)

    # 建立分词器
    _token_dict = load_vocab(os.path.join(args.model_name_or_path, 'vocab.txt'))
    # keep_words是在bert中保留的字表
    token_dict, keep_words = {}, []
    if args.cut_vocab:
        if os.path.exists('./seq2seq_config.json'):
            chars = json.load(open('./seq2seq_config.json', encoding='utf-8'))
        else:
            chars = {}
            for a in tqdm(read_text(args.max_input_len, args.max_output_len), desc='构建字表中'):
                for b in a:
                    for w in b:
                        chars[w] = chars.get(w, 0) + 1
            chars = [(i, j) for i, j in chars.items() if j >= args.min_count]
            chars = sorted(chars, key=lambda c: - c[1])
            chars = [c[0] for c in chars]
            json.dump(
                chars,
                codecs.open('./seq2seq_config.json', 'w', encoding='utf-8'),
                indent=4,
                ensure_ascii=False
            )
        for c in ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[unused1]']:
            token_dict[c] = len(token_dict)
            keep_words.append(_token_dict[c])

        for c in chars:
            if c in _token_dict:
                token_dict[c] = len(token_dict)
                keep_words.append(_token_dict[c])

    tokenizer = SimpleTokenizer(token_dict if args.cut_vocab else _token_dict)

    config = BertConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=[0, 1],
        finetuning_task='unilm',
        cache_dir=None,
    )
    config.keep_words = keep_words
    model = UniLMModel.from_pretrained(
        args.model_name_or_path,
        config=config,
        cache_dir=None,
    )
    model.to(args.device)
    # 精简词表
    if args.cut_vocab:
        model.resize_token_embeddings(new_num_tokens=len(keep_words), keep_words=keep_words)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        t_total = args.max_steps
        tb_writer = SummaryWriter('./tensorboardX')
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)

        train_epochs = trange(args.num_train_epochs, desc='开始训练epoch')
        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()
        for epoch in train_epochs:
            for step, batch in enumerate(data_generator(tokenizer,
                                                        batch_size=args.batch_size,
                                                        max_output_len=args.max_output_len,
                                                        max_input_len=args.max_input_len)):
                model.train()

                input_ids = torch.tensor(batch[0][0], dtype=torch.long).to(device)
                token_type_ids = torch.tensor(batch[0][1], dtype=torch.long).to(device)

                outputs = model(input_ids, token_type_ids=token_type_ids)
                loss = outputs[0]

                # 只计算摘要部分的输出loss
                y_mask = token_type_ids[:, 1:].reshape(-1).contiguous()
                loss = torch.sum(loss * y_mask) / torch.sum(y_mask)

                loss.backward()
                tr_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)  # 梯度截断

                    optimizer.step()
                    model.zero_grad()
                    global_step += 1

                    if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        logs = {}
                        loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                        logs["loss损失"] = loss_scalar
                        logging_loss = tr_loss

                        for key, value in logs.items():
                            tb_writer.add_scalar(key, value, global_step)
                        print(json.dumps({**logs, **{"step": global_step}}))

                    if args.save_steps > 0 and global_step % args.save_steps == 0:
                        output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )
                        model_to_save.save_pretrained(output_dir)

                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        logger.info("Saving model checkpoint to %s", output_dir)

                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        logger.info("Saving optimizer states to %s", output_dir)

                if args.max_steps > 0 and global_step > t_total:
                    break

            if args.max_steps > 0 and global_step > t_total:
                train_epochs.close()
                break
        tb_writer.close()
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss / global_step)

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )
        model_to_save.save_pretrained(args.output_dir)

        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
        logger.info("Saving model checkpoint to %s", args.output_dir)

        torch.save(optimizer.state_dict(), os.path.join(args.output_dir, "optimizer.pt"))
        logger.info("Saving optimizer states to %s", args.output_dir)

    if args.do_show:
        config = BertConfig.from_pretrained(
            args.output_dir,
            num_labels=[0, 1],
            finetuning_task='unilm',
        )
        # config.keep_words = keep_words
        # config.vocab_size = len(keep_words)
        model = UniLMModel.from_pretrained(args.output_dir, config=config, cache_dir='../../data/output')
        model.to(args.device)

        test_data = pd.read_csv('../../data/input/test.csv', encoding='utf-8')

        result = []

        index = 0
        print('length: {}'.format(test_data.shape[0]))
        for s in tqdm(test_data['article'].values):
            index += 1
            current_result = gen_sent(s, tokenizer, model, args)
            print(current_result)
            result.append(current_result)

        pd.DataFrame(result).to_csv('../../data/output/result.csv', header=None)


if __name__ == '__main__':
    main()

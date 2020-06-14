# -*- coding:utf-8 -*-
from __future__ import absolute_import, division, print_function
import os
import sys
import pathlib

parent_dir = pathlib.Path(os.path.abspath(__file__)).parent.parent.parent

sys.path.append(parent_dir)
os.chdir(sys.path[0])

data_dir = os.path.join(parent_dir, 'data')
input_dir = os.path.join(data_dir, 'input')
output_dir = os.path.join(data_dir, 'output')

import pandas as pd
# coding:utf8
from rouge import Rouge
from tqdm import trange


class EDA(object):
    def __init__(self, index):
        self.train_csv_path = os.path.join(input_dir, 'train.csv')
        self.test_csv_path = os.path.join(input_dir, 'test.csv')
        self.result_txt_path = os.path.join(output_dir, 'result.csv')
        self.result_match_txt_path = os.path.join(output_dir, 'result_match_' + str(index) + '.csv')
        self.index = index * 1000

    def calculate(self, a, b):
        """
        f:F1值  p：查准率  R：召回率
        """
        rouge = Rouge()
        rouge_score = rouge.get_scores(a, b)
        return rouge_score[0]["rouge-2"]['f']

    def eda(self):
        print('index: {}'.format(self.index))
        # article,summarization
        train_data = pd.read_csv(filepath_or_buffer=self.train_csv_path, encoding='utf-8')
        # article
        test_data = pd.read_csv(filepath_or_buffer=self.test_csv_path, encoding='utf-8')
        # result
        result_data = pd.read_csv(filepath_or_buffer=self.result_txt_path, encoding='utf-8', header=None)
        result_data.columns = ['indexes', 'summarization']

        count = 0
        for test_index in trange(self.index, self.index + 1000):
            max_rouge_2 = -1
            article = test_data.iloc[test_index, 0]
            article = article.replace('\t', ' ')
            article = article.replace('\n', ' ')
            # article = article[:256]
            most_similar_summarization = ''

            for train_index in range(train_data.shape[0]):
                current_article = train_data.iloc[train_index, 0]
                current_summarization = train_data.iloc[train_index, 1]

                current_article = current_article.replace('\t', ' ')
                current_article = current_article.replace('\n', ' ')
                # current_article = current_article[:256]

                current_rouge_2 = self.calculate(a=article, b=current_article)

                if current_rouge_2 > max_rouge_2:
                    max_rouge_2 = current_rouge_2
                    most_similar_summarization = current_summarization

            if max_rouge_2 > 0.8:
                count += 1
                result_data.iloc[test_index, 1] = most_similar_summarization

        print('count: {}'.format(count))
        result_data[self.index:self.index + 1000].to_csv(self.result_match_txt_path, encoding='utf-8', index=None,
                                                         header=None)


if __name__ == '__main__':
    # ######################## 256 ########################
    # 56
    # eda = EDA(index=0)
    # 37
    # eda = EDA(index=1)
    # 39
    # eda = EDA(index=2)
    # 37
    # eda = EDA(index=3)
    # 44
    # eda = EDA(index=4)
    # 40
    # eda = EDA(index=5)
    # 37
    # eda = EDA(index=6)
    # 42
    # eda = EDA(index=7)
    # 35
    # eda = EDA(index=8)
    # 46
    # eda = EDA(index=9)

    # ######################## 全 ########################
    #
    # eda = EDA(index=0)
    #
    # eda = EDA(index=1)
    #
    # eda = EDA(index=2)
    #
    # eda = EDA(index=3)
    #
    # eda = EDA(index=4)
    #
    # eda = EDA(index=5)
    #
    # eda = EDA(index=6)
    #
    # eda = EDA(index=7)
    #
    # eda = EDA(index=8)
    #
    eda = EDA(index=9)

    eda.eda()

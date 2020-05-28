import os
import sys
import re
import pandas as pd

from sklearn.model_selection import train_test_split


class Utils(object):
    @staticmethod
    def deal_text(text: str):
        """
        处理文本
        """
        text = text.replace('<Paragraph>', ' ')

        for special_char in ['■', '●', '\t', '\n']:
            text = text.replace(special_char, ' ')

        for (ch_char, en_char) in zip(['０', '１', '２', '３', '４', '５', '６', '７', '８', '９'],
                                      ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']):
            text = text.replace(ch_char, en_char)

        return text

    @staticmethod
    def generate_train_val_test(input_train_csv, input_test_csv, output_train_csv, output_val_csv, output_test_csv):
        """
        生成训练/测试/验证集
        """
        # article, summarization
        input_train = pd.read_csv(input_train_csv, encoding='utf-8')
        # article
        input_test = pd.read_csv(input_test_csv, encoding='utf-8')

        input_train['article'] = input_train['article'].apply(lambda a: Utils.deal_text(a))
        input_train['summarization'] = input_train['summarization'].apply(lambda a: Utils.deal_text(a))

        input_test['article'] = input_test['article'].apply(lambda a: Utils.deal_text(a))

        X_train, X_val, y_train, y_val = train_test_split(input_train['article'], input_train['summarization'],
                                                          test_size=0.2, random_state=42)

        train = pd.DataFrame({'article': X_train, 'summarization': y_train})
        val = pd.DataFrame({'article': X_val, 'summarization': y_val})

        train.to_csv(output_train_csv, index=None, encoding='utf-8')
        val.to_csv(output_val_csv, index=None, encoding='utf-8')
        input_test.to_csv(output_test_csv, index=None, encoding='utf-8')


if __name__ == '__main__':
    input_train_csv = '../../data/input/train.csv'
    input_test_csv = '../../data/input/test.csv'

    output_train_csv = '../../data/output/train.csv'
    output_val_csv = '../../data/output/val.csv'
    output_test_csv = '../../data/output/test.csv'

    Utils.generate_train_val_test(input_train_csv=input_train_csv,
                                  input_test_csv=input_test_csv,
                                  output_train_csv=output_train_csv,
                                  output_val_csv=output_val_csv,
                                  output_test_csv=output_test_csv)

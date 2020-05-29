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
        for special_char in ['<Paragraph>', '■', '●', '\n', '\t']:
            while special_char in text:
                text = text.replace(special_char, ' ')

        for (ch_char, en_char) in zip(['０', '１', '２', '３', '４', '５', '６', '７', '８', '９'],
                                      ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']):
            while ch_char in text:
                text = text.replace(ch_char, en_char)

        # 关键
        text = text.strip()

        return text

    @staticmethod
    def delete_date(text: str):
        """
        删除日期
        """
        post_time = re.search(
            '(20\d{2}([\.\-/|年月\s]{1,3}\d{1,2}){2}日?(\s?\d{2}:\d{2}(:\d{2})?)?)|(\d{1,2}\s?(分钟|小时|天)前)', text)

        while post_time:
            post_time = re.search(
                '(20\d{2}([\.\-/|年月\s]{1,3}\d{1,2}){2}日?(\s?\d{2}:\d{2}(:\d{2})?)?)|(\d{1,2}\s?(分钟|小时|天)前)', text)
            if post_time is not None:
                text = text.replace(post_time.group(), '')
            else:
                break

        if len(text) < 510:
            return text
        else:
            return text[:300] + text[-210:]

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

    @staticmethod
    def generate_train_src_tgt(input_train_csv, input_test_csv, output_train_src_csv, output_train_tgt_csv,
                               output_test_csv):
        """
        生成训练/测试/验证集
        """
        # article, summarization
        input_train = pd.read_csv(input_train_csv, encoding='utf-8')
        # article
        input_test = pd.read_csv(input_test_csv, encoding='utf-8')

        input_train['article'] = input_train['article'].apply(lambda a: Utils.deal_text(a))
        input_test['article'] = input_test['article'].apply(lambda a: Utils.deal_text(a))
        input_train['summarization'] = input_train['summarization'].apply(lambda a: Utils.deal_text(a))

        input_train['article'] = input_train['article'].apply(lambda a: Utils.delete_date(a))
        input_test['article'] = input_test['article'].apply(lambda a: Utils.delete_date(a))

        input_train.dropna(inplace=True)

        input_train['article'].to_csv(output_train_src_csv, index=None, header=None, encoding='utf-8')
        input_train['summarization'].to_csv(output_train_tgt_csv, index=None, header=None, encoding='utf-8')
        input_test.to_csv(output_test_csv, index=None, header=None, encoding='utf-8')

    @staticmethod
    def delete_date_sub(input_sub_csv, output_sub_csv):
        data = pd.read_csv(input_sub_csv, encoding='utf-8', header=None)
        data.columns = ['indexs', 'summarization']

        # time_test = ['2017年09月25日', '2017-09-25', '2017/09/25', '2017 09 25', '2017年09月25日19:13:33',
        #              '2017-09-25 19:13:40', '1小时前', '1 天前']

        summarizations = []
        for i in data['summarization']:
            i = Utils.delete_date(i)
            summarizations.append(i)

        data['summarization'] = summarizations
        data.to_csv(path_or_buf=output_sub_csv, encoding='utf-8', header=None, index=None)


if __name__ == '__main__':
    input_train_csv = '../../data/input/train.csv'
    input_test_csv = '../../data/input/test.csv'
    input_sub_csv = '../../data/output/sub.csv'

    output_train_csv = '../../data/output/train.csv'
    output_val_csv = '../../data/output/val.csv'
    output_test_csv = '../../data/output/test.csv'
    output_train_src_csv = '../../data/output/train_src.csv'
    output_train_tgt_csv = '../../data/output/train_tgt.csv'
    output_sub_csv = '../../data/output/sub_delete_time.csv'

    # Utils.generate_train_val_test(input_train_csv=input_train_csv,
    #                               input_test_csv=input_test_csv,
    #                               output_train_csv=output_train_csv,
    #                               output_val_csv=output_val_csv,
    #                               output_test_csv=output_test_csv)

    Utils.generate_train_src_tgt(input_train_csv,
                                 input_test_csv,
                                 output_train_src_csv,
                                 output_train_tgt_csv,
                                 output_test_csv)

    # Utils.delete_date_sub(input_sub_csv=input_sub_csv, output_sub_csv=output_sub_csv)

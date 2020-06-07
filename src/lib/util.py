import re
import pandas as pd

from sklearn.model_selection import train_test_split

import os
import sys

sys.path.append('/content/2020-AI-Know-The-Text-Summary')
os.chdir(sys.path[0])
import matplotlib.pyplot as plt


class Utils(object):

    # @staticmethod
    # def delete_date(text: str, left_length: int):
    #     """
    #     删除日期
    #     """
    #     if len(text) < 510:
    #         return text
    #     else:
    #         return text[:left_length] + text[-(510 - left_length):]

    # @staticmethod
    # def generate_train_val_test(input_train_csv, input_test_csv, output_train_csv, output_val_csv, output_test_csv):
    #     """
    #     生成训练/测试/验证集
    #     """
    #     # article, summarization
    #     input_train = pd.read_csv(input_train_csv, encoding='utf-8')
    #     # article
    #     input_test = pd.read_csv(input_test_csv, encoding='utf-8')
    #
    #     input_train['article'] = input_train['article'].apply(lambda a: Utils.deal_text(a))
    #     input_train['summarization'] = input_train['summarization'].apply(lambda a: Utils.deal_text(a))
    #
    #     input_test['article'] = input_test['article'].apply(lambda a: Utils.deal_text(a))
    #
    #     X_train, X_val, y_train, y_val = train_test_split(input_train['article'], input_train['summarization'],
    #                                                       test_size=0.2, random_state=42)
    #
    #     train = pd.DataFrame({'article': X_train, 'summarization': y_train})
    #     val = pd.DataFrame({'article': X_val, 'summarization': y_val})
    #
    #     train.to_csv(output_train_csv, index=None, encoding='utf-8')
    #     val.to_csv(output_val_csv, index=None, encoding='utf-8')
    #     input_test.to_csv(output_test_csv, index=None, encoding='utf-8')

    # @staticmethod
    # def delete_date_sub(input_sub_csv, output_sub_csv):
    #     data = pd.read_csv(input_sub_csv, encoding='utf-8', header=None)
    #     data.columns = ['indexs', 'summarization']
    #
    #     # time_test = ['2017年09月25日', '2017-09-25', '2017/09/25', '2017 09 25', '2017年09月25日19:13:33',
    #     #              '2017-09-25 19:13:40', '1小时前', '1 天前']
    #
    #     summarizations = []
    #     for i in data['summarization']:
    #         i = Utils.delete_date(i, left_length=300)
    #         summarizations.append(i)
    #
    #     data['summarization'] = summarizations
    #     data.to_csv(path_or_buf=output_sub_csv, encoding='utf-8', header=None, index=None)

    @staticmethod
    def deal_text(text: str, split_index=256):
        """
        处理文本
        """
        post_time = re.search('(20\d{2}([\.\-/]{1,3}\d{1,2}){2}?(\s?\d{2}:\d{2}(:\d{2})?)?)', text)

        while post_time:
            post_time = re.search('(20\d{2}([\.\-/\s]{1,3}\d{1,2}){2}?(\s?\d{2}:\d{2}(:\d{2})?)?)', text)
            if post_time is not None:
                text = text.replace(post_time.group(), '')
            else:
                break

        for special_char in ['■', '●', '\n', '\t']:
            while special_char in text:
                text = text.replace(special_char, ' ')

        for (ch_char, en_char) in zip(['０', '１', '２', '３', '４', '５', '６', '７', '８', '９'],
                                      ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']):
            while ch_char in text:
                text = text.replace(ch_char, en_char)

        # 关键
        text = text.strip()

        if len(text) < 510:
            result = text.replace('<Paragraph>', ' ')
        else:
            while text[split_index] in list('<Paragraph>'):
                split_index -= 1

            left = text[:split_index]
            right = text[split_index:]

            result = left.replace('<Paragraph>', ' ')

            right_list = right.split('。')
            for item in right_list:
                if '<Paragraph>' in item:
                    item = item.replace('<Paragraph>', ' ')
                    result += item

                if len(result) > 510:
                    result = result[:510]
                    break

            if len(result) < 510:
                text = text.replace('<Paragraph>', ' ')
                result += text[len(result):]
                result = result[:510]

        # 关键
        result = result.strip()

        return result

    @staticmethod
    def generate_train_src_tgt(input_train_csv, input_test_csv, output_train_src_csv, output_train_tgt_csv,
                               output_test_csv, left_length):
        """
        生成训练/测试/验证集
        """
        # article, summarization
        input_train = pd.read_csv(input_train_csv, encoding='utf-8')
        # article
        input_test = pd.read_csv(input_test_csv, encoding='utf-8')

        input_train['article'] = input_train['article'].apply(lambda a: Utils.deal_text(a))
        input_test['article'] = input_test['article'].apply(lambda a: Utils.deal_text(a))

        input_train.dropna(inplace=True)

        input_train['article'].to_csv(output_train_src_csv, index=None, header=None, encoding='utf-8')
        input_train['summarization'].to_csv(output_train_tgt_csv, index=None, header=None, encoding='utf-8')
        input_test.to_csv(
            os.path.splitext(output_test_csv)[0] + '_' + str(left_length) + os.path.splitext(output_test_csv)[1],
            index=None, header=None, encoding='utf-8')

    @staticmethod
    def deal_sub_csv(input_sub_csv, output_sub_csv):
        data = pd.read_csv(input_sub_csv, encoding='utf-8', header=None)
        data.columns = ['indexs', 'summarization']

        summarizations = []
        for i in data['summarization']:
            i = i.replace('[UNK]', '')
            i = i.replace(' ', '')
            summarizations.append(i)

        data['summarization'] = summarizations
        data.to_csv(path_or_buf=output_sub_csv, encoding='utf-8', header=None, index=None)

    @staticmethod
    def calculate_tgt(input_train_tgt_csv):
        data = pd.read_csv(input_train_tgt_csv, header=None, encoding='utf-8')
        if len(data.columns) == 1:
            data.columns = ['tgt']
        elif len(data.columns) == 2:
            data.columns = ['indexes', 'tgt']

        result = {}
        for i in data['tgt'].values:
            length = len(i)
            if length not in result:
                result[length] = 1
            else:
                result[length] += 1

        print('mean: {}'.format(sum(list([k * v for k, v in result.items()])) // sum(list(result.values()))))

        key_value = {}
        for key, value in sorted(result.items(), key=lambda x: x[0]):
            key_value[key] = value

        plt.plot(key_value.keys(), key_value.values(), 's-', color='r', label="ATT-RLSTM")  # s-:方形
        plt.show()


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
    output_result_csv = '../../data/output/result.csv'

    Utils.deal_text(
        text='2015-06-14<Paragraph>09:31<Paragraph>海西晨报<Paragraph>显示图片经常往返海沧和岛内的读者，你是否经常被堵在海沧大桥？好消息来了，到2019年，你多了一个选择，可以走隧道往返海沧和岛内。晨报讯（记者<Paragraph>刘宇瀚<Paragraph>通讯员<Paragraph>李春妮）在不久的未来，进出厦门岛将再添一条通道！记者昨日从厦门市发改委获悉，厦门第二西通道“可研”已于6月10日获国家发改委批复，正式获得国家发改委立项。这条连接海沧与湖里的海底隧道，项目总投资估算为57.05亿元，计划今年底开工建设，2019年建成通车。建成后，将有效缓解出入厦门岛的交通压力，为海沧大桥“减负”，完善海西区域路网。建6000多米长隧道据悉，连接厦门海沧区和厦门岛的厦门第二西通道，是厦门“两环八射”快速路系统中内环道路的重要组成部分。第二西通道路线起自海沧区海沧大道和马青路交叉口东侧，通过海沧端接线工程连接马青路、海沧大道及拟建的海沧疏港快速路，在海沧大道与拥军路交叉口附近，以海底隧道形式穿越厦门西海域，在象屿码头14号泊位附近进入厦门岛，沿兴湖路前行，下穿石鼓山立交后（隧道洞口）终于火炬北路，通过本岛端接线工程连接成功大道、枋钟路及拟建的第二东通道。第二西通道全长7.1公里，隧道长6335米。全线采用双向6车道一级公路标准（结合城市快速路功能）建设，设计速度采用80公里/小时，路基宽度31.5米。为海沧大桥“减负”据厦门市发改委介绍，第二西通道是《海峡西岸经济区高速公路网规划》的海西城市联络线之一，已纳入了《厦门市总体规划》。目前，海沧大桥是连接厦门岛内与海沧的惟一西通道，但海沧大桥交通量已远超设计通行能力，高峰时段时常发生堵车现象。厦门市发改委相关负责人表示，建设第二西通道已经迫在眉睫。“第二西通道连接厦门本岛与海沧区，向西连通厦成高速公路，向东与兴湖路相接，形成岛内北部新的进出岛快速通道。项目的建设对完善国家和区域路网、实现海峡西岸经济区发展规划战略目标、扩大厦门对外经济辐射、增强区域城市沟通联系、构建现代化交通基础设施和厦门进出岛交通通道体系、满足厦门市交通运输综合发展都具有很重要的意义。”比翔安隧道长<Paragraph>耗时约5分钟解密同样是海底隧道，今年要开工的第二西通道和厦门人已非常熟悉的翔安隧道有什么不同？晨报记者梳理了规划图和资料，详细为您解密。1.也是三孔隧道<Paragraph>中为服务隧道从第二西通道的主体工程横、纵断面图可以看出，它同中国大陆首条海底隧道———翔安隧道一样，也是三孔隧道，左右两孔分别是进出岛方向的行车主道，中间的孔则是服务隧道。服务隧道也是检修通道，除了可用作电力通道，隧道管线也会放在这里。此外，服务隧道还设有人横通道，可直接通往两边的主隧道。在关键时刻，这里将成为逃生通道。2.时速80公里<Paragraph>耗时约5分钟第二西通道全长7.1公里，海底隧道长6335米。以设计时速80公里计算的话，穿过整条海底隧道需耗时5分钟左右。记者查阅公开报道发现，翔安隧道海底隧道长度约为6公里，第二西通道的长度比翔安隧道还要略长一些。3.新阳到机场<Paragraph>时间缩短一半第二西通道建成通车后，将大大缩短马銮湾片区、新阳工业区等到厦门岛西北部距离。据测算，以往从新阳工业区到厦门机场开车需要半小时左右，隧道建成后，开车时间预计将缩短一半以上。海沧端接线2012年已开工回放据悉，厦门第二西通道海底隧道划为主体工程、海沧端接线工程和岛内端接线工程三个项目。其中，海沧端接线工程已于2012年开工。海沧端接线起于海沧吴冠采石场，与海沧疏港快速路相接，下穿马青路，与厦门第二西通道海底隧道连接，主要工程为马青路吴冠互通。其中，马青路吴冠互通位于马青路与海沧大道的交汇处，主要结构物包含马青路跨线桥1座、主线桥1座、匝道桥2座以及现浇人行天桥3座、涵洞18道。而作为厦门第二西通道海底隧道另一重大项目———岛内端接线工程，则涉及石鼓山立交、兴湖路及机场等部分路段。,厦门第二西通道计划今年底开工建设，2019年建成通车，总投资57.05亿，建成后将有效缓解出入厦门岛的交通压力。')

    # Utils.generate_train_val_test(input_train_csv=input_train_csv,
    #                               input_test_csv=input_test_csv,
    #                               output_train_csv=output_train_csv,
    #                               output_val_csv=output_val_csv,
    #                               output_test_csv=output_test_csv)

    # Utils.delete_date_sub(input_sub_csv=input_sub_csv, output_sub_csv=output_sub_csv)

    # Utils.calculate_tgt(output_train_tgt_csv)

    # Utils.calculate_tgt(output_result_csv)

    # Utils.generate_train_src_tgt(input_train_csv=input_train_csv,
    #                              input_test_csv=input_test_csv,
    #                              output_train_src_csv=output_train_src_csv,
    #                              output_train_tgt_csv=output_train_tgt_csv,
    #                              output_test_csv=output_test_csv,
    #                              left_length=500)

    Utils.deal_sub_csv(input_sub_csv='../../data/output/sub.csv', output_sub_csv='../../data/output/result.csv')

"""
南方都市报2015-05-22<Paragraph>00:00心电图显示龙宝中患有严重的心肌梗塞的症状，但医生的病历本只注明诊断是胆囊炎。南都记者<Paragraph>何永华<Paragraph>摄●5月19日<Paragraph>●南城人民医院南都讯<Paragraph>记者何永华<Paragraph>正在输液治疗胆囊炎，却因心肌梗塞而死。5月19日早上62岁的龙宝中因为下腹疼痛被家人送到南城人民医院，医生并没有将检查出来的心肌梗塞症状写入病历，也没有开治疗心肌梗塞的药，随后老翁因此死亡。家属质疑医院医生存在延误救治。南城医院承认在此事件中医生存在失误，但建议家属走司法程序解决。做了两次心电图<Paragraph>却只写胆囊炎62岁的龙宝中是河南人。生前，龙宝中夫妇被唯一的女儿接到东莞，与女儿女婿一家共同生活。据龙宝中的女婿方先生说，岳父生前患有胃病多年，这些年一直在调养。5月19日凌晨5点多，岳父突然感觉到下腹疼痛。“他原本想忍一忍的，后来实在是痛得难以忍受。在早上6点多，我和我老婆就开车把他送到南城医院。”方先生说。医院的病历本上显示，龙宝中是在5月19日早上6点40分进到南城人民医院急诊室的，给他看病的医生姓秦。方先生说，进到医院后，医生就给岳父开了多项检查。“其中包括心电图以及抽血。”方先生感到诧异的是，做完第一个心电图检查后，医生又给岳父做了个心电图检查。“我们家属又看不懂心电图，所以也不知道结果是什么。”龙宝中的病历本上显示，医生给他的临床诊断是胆囊炎。随后，医生就给他开了治疗胃病的胶囊以及治疗胆囊炎的消炎针剂。“看那个病历本，我以为岳父只是胆囊炎引起腹痛，并没有其他的病症。”方先生说，早上7点左右，他就陪着岳父到输液室里输液。可输了10分钟，他就发现岳父呼吸困难，嘴唇发黑，捂着胸口痛苦不堪。方先生连忙去叫医生。输液十分钟<Paragraph>因心肌梗塞而死医生来到后，连忙将龙宝中推进了抢救室。龙宝中的病历本上显示“患者突发口唇发黑，呼吸减慢，呼之不应。即重，神志不清，口唇发黑，颈动脉搏动消失，心跳过速，心率弱，拟心肌梗死。即予胸外接压，面罩吸氧，转入内科IC<Paragraph>U继续抢救。”方先生说，岳父在重症监护室里抢救了十来分钟，医生就告诉他已经宣告死亡了。“当时还输着液呢。”“治疗胆囊炎怎么会把人给治死呢？”方先生说，事后他拿着岳父的心电图报告和病历本去找其他医院相熟的医生询问，结果别的医生说“心电图显示我岳父当时的病症是患有严重的心肌梗塞症状”。方先生说，整个治疗过程中，医生都没有告诉他这个情况。“不仅没有在病历本上写，更是明知道有心肌梗塞的症状，却没有开具类似救心丸之类的药。”据此，方先生认为，医生发现病情后，没有及时地用药，对他岳父的死亡要承担一定的责任。“如果是早点用药的话，说不定还可以救我岳父一命。”方先生说，这两天他们家属一直在找医院要个说法，但至今都没有。对此，东莞市一所三甲医院急诊科负责人说，心肌梗塞病发得又快又重，及时采取措施的话，还是有生还的可能。“只是种可能。”[医院回应]承认过失<Paragraph>愿意承担责任前日下午，南城人民医院的相关负责人回应称，的确从医生给龙宝中所做的两次心电图检查来看，当时患者是有严重的心肌梗塞的症状。“院方目前还不清楚，医生当时为什么不把这样的症状写进诊疗过程当中去，也的确是没有开具相关治疗心肌梗塞的药。”该负责人说，从这个角度上看，医生存在一定的失误，医院也会根据事态的进展，对当事的医生进行处理。该负责人建议家属走司法途径解决此事。“我们愿意承担责任，但需根据法院的判决来定。”【想看更多新鲜资讯请浏览奥一网（oeeee.com）或关注奥一网官方微信。（微信号：oeeeend）】	
东莞62岁老翁查出心肌梗塞，医生未予治疗，病历本只注明胆囊炎，患者输液时猝死；医院承认过失愿担责(图)


"""

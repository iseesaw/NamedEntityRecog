# -*- coding: utf-8 -*-
'''
模型预测
Requirements:
    需要将输入的句子转为词向量和字符级别向量, 需要单词表word2index、字符表char2index
    模型参数文件 BiLSTM_CRF.h5
    模型输入需要相关参数 max_len_seq, max_len_word
    预测结果进行解码需要标签表, index2tag
'''
from CRF_BiLSTM import *
import time

class Predict:
    def __init__(self, filename, includeBound=True):
        self.filename = filename

        self.load_testdata(filename, includeBound)

        self.load_params()

        self.predict()

    '''
    读取文件, 按句保存
    :@param includeBound 是否包含句子边界
    '''

    def load_testdata(self, filepath, includeBound):
        # 读取
        with open(filepath, 'r') as f:
            lines = f.readlines()
        # 保存所有句子
        seqs = []
        # 保存每个句子
        seq = []
        # 不包含句子边界
        if not includeBound:
            for line in lines:
                # \n换行, 读取时不会自动替换
                line = line.replace('\n', '')
                # 属于同一个句子
                if line != '':
                    seq.append(line)
                else:
                    # 标志一个句子的完结
                    seqs.append(seq)
                    seq = []
        # 有句子边界
        else:
            # 上一个是句子边界
            lastB = False
            for line in lines:
                line = line.replace('\n', '')
                # 是句子边界则直接保存
                if '###' in line:
                    seqs.append(line)
                    lastB = True
                elif line != '':
                    seq.append(line)
                    lastB = False
                # 句子边界紧接着的空行
                elif line == '' and lastB:
                    pass
                # 句子完结
                else:
                    seqs.append(seq)
                    seq = []
        self.seqs = seqs

    def load_params(self):
        # 读取单词表、字符表、标签表以及标签反表
        self.word2index = self.load_json('files/word2index.json')
        self.char2index = self.load_json('files/char2index.json')
        self.tag2index = self.load_json('files/tag2index.json')
        self.index2tag = self.load_json('files/index2tag.json')

        # 读取搭建网络所需参数
        params = self.load_json('files/params.json')
        self.max_len_word = params.get('max_len_word')
        self.max_len_seq = params.get('max_len_seq')
        self.n_words = params.get('n_words')
        self.n_tags = params.get('n_tags')
        self.n_chars = params.get('n_chars')

    '''
    对输入句子集合进行测试
    
    构建模型 => 读取参数 => 对每个句子进行预测 => 对预测结果进行解码
    
    Requirements:
        model, model_params
        params: max_len_seq, max_len_word, n_words, n_chars, n_tags
                word2index, char2index, tag2index, index2tag(标签解码)
        xseq => X_word, X_char
    
    '''

    def predict(self):
        # 搭建神经网络
        model = create_model(self.max_len_seq,
                             self.max_len_word,
                             self.n_words,
                             self.n_tags,
                             self.n_chars)

        # 读取模型并重建模型
        model.load_weights('files/model.txt')

        # 由于边界情况的问题, 因此对句子一个一个进行预测
        seq_labels = []

        count = 0
        start = time.time()
        for seq in self.seqs:
            if '###' in seq:
                seq_labels.append(seq)
            else:
                # 对句子集合操作, 便于后面的测试
                # 向量化
                X_word, X_char = self.vetorization_padding([seq])
                # 进行标签预测
                y_pred = model.predict([X_word,
                                        np.asarray(X_char).reshape(len(X_char),
                                                                   self.max_len_seq, self.max_len_word)])

                # 对预测结果进行解码
                yseq = self.decode_tags(y_pred)

                seq_labels.append(yseq[:len(seq)])
            count += 1
            if not count % 100:
                print('{:.0f}s... {}/{}'.format(time.time() - start, count, len(self.seqs)))

        # 保存预测结果
        self.save_predict(seq_labels)

    '''
    y_pred的形式为:
    [[[],[],[]],
    [[],[],[]],
    [[],[],[]]]
    每个词有tags+2种概率预测
    取每个词最大概率的预测标签下标, 并转为标签

    需要注意去除填充
    '''

    def decode_tags(self, y_pred):
        # 取最后一个维度的最大值, 即取每个词的最大概率标记的下标
        y_idx = np.argmax(y_pred, axis=-1)
        # 将每个词的标签下标转为标签
        # 保存的key值是字符串类型
        yseq = [[self.index2tag[str(w)] for w in s] for s in y_idx]
        return yseq[0]

    def vetorization_padding(self, xseqs):
        # # 注意未登录词情况
        X_word = [[self.word2index.get(word, self.word2index['UNK']) for word in xseq] for xseq in xseqs]
        # 填充（从后补齐或者从后面截断）
        X_word = pad_sequences(maxlen=self.max_len_seq,
                               sequences=X_word,
                               value=self.word2index['PAD'],
                               padding='post')


        # 将每个句子中每个单词的字符转换为编号
        # 此时每个单词由一个构成字符编号列表组成
        X_char = []
        for xseq in xseqs:
            xseq_pad = []
            # 填充每个句子为max_len_seq长度
            for i in range(self.max_len_seq):
                word = []
                # 填充每个单词为max_len_word长度
                for j in range(self.max_len_word):
                    try:
                        # 获取字符的编号
                        # 注意未登录词情况
                        word.append(self.char2index.get(xseq[i][j], self.char2index['UNK']))
                    # 产生异常则需要填充
                    except:
                        word.append(self.char2index['PAD'])

                xseq_pad.append(word)
            X_char.append(xseq_pad)

        return X_word, X_char

    def load_json(self, filename):
        with open(filename, 'r') as f:
            json_data = json.load(f)
        return json_data

    def save_predict(self, seqs_labels):
        with open('result.txt', 'w') as f:
            # 遍历得到每句及该句标记
            for seq, seq_labels in zip(self.seqs, seqs_labels):
                if '###' in seq:
                    f.write(seq + '\n')
                else:
                    # 遍历每句
                    for word, label in zip(seq, seq_labels):
                        f.write(word + '\t' + label + '\n')
                # 每句用空行分割
                f.write('\n')

if __name__ == '__main__':
    # 测试包含句子边界
    filename = 'files/Genia4EReval2.raw'
    Predict(filename, True)

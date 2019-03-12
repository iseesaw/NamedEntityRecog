# -*- coding: utf-8 -*-
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Embedding, LSTM, Bidirectional, TimeDistributed, Dropout, concatenate, \
    SpatialDropout1D
from keras_contrib.layers import CRF
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, plot_model
import json
import pandas as pd

'''
读取训练集
seqs = [[[w1, label1],[w2, label2],[w3, label3], ...], 
        [...],
         ...]
words - all word in dataset
tags - all tag in dataset
'''

def load_data(filename='files/Genia4ERtask1.iob2'):
    with open(filename, 'r') as f:
        lines = f.readlines()
    # 得到所有句子及对应的标签
    xseqs = []
    xseq = []
    # 标签
    yseqs = []
    yseq = []
    # 所有出现的字、标签集合
    words = set()
    tags = set()
    for line in lines:
        line = line.replace('\n', '')
        results = line.split('\t')
        # 属于同一个句子
        if len(results) == 2:
            # 同一句子及其标签
            xseq.append(results[0])
            words.add(results[0])

            # 所有出现的词和标签
            yseq.append(results[1])
            tags.add(results[1])

        # 句子边界
        else:
            # 句子序列
            xseqs.append(xseq)
            xseq = []

            # 标签序列
            yseqs.append(yseq)
            yseq = []

    return xseqs, yseqs, list(words), list(tags)

'''
将句子进行向量化
@:return X_word 将句子中每个单词转为数字编号
@:return X_char 将句子中每个单词的字符转为数字编号
@:return Y 
@:return max_len_word
@:return max_len_char
'''

def vetorization_padding(xseqs, yseqs, words, tags):
    # 词总数和标签总数
    n_words = len(words)
    n_tags = len(tags)

    # TODO: 确定最大长度
    # 最大句子长度
    # max_len_seq = max([len(xseq) for xseq in xseqs])
    max_len_seq = 250
    # 最大词长度
    # max_len_word = max([len(word) for word in words])
    # 设为10, 单词最长为10
    max_len_word = 20

    # 单词表和标签表： key-word/tag  value-index
    word2index = {word: idx + 2 for idx, word in enumerate(words)}
    # 填充以及unkown未登录词的编号
    word2index['PAD'] = 0
    word2index['UNK'] = 1

    tag2index = {tag: idx + 1 for idx, tag in enumerate(tags)}
    tag2index['PAD'] = 0

    # 字符和标签表： key-char value-index
    chars = set([c for word in words for c in word])
    n_chars = len(chars)
    char2index = {c: idx + 2 for idx, c in enumerate(chars)}
    # 填充
    char2index['PAD'] = 0
    char2index['UNK'] = 1

    # 将每个句子中单词转为编号
    print('word2index...')
    # model = Word2Vec.load('w2v.model')
    # X_word = []
    # padding = np.ones(100) * 0.1
    # for xseq in xseqs:
    #     xseq_pad = []
    #     for i in range(max_len_seq):
    #         if len(xseq) > i:
    #             try:
    #                 xseq_pad.append(model.wv.word_vec(xseq[i]))
    #             except:
    #                 xseq_pad.append(padding)
    #         else:
    #             xseq_pad.append(np.zeros(100))
    #     X_word.append(xseq_pad)

    X_word = [[word2index[word] for word in xseq] for xseq in xseqs]
    # 填充（从后补齐）
    X_word = pad_sequences(maxlen=max_len_seq, sequences=X_word, value=word2index['PAD'], padding='post')

    # 将每个句子中每个单词的字符转换为编号
    # 此时每个单词由一个构成字符编号列表组成
    print('char2index...')
    X_char = []
    for xseq in xseqs:
        xseq_pad = []
        # 填充每个句子为max_len_seq长度
        for i in range(max_len_seq):
            word = []
            # 填充每个单词为max_len_word长度
            # 可能会进行截断
            for j in range(max_len_word):
                try:
                    # 获取字符的编号
                    word.append(char2index[xseq[i][j]])
                # 产生异常则需要填充
                except:
                    word.append(char2index['PAD'])

            xseq_pad.append(word)
        X_char.append(xseq_pad)

    # 将标签转为编号
    print('tag2index...')
    Y = [[tag2index[tag] for tag in yseq] for yseq in yseqs]
    # 填充
    Y = pad_sequences(maxlen=max_len_seq, sequences=Y, value=tag2index['PAD'], padding='post')

    # 将Y转为热编码 0-000 1-010 2-001
    # TODO:是否需要将标签转为热编码
    # TODO: 输入标签Y和CRF预测输出的概率是否有关系
    Y = [to_categorical(y, num_classes=n_tags + 1) for y in Y]
    # 保存相关参数
    params = {
        'max_len_seq': max_len_seq,
        'max_len_word': max_len_word,
        'n_words': n_words,
        'n_tags': n_tags,
        'n_chars': n_chars
    }

    # 单词表、标签表、字符表需要在后续测试集中使用, 需要保存
    save_dict(word2index, tag2index, char2index, params)

    return X_word, X_char, Y, params

def save_dict(word2index, tag2index, char2index, params):
    # 单词表
    with open('files/word2index.json', 'w') as f:
        json.dump(word2index, f, indent=2)

    # 标签表
    with open('files/tag2index.json', 'w') as f:
        json.dump(tag2index, f, indent=2)

    # 字符表
    with open('files/char2index.json', 'w') as f:
        json.dump(char2index, f, indent=2)

    # 参数
    with open('files/params.json', 'w') as f:
        json.dump(params, f, indent=2)

    # 标签反表 - 将模型预测结果转为标签
    index2tag = {tag2index.get(key): key for key in tag2index.keys()}
    with open('files/index2tag.json', 'w') as f:
        json.dump(index2tag, f, indent=2)

'''
搭建LSTM+CRF神经网络
input and embedding part
    word Embedding layer
    char Embedding layer
       - train by LSTM
       - should operation on each char by TimeDistribute
main part
BiLSTM
CRF

'''

def create_model(max_len_seq, max_len_word, n_words, n_tags, n_chars):
    # 词嵌入
    # 将输入的稀疏的词向量进行训练
    # shape=(max_len_seq, ) 表示不限样本数量, 但每个样本（句子）有max_len_seq长
    # mask_zero表示将0默认视为填充
    word_in = Input(shape=(max_len_seq,))
    # (None, max_len_seq) => (None, max_len_seq, 20)
    # input_dim表示词表大小
    word_emb = Embedding(input_dim=n_words + 2,
                         output_dim=64,
                         input_length=max_len_seq,
                         mask_zero=True, name='Word_Embedding_Layer')(word_in)
    word_emb = Dropout(0.25, name='Dropout_Layer')(word_emb)
    # 字符级别词嵌入
    char_in = Input(shape=(max_len_seq, max_len_word,))
    # TimeDistributed对每个字符级别进行操作
    char_emb = TimeDistributed(Embedding(input_dim=n_chars + 2,
                                         output_dim=128,
                                         input_length=max_len_word,
                                         mask_zero=True, name='Char_Embedding_Layer'))(char_in)
    # 使用LSTM对字符级别向量进行训练 - 就是提取特征
    # return_sequences 是返回输出序列中的最后一个输出, 还是全部序列
    # recurrent_droput 在0和1之间的浮点数, 单元的丢弃比例, 用于循环层状态的线性转换
    char_enc = TimeDistributed(LSTM(units=64,
                                    return_sequences=False,
                                    recurrent_dropout=0.5, name='Char_Encode_Layer'))(char_emb)

    # 将词向量和字符级别向量连接作为输入整体
    x = concatenate([word_emb, char_enc])
    x = SpatialDropout1D(0.3, name='SpatialDropout1D_Layer')(x)

    # LSTM + CRF模块
    x = Bidirectional(LSTM(units=64,
                           return_sequences=True,
                           recurrent_dropout=0.5, name='BiLSTM'))(x)

    x = TimeDistributed(Dense(32, activation='relu', name='Dense'))(x)
    # TODO CRF Keras function api
    # sparse_target=False - hot code
    crf = CRF(n_tags + 1)
    out = crf(x)

    # 创建模型输入和输出
    # 输入词向量和字向量
    model = Model([word_in, char_in], out)

    # 设置优化器以及损失函数和评价指标
    model.compile(optimizer='rmsprop', loss=crf.loss_function, metrics=[crf.accuracy])

    # 打印模型信息
    print(model.summary())
    plot_model(model, to_file='model.png',show_shapes=True)

    return model

# 训练模型
def train_model(X_word, X_char, Y, model, max_len_seq, max_len_word):
    # epochs为2时就过拟合了
    history = model.fit([X_word,
                         np.asarray(X_char).reshape((len(X_char), max_len_seq, max_len_word))],
                        np.array(Y),
                        batch_size=32,
                        epochs=10,
                        verbose=1)

    # 保存训练过程
    hist = pd.DataFrame(history.history)
    hist.to_csv('files/history.csv')

    # 保存模型权重
    model.save('files/model.txt')

if __name__ == '__main__':
    # 得到句子和标签
    print('loading data...')
    xseqs, yseqs, words, tags = load_data()
    print('loading over...')

    # 对句子进行向量化（将句子中的字或字符转为数字编号）并填充
    # params max_len_seq, max_len_word, n_words, n_tags, n_chars
    print('vertoring sequences...')
    X_word, X_char, Y, params = vetorization_padding(xseqs, yseqs, words, tags)
    print('vertoring over...')

    n_words = params.get('n_words')
    n_tags = params.get('n_tags')
    n_chars = params.get('n_chars')
    max_len_seq = params.get('max_len_seq')
    max_len_word = params.get('max_len_word')

    # 搭建模型
    print('creating model...')
    model = create_model(max_len_seq, max_len_word, n_words, n_tags, n_chars)
    print('creating over...')

    print('training model...')
    train_model(X_word, X_char, Y, model, max_len_seq, max_len_word)
    print('training over...')

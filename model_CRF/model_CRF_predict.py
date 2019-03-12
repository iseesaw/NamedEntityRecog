# -*- coding: utf-8 -*-
'''
读取测试文件
调用模型进行标记并输出
'''
import pycrfsuite
import re
from extract_feature import seq2feature
'''
读取文件, 按句保存
:@param includeBound 是否包含句子边界
'''

def load_testdata(filepath, includeBound):
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
    return seqs

if __name__ == '__main__':
    filepath_withB = 'files/Genia4EReval2.raw'

    tagger = pycrfsuite.Tagger()
    tagger.open('files/model.txt')

    # 读取句子
    print('loading testdata...')
    seqs = load_testdata(filepath_withB, True)
    print('extracting features...')
    # 获取句子特征 句子边界不算
    seqs_features = [seq if '###' in seq else seq2feature(seq) for seq in seqs]
    # 进行标记
    print('predicting...')
    seqs_labels = [seq_features if '###' in seq_features else tagger.tag(seq_features) for seq_features in seqs_features]
    print('writing predict...')
    # 写入文件
    with open('result.txt', 'w') as f:
        # 遍历得到每句及该句标记
        for seq, seq_labels in zip(seqs, seqs_labels):
            if '###' in seq:
                f.write(seq + '\n')
            else:
                # 遍历每句
                for word, label in zip(seq, seq_labels):
                    f.write(word + '\t' + label + '\n')
            # 每句用空行分割
            f.write('\n')

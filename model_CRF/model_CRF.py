# -*- coding: utf-8 -*-
import pycrfsuite
from extract_feature import seq2feature
import matplotlib.pyplot as plt

'''
读取训练集
xseqs 句子
yseqs 标签
'''

def load_training(filename='files/Genia4ERtask1.iob2'):
    with open(filename, 'r') as f:
        lines = f.readlines()
    # 特征集合
    # 训练街->句子组成->词组成->词特征组成
    xseqs = []
    yseqs = []
    xseq = []
    yseq = []
    for line in lines:
        line = line.replace('\n', '')
        results = line.split('\t')
        if len(results) == 2:
            xseq.append(results[0])
            yseq.append(results[1])
        # 一个句子完结
        else:
            xseqs.append(xseq)
            yseqs.append(yseq)
            xseq = []
            yseq = []
    return xseqs, yseqs

'''
训练模型
'''

def train_model():
    # 读取训练集特征和标签
    print('load training dataset...')
    xseqs, yseqs = load_training()
    print('extract feature...')
    xseqs = [seq2feature(xseq) for xseq in xseqs]

    # 添加训练样本
    trainer = pycrfsuite.Trainer(verbose=True)
    for xseq, yseq in zip(xseqs, yseqs):
        trainer.append(xseq, yseq)
    # 设置训练参数
    trainer.set_params({
        # L1、L2超参数
        # c1越小越容易过拟合
        'c1': 1,
        'c2': 1e-3,
        # 最大迭代次数
        'max_iterations': 200,
        # 'feature.minfreq':1e-10,
        # 'linesearch':0.5,
        #'feature.possible_states' : True,
        # 是否包含转移概率、隐式设置不可见
        #'feature.possible_transitions': True
    })
    # trainer.select('')
    # 3 - 69.24
    trainer.train('model.txt')

# 训练模型
if __name__ == '__main__':
    train_model()
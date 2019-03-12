# -*- coding: utf-8 -*-
'''
特征提取
输入文件：训练文件
输出：特征文件
    named_entity.txt, 文件每行对应一个样本及其对应的实体标记

特征种类
    DNA
    RNA
    protein
    cell_line
    cell_type
B-class, I-class or 0
'''
import re
import nltk

'''
提取特征
每行对应一个样本及其对应的实体标记
构建特征文件
like 
tag word feature1 feature2 feature3 ....

特征
tag -- 实体标记 like. B-DNA, I-DNA, O
word -- 当前词
word.istitle -- 当前词首字母是否大写
word.containdigit -- 当前词是否包含数字
word.contain'-' -- 当前词是否包含'-'
word-1 -- 前一个词
word-1.tag -- 前一个词的实体标记
word+1 -- 后一个词
word+1.tag -- 后一个词的实体标记

传入当前词、当前词下标及句子
获得当前词的特征
'''

def word2features(index, word, xseq, postags=None):
    feature = [
        'word=' + word,
        'word.lower=' + word.lower(),
        # 'word.isupper=%s' % word.isupper(),
        # 'word.istitle=%s' % word.istitle(),
        # 'word.isdigit=%s' % word.isdigit(),
        'word.contain_digit=%s' % bool(re.search(r'\d', word)),
        'word.contain_signal=%s' % ('-' in word or '/' in word or '_' in word),
        'word.regular1=%s' % (re.match(r'\w+[\-_/]\w+', word) != None),
        'word.regular2=%s' % (re.match(r'[A-Z]+-\w+', word) != None),
        'word[-3:]=%s' % word[-3:],
        'word[:3]=%s' % word[:3],
        'word[-2:]=%s' % word[-2:],
        'word[:2]=%s' % word[:2],
        'word.pos=%s' % postags[index][1]
    ]
    # 前一个词的特征
    if index != 0:
        word = xseq[index - 1]
        feature.extend([
            '-1:word=' + word,
            '-1:word.lower=' + word.lower(),
            # '-1:word.istitle=%s' % word.istitle(),
            # '-1:word.isupper=%s' % word.isupper(),
            # '-1:word.isdigit=%s' % word.isdigit(),
            '-1:word.contain_digit=%s' % bool(re.search(r'\d', word)),
            '-1:word.contain_signal=%s' % ('-' in word or '/' in word or '_' in word),
            '-1:word.regular1=%s' % (re.match(r'\w+[\-_/]\w+', word) != None),
            '-1:word.regular2=%s' % (re.match(r'[A-Z]+-\w+', word) != None),
            '-1:word[-3:]=%s' % word[-3:],
            '-1:word[-2:]=%s' % word[-2:],
            '-1:word[:3]=%s' % word[:3],
            '-1:word[:2]=%s' % word[:2],
            '-1:word.pos=%s' % postags[index - 1][1]
        ])
    else:
        feature.append('BOS')
    # 后一个词的特征
    if index != len(xseq) - 1:
        word = xseq[index + 1]
        feature.extend([
            '+1:word=' + word,
            '+1:word=' + word.lower(),
            # '+1:word.istitle=%s' % word.istitle(),
            # '+1:word.isupper=%s' % word.isupper(),
            # '+1:word.isdigit=%s' % word.isdigit(),
            '+1:word.contain_digit=%s' % bool(re.search(r'\d', word)),
            '+1:word.contain_signal=%s' % ('-' in word or '/' in word or '_' in word),
            '+1:word.regular1=%s' % (re.match(r'\w+[\-_/]\w+', word) != None),
            '+1:word.regular2=%s' % (re.match(r'[A-Z]+-\w+', word) != None),
            '+1:word[-3:]=%s' % word[-3:],
            '+1:word[-2:]=%s' % word[-2:],
            '+1:word[:3]=%s' % word[:3],
            '+1:word[:2]=%s' % word[:2],
            '+1:word.pos=%s' % postags[index + 1][1]
        ])
    else:
        feature.append('EOS')

    if index >= 2:
        word = xseq[index - 2]
        feature.extend([
            '-2:word=' + word,
            '-2:word=' + word.lower(),
            # '-2:word.istitle=%s' % word.istitle(),
            # '-2:word.isupper=%s' % word.isupper(),
            # '-2:word.isdigit=%s' % word.isdigit(),
            '-2:word.contain_digit=%s' % bool(re.search(r'\d', word)),
            '-2:word.contain_signal=%s' % ('-' in word or '/' in word or '_' in word),
            '-2:word.regular1=%s' % (re.match(r'\w+[\-_/]\w+', word) != None),
            '-2:word.regular2=%s' % (re.match(r'[A-Z]+-\w+', word) != None),
            '-2:word[-3:]=%s' % word[-3:],
            '-2:word[-2:]=%s' % word[-2:],
            '-2:word[:3]=%s' % word[:3],
            '-2:word[:2]=%s' % word[:2],
            '-2:word.pos=%s' % postags[index - 2][1]
        ])
    else:
        feature.append('BBOS')

    if index <= len(xseq) - 3:
        word = xseq[index + 2]
        feature.extend([
            '+2:word=' + word,
            '+2:word=' + word.lower(),
            # '+2:word.istitle=%s' % word.istitle(),
            # '+2:word.isupper=%s' % word.isupper(),
            # '+2:word.isdigit=%s' % word.isdigit(),
            '+2:word.contain_digit=%s' % bool(re.search(r'\d', word)),
            '+2:word.contain_signal=%s' % ('-' in word or '_' in word or '/' in word),
            '+2:word.regular1=%s' % (re.match(r'\w+[\-_/]\w+', word) != None),
            '+2:word.regular2=%s' % (re.match(r'[A-Z]+-\w+', word) != None),
            '+2:word[-3:]=%s' % word[-3:],
            '+2:word[-2:]=%s' % word[-2:],
            '+2:word[:3]=%s' % word[:3],
            '+2:word[:2]=%s' % word[:2],
            '+2:word.pos=%s' % postags[index + 2][1]
        ])
    else:
        feature.append('EEOS')

    return feature

'''
将输入句子转换为每个词的特征的集合
'''

def seq2feature(xseq):
    #pos_tags = None
    pos_tags = nltk.pos_tag(xseq)
    return [word2features(index, word, xseq, pos_tags) for index, word in enumerate(xseq)]

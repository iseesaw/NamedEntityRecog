### files
model.txt  模型文件  
char2index.json  训练集字符表  
word2index.json  训练集单词表  
tag2index.json 实体标记表  
index2tag.json  实体标记反表  
params.json 训练集相关参数  
Genia4ERtask1.iob2  不带边界的训练集  
Genia4EReval.raw  带边界的测试集  

### CRF_BiLSTM.py  
CRF+BiLSTM模型源代码   
  
实现思路, 首先得到训练集单词表和字符表(set保存,因此每次运行模型, 单词和字符对对应的下标都会不一样);   
然后得到词向量输入(word_in将句子中每个单词用单词表中对应下标表示)及字符级别字向量输入(char_in将每个句子中每个单词的组成字符用字符表中对于下标表示);  
对word_in和char_in进行词嵌入(embedding)得到word_emb和char_emb, 然后用LSTM训练char_emb得到char_enc,学习每个词的特征;将word_emb和char_enc拼接作为BiLSTM的输入, 最后通过CRF层输出。

经过多次测试, 发现word_in/emb和char_in/emb/enc的维度对模型性能有着较大的影响, 当维度越大时, 模型性能越好, 但模型训练时间明显加长(使用GPU GTX 1050Ti跑1个epoch需要10分钟左右, 实验中最高达到75.91%/67.53%/71.47%(R/P/F1)训练了200个epoch,用时3个小时, 其中word_emb维度为64、char_emb维度为128、char_enc维度为64)  

*Input:* files/Genia4ERtask1.iob2(不带边界的训练集)    
*Ouput:* files/*.json model.txt(模型相关文件)  

### CRF_BiLSTM_predict.py
调用CRF+BiLSTM模型预测源代码  
模型预测每100句需要10s左右, 总共需要7分钟左右 

*Input:* files/Genia4EReval.raw(带边界的测试集) model.txt(模型文件)  
*Output:* result.txt(带边界测试集预测文件)

### model.png
CRF+BiLSTM神经网络模型结构图


### result.txt
CRF+BiLSTM模型预测结果  

### evaluation_result.txt
使用SharedTaskEval.pl对result.txt的评测结果  
详细显示各时期的各命名实体的识别结果  

### evalIOB2_result.txt
使用evalIOB2.pl对result.txt的评测结果  
显示总体的各命名实体的识别结果　　

|     R   |     P   |    F1   |
| ------- | ------- | ------- |
|  75.91% |  67.53% |  71.47% |
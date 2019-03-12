### files
Genia4EReval2.raw  带边界测试集  
Genia4ERtask1.iob2  不带边界训练集  
model.txt  CRF模型文件  

### extract_feature.py
特征提取源代码(调用了nltk词性标注, 需要安装nltk库以及所需数据库  
如果不使用词性标注, 可以注释代码中17行import nltk、56/75/96/117/138行的词性特征、151行的词性调用、显示150行的pos_tags变量。性能会有所下降)  

### model_CRF.py
CRF模型实现源代码  
可以直接运行训练模型  
训练时, 调用extract_feature.py提取特征(无边界) 
用到比较多的特征, 迭代200次大概需要10分钟左右   
*Input*: files/Genia4ERtask1.iob2(无边界训练集)  
*Output*: files/model.txt(CRF模型文件)  

### model_CRF_predict
读取files/model.txt进行预测  
可以直接运行以输出预测结果  

*Input*: files/Genia4EReval2.raw(有边界测试集)  　
*Output*: result.txt(预测结果)　　

### result.txt
CRF模型预测输出文件(带边界, 可直接使用SharedTaskEval.pl评测)  

### evaluation_result.txt
使用SharedTaskEval.pl对result.txt的评测结果  
详细显示各时期的各命名实体的识别结果  

### evalIOB2_result.txt
使用evalIOB2.pl对result.txt的评测结果  
显示总体的各命名实体的识别结果  

|    R    |    P    |    F1   |
| ------- | ------- | ------- |
|  70.61% |  69.16% |  69.88% |

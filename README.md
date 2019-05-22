
## 基于CRF+BiLSTM的命名实体识别
>详见文档[document.pdf](document.pdf)
### 特征提取
extract_feature.py 特征提取程序源代码  
feature.txt 特征文件(直接提取时不包含词性, 在CRF模型的实验中为了达到更好的效果, 调用了nltk的词性标注接口, F1值会有所提高)  

### 实体识别模型
先后实现了CRF模型和CRF_BiLSTM模型  
其中CRF+BiLSTM是最终模型,性能优于CRF模型  
分别包含在以下文件夹下:  

#### model_CRF_BiLSTM
最终实现的实体识别模型  
具体参考model_CRF_BiLSTM.md
所依赖库参考requirements.txt
文件夹下包含模型文件、测试文件识别结果及评价文件  
使用evalIOB2.pl评价：  

|    R   |    P   |   F1   |
| ------ | ------ | ------ |
| 75.91% | 67.53% | 71.47% |

#### model_CRF
过程中实现的实体识别模型  
用到的特征比较多, 比feature.txt中多了词性相关的特征  
具体参考model_CRF/README.md  
所依赖库参考requirements.txt  
文件夹下包含模型文件、测试文件识别结果及评价文件  
使用evalIOB2.pl评价：

|    R   |    P   |   F1   |
| ------ | ------ | ------ |
| 70.61% | 69.16% | 69.88% |  

### dataset
包含完整数据  
其中dataset/JNLPBA2004_eval文件夹下包含CRF和CRF+BiLSTM两个模型预测结果CRF_result.txt和CRF_BiLSTM_result.txt, 可以进行相关评测   

### requirements
实验所依赖库, 程序复现时所需  

### 其他
最终模型文件 model_CRF_BiLSTM/files/model.txt
最终模型识别结果文件 model_CRF_BiLSTM/result.txt
最终模型评价结果文件 model_CRF_BiLSTM/evaluation_result.txt

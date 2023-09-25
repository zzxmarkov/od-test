# 2021-11-30  0:50
import re
import time
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
import jieba
jieba.setLogLevel(jieba.logging.INFO)
from sklearn.feature_selection import VarianceThreshold
import os
"""
s = '我是我'
print(jieba.lcut(s))
"""
# s = '我'
# a = [['我', 'aa', 'bb'], ['NLP', '商汤科技', 'rnn']]
# for c, w in enumerate(a):
#     print(c,w)

# import time
# lst = ["\\", "|", "/", "———"]
# for i in range(20):
#     j = i % 4
#     print(j)
#     time.sleep(0.2)

# s = "枝上柳绵吹又少，天涯何处无芳草"
# l = len(s)
# for i in range(l):
#     print("\r" + s[:l-1-i] + "|", end="")
#     time.sleep(3)
# print('\r\n1\r2\n3\r\n4'.splitlines())
"""def remove_noise(document):
    noise_pattern = re.compile('|'.join(['http\S+', '\@\w+', '\#\w+']))
    clean_text = re.sub(noise_pattern, '', document)
    return clean_text.strip()

s = remove_noise('Trump images are now more popular than cat gifs. @trump #trends http://www.trumptrends.html')
print(s)"""
"""
L1 = ['a', 'b', 'c']
L2 = ['d', 'e', 'f']
L3 = [1, 2, 3]
L4 = [5, 6, 7]
aa, bb, cc, dd = zip(*[('a', 'd', 1, 5), ('b', 'e', 2, 6), ('c', 'f', 3, 7)])
print(aa, bb, cc, dd, type(aa))
print(dict(zip(L1, L2)))
"""

"""vec = CountVectorizer(lowercase=True, analyzer='char_wb', ngram_range=(1, 2), max_features=1000,
                      preprocessor=remove_noise)

s = vec.fit(['Trump images are now more popular than cat gifs. @trump #trends http://www.trumptrends.html'])

def get_features(x):
    vec.transform(x)

result = vec.transform(['Trump images are now more popular than cat gifs. @trump #trends http://www.trumptrends.html'])

print(type(s))"""

# a = [['我', 'aa', 'bb'], ['NLP', '商汤科技', 'rnn']]
# print(a[0:1])

"""def func(sen, L):
    for l in L:
        sen.append(l)

sen = []
L1 = ['a', 'b', 'c']
L2 = ['d', 'e', 'f']
L3 = [1, 2, 3]
L4 = [5, 6, 7]

func(sen, L1)
print(sen)
func(sen, L2)
print(sen)
func(sen, L3)
print(sen)
func(sen, L4)
print(sen)"""


#CountVectorizer详解

"""texts=["dog cat fish","dog cat cat","fish bird", 'bird'] # “dog cat fish” 为输入列表元素,即代表一个文章的字符串
cv = CountVectorizer(ngram_range=(1, 2))#创建词袋数据结构
cv_fit=cv.fit_transform(texts)
#上述代码等价于下面两行
#cv.fit(texts)
#cv_fit=cv.transform(texts)

print(cv.get_feature_names())    #['bird', 'cat', 'dog', 'fish'] 列表形式呈现文章生成的词典

print(cv.vocabulary_)              # {‘dog’:2,'cat':1,'fish':3,'bird':0} 字典形式呈现，key：词，value:词索引

print(cv_fit)
# （0,3） 1   第0个列表元素，**词典中索引为3的元素**， 在这句话中的词频
#（0,1）1
#（0,2）1
#（1,1）2
#（1,2）1
#（2,0）1
#（2,3）1
#（3,0）1

print(cv_fit.toarray()) #.toarray() 是将结果转化为稀疏矩阵矩阵的表示方式；
#[[0 1 1 1]
# [0 2 1 0]
# [1 0 0 1]
# [1 0 0 0]]

print(cv_fit.toarray().sum(axis=0))  #每个词在所有文档中的词频
#[2 3 2 2]
print(type(cv_fit))"""

# r1 = ['a', 'b', 'c']
# r = r1[:]
# print(r)
"""
student_df = pd.DataFrame([{'姓名': '张三', '专业': '计算机'},
                        {'姓名': '李四', '专业': '会计'},
                        {'姓名': '王五', '专业': '市场营销'}])

print(student_df['专业'], type(student_df['专业'].values))

def func(sentences, category):
    for i in range(4):
        sentence = []
        for i in range(5):
            sentence.append(str(i))

        sentences.append(('  '.join(sentence), category))

sentences = []
func(sentences, 'a')
print(sentences)
func(sentences, 'b')
print(sentences)

"""

"""
s = '我 爱 NLP'
a = s.split(' ')

print(a)  # ['我', '爱', 'NLP']，字符串转成list


s = ['我', '爱', 'NLP']
print('  '.join(s))    #我  爱  NLP, 可迭代对象转成字符串
"""
"""
s = [('w', '我'), ('n', '你'), ('t', '他')]
print([a[0] for a in s])         #['w', 'n', 't']

b = '我 爱 中国'
print([ss for ss in b])          #['我', ' ', '爱', ' ', '中', '国']

c = {'姓名': '张三', '专业': '计算机'}

print(c['姓名'])

"""
"""
stopwords1 = pd.read_csv('../../../NLP实战/News-Classifier-Machine-Learning-and-Deep-Learning/origin_data/stopwords.txt',
                        index_col=False, names=['stopword'], quoting=3, encoding='utf-8', engine='python')

stopwords = pd.read_csv('../../../NLP实战/News-Classifier-Machine-Learning-and-Deep-Learning/origin_data/stopwords.txt',
                        index_col=False, names=['stopword'], quoting=3, sep='\t', encoding='utf-8', engine='python')
print(stopwords['stopword'].values.tolist()[11])

print(stopwords1['stopword'].values.tolist()[11])
"""
"""
y_train = [1, 2, 1, 3, 4]
label_color_dict = {1: 'red', 2: 'green', 3: 'blue', 4: 'yellow'}
colors = list(map(lambda label: label_color_dict[label], y_train))
print(colors)
"""
"""
s = np.zeros((2, 3))
print(s)

a = np.random.randint(100, size=(2, 3))
print(a)
"""

"""
X_train = np.array([['male', 'low'],
                  ['female', 'low'],
                  ['female', 'middle'],
                  ['male', 'low'],
                  ['female', 'high'],
                  ['male', 'low'],
                  ['female', 'low'],
                  ['female', 'high'],
                  ['male', 'low'],
                  ['male', 'high']])

X_test = np.array([['male', 'low'],
                  ['male', 'low'],
                  ['female', 'middle'],
                  ['female', 'low'],
                  ['female', 'high']])

LE = LabelEncoder()
LE1 = LE.fit_transform(X_train[:, 0]).reshape(-1, 1)  # <class 'numpy.ndarray'>
OE1 = OneHotEncoder().fit_transform(LE1)   # 稀疏矩阵
OOE1 = OE1.todense()

LE2 = LabelEncoder().fit_transform(X_train[:, 1]).reshape(-1, 1)
s = np.hstack((OOE1, LE2))

print(s)
# print(OE1)
# print(OOE1)

LE11 = LE.transform(X_test[:, 0]).reshape(-1, 1)  # <class 'numpy.ndarray'>
OE11 = OneHotEncoder().fit_transform(LE11)   # 稀疏矩阵
OOE11 = OE11.todense()

LE21 = LabelEncoder().fit_transform(X_test[:, 1]).reshape(-1, 1)
s1 = np.hstack((OOE11, LE21))

print(s1)
"""
"""
X = [[0, 0, 1],
     [0, 1, 0],
     [1, 0, 0],
     [0, 1, 1],
     [0, 1, 0],
     [0, 1, 1]]

# 根据方差保留80%的向量
# 由于特征是伯努利随机变量(两个取值)，方差计算公式：var_thresh = p(1-p)
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
X_new = sel.fit_transform(X)
print(X_new)
"""
"""
train = [1493436143, 1493436123, 1493436113, 1493436153]

# 时间戳转字符串时间
def stamp2str(stamp, strTimeFormat="%Y-%m-%d %H:%M:%S"):
     
     


for i in range(len(X_train)):
     X_train[i] = stamp2str(X_train[i], strTimeFormat="%Y-%m-%d")

print(X_train)
"""
'''
s = 'helloaaawordatoaapython!'
s1 = re.split(r'[^a]+', s)
s11 = s1[1:len(s1)-1]
s22 = re.split(r'[a]+', s)[::-1]

new_list = []
for i in range(max(len(s11), len(s22))):
    if s22:
        new_list.append(s22.pop(0))
    if s11:
        new_list.append(s11.pop(0))

new_str = ''.join(new_list)

print(new_str)
'''





















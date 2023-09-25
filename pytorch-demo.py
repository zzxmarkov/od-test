# -*- coding: utf-8 -*-
# @Time   : 2022/9/11 16:56

import logging
import sys
from collections import Counter
import os
import re
import random
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, classification_report
from torchvision.datasets import MNIST
from torchvision import transforms
from torchtext.datasets import IMDB, WikiText2, Multi30k
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import jieba
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import math
from transformers import BertTokenizer, BertForSequenceClassification, BertModel, BertTokenizerFast, AutoTokenizer
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('max_colwidth', None)
from matplotlib import pyplot as plt


'''
t1 = torch.tensor([[1, 2, 3]], dtype=torch.float).dtype


t2 = torch.arange(24).reshape(2, 3, 4)
t2.transpose_(1, 2)

t3 = torch.tensor(np.array(12, dtype=np.int32))


if torch.cuda.is_available():
    device = torch.device('cuda')
    x = torch.ones_like(t2, device=device)
    x = x.to(device)
    t2 = t2.to(device)
    z = x + t2
    print(z)
'''

'''
a = torch.randn(2, 2)
a = (3*a / (a-1))
a.requires_grad_(True)
b = (a*a).sum()
print(b.grad_fn)
'''

'''
x = torch.tensor([[1, 2],
                  [3, 4]], dtype=torch.float, requires_grad=True)
z = 3 * (x + 2) ** 2
out = z.mean()
# 当out是标量的时候，直接out.backward()就能反向传播，out就是loss，一般loss都是标量
out.backward()
# x.gard是梯度的累加，每次反向传播之前都要先归零再计算
print(x.grad)
'''

'''
# 线性回归demo

# 构建数据
x = torch.rand([50])
y_true = 3 * x + 0.8

# 初始化参数
w = torch.rand(1, requires_grad=True)
b = torch.rand(1, requires_grad=True)
learning_rate = 0.01

# 计算loss
def loss_fn(y_true, y_predict):
    loss = ((y_true-y_predict)**2).mean()
    if w.grad:
        w.grad.data.zero_()
    if b.grad:
        b.grad.data.zero_()
    loss.backward()
    return loss

# 开始训练

for i in range(2000):
    y_predict = x * w + b
    loss = loss_fn(y_true, y_predict)
    if i % 100 == 0:
        print('i, w, b, loss', i, w.item(), b.item(), loss.item(), type(loss), type(w.data), type(w.grad))
    #     全部是<class 'torch.Tensor'> <class 'torch.Tensor'> <class 'torch.Tensor'>

    w.data -= learning_rate * w.grad.data
    b.data -= learning_rate * b.grad.data

# plt.figure(figsize=(30, 30))
# plt.scatter(x, y_true, c='r')
# y_predict = x * w + b
# plt.plot(x, y_predict.detach().numpy())
# plt.show()
'''

'''
# 神经网络实现线性模型

# 数据准备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = torch.rand([500, 1]).to(device)
y_true = (3 * x + 0.8).to(device)

# 搭建模型
class Lr(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        out = self.linear(x)
        return out

# 实例化
model = Lr().to(device)
loss_fn = nn.MSELoss()
optimers = optim.SGD(model.parameters(), lr=0.01)

# 训练模型
for i in range(1, 2000):
    y_predict = model(x)
    loss = loss_fn(y_true, y_predict)
    optimers.zero_grad()
    loss.backward()
    optimers.step()
    if i % 100 == 0:
        params = list(model.parameters())
        print('Epoch[{}/2000], loss: {}, w: {}, b: {}'.format(i, loss.data, params[0].item(), params[1].item()))

# 模型评估
# model.eval()
# y_predict = model(x)
# plt.figure(figsize=(30, 30))
# plt.scatter(x, y_true, c='r')
# plt.plot(x, y_predict.detach().numpy())
# plt.show()
'''

'''
# 数据加载

data_path = './smsspamcollection/SMSSpamCollection'

class MyDataset(Dataset):
    def __init__(self):
        lines = open(data_path, 'r', encoding='utf-8')
        data = [[line[:4].strip(), line[4:].strip()] for line in lines]
        self.data = pd.DataFrame(data, columns=['label', 'text'])

    def __getitem__(self, index):
        item = self.data.iloc[index]
        return item[0], item[1]

    def __len__(self):
        return len(self.data)

data = MyDataset()
d = DataLoader(dataset=data, batch_size=2, shuffle=True)
for i, (label, text) in enumerate(d):
    print(i, label, text)
'''

# 手写数字
'''
BATCH_SIZE = 128
# 加载数据
def get_dataset(mode):
    mnist = MNIST(root='./data', train=mode,
                  transform=transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.1307,), (0.3081,))]))
    data_loader = DataLoader(dataset=mnist, batch_size=BATCH_SIZE, shuffle=True)
    return data_loader

# 搭建模型
class Mmodel(nn.Module):
    def __init__(self):
        super(Mmodel, self).__init__()
        self.line1 = nn.Linear(28*28*1, 28)
        self.line2 = nn.Linear(28, 10)

    def forward(self, x):
        # x.size=[batch_size, C, H, W]   128*1*28*28
        # 因为要用Linear，所以要转成二维
        x = x.view([-1, 28*28*1])
        x = self.line1(x)
        x = F.relu(x)
        out = self.line2(x)
        return F.log_softmax(out, dim=-1)

# 模型的训练
def train():
    mode = True
    model.train(mode)
    train_data = get_dataset(mode)
    # train_data.size=[batch_size, C, H, W]   128*1*28*28
    # nll_loss + log_softmax == cross_entropy
    for i, (input, target) in enumerate(train_data):
        optimizer.zero_grad()
        output = model(input)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if i % 500 == 0:
            print('Train set: [{}/{} ({:.0f}%)], loss: {:.6f}'.format(i * len(input), len(train_data.dataset),
                                                                      100. * i / len(train_data), loss.item()))

# 模型评估
def test():
    test_loss = 0
    acc_sum = 0
    model.eval()
    test_data = get_dataset(False)
    with torch.no_grad():
        for data, target in test_data:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=-1, keepdim=True)
            acc_sum += pred.eq(target.view_as(pred)).sum()
    test_loss /= len(test_data.dataset)
    print('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(test_loss,
                                                                          acc_sum, len(test_data.dataset),
                                                                          100.*acc_sum/len(test_data.dataset)))




if __name__ == '__main__':

    model = Mmodel()
    optimizer = Adam(params=model.parameters(), lr=0.001)
    for i in range(10):
        train()
        test()
        print('***************')
'''

'''
# 加载数据
def get_dataset(flag):
    mnist = MNIST('./data', train=flag, transform=transforms.Compose([transforms.ToTensor(),
                                                                     transforms.Normalize((0.1307,), (0.3081,))]))
    train_data = DataLoader(dataset=mnist, batch_size=BATCH_SIZE, shuffle=True)
    return train_data


# 搭建模型
class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.line1 = nn.Linear(28*28*1, 28)
        self.line2 = nn.Linear(28, 10)

    def forward(self, x):
        x = x.view([-1, 28*28*1])
        x = self.line1(x)
        x = F.relu(x)
        x = self.line2(x)
        return F.log_softmax(x, dim=-1)


# 模型训练
def train():
    model.train()

    for i, (data, target) in enumerate(train_data):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if i % 50 == 0:
            print('Train set [{}/{}, {:.0f}%], loss: {:.4f}'.format(i*len(data), len(train_data.dataset),
                                                                    100. * i / len(train_data), loss.item()))


# 模型评估
def test():
    model.eval()
    test_loss = 0
    acc_sum = 0
    with torch.no_grad():
        for data, target in test_data:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum')
            pred = output.argmax(dim=-1, keepdim=True)
            acc_sum += pred.eq(target.view_as(pred)).sum()
    test_loss_avg = test_loss / len(test_data.dataset)
    acc = acc_sum / len(test_data.dataset)
    print('Test set Avge_loss: {:.4f}, acc: [{}/{}], {:.2f}%'.format(test_loss_avg, acc_sum,
                                                                     len(test_data.dataset), 100. * acc))


if __name__ == '__main__':
    BATCH_SIZE = 128
    model = MnistModel()
    optimizer = Adam(model.parameters(), lr=0.001)
    train_data = get_dataset(flag=True)
    test_data = get_dataset(flag=False)
    for i in range(10):
        train()
        test()
'''

'''
text = "An adaptation of 'Union Street'...no.<br /><br />The women of Union Street \t \n NewYork"
fileters = ['!', '"', '#', '$', '%', '&', '\(', '\)', '\*', '\+',
            ',', '-', '\.', '/', ':', ';', '<', '=', '>', '\?', '@', '\[', '\\', '\]',
            '^', '_', '`', '\{', '\|', '\}', '~', '\t', '\n', '\x97', '\x96', '“', '”']

text = re.sub('<.*?>', ' ', text, flags=re.S)
text = re.sub('|'.join(fileters), ' ', text, flags=re.S)
a = [i.strip() for i in text.split()]
b = text.split()
print(a)
print(b)
print(a == b)
'''

'''
def tokenizer(context):
    fileters = ['!', '"', '#', '$', '%', '&', '\(', '\)', '\*', '\+',
                ',', '-', '\.', '/', ':', ';', '<', '=', '>', '\?', '@', '\[', '\\', '\]',
                '^', '_', '`', '\{', '\|', '\}', '~', '\t', '\n', '\x97', '\x96', '“', '”']
    context = re.sub('<.*?>', ' ', context)
    context = re.sub("'s", ' is', context)
    context = re.sub("'m", ' am', context)
    context = re.sub('|'.join(fileters), ' ', context)
    context = [word.strip().lower() for word in context.split()]
    return context


class ImdbData(Dataset):
    def __init__(self):
        train_data_path = './sentiment-demo/aclImdb/train'
        test_data_path = './sentiment-demo/aclImdb/test'
        train_data_file_path = [os.path.join(train_data_path, 'pos'), os.path.join(train_data_path, 'neg')]
        self.files_path = []
        for files in train_data_file_path:
            file_path = [os.path.join(files, i) for i in os.listdir(files) if i.endswith('.txt')]
            self.files_path.extend(file_path)

    def __getitem__(self, index):
        file_path = self.files_path[index]
        label = 1 if file_path.split('\\')[-2] == 'pos' else 0
        with open(file_path, 'r', encoding='utf-8') as f:
            context = f.read()
            context = tokenizer(context)

        return label, context

    def __len__(self):
        return len(self.files_path)


def myCollate(batch):
    # batch = [(__getitem__), (), ..., ()], a list of tuple , the tuple is return of __getitem__.
    batch = list(zip(*batch))
    label = batch[0]
    context = batch[1]
    return label, context


# data = ImdbData()
# data_loader = DataLoader(dataset=data, batch_size=2, shuffle=True, collate_fn=myCollate)
# for i in data_loader:
#     print(i)
#     break


class Word2Sequence:
    UNK_TAG = 'UNK'
    PAD_TAG = 'PAD'
    UNK = 0
    PAD = 1

    def __init__(self):
        self.dic = {self.UNK_TAG: self.UNK,
                    self.PAD_TAG: self.PAD}

        self.count = {}
        self.inverse_dic = {}

    def __len__(self):
        return len(self.dic)

    def fit(self, sentence):
        for word in sentence:
            if word not in self.count:
                self.count[word] = 1
            else:
                self.count[word] += 1

    def build_vocab(self, min_count=1, max_count=None, max_len=None):
        if min_count:
            self.count = {k: v for k, v in self.count.items() if v >= min_count}
        if max_count:
            self.count = {k: v for k, v in self.count.items() if v <= max_count}
        if max_len and max_len < len(self.count):
            temp = sorted(self.count.items(), key=lambda x: x[-1], reverse=True)[:max_len]
            self.count = dict(temp)
        for word in self.count:
            self.dic[word] = len(self.dic)

        self.inverse_dic = {k: v for v, k in self.dic.items()}

    def transform(self, sentence, max_len=None):
        embedding_list = []
        if max_len:
            if max_len > len(sentence):
                sentence += ['PAD'] * (max_len - len(sentence))
            else:
                sentence = sentence[:max_len]
        for word in sentence:
            if word in self.dic:
                embedding_list.append(self.dic[word])
            else:
                embedding_list.append(self.UNK)
        return embedding_list

    def inverse_transform(self, encodes):
        word_list = []
        for encode in encodes:
            if encode in self.inverse_dic:
                word_list.append(self.inverse_dic[encode])
            else:
                word_list.append(self.UNK_TAG)
        return word_list


if __name__ == '__main__':
    # ws = Word2Sequence()
    # ws.fit(['我', '爱', '中国'])
    # ws.fit(['我', '是', '谁'])
    # ws.build_vocab()
    # print(ws.dic)
    # print(ws.count)
    # ret = ws.transform(['我', '爱', '北京'], max_len=10)
    # print(ret)
    # r = ws.inverse_transform(ret)
    # print(r)

    path = './sentiment-demo/aclImdb/train'
    ws = Word2Sequence()
    train_data_folder_path = [os.path.join(path, 'pos'), os.path.join(path, 'neg')]
    for folder in train_data_folder_path:
        train_file_name = os.listdir(folder)
        for i in tqdm(train_file_name):
            train_data_path = os.path.join(folder, i)
            sentence = tokenizer(open(train_data_path, 'r', encoding='utf-8').read())
            ws.fit(sentence)
    ws.build_vocab(min_count=10)
    pickle.dump(ws, open('./sentiment-demo/model/ws.pkl', 'wb'))
    print(len(ws))
'''


# LSTM的输出，output.shape = [seq_len, batch_size, hidden_size], 即最后一层的输出
# h_n = [num_layers*num_directions, batch_size, hidden_size], 即最后一个子的hidden,
# output = [[第一时间步正向， 最后时间步反向],
#           [最后时间步正向，第一时间步反向]]

'''
batch_size = 10
seq_len = 20

x = torch.randint(low=0, high=100, size=(batch_size, seq_len))
emdedding_layer = torch.nn.Embedding(100, 30)
lstm_layer = torch.nn.LSTM(input_size=30, hidden_size=18, num_layers=2, bidirectional=True, batch_first=True)

lstm_input = emdedding_layer(x)
print(lstm_input.shape)

# lstm_input.transpose_(1, 0)
out_put, (h_n, c_n) = lstm_layer(lstm_input)
# [20, 10, 18*2]
print(out_put.shape)
# [2*2, 10, 18]
print(h_n.shape)
# [2*2, 10, 18]
print(c_n.shape)

# a = out_put[-1, :, :3]
# b = out_put[0, :, 3:]
# x = torch.cat([a, b], dim=-1)
# c = torch.cat([h_n[-2, :, :], h_n[-1, :, :]], dim=-1)
# print(x)
# print(c)
# print(x == c)
'''

# 写文件
'''
data_path = 'data/stopwords-master'
s = [os.path.join(data_path, i) for i in os.listdir(data_path) if i.endswith('txt')]
res = []
for file in s:
    f = open(file, 'r', encoding='utf-8')
    p = f.readlines()
    for i in p:
        i = re.sub(r'^[A-Za-z]+\'*[A-Za-z]+$', '', i)
        if i not in res:
            res.append(i)
    f.close()

res = ''.join(res)


with open('./data/stopwords-master/my_stopwords.txt', 'w', encoding='utf-8') as f:
    p = f.write(res)
    f.seek(0)
    print(p)
'''


# fasttext语料准备
'''
cate_dic = {'technology': 1, 'car': 2, 'entertainment': 3, 'military': 4, 'sports': 5}
path = r'./data/news'
new_name = [os.path.join(path, i) for i in os.listdir(path) if i.endswith('.csv')]
stopwords = pd.read_csv('./data/stopwords-master/my_stopwords.txt', index_col=False, quoting=3,
                        sep='\t', names=['stopword'], encoding='utf-8')
stopwords = stopwords['stopword'].values

flag = [0, 0, 0, 0, 1]
contents = []
def preprocess_text(sentences, category):
    res = []
    for sentence in sentences:
        words = jieba.lcut(sentence)
        words = [word for word in words if len(word) > 1]
        res.append('__label__' + str(category) + ' , ' + ' '.join(words))
    return res

for new in tqdm(new_name):
    new_df = pd.read_csv(new, encoding='utf-8').dropna()
    sentences = new_df.content.values.tolist()
    category = new.split('\\')[-1]
    category = re.match(r'[a-z]+', category).group()
    cate = cate_dic[category]
    content = preprocess_text(sentences, cate)
    contents.extend(content)

random.shuffle(contents)
f_train = open('./data/news/train_data.txt', 'w', encoding='utf-8')
f_test = open('./data/news/test_data.txt', 'w', encoding='utf-8')
for i in tqdm(contents):
    m = random.choice(flag)
    if m == 0:
        f_train.write(i + '\n')
    else:
        f_test.write(i + '\n')
f_train.seek(0)
f_test.seek(0)
f_train.close()
f_test.close()
'''


'''
from .seq2seq_demo.word_seq import num_sequence
class NumEncoder(nn.Module):
    def __init__(self):
        super(NumEncoder, self).__init__()
        self.vocab_size = len(num_sequence)
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size,
                                      embedding_dim=100,
                                      padding_idx=0)
        self.gru = nn.GRU(input_size=100,
                          hidden_size=64,
                          num_layers=1,
                          batch_first=True)

    def forward(self, input, input_len):
        """

        :param input: [batch_size, 10]
        :param input_len: [batch_size]
        :return:
        """

        # embedded: [batch_size, 10, embedding_dim]
        embedded = self.embedding(input)

        embedded = pack_padded_sequence(embedded, lengths=input_len, batch_first=True)
        out, h_n = self.gru(embedded)

        # out: [batch_size, 10, hidden_size]
        out = pad_packed_sequence(out, batch_first=True, padding_value=0)
        return out, h_n

Batch_size = 


class NumDecoder(nn.Module):
    def __init__(self):
        super(NumDecoder, self).__init__()
        self.vocab_size = len(num_sequence)
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size,
                                      embedding_dim=100,
                                      padding_idx=0)
        self.gru = nn.GRU(input_size=100,
                          hidden_size=64,
                          num_layers=1,
                          batch_first=True)
        self.fc = nn.Linear(64, self.vocab_size)

    def forward(self, encoder_hidden, target, target_len):
        """

        :param encoder_hidden: [1, batch_size, hidden_size]
        :param target: [batch_size, 10]
        :param target_len: [batch_size, 1]
        :return:
        """
        
        # 把BOS做为decoder的第一个输入
        decoder_input = torch.LongTensor([[2]] * Batch_size)
        
        # 准备个矩阵，保存结果
        decoder_outputs = torch.zeros(Batch_size, 10, self.vocab_size)
        decoder_hidden = encoder_hidden
        
        for t in range(10):
            decoder_output_t, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs[:, t, :] = decoder_output_t
            
            use_teacher_forcing = random.random() > 0.5
            if use_teacher_forcing:
                # 下次使用真实值做为输入
                decoder_input = target[: t].unsqueeze(1)
            else:
                # decoder_input = torch.argmax(decoder_output_t, keepdim=True, dim=-1)
                value, index = torch.topk(decoder_output_t, 1)
                decoder_input = index
        return decoder_outputs, decoder_hidden
            
    def forward_step(self, decoder_input, decoder_hidden):
        """
        
        :param decoder_input: [batch_size, 1]
        :param decoder_hidden: [1, batch_size, hidden_size]
        :return: 
        """
        
        # embedded: [batch_size, 1, embedding_dim]
        embedded = self.embedding(decoder_input)
        
        # embedded: [batch_size, 1, hidden_size], output_hidden: [1, batch_size, hidden_size]
        embedded, output_hidden = self.gru(embedded, decoder_hidden)
        
        # embedded: [batch_size, hidden_size]
        embedded = embedded.squeeze()
        
        # out: [batch_size, vocab_size]
        out = F.log_softmax(self.fc(embedded), dim=-1)
        
        return out, output_hidden
'''

from sklearn.feature_extraction.text import TfidfVectorizer

'''
query = "走私了两万元，在法律上应该怎么量刑？"
lines_cut = jieba.lcut(query)

# ['走私', '了', '两万元', '，', '在', '法律', '上', '应该', '怎么', '量刑', '？']
print(lines_cut)
tf = TfidfVectorizer()
features = tf.fit(lines_cut)
result = features.transform(lines_cut)

# {'走私': 4, '两万元': 0, '法律': 3, '应该': 1, '怎么': 2, '量刑': 5}
print(features.vocabulary_)
print(tf.get_feature_names_out())
# (0, 4)	1.0   第0个位置（走私）对应字典的4
#   (2, 0)	1.0   第二个位置（两万元）对应字典的0
#   (5, 3)	1.0
#   (7, 1)	1.0
#   (8, 2)	1.0
#   (9, 5)	1.0
print(result)
'''

'''
document_list = ["行政机关强行解除行政协议造成损失，如何索取赔偿？",
                     "借钱给朋友到期不还得什么时候可以起诉？怎么起诉？",
                     "我在微信上被骗了，请问被骗多少钱才可以立案？",
                     "公民对于选举委员会对选民的资格申诉的处理决定不服，能不能去法院起诉吗？",
                     "有人走私两万元，怎么处置他？",
                     "法律上餐具、饮具集中消毒服务单位的责任是不是对消毒餐具、饮具进行检验？"]
doc_list = [list(jieba.cut(doc)) for doc in document_list]
doc_list = [' '.join(i) for i in doc_list]
query = "走私了两万元，在法律上应该怎么量刑？"
query_cut = list(jieba.cut(query))
tf = TfidfVectorizer()
# tf.fit(doc_list)
result = tf.fit_transform(doc_list)
res = tf.transform(query_cut)
print(result)
print(res)
'''

'''
loss = nn.MSELoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5)
output = loss(input, target)
print(input)
print(target)
print(output)
output.backward()
'''


'''
data_path = './BERT-demo/data/news.xlsx'
pretrained_model_path = './BERT-demo/bert-base-chinese'
bert_tokenizer = BertTokenizer.from_pretrained(pretrained_model_path)

data = pd.read_excel(data_path, sheet_name='train').iloc[:, 1].tolist()
# s = bert_tokenizer.tokenize(data[0])
# s = ['[CLS]'] + s + ['[SEP]']
# text = ['这是中国制造宝马汽车',
#         '美国总统特朗普演讲',
#         '中国男篮是冠军',
#         '华为手机发布会将于明天举行']
token = bert_tokenizer.encode_plus(data[0], padding='max_length', truncation=True, max_length=5, return_tensors='pt')

# dic = {}
# for i, sentence in enumerate(data):
#     if len(sentence) in dic:
#         dic[len(sentence)] += 1
#     else:
#         dic[len(sentence)] = 1
# dic = dict(sorted(dic.items(), key=lambda x: x[0], reverse=False))
# n = len(data)
# print(dic)
# rate = 0
# res = 0
# for k, v in dic.items():
#     rate += v / n
#     if rate > 0.98:
#         res = k
#         break
# print(res)

print(token['input_ids'])
'''

'''
import argparse
parse = argparse.ArgumentParser(prog='argparseDemo', description='the message info before help info',
                                epilog="over")

parse.add_argument('--aaa')
parse.add_argument('--bbb')
a = parse.parse_args()
print(a)
# parse.print_help()
# print(parse)
'''

'''
def get_positional_encoding(max_seq_len, embed_dim):
    # 初始化一个positional encoding
    # embed_dim: 字嵌入的维度
    # max_seq_len: 最大的序列长度
    positional_encoding = np.array([
        [pos / np.power(10000, 2 * i / embed_dim) for i in range(embed_dim)]
        if pos != 0 else np.zeros(embed_dim) for pos in range(max_seq_len)])
    positional_encoding[1:, 0::2] = np.sin(positional_encoding[1:, 0::2])  # dim 2i 偶数
    positional_encoding[1:, 1::2] = np.cos(positional_encoding[1:, 1::2])  # dim 2i+1 奇数
    # 归一化, 用位置嵌入的每一行除以它的模长
    # denominator = np.sqrt(np.sum(position_enc**2, axis=1, keepdims=True))
    # position_enc = position_enc / (denominator + 1e-8)
    return positional_encoding

positional_encoding = get_positional_encoding(max_seq_len=2, embed_dim=5)
print(positional_encoding)
'''

'''
tokenizer = BertTokenizer.from_pretrained('./BERT-demo/bert-base-chinese')
data = pd.read_excel('./BERT-demo/data/dianping_train_test.xls')
context = data.iloc[:, 1].tolist()
label = data.iloc[:, -1].tolist()

token_seq = list(map(tokenizer.tokenize, context))


def trunating_and_padding(seq, max_len):
    if len(seq) > max_len - 2:
        seq = seq[:max_len-2]
    seq = ['[CLS]'] + seq + ['[SEP]']

    seq_ids = tokenizer.convert_tokens_to_ids(seq)

    padding = [0] * (max_len - len(seq))
    seq_ids += padding
    attention_mask = [1] * len(seq) + padding
    seq_segment = [0] * len(seq) + padding

    assert len(seq_ids) == max_len
    assert len(attention_mask) == max_len
    assert len(seq_segment) == max_len
    return seq_ids, attention_mask, seq_segment

result = map(trunating_and_padding, token_seq, [10] * len(token_seq))
seq_ids, attention_mask, seq_segment = zip(*result)
seq_ids = torch.tensor(seq_ids)
attention_mask = torch.tensor(attention_mask)
seq_segment = torch.tensor(seq_segment)

train_data = TensorDataset(seq_ids, attention_mask, seq_segment)
data_loader = DataLoader(dataset=train_data, batch_size=8)
for i in data_loader:
    print(i)
    break
'''

'''
dtype = torch.FloatTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sentences = ["jack like dog", "jack like cat", "jack like animal",
  "dog cat animal", "banana apple cat dog like", "dog fish milk like",
  "dog cat animal like", "jack like apple", "apple like", "jack like banana",
  "apple banana jack movie book music like", "cat dog hate", "cat dog like"]

word_sequence = " ".join(sentences).split() # ['jack', 'like', 'dog', 'jack', 'like', 'cat', 'animal',...]
vocab = list(set(word_sequence)) # build words vocabulary
word2idx = {w: i for i, w in enumerate(vocab)} # {'jack':0, 'like':1,...}

# Word2Vec Parameters
batch_size = 8
embedding_size = 2  # 2 dim vector represent one word
C = 2 # window size
voc_size = len(vocab)

skip_grams = []
for idx in range(C, len(word_sequence) - C):
  center = word2idx[word_sequence[idx]] # center word
  context_idx = list(range(idx - C, idx)) + list(range(idx + 1, idx + C + 1)) # context word idx
  context = [word2idx[word_sequence[i]] for i in context_idx]
  for w in context:
    skip_grams.append([center, w])

def make_data(skip_grams):
  input_data = []
  output_data = []
  for i in range(len(skip_grams)):
    input_data.append(np.eye(voc_size)[skip_grams[i][0]])
    output_data.append(skip_grams[i][1])
  print(input_data)
  return input_data, output_data


make_data(skip_grams)
'''



# tokenizer = BertTokenizer.from_pretrained('./relation-extraction/bert-base-uncased')
# df = pd.read_csv('./relation-extraction/data/train_clean.csv')
# text = df['text'].tolist()
# x = tokenizer.encode_plus(text)['input_ids']
# y = tokenizer.convert_ids_to_tokens(x)
# e1 = df['e2'].tolist()[4]
# e2 = df['raw_text'].tolist()[4].split()
# x = True if e1 in e2 else False
#
# s = ['par', '##odies', 'word']
# print(text)

# s = ['[CLS]', 'the', 'harm', 'has', 'been', 'caused', 'by', 'the', 'invitation', 'system', '.', '[SEP]']
# e1 = ['harm']
#
# e1_start, e1_len, flag = 0, 1, True
# for i, token in enumerate(s):
#     if token not in e1 and not flag:
#         break
#     if token in e1 and flag:
#         e1_start = i
#         flag = False
#     if token in e1 and not flag:
#         e1_len += 1
#
# print(e1_start)
# print(e1_len)


# x = np.array([13 >> d & 1 for d in range(10)][::-1])
# print(x)

'''
x = [0, 0, 0, 1, 2, 3, 0, 4, 0, 0, 5, 6, 0, 0, 0]
y = [0, 0, 0, 1, 1, 1, 0, 4, 0, 0, 5, 6, 0, 0, 0]


print(accuracy_score(x, y))
print(recall_score(x, y, average='macro'))
print(precision_score(x, y, average='macro'))
print(f1_score(x, y, average='macro'))
print(classification_report(x, y))
'''

'''
tokenizer = AutoTokenizer.from_pretrained('./People Daily NER/hfl/rbt6')
text = [
        '海', '钓', '比', '赛', '地', '点', '在', '厦', '门', '与', '金', '门', '之', '间',
        '的', '海', '域', '。'
    ]

s = tokenizer(text)
print(s)
'''


'''
tokenizer = BertTokenizerFast.from_pretrained('./BERT-demo/bert-base-chinese')
x = '张国荣Forever专辑'
input_ids = tokenizer.tokenize(x)
print('input_ids:', input_ids)

# sub_pos = [4, 4]
# sub_ids = [id for i, id in enumerate(input_ids) if i <= sub_pos[1] and i >= sub_pos[0]]
# print('sub_ids:', sub_ids)
#
# sub_text = tokenizer.decode(sub_ids).replace(' ', '')
# print(sub_text)

tokened = tokenizer(x, return_offsets_mapping=True)
print(tokened)
# offset_mapping = tokened['offset_mapping']
# head, tail = offset_mapping[4]
# print(x[head:tail])
'''

# 初级使用
'''
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger()
logger.info('This is a log info')
logger.debug('Debugging')
logger.warning('Warning exists')
logger.info('Finish')
'''


# 输出到控制台

"""
logger = logging.getLogger()
logger.setLevel(level=logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console = logging.StreamHandler(sys.stdout)
console.setFormatter(formatter)
# console.setLevel(level=logging.INFO)
logger.addHandler(console)
logger.info('This is a log info')
logger.debug('Debugging')
logger.warning('Warning exists')
logger.info('Finish')
"""


# 输出到文件

"""
logger = logging.getLogger()
logger.setLevel(level=logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file = logging.FileHandler(r'C:\\Users\\15421\\Desktop\\output.log', encoding='utf-8')
file.setLevel(level=logging.INFO)
file.setFormatter(formatter)
logger.addHandler(file)
logger.info('This is a log info')
logger.debug('Debugging')
logger.warning('Warning exists')
logger.info('Finish')
"""


'''
#编程的方式记录日志

#记录器
logger1 = logging.getLogger("logger1")
logger1.setLevel(logging.DEBUG)
print(logger1)
print(type(logger1))

logger2 = logging.getLogger("logger2")
logger2.setLevel(logging.INFO)
print(logger2)
print(type(logger2))


#处理器
#1.标准输出
sh1 = logging.StreamHandler()

sh1.setLevel(logging.WARNING)

sh2 = logging.StreamHandler()


# 2.文件输出
# 没有设置输出级别，将用logger1的输出级别(并且输出级别在设置的时候级别不能比Logger的低!!!)，设置了就使用自己的输出级别
fh1 = logging.FileHandler(filename="fh.log",mode='w')

fh2 = logging.FileHandler(filename="fh.log",mode='a')
fh2.setLevel(logging.WARNING)

# 格式器
fmt1 = logging.Formatter(fmt="%(asctime)s - %(levelname)-9s - %(filename)-8s : %(lineno)s line - %(message)s")

fmt2 = logging.Formatter(fmt="%(asctime)s - %(name)s - %(levelname)-9s - %(filename)-8s : %(lineno)s line - %(message)s"
                        ,datefmt="%Y/%m/%d %H:%M:%S")

#给处理器设置格式
sh1.setFormatter(fmt1)
fh1.setFormatter(fmt2)
sh2.setFormatter(fmt2)
fh2.setFormatter(fmt1)

#记录器设置处理器
logger1.addHandler(sh1)
logger1.addHandler(fh1)
logger2.addHandler(sh2)
logger2.addHandler(fh2)

#打印日志代码
logger1.debug("This is  DEBUG of logger1 !!")
logger1.info("This is  INFO of logger1 !!")
logger1.warning("This is  WARNING of logger1 !!")
logger1.error("This is  ERROR of logger1 !!")
logger1.critical("This is  CRITICAL of logger1 !!")

logger2.debug("This is  DEBUG of logger2 !!")
logger2.info("This is  INFO of logger2 !!")
logger2.warning("This is  WARNING of logger2 !!")
logger2.error("This is  ERROR of logger2 !!")
logger2.critical("This is  CRITICAL of logger2 !!")

'''

'''
ADDITIONAL_SPECIAL_TOKENS = ["<e1>", "</e1>", "<e2>", "</e2>"]
tokenizer = BertTokenizer.from_pretrained('./R-BERT/bert-base-uncased')
tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
path = './R-BERT/data/train.tsv'

data = pd.read_csv(path, encoding='utf-8', header=None, sep='\t')
label_list = data.iloc[:, 0].tolist()
text_list = data.iloc[:, -1].tolist()[0]
# text_list = text_list.replace('<e1>', '$')
# text_list = text_list.replace('</e1>', '$')
# text_list = text_list.replace('<e2>', '#')
# text_list = text_list.replace('</e2>', '#')
text = tokenizer.tokenize(text_list)
print(text)
e11_p = text.index('<e1>')
e12_p = text.index('</e1>')
e21_p = text.index('<e2>')
e22_p = text.index('</e2>')

text[e11_p] = '$'
text[e12_p] = '$'
text[e21_p] = '#'
text[e22_p] = '#'
print(text)

e11_p += 1
e12_p += 1
e21_p += 1
e22_p += 1
print(e11_p, e12_p, e21_p, e22_p)
if len(text) > 510:
    text = text[:510]

text = ['[CLS]'] + text + ['[SEP]']
print(text)
text = tokenizer.convert_tokens_to_ids(text)
print(text)
'''

'''
bert_model = BertModel.from_pretrained('./BERT-demo/bert-base-chinese', output_hidden_states=True, output_attentions=True)
tokenizer = BertTokenizer.from_pretrained('./BERT-demo/bert-base-chinese')

text = '让我们来看一下bert的输出都有哪些'
input_ids = torch.tensor([tokenizer.encode(text)], dtype=torch.long)
outputs = bert_model(input_ids)
# print(outputs)
print(outputs.keys())
print(len(outputs['hidden_states']))
print(len(outputs['attentions']))
for i in outputs['hidden_states']:
    print(i.shape)
    break
for i in outputs['attentions']:
    print(i.shape)
    break
# print(input_ids)
# print(len(outputs[2:]))
for i, j in bert_model.named_parameters():
    print(i)
'''

'''
a = torch.arange(12).reshape(2, 6)
x1 = torch.arange(48).reshape(2, 4, 6)
x2 = torch.arange(48, 96).reshape(2, 4, 6)
x3 = torch.arange(96, 144).reshape(2, 4, 6)

y1 = torch.arange(96).reshape(2, 3, 4, 4)
y2 = torch.arange(96, 192).reshape(2, 3, 4, 4)


b = ((x1, x2, x3), (y1, y2))
s = (a, ) + b
print(len(s))
'''

'''
no_decay = ["bias", "LayerNorm.weight"]
s = ['embeddings.word_embeddings.weight',
     'embeddings.position_embeddings.weight',
     'embeddings.token_type_embeddings.weight',
     'embeddings.LayerNorm.weight',
     'embeddings.LayerNorm.bias']

a = [i for i in s if not any(j in i for j in no_decay)]
print(a)
'''

'''
x = torch.arange(24).reshape(2, 3, 4)
# y = x.repeat_interleave(2, dim=-1)
# z = x.repeat(1, 1, 2)
a = x[..., None, ::2]
b = x[..., ::2, :]
print(a)
print(b)
'''

'''
x = torch.arange(120).reshape(2, 3, 4, 5)
y = torch.arange(120, 240).reshape(2, 3, 4, 5)
a = torch.einsum('bmhd,bnhd->bhmn', x, y)
print(a)
x = x.permute(0, 2, 1, 3)
y = y.permute(0, 2, 3, 1)
z = x.matmul(y)
print(z)
print(a == z)
'''
# tokenizer = BertTokenizer.from_pretrained('./BERT-demo/bert-base-chinese')
# text = json.load(open('./CasRel-demo/data/CMED/train_triples.json', 'r', encoding='utf-8'))
# s = text[0]['text']
# # gold_triples = text[0]['triple_list']
# # gold_triples_set = set()
# # for gold_triple in gold_triples:
# #     sub = tokenizer.tokenize(gold_triple[0])
# #     obj = tokenizer.tokenize(gold_triple[2])
# #     gold_triples_set.add((sub, gold_triple[1], obj))
# token = tokenizer.tokenize(s)
# token_ids = tokenizer.encode(token)
# print(token_ids)

'''
# 深拷贝和浅拷贝
import copy
a = [[1, 2, 3], "张小鸡"]
b = a
c = copy.copy(a)
d = copy.deepcopy(a)
print(id(a))
print(id(b))
print(id(c))
print(id(d))
print(id(a[0]))
print(id(b[0]))
print(id(c[0]))
print(id(d[0]))
print(id(a[1]))
print(id(b[1]))
print(id(c[1]))
print(id(d[1]))
a[0].append(4)
a.append(6)
print('*******')
print(a)
print(b)
print(c)
print(d)
print(id(a))
print(id(b))
print(id(c))
print(id(d))
print(id(a[0]))
print(id(b[0]))
print(id(c[0]))
print(id(d[0]))
print(id(a[1]))
print(id(b[1]))
print(id(c[1]))
print(id(d[1]))
'''

a, b, c, d, e, f = [], [], [], [], [], []
# s = {'hour': a, 'status': b, 'P_gas': c, 'P_grid': d, 'P_gshp': e, 'lowest_cost': f}
x1 = {'hour': 21, 'status': 'Optimal', 'P_gas': 132.0, 'P_grid': 214.5, 'P_gshp': 0.0, 'lowest_cost': 242.21315983477578}
x2 = {'hour': 22, 'status': 'Optimal', 'P_gas': 0.0, 'P_grid': 300.0, 'P_gshp': 0.0, 'lowest_cost': 90.69}
x3 = {'hour': 23, 'status': 'Optimal', 'P_gas': 0.0, 'P_grid': 300.0, 'P_gshp': 0.0, 'lowest_cost': 90.69}

res = [x1, x2, x3]
for i in res:
    a.append(i['hour'])
print(a)







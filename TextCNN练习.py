# 2021-12-08  21:28
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras import Input, Model, losses, optimizers
from tensorflow.keras.layers import Embedding, Dense, Conv1D, GlobalMaxPool1D, Concatenate, Dropout
from utils import *
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint



# 定义网络结构
class TextCNN(object):
    def __init__(self, maxlen, max_features, embedding_dims, class_num=5, last_activation='softmax'):
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.class_num = class_num
        self.last_activation = last_activation

    def get_model(self):
        input = Input((self.maxlen, ))
        embedding = Embedding(self.max_features, self.embedding_dims, input_length=self.maxlen)(input)
        convs = []
        for kernel_size in [3, 4, 5]:
            c = Conv1D(128, kernel_size, activation='relu')(embedding)
            c = GlobalMaxPool1D()(c)
            convs.append(c)
        x = Concatenate()(convs)

        output = Dense(self.class_num, activation=self.last_activation)(x)
        model = Model(inputs=input, outputs=output)
        return model

# 数据预处理

data_dir = '../../../NLP实战/News-Classifier-Machine-Learning-and-Deep-Learning/processed_data'
vocab_file = '../../../NLP实战/News-Classifier-Machine-Learning-and-Deep-Learning/vocab/vocab.txt'
vocab_size = 40000

 # embedding层的参数
max_features = 40001
maxlen = 100
batch_size = 64
embedding_dims = 50
epochs = 8

print('数据预处理与加载数据...')

 #如果没有词汇表，重建
if not os.path.exists(vocab_file):
    build_vocab(data_dir, vocab_file, vocab_size)

 #获得词典、类别的映射字典
categories, cat_to_id = read_category()
words, word_to_id = read_vocab(vocab_file)

 #读全部数据
x, y = read_files(data_dir)   #x是特征，y是标签
data = list(zip(x, y))     #list of tuple
del x, y
random.shuffle(data)

 #划分训练集和测试集（只是切分数据，包括标签），这里的data = [('', table),...,('', table)]
 #Allowed inputs are lists, numpy arrays, scipy-sparse matrices or pandas dataframes.
train_data, test_data = train_test_split(data)

 #对文本的词和类别，进行编码
x_train = encode_sentences([content[0] for content in train_data], word_to_id)   #[[id,.....,id],...,[id,.....,id]]
y_train = to_categorical(encode_cate([content[1] for content in train_data], cat_to_id))  #将整型类别列表onehot

x_test = encode_sentences([content[0] for content in test_data], word_to_id)
y_test = to_categorical(encode_cate([content[1] for content in test_data], cat_to_id))

print('对序列进行补齐，padding')

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)    #对list of sequences进行补齐，默认补到pre,或者截断pre
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('构建模型')

model = TextCNN(maxlen, max_features, embedding_dims).get_model()
model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

 #设定回调函数
my_callbacks = [ModelCheckpoint('../../../NLP实战/News-Classifier-Machine-Learning-and-Deep-Learning/CNN_model.h5',
                                verbose=1), EarlyStopping(monitor='val_accuracy', patience=2, mode='max')]

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=my_callbacks,
                    validation_data=(x_test, y_test))

print(history)
















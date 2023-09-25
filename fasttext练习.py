# 2021-12-22  21:55
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
import jieba
jieba.setLogLevel(jieba.logging.INFO)
import random
import fasttext



#设定类别的字典
cate_dic = {'technology':1, 'car':2, 'entertainment':3, 'military':4, 'sports':5}

df_entertainment = pd.read_csv('../../../NLP实战/'
                               'News-Classifier-Machine-Learning-and-Deep-Learning/origin_data/entertainment_news.csv',
                               encoding='utf-8', engine='python')
df_entertainment.dropna(inplace=True)

df_technology = pd.read_csv('../../../NLP实战/'
                               'News-Classifier-Machine-Learning-and-Deep-Learning/origin_data/technology_news.csv',
                               encoding='utf-8', engine='python')
df_technology.dropna(inplace=True)

df_car = pd.read_csv('../../../NLP实战/'
                               'News-Classifier-Machine-Learning-and-Deep-Learning/origin_data/car_news.csv',
                               encoding='utf-8', engine='python')
df_car.dropna(inplace=True)

df_military = pd.read_csv('../../../NLP实战/'
                               'News-Classifier-Machine-Learning-and-Deep-Learning/origin_data/military_news.csv',
                               encoding='utf-8', engine='python')
df_military.dropna(inplace=True)

df_sports = pd.read_csv('../../../NLP实战/'
                               'News-Classifier-Machine-Learning-and-Deep-Learning/origin_data/sports_news.csv',
                               encoding='utf-8', engine='python')
df_sports.dropna(inplace=True)


entertainment = df_entertainment['content'].values.tolist()[:20000]   #df['']<class 'pandas.core.series.Series'>
technology = df_technology['content'].values.tolist()[1000:21000]  #df[''].values <class 'numpy.ndarray'>
car = df_car['content'].values.tolist()[1000:21000]
military = df_military['content'].values.tolist()[:20000]
sports = df_sports['content'].values.tolist()[:20000]

#载入停用词
stopwords = pd.read_csv('../../../NLP实战/News-Classifier-Machine-Learning-and-Deep-Learning/origin_data/stopwords.txt',
                        index_col=False, names=['stopword'], quoting=3, sep='\t', encoding='utf-8', engine='python')

stopwords = stopwords['stopword'].values    #type(stopwords)  numpy.ndarray

def preprocess_text(content_lines, sentences, category):
    for line in content_lines:
        try:
            segs = jieba.lcut(line)
            sentence = []
            for seg in segs:
                if len(seg) > 1 and seg not in stopwords:
                    sentence.append(seg)
            sentences.append('__label__' + str(category) + ' , ' + ' '.join(sentence))

        except:
            print(line)
            continue

sentences = []
preprocess_text(entertainment, sentences, cate_dic['entertainment'])
preprocess_text(technology, sentences, cate_dic['technology'])
preprocess_text(car, sentences, cate_dic['car'])
preprocess_text(sports, sentences, cate_dic['sports'])

random.shuffle(sentences)

# print(sentences[:10])

#写文件
print('writing data to fasttext format...')

out_f = open('../../../train_data.txt', 'w',
             encoding='utf-8')

for line in sentences:
    out_f.write(line + '\n')

out_f.close()
print('done!')

#fasttext模型生成
classifier = fasttext.train_supervised(input='../../../train_data.txt',
                                       dim=100, epoch=5, lr=0.1, loss='softmax',
                                       wordNgrams=2)
classifier.save_model('../../../classifier.model')

result = classifier.test('../../../train_data.txt')
#  return [样本个数, 准确率, 召回率]
print(type(result))
print('P@1:', result[1])
print('R@1:', result[2])
print('Number of examples:', result[0])


classifier = fasttext.load_model('../../../classifier.model')
label_to_cate = {'__label__1':'technology', '__label__2':'car', '__label__3':'entertainment',
                 '__label__4':'military', '__label__5':'sports'}

texts = '这 是 中国 制造 宝马 汽车'
labels = classifier.predict(texts)
print(type(labels))
print(label_to_cate[labels[0][0]])

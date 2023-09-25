# 2021-12-02  22:15
import warnings
warnings.filterwarnings('ignore')
import jieba
jieba.setLogLevel(jieba.logging.INFO)
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)
import random
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score

#文本读取
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

entertainment = df_entertainment['content'].values.tolist()[:20000]
technology = df_technology['content'].values.tolist()[1000:21000]
car = df_car['content'].values.tolist()[1000:21000]
military = df_military['content'].values.tolist()[:20000]
sports = df_sports['content'].values.tolist()[:20000]
# print(entertainment[10])

# 去停用词
stopwords = pd.read_csv('../../../NLP实战/News-Classifier-Machine-Learning-and-Deep-Learning/origin_data/stopwords.txt',
                        index_col=False, names=['stopwords'], sep='\t', quoting=3, encoding='utf-8', engine='python')
stopwords = stopwords['stopwords'].values
# print(stopwords, type(stopwords))     不加values，是series，.values后是<class 'numpy.ndarray'>

def preprocess_text(content_lines, sentences, category, target_path):
    out_f = open('../../../NLP实战/News-Classifier-Machine-Learning-and-Deep-Learning/'
                 + target_path + '/' + category + '.txt', 'w')

    for line in content_lines:
        try:
            segs = jieba.lcut(line)
            sentence = []
            for seg in segs:
                if len(seg) > 1 and seg not in stopwords:
                    sentence.append(seg)
            sentences.append((' '.join(sentence), category))      #join(iterable),iterable里面的元素必须是str
            out_f.write(' '.join(sentence) + '\n')
        except:
            print(line)
            continue
    out_f.close()

sentences = []
preprocess_text(technology, sentences, 'technology', 'processed_data')
preprocess_text(car, sentences, 'car', 'processed_data')
preprocess_text(entertainment, sentences, 'entertainment', 'processed_data')
preprocess_text(military, sentences, 'military', 'processed_data')
preprocess_text(sports, sentences, 'sports', 'processed_data')

random.shuffle(sentences)

# print(sentences[:10])
X, y = zip(*sentences)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1234)
print(len(X), type(X), len(X_train))

vec = CountVectorizer(analyzer='word', max_features=20000, ngram_range=(1, 4))
vec.fit(X_train)
data_x_train = vec.transform(X_train)
data_x_test = vec.transform(X_test)

classifier = MultinomialNB()
classifier.fit(data_x_train, y_train)

score = classifier.score(data_x_test, y_test)
print('准确率为{:.3f}'.format(score), type(score))

y_pred = classifier.predict(data_x_test)
acc = accuracy_score(y_test, y_pred)
print('准确率为%.3f'%acc, type(acc))

#交叉验证
def stratifiedkfold(x, y, clf_class, **kwargs):
    strKFold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    y_pr = y[:]
    for train_index, test_index in strKFold.split(x, y):
        X_train, X_test = x[train_index], x[test_index]
        y_train = y[train_index]
        clf = clf_class(**kwargs)
        clf.fit(X_train, y_train)
        y_pr[test_index] = clf.predict(X_test)
    return y_pr

NB = MultinomialNB
print(precision_score(y, stratifiedkfold(vec.transform(X), np.array(y), NB), average='macro'))















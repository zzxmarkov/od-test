# 2021-11-13  23:59
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from joblib import dump, load
import numpy as np
import re
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)




"""in_f = pd.read_csv('../../../NLP实战/language_detector/notebooks/data.csv')
in_f.iloc[:, 0] = in_f.iloc[:, 0].str.strip()   #loc只能输名字，iloc输索引，他们都是默认按行取
print(in_f.head(5))

# 数据读入
in_f = open('../../../NLP实战/language_detector/notebooks/data.csv')
lines = in_f.readlines()
in_f.close()
dataset = [(line.strip()[:-3], line.strip()[-2:]) for line in lines]
print(dataset[:5])

# 划分训练集和测试集
x, y = zip(*dataset)  #tuple：((feature),(tabel))

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)

def remove_noise(document):
    noise_pattern = re.compile('|'.join(['http\S+', '\@\w+', '\#\w+']))
    clean_text = re.sub(noise_pattern, '', document)
    return clean_text.strip()

s = remove_noise('Trump images are now more popular than cat gifs. @trump #trends http://www.trumptrends.html')
print(s)

vec = CountVectorizer(lowercase=True, analyzer='char_wb', ngram_range=(1, 2), max_features=1000,
                      preprocessor=remove_noise)

vec.fit(x_train)

def get_features(x):
    vec.transform(x)

result = vec.transform(['Trump images are now more popular than cat gifs. @trump #trends http://www.trumptrends.html'])

classifier = MultinomialNB().fit(vec.transform(x_train), y_train)
acc = classifier.score(vec.transform(x_test), y_test)
# y_pre = classifier.predict(vec.transform(x_test))
# acc = accuracy_score(y_pre, y_test)
print(acc)"""

class LanguageDetector():
    def __init__(self, classifier=MultinomialNB()):
        self.classifier = classifier
        self.vectorizer = CountVectorizer(lowercase=True, ngram_range=(1, 2),
                                          max_features=1000, analyzer='char_wb', preprocessor=self._remove_noise)

    def _remove_noise(self, document):
        noise_pattern = re.compile('|'.join(['http\S+', '\@\w+', '\#\w+']))
        clean_text = re.sub(noise_pattern, '', document)
        return clean_text

    def features(self, X):
        return self.vectorizer.transform(X)

    def fit(self, X, y):
        self.vectorizer.fit(X)
        self.classifier.fit(self.features(X), y)


    def predict(self, X):
        return self.classifier.predict(self.features(X))

    def score(self, X, y):
        return self.classifier.score(self.features(X), y)

    def save_model(self, path):
        dump((self.classifier, self.vectorizer), path)

    def load_model(self,path):
        self.classifier, self.vectorizer = load(path)


in_f = open('../../../NLP实战/language_detector/notebooks/data.csv')
lines = in_f.readlines()
in_f.close()
dataset = [(line.strip()[:-3], line.strip()[-2:]) for line in lines]

X, y = zip(*dataset)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

language_detector = LanguageDetector()
language_detector.fit(X_train, y_train)
print(language_detector.score(X_test, y_test))
print(language_detector.predict(["This is an English sentence"]))   #predict的参数，必须为可迭代对象

# model_path = '../../../language_detector.model'
# language_detector.save_model(model_path)

# new_language_detector = LanguageDetector()
# new_language_detector.load_model(model_path)

# print(new_language_detector.predict(["10 der welt sind bei"]))












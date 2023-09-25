# 2021-11-29  20:29
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import jieba
jieba.setLogLevel(jieba.logging.INFO)
import jieba.analyse as analyse
import gensim
from gensim import corpora, models, similarities

#读取文本
"""df = pd.read_csv('../../../NLP实战/News-Classifier-Machine-Learning-and-Deep-Learning/origin_data/technology_news.csv',
                 encoding='utf-8', engine='python')
df.dropna(inplace=True)
# print(df.columns.values)
lines = df['content'].values.tolist()
content = ''.join(lines)

#抽取关键词
keywords = analyse.extract_tags(content, topK=30, withWeight=False, allowPOS=())
print(keywords)"""


"""df = pd.read_csv('../../../NLP实战/News-Classifier-Machine-Learning-and-Deep-Learning/origin_data/military_news.csv',
                 encoding='utf-8', engine='python')

df.dropna(inplace=True)
# print(df.columns.values)
lines = df['content'].values.tolist()
content = ''.join(lines)
# 基于TF-IDF
keywords = '  '.join(analyse.extract_tags(content, topK=30, withWeight=False, allowPOS=()))
print(keywords)
#基于textrank
keywords = '  '.join(analyse.textrank(content, topK=30, withWeight=False))
print(keywords)"""

#载入停用词
stopwords = pd.read_csv('../../../NLP实战/News-Classifier-Machine-Learning-and-Deep-Learning/origin_data/stopwords.txt',
                        names=['stopwords'], index_col=False, sep='\r\t', quoting=3, encoding='utf-8', engine='python')
stopwords = stopwords['stopwords'].values
# print(type(stopwords))
df = pd.read_csv('../../../NLP实战/News-Classifier-Machine-Learning-and-Deep-Learning/origin_data/technology_news.csv',
                 encoding='utf-8', engine='python')
df.dropna(inplace=True)
lines = df['content'].values.tolist()

sentences = []
# for line in lines:
#     try:
#         segs = jieba.lcut(line)
#         sentence = []
#         for seg in segs:
#             if len(seg) > 1 and seg not in stopwords:
#                 sentence.append(seg)
#         sentences.append(sentence)
#     except:
#         print(line)
#         continue
for line in lines:
    try:
        segs = jieba.lcut(line)
        segs = list(filter(lambda x: len(x) > 1, segs))
        segs = list(filter(lambda x: x not in stopwords, segs))
        sentences.append(segs)
    except:
        print(line)
        continue
# print(sentences[:2])

# for word in sentences[5]:
#     print(word, end='\t')

# 词袋模型
# 先生成个字典
dictionary = corpora.Dictionary(sentences)
# print(dictionary.items())
#统计每句话中，每个词的frequence，生成[(id, frequence)]
corpus = [dictionary.doc2bow(sentence) for sentence in sentences]
# print(corpus[:2])

# LDA模型
lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=20)
# #return a string
# print(lda.print_topic(19))
# #return a list of (id, probability)
# print(lda.get_topic_terms(19))
# #return a list of (word, probability)
# print(lda.show_topic(19))
for topic in lda.show_topic(3):
    print(topic[0])
# print(lda.print_topics(num_topics=10, num_words=4)) # a list of (id, print_topic())
# print(lda.show_topics(num_topics=10, num_words=4)) # a list of (id, print_topic())


text5 = ['徐立', '商汤', '科技', 'CEO', '谈起', '本次参展', '谈到', '成立', '刚刚', '两年', '创业',
         '公司', '参展', '展示', '最新', '人工智能', '技术', '产品', '表达', '人工智能', '理解', '人工智能',
         '特定', '领域', '超越', '人类', '广泛应用', '标志', 'Master', '胜利', '围棋', '世界', '开拓', '局面', '不谋而合']

bow = dictionary.doc2bow(text5)
print(list(bow))
ndarray = lda.inference([bow])[0]
print(ndarray)
for e, v in enumerate(ndarray[0]):
    print('主题%s的可能性%.3f'%(e, v))




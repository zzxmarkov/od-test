# 2021-11-23  22:14
import warnings
warnings.filterwarnings("ignore")
import jieba
jieba.setLogLevel(jieba.logging.INFO)    #setLogLevel():default_logger.setLogLevel(logging.DEBUG)
import numpy as np
import pandas as pd
import codecs
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.figsize'] = (10.0, 5.0)
from wordcloud import WordCloud, ImageColorGenerator
from imageio import imread
pd.set_option('display.max_columns', None)
pd.set_option('display.max_row', None)
pd.set_option('display.width', None)

"""
#娱乐新闻
df = pd.read_csv('../../../NLP实战/News-Classifier-Machine-Learning-and-Deep-Learning/origin_data/entertainment_news.csv',
                 encoding='UTF-8', engine='python')
df = df.dropna()
content = df['content'].values.tolist()

# s = '我爱自然语言处理'
# print(type('|'.join(jieba.cut(s))))  #jieba.cut()：print('str'), jieba.lcut():print('list')

segment = []
for line in content:
    try:
        segs = jieba.lcut(line)   #必须是'str'
        for seg in segs:
            if len(seg) >1 and seg != '\r\n':
                segment.append(seg)
    except:
        print(line)

#去停用词
word_df = pd.DataFrame({'segment': segment})
# print(word_df.shape)
stop_word_df = pd.read_csv('../../../NLP实战/News-Classifier-Machine-Learning-and-Deep-Learning/origin_data/stopwords.txt',
                           index_col=False, names=['stopword'], quoting=3, sep='\r\t', encoding='UTF-8', engine='python')
# print(stop_word_df.shape)
word_df = word_df[~word_df['segment'].astype(str).isin(stop_word_df['stopword'].astype(str))]
# print(word_df.shape)

# 词频统计
word_stat = word_df.groupby('segment')['segment'].agg([('计数', np.size)]) #agg()想要增加列，不能传字典，只能传元组的列表！！！
word_stat = word_stat.reset_index().sort_values(by='计数', ascending=False)

#设定图像尺寸
# matplotlib.rcParams['figure.figsize'] = (12.0, 12.0)
plt.figure(figsize=(12, 12))
# 设置词云的字体
# wordcloud = WordCloud(font_path='../../../NLP实战/News-Classifier-Machine-Learning-and-Deep-Learning/origin_data/simhei.ttf',
#                       background_color='white', max_font_size=80)

# word_frequence = {x[0]: x[1] for x in word_stat.head(1000).values}
# fit_words()必须得是字典
# wordcloud = wordcloud.fit_words(word_frequence)
# plt.imshow(wordcloud)
# plt.show()
#读取图片
bimg = imread('../../../NLP实战/News-Classifier-Machine-Learning-and-Deep-Learning/img/entertainment.jpeg')
wordcloud = WordCloud(font_path='../../../NLP实战/News-Classifier-Machine-Learning-and-Deep-Learning/origin_data/simhei.ttf',
                      background_color='white', max_font_size=80, mask=bimg)

# 获取词频，这里必须转成字典
wordfrequence = {x[0]: x[1] for x in word_stat.head(1000).values}
wordcloud = wordcloud.fit_words(wordfrequence)
# 图片生成器获取图片颜色
bimgcolor = ImageColorGenerator(bimg)

plt.imshow(wordcloud.recolor(color_func=bimgcolor))
plt.show()"""

# 体育新闻
#读取文本
df = pd.read_csv('../../../NLP实战/News-Classifier-Machine-Learning-and-Deep-Learning/origin_data/sports_news.csv',
                 encoding='utf-8', engine='python')
df.dropna(inplace=True)
content = df["content"].values.tolist()

#分词
segment = []
for line in content:
    try:
        segms = jieba.lcut(line)
        for segm in segms:
            if len(segm) > 1 and segm != '\r\n':
                segment.append(segm)
    except:
        print(line)
        continue

#去停用词
words_df = pd.DataFrame({'segment': segment})   #注意写成字典

stop_words = pd.read_csv('../../../NLP实战/News-Classifier-Machine-Learning-and-Deep-Learning/origin_data/stopwords.txt',
                         index_col=False, names=['stopwords'], quoting=3, sep='\r\t', encoding='utf-8', engine='python')
words_df = words_df[~words_df['segment'].astype(str).isin(stop_words['stopwords'].astype(str))]

#统计词频
word_stat = words_df.groupby('segment')['segment'].agg([('计数', np.size)])
word_stat = word_stat.reset_index().sort_values(by='计数', ascending=False)
# print(word_stat.head())

#可视化
# wordcloud = WordCloud(font_path='../../../NLP实战/News-Classifier-Machine-Learning-and-Deep-Learning/origin_data/simhei.ttf',
#                       background_color='black', max_font_size=80)
#
# wordfrequence = {x[0]: x[1] for x in word_stat.head(1000).values}
# print(wordfrequence)
# wordcloud = wordcloud.fit_words(wordfrequence)
#
# plt.figure(figsize=(10, 8))
# plt.imshow(wordcloud)
# plt.show()

img = imread('../../../NLP实战/News-Classifier-Machine-Learning-and-Deep-Learning/img/sports.jpeg')
wordcloud = WordCloud(font_path='../../../NLP实战/News-Classifier-Machine-Learning-and-Deep-Learning/origin_data/simhei.ttf',
                      max_font_size=200, mask=img, background_color='black')

wordfrequence = {x[0]: x[1] for x in word_stat.head(1000).values}
wordcloud = wordcloud.fit_words(wordfrequence)

img_color = ImageColorGenerator(img)
plt.figure(figsize=(15, 12))
plt.axis('off')
plt.imshow(wordcloud.recolor(color_func=img_color))
plt.show()




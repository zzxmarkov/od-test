# -*- coding: utf-8 -*-
# @Time   : 2022/9/9 21:32


import numpy as np
from collections import Counter

"""
IFIDF: 返回一个列表，有几个文档就有几个元素，[query里的每个字在第一个文档中的ifidf值之和，...query里的每个字在最后一个文档中的ifidf值之和]
self.tf = [{sentence1 每个词的词频}, {sentence2 每个词的词频}, ... ,{sentencen 每个词的词频}]
df:每句话里的词，只算一次，统计的是这个词在多少个文档里出现过
self.idf = {所有词的idf}
"""


class TFIDF_Sim(object):
    def __init__(self, docs_lst):
        self.docs_lst = docs_lst  # word cutted
        self.docs_number = len(docs_lst)
        self.tf = []
        self.idf = {}
        self.init()

    def init(self):
        df = {}
        for document in self.docs_lst:
            temp = {}
            for word in document:
                temp[word] = temp.get(word, 0) + 1/len(document)
            self.tf.append(temp)
            for key in temp.keys():
                df[key] = df.get(key, 0) + 1
        for key, value in df.items():
            self.idf[key] = np.log(self.docs_number / (value + 1))

    def get_score(self, index, query):
        score = 0.0
        for q in query:
            if q not in self.tf[index]:
                continue
            score += self.tf[index][q] * self.idf[q]
        return score

    def get_docs_score(self, query):
        score_list = []
        for i in range(self.docs_number):
            score_list.append(self.get_score(i, query))
        return score_list


class BM25_Sim(object):
    def __init__(self, docs_lst, k1=2, k2=1, b=0.75):
        self.docs_lst = docs_lst
        self.docs_number = len(docs_lst)
        self.avg_docs_len = sum([len(document) for document in docs_lst]) / self.docs_number
        self.tf = []
        self.idf = {}
        self.k1 = k1
        self.k2 = k2
        self.b = b
        self.init()

    def init(self):
        df = {}
        for document in self.docs_lst:
            temp = {}
            for word in document:
                temp[word] = temp.get(word, 0) + 1
            self.tf.append(temp)
            for key in temp.keys():
                df[key] = df.get(key, 0) + 1
        for key, value in df.items():
            self.idf[key] = np.log((self.docs_number - value + 0.5) / (value + 0.5))

    def get_score(self, index, query):
        score = 0.0
        document_len = len(self.tf[index])
        qf = Counter(query)
        for q in query:
            if q not in self.tf[index]:
                continue
            score += self.idf[q] * (self.tf[index][q] * (self.k1 + 1) / (
                        self.tf[index][q] + self.k1 * (1 - self.b + self.b * document_len / self.avg_docs_len))) * (
                                 qf[q] * (self.k2 + 1) / (qf[q] + self.k2))

        return score

    def get_docs_score(self, query):
        score_list = []
        for i in range(self.docs_number):
            score_list.append(self.get_score(i, query))
        return score_list


if __name__ == "__main__":
    import jieba

    document_list = ["行政机关强行解除行政协议造成损失，如何索取赔偿？",
                     "借钱给朋友到期不还得什么时候可以起诉？怎么起诉？",
                     "我在微信上被骗了，请问被骗多少钱才可以立案？",
                     "公民对于选举委员会对选民的资格申诉的处理决定不服，能不能去法院起诉吗？",
                     "有人走私两万元，怎么处置他？",
                     "法律上餐具、饮具集中消毒服务单位的责任是不是对消毒餐具、饮具进行检验？"]
    doc_list = [list(jieba.cut(doc)) for doc in document_list]
    query = "走私了两万元，在法律上应该怎么量刑？"
    query_cut = list(jieba.cut(query))
    tf_idf_model = TFIDF_Sim(doc_list)
    # print(f"document_list:{tf_idf_model.docs_lst}")
    # print(f"document_number:{tf_idf_model.docs_number}")
    # print(f"TF:{tf_idf_model.tf}")
    # print(f"IDF:{tf_idf_model.idf}")
    scores = tf_idf_model.get_docs_score(query_cut)

    print(f"document_scores:{scores}")
    print(f"The most similarity to {query} is {document_list[scores.index(max(scores))]}")


    # bm25_sim = BM25_Sim(doc_list)
    # print(f"document_list:{bm25_sim.docs_lst}")
    # print(f"document_number:{bm25_sim.docs_number}")
    # print(f"TF:{bm25_sim.tf}")
    # print(f"IDF:{bm25_sim.idf}")
    # scores = bm25_sim.get_docs_score(query_cut)
    #
    # print(f"document_scores:{scores}")
    # print(f"The most similarity to {query} is {document_list[np.argmax(scores)]}")
# 2022-03-15  11:43
import warnings
warnings.filterwarnings('ignore')
import numpy as np

from pyhanlp import *
'''
for term in HanLP.segment('下雨天地面积水'):
    print('{}\t{}'.format(term.word, term.nature), end=' ')
'''


dics = ['我', '北京', '研究生', '大学', '生命', '起', '生', '源', '京',
        '北', '命', '研究', '究', '研', '读', '北京大学', '就读', '就', '起源']
text = '就读北京大学研究生命起源'
'''
def fully_segment(text, dics):
    words = []
    for i in range(len(text)):
        for j in range(i + 1, len(text) + 1):
            word = text[i:j]
            if word in dics:
                words.append(word)
    return words

print(fully_segment(text, dics))
'''

def forward_segment(text, dics):
    words = []
    i = 0
    while i < len(text):
        longest_word = text[i]
        j = i + 1
        while j <= len(text):
            word = text[i:j]
            if word in dics and len(word) > len(longest_word):
                longest_word = word

            j += 1


        words.append(longest_word)
        i += len(longest_word)
    return words


def backward_segment(text, dics):
    words = []
    i = len(text) - 1
    while i >= 0:
        longest_word = text[i]
        for j in range(0, i):
            word = text[j:i + 1]
            if word in dics:
                longest_word = word
                break

        words.insert(0, longest_word)
        i -= len(longest_word)
    return words


def count_single_char(word_list):
    return sum(1 for i in word_list if len(i) == 1)

def bidirectional_segment(text, dics):
    f = forward_segment(text, dics)
    b = backward_segment(text, dics)
    if len(f) < len(b):
        return f
    elif len(f) > len(b):
        return b

    else:
        if count_single_char(f) < count_single_char(b):
            return f
        else:
            return b


print(bidirectional_segment(text, dics))




























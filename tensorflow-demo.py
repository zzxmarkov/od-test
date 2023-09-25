# -*- coding: utf-8 -*-
# @Time   : 2022/10/17 7:24

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('max_colwidth', None)


'''
# x = tf.ones([1, 3])
# y = tf.ones([2, 1])

x = tf.constant([[1, 2], [4, 7]])
y = tf.constant([[3, 4], [5, 6]])


# tf.Tensor(
# [[1. 1.]
#  [1. 1.]], shape=(2, 2), dtype=float32)
# 广播操作，broadcast
z = tf.multiply(x, y)

# tf.Tensor([[2.]], shape=(1, 1), dtype=float32)
# 就是正常的矩阵乘法
# z = tf.matmul(x, y)
print(x)
print(y)
print(z)
'''

'''
x = np.arange(12).reshape(3, 4)
dataset = tf.data.Dataset.from_tensor_slices(x).repeat(2).shuffle(buffer_size=100).batch(2)
for data in dataset:
    print(data)

print(x)
'''

# 数据生成器
'''
def data_generator():
    data = np.arange(12).reshape(3, 4)
    for y, x in enumerate(data):
        yield x, y

def dataset():
    # ds = tf.data.Dataset.from_generator(lambda: data_generator(), output_signature=(tf.TensorSpec(shape=(4,), dtype=tf.int32),
    #                                                                                 tf.RaggedTensorSpec(([]), tf.int32, -1, tf.int64)))
    ds = tf.data.Dataset.from_generator(lambda: data_generator(), output_types=(tf.int32, tf.int32), output_shapes=(4, ()))
    ds = ds.shuffle(3)
    ds = ds.batch(2)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    # return ds
    return list(ds.take(1))

print(dataset())
'''




























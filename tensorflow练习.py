# 2021-12-05  19:50
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, losses



# a = tf.constant(2.)
# b = tf.constant(3.)
# c = tf.add(a, b)
# print(float(c))
"""
a = np.arange(5)
print(tf.is_tensor(a))
b = tf.convert_to_tensor(a)
print(tf.is_tensor(b))
print(b.ndim, b.shape, b.dtype)

tf.cast(b, dtype=tf.int32)

print(b.dtype)


a = tf.range(5)
print(a)
b = tf.Variable(a, name='aaa')
print(b)
print(tf.is_tensor(b))
print(isinstance(b, tf.Variable))
print(isinstance(b, tf.Tensor)) #不推荐用，会输出False
b = a.numpy()
print(b)

a = np.random.rand(2, 2)

c = tf.ones(())  #维数写空，就是0维
print(c, c.ndim)
d = c.numpy()
print(d, d.ndim)

e = int(c)  #必须是scalars才能转
print(e, type(e))
"""

"""
a = np.ones((2, 2))
print(a)

b = np.array([1, 2])
print(b.shape)

print(tf.ones(1))
print(tf.ones([1]))
"""
"""
print(tf.fill((2, 2), 8))
# 初始化
w = tf.random.normal((3, 3), mean=20, stddev=1)
print(w)

b = tf.random.truncated_normal((3, 3), mean=20, stddev=1)
print(b)
"""
"""
out = tf.random.uniform((4, 10))
y = tf.range(4)
y = tf.one_hot(y, depth=10)
print(y)

loss = tf.keras.losses.mse(y, out)
loss = tf.reduce_mean(loss)

print(loss)
"""

# net = tf.keras.layers.Dense(10)
# net.build((4, 8))
# print(net.kernel)
"""
model = tf.keras.Sequential()

model.add(layers.Dense(16))
model.add(layers.Dense(32))
model.add(layers.Dense(1))
model.build(input_shape=(20, 80))
print(model.summary())
"""

input_arrary = np.random.randint(1000, size=(1, 4, 10))
print(input_arrary)
print(input_arrary.shape)

"""
model = tf.keras.Sequential()
model.add(layers.Embedding(1000, 64, input_length=10))



input = (4, 10, 128)
print(input[1:])
"""

"""
x = []
for i in range(5):
    j = [1, 2, 3]
    x.append(j)


print(tf.keras.layers.Concatenate()(x))
"""
"""
# Example 1: (batch_size = 1, number of samples = 4)
y_true = [0, 1, 0, 0]
y_pred = [-18.6, 0.51, 2.94, -12.8]
bce = losses.BinaryCrossentropy(from_logits=True)
print(bce(y_true, y_pred).numpy())
"""
"""
y_true = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
y_pred = [[0.3, 0.3, 0.4], [0.3, 0.4, 0.3], [0.1, 0.2, 0.7]]
# Using 'auto'/'sum_over_batch_size' reduction type.
cce = tf.keras.losses.CategoricalCrossentropy()
print(cce(y_true, y_pred))
"""

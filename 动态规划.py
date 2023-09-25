# 2022-04-21  21:22
import numpy as np
import time
'''
def func(s):
    s1 = s.split(' ')
    i = len(s1) - 1
    s2 = []
    s4 = []
    lst = []
    while i >= 0:
        if s1[i] != '':
            s2.append(s1[i])
        i -= 1
    j = 0
    while j < len(s1):
        if s1[j] == '':
            s4.append(' ')
        j += 1

    # for k in range(len(s2) - 1):
    #     lst.append(s2.pop(0))
    #     lst.append(s4.pop(0))
    # lst.append(s2.pop(0))
    # lst = ' '.join(lst)

    return s1[1]

s = 'i am  NLP   ML'
print(func(s))
'''
'''
def func(s, t):
    dict1 = {}
    dict2 = {}
    for i in range(len(s)):
        dict1[s[i]] = i
        dict2[t[i]] = i

    # if list(dict1.values()) == list(dict2.values()):
    #     return True
    # else:
    #     return False
    return dict1, dict2
'''
'''
def func(li):
    even_id = 0
    odd_id = 1
    while even_id < len(li) and odd_id < len(li):

        if li[-1] % 2 == 0:
            li[even_id], li[-1] = li[-1], li[even_id]
            even_id += 2
        else:
            li[odd_id], li[-1] = li[-1], li[odd_id]
            odd_id += 2
    return li

li = [2, 4, 1, 2, 5, 7, 9, 11]
print(func(li))
'''
'''
def func(li):
    even_list = []
    odd_list = []
    for i in range(len(li)):
        if li[i] % 2 == 0:
            even_list.append(li[i])
        else:
            odd_list.append(li[i])
    res = []
    for j in range(len(even_list) + len(odd_list)):
        if j % 2 == 0 and len(even_list) >0 and len(odd_list) > 0:
            res.append(even_list.pop(0))
        elif j % 2 != 0 and len(even_list) >0 and len(odd_list) > 0:
            res.append(odd_list.pop(0))
    while len(even_list) > 0:
        res.append(even_list.pop(0))
    while len(odd_list) > 0:
        res.append(odd_list.pop(0))

    return res

li = [2, 4, 1, 2, 5, 7, 9, 11]
print(func(li))
'''
'''
def func(s):
    res = []
    for i in range(len(s)):
        for j in range(i, len(s)):
            res.append(s[i:j + 1])
    return res

s = 'abcd'
print(func(s))
'''
# 字符串子序列，位运算法
'''
def sub(s):
    size = len(s)
    sub_len = 1 << size
    res = []
    for i in range(sub_len):
        sub_str = ''
        for j in range(size):
            if (i >> j) % 2 == 1:
                sub_str += s[j]
        res.append(sub_str)
    return res

s = 'abcd'
print(sub(s))
'''

'''
def sub(s, i, res, ans):
    if i == len(s):

        return ans.add(res)
    else:
        sub(s, i + 1, res, ans)
        sub(s, i + 1, res + s[i], ans)
'''
'''
def func(li, L, R):
    if L == R:
        return li[L]

    mid = L + ((R - L) >> 1)
    leftMax = func(li, L, mid)
    rightMax = func(li, mid + 1, R)
    return max(leftMax, rightMax)


def m(li):
    return func(li, 0, len(li) - 1)


li = [3, 6, 5, 7]
print(m(li))
'''
# 递归 汉诺塔
'''
def hanoi(n):
    if n > 0:
        func(n, 'left', 'right', 'mid')


def func(n, _from, to, other):
    if n == 1:
        print('Move 1 from ' + _from + ' to ' + to)
        return

    func(n - 1, _from, other, to)
    print('Move ' + str(n) + ' from ' + _from + ' to ' + to)
    func(n - 1, other, to, _from)

print(hanoi(3))
'''
'''
def m(s):

    res = []
    s1 = list(s)
    if len(s1) == 0:
        return res

    process(s1, 0, res)
    return res


def isrepeat(s, i, j):
    b = True
    for m in range(i, j):
        if s[i] == s[j]:
            b = False
            break
    return b


def process(s1, i, res):
    if i == len(s1):
        res.append(''.join(s1))
        return
    
    for j in range(i, len(s)):
        if not isrepeat(s1, i, j):
            continue

        s1[i], s1[j] = s1[j], s1[i]
        process(s1, i + 1, res)
        s1[i], s1[j] = s1[j], s1[i]

s = 'alibaba'
print(len(m(s)))
'''
# 字符串全排列
'''
def m(s):
    res = []
    s1 = list(s)
    begin = 0
    process(s1, begin, len(s1), res)
    return resort(res)


def process(s1, begin, end, res):
    if begin == end:
        res.append(''.join(s1))

    for i in range(begin, end):
        if begin == i or (s1[i] not in s1[begin:i]):
            s1[i], s1[begin] = s1[begin], s1[i]
            process(s1, begin + 1, end, res)
            s1[i], s1[begin] = s1[begin], s1[i]


def resort(li):
    for i in range(len(li) - 1):
        pos = i
        for j in range(i, len(li)):
            if li[j] < li[pos]:
                pos = j
        li[i], li[pos] = li[pos], li[i]
    return li

s = 'abbc'
print(m(s))
'''


# 背包问题
'''
def m(w, v, bag):
    return process(w, v, 0, bag)


def process(w, v, index, rest):
    if rest < 0:
        return -1
    if index == len(w):
        return 0

    p1 = process(w, v, index + 1, rest)
    # 判断是否超重，如果没有，那就加上他的价值
    p2Next = process(w, v, index + 1, rest - w[index])
    p2 = -1
    if p2Next != -1:
        p2 = p2Next + v[index]
    return max(p1, p2)

w = [2, 4]
v = [3, 4]
bag = 6
print(m(w, v, bag))
'''
# n皇后问题
'''
def m1(n):
    if n == 0:
        return 0
    record = [-1] * n
    return process1(0, record, n)


def process1(i, record, n):
    if i == n:
        return 1
    res = 0
    for j in range(n):
        if isVaild(record, i, j):
            record[i] = j
            res += process1(i + 1, record, n)
    return res


def isVaild(record, i, j):
    # 考虑K行（0~i-1）和i行上，列的关系。(k, record[k]) 和(i, j)
    for k in range(i):
        if (abs(record[k] - j) == abs(i - k)) or j == record[k]:
            return False
    return True




def m2(n):
    limit = (1 << n) - 1
    return process2(limit, 0, 0 ,0)


def process2(limit, colLim, leftLim, rightLim):
    if limit == colLim:
        return 1

    pos = limit & (~(colLim|leftLim|rightLim))
    res = 0
    mostRightOne = 0
    while pos != 0:
        mostRightOne = pos & (~pos + 1)
        pos = pos - mostRightOne
        res += process2(limit, colLim|mostRightOne,
                       (leftLim|mostRightOne) << 1,
                       (rightLim|mostRightOne) >> 1)
    return res

n = 12
start = time.time()
print(m1(n))
print(time.time() - start)

start = time.time()
print(m2(n))
print(time.time() - start)
'''

# ---------------------动态规划---------------------------
# 背包问题
'''
def m(w, v, bag):
    return process(w, v, 0, bag)


def process(w, v, index, rest):
    if rest < 0:
        return -1
    if index == len(w):
        return 0

    p1 = process(w, v, index + 1, rest)
    p2 = process(w, v, index + 1, rest - w[index])
    p2Next = -1
    if p2 != -1:
        p2Next = v[index] + p2
    return max(p1, p2Next)

w = [2, 4, 5, 3]
v = [3, 4, 2, 5]
bag = 6
print(m(w, v, bag))
'''
'''
def m(w, v, bag):
    dp = [[0] * (bag + 1) for i in range(len(w) + 1)]
    for item in range(1, len(w) + 1):
        for rest in range(1, bag + 1):
            if rest - w[item - 1] >= 0:
                dp[item][rest] = max(dp[item - 1][rest - w[item - 1]] + v[item - 1],
                                     dp[item - 1][rest])
            else:
                dp[item][rest] = dp[item - 1][rest]
    return dp[-1][-1]

w = [3, 2, 4, 7]
v = [5, 6, 3, 19]
bag = 11
print(m(w, v, bag))
'''
# 要从最后倒着遍历，不然w，v的值和索引对不上
'''
def m(w, v, bag):
    dp = np.zeros([len(w) + 1, bag + 1], dtype=int)
    index = len(w) - 1
    while index >= 0:
        for rest in range(bag + 1):
            if rest >= w[index]:
                dp[index][rest] = max(dp[index + 1][rest],
                                      dp[index + 1][rest - w[index]] + v[index])
            else:
                dp[index][rest] = dp[index + 1][rest]
        index -= 1
    return dp[0][-1]

w = [3, 2, 4, 7]
v = [5, 6, 3, 19]
bag = 11
print(m(w, v, bag))
'''
# 字符表示
'''
def m(s):
    s1 = list(s)
    return process(s1, 0)


def process(s1, i):
    if i == len(s1):
        return 1
    if s1[i] == '0':
        return 0

    if s1[i] == '1':
        res = process(s1, i + 1)
        if i + 1 < len(s1):
            res += process(s1, i + 2)
        return res

    if s1[i] == '2':
        res = process(s1, i + 1)
        if i + 1 < len(s1) and s1[i + 1] >= '0' and s1[i + 1] <= '6':
            res += process(s1, i + 2)
        return res
    return process(s1, i + 1)

s = '11111'
print(m(s))
'''
'''
def m(s):
    s1 = list(s)
    N = len(s1)
    dp = [-1] * (N + 1)
    dp[N] = 1
    i = N - 1
    while i >= 0:
        if s1[i] == '0':
            dp[i] = 0
        elif s1[i] == '1':
            dp[i] = dp[i+1]
            if i + 1 < N:
                dp[i] += dp[i+2]
        elif s1[i] == '2':
            dp[i] = dp[i+1]
            if i + 1 < N and s1[i] <= '6' and s1[i] >= '0':
                dp[i] += dp[i+2]
        else:
            dp[i] = dp[i+1]
        i -= 1
    return dp

s = '1111'
print(m(s))
'''
# 一维解，背包问题
'''
def m(w, v, bag):
    df = [0] * (bag + 1)
    n = len(w)
    for item in range(1, n+1):
        # 必须要倒序，不然之前的会被修改
        for rest in range(bag, -1, -1):
            if rest >= w[item-1]:
                df[rest] = max(df[rest], df[rest - w[item-1]] + v[item-1])

    return df[-1]

w = [3, 2, 4, 7]
v = [5, 6, 3, 19]
bag = 11
print(m(w, v, bag))
'''
# 完全背包问题
'''
def m(w, v, bag):
    dp = [[0] * (bag + 1) for i in range(len(w) + 1)]
    for item in range(1, len(w)+1):
        for rest in range(0, bag+1):
            if rest >= w[item-1]:
                k = 1
                while k * w[item-1] <= rest:
                    dp[item][rest] = max(dp[item-1][rest], dp[item-1][rest - k * w[item-1]] + k * v[item-1])
                    k += 1
            else:
                dp[item][rest] = dp[item-1][rest]
    return dp[-1][-1]

w = [3, 2, 4, 7]
v = [5, 6, 3, 19]
bag = 11
print(m(w, v, bag))
'''

'''
def m(w, v, bag):
    n = len(w)
    dp = [[0] * (bag+1) for i in range(n+1)]
    for item in range(1, n+1):
        for rest in range(1, bag+1):
            if rest >= w[item-1]:
                dp[item][rest] = max(dp[item-1][rest], dp[item][rest-w[item-1]] + v[item-1])
            else:
                dp[item][rest] = dp[item-1][rest]
    return dp[-1][-1]


w = [3, 2, 4, 7]
v = [5, 6, 3, 19]
bag = 11
print(m(w, v, bag))
'''

'''
def m(w, v, bag):
    n = len(w)
    dp = [0] * (bag+1)
    for i in range(1, n+1):
        for rest in range(1, bag+1):
            if rest >= w[i-1]:
                dp[rest] = max(dp[rest], dp[rest-w[i-1]] + v[i-1])
    return dp[-1]


w = [3, 2, 4, 7]
v = [5, 6, 3, 19]
bag = 11
print(m(w, v, bag))
'''

'''
def m(li, aim):
    if len(li) == 0 or aim < 0 or li == None:
        return 0

    dp = [[0] * (aim + 1) for i in range(len(li) + 1)]
    dp[len(li)][0] = 1

    for item in range(len(li) - 1, -1, -1):
        for rest in range(aim + 1):
            if li[item] <= rest:
                dp[item][rest] = dp[item][rest - li[item]] + dp[item + 1][rest]
            else:
                dp[item][rest] = dp[item + 1][rest]

    return dp

li = [5, 2, 1]
aim = 4
print(m(li, aim))
'''

# s = ['abcd', 'ac', 'cdbb']
# map = [[0] * 26 for i in range(len(s))]
# for i in range(len(s)):
#     for str in s[i]:
#         map[i][ord(str) - ord('a')] += 1
# print(map)

# 求最长公共子序列
'''
def m(str1, str2):
    dp = [[0] * len(str2) for i in range(len(str1))]
    dp[0][0] = 1 if str1[0] == str2[0] else 0
    # 先填第一行和第一列的值，因为递推需要，从左到右，从上往下是递增的
    for i in range(1, len(str1)):
        dp[i][0] = max(dp[i - 1][0], 1 if str1[i] == str2[0] else 0)
    for j in range(1, len(str2)):
        dp[0][j] = max(dp[0][j - 1], 1 if str1[0] == str2[j] else 0)

    for i in range(1, len(str1)):
        for j in range(1, len(str2)):
            # 如果str1[i] != str2[j]，有三种情况，1):'ab12cd3f', 'sd123dsa';
            # 2):'ab12cd3', 'sd123ds'; 3):'ab12cd3f', 'sd123'.
            # 也就是max([i][j-1], [i-1][j], [i-1][j-1]) 因为是递增的，所以[i-1][j-1]肯定是最小的
            # 如果str1[i] == str2[j]，即:'ab12cd3', 'sd123'。[i-1][j-1]+1，也就是四种情况求最大值。
            dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
            if str1[i] == str2[j]:
                dp[i][j] = max(dp[i][j], dp[i -1][j - 1] + 1)
    return dp[-1][-1]

str1 = 'ab12cd3'
str2 = 'sd123ds'
print(m(str1, str2))
'''
'''
def m(str1, str2):
    dp = [[0] * (len(str2) + 1) for i in range(len(str1) + 1)]
    dit = [[0] * (len(str2) + 1) for i in range(len(str1) + 1)]
    # dp[0][0] = 1 if str1[0] == str2[0] else 0
    # for i in range(len(str1)):
    #     dp[i][0] = max(dp[i-1][0], 1 if str1[i] == str2[0] else 0)
    # for j in range(len(str2)):
    #     dp[0][j] = max(dp[0][j-1], 1 if str1[0] == str2[j] else 0)

    for i in range(1, len(str1)+1):
        for j in range(1, len(str2)+1):
            dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            if dp[i-1][j] >= dp[i][j-1]:
                dit[i][j] = 1
            else:
                dit[i][j] = 2
            if str1[i-1] == str2[j-1]:
                dp[i][j] = max(dp[i][j], dp[i-1][j-1] + 1)
                dit[i][j] = 3

    LCS = ''
    x = len(str1)
    y = len(str2)
    while x != 0 and y != 0:
        if str1[x-1] == str2[y-1]:
            LCS += str1[x-1]
        if dit[x][y] == 1:
            x -= 1
        elif dit[x][y] == 2:
            y -= 1
        elif dit[x][y] == 3:
            x -= 1
            y -= 1

    return dit

str1 = 'ab12cd3'
str2 = 'bd123ds'
print(m(str1, str2))
'''










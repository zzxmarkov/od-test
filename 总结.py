# 2022-05-17  20:07
# 字符串的所有子序列

'''
def f(s):
    l = len(s)
    str_l = 1 << l
    res = []
    for i in range(str_l):
        st = ''
        for j in range(l):
            if (i >> j) % 2 == 1:
                st += s[j]
        res.append(st)
    return res

s = 'abcd'
print(f(s))
'''
# 有重复的，无重复的时候用set()
'''
def f(s, i, res, a):
    if i == len(s):
        return a.add(res)

    f(s, i + 1, res, a)
    f(s, i + 1, res + s[i], a)


def m(s):
    res = ''
    a = set()
    f(s, 0, res, a)
    return a

s = 'abbd'
print(m(s))
'''
# 字符串的所有子串
'''
def f(s):
    res = []
    for i in range(len(s)):
        for j in range(i + 1, len(s) + 1):
            res.append(s[i:j])

    return res

s = 'abcd'
print(f(s))
'''
# 字符串全排列
'''
def m(s):
    s = list(s)
    end = len(s)
    res = []
    f(s, 0, end, res)
    return res

def f(s, start, end, res):
    if start == end:
        return res.append(''.join(s))
    # 比如'abc'，把'a'定住，分别用'a','b','c'给'a'交换。这里start='a', i 分别是'a', 'b', 'c'
    for i in range(start, end):
        if s[i] not in s[start:i]:
            s[start], s[i] = s[i], s[start]
            # 把'b'定住，分别用'b','c'给'b'交换。这里start='b', i 分别是'b', 'c'
            f(s, start + 1, end, res)
            s[start], s[i] = s[i], s[start]

s = 'alibaba'
print(len(m(s)))
'''


# 最长公共子串
'''
def f(str1, str2):
    p = 0
    max_l = 0
    dp = [[0] * len(str2) for i in range(len(str1))]
    dp[0][0] = 1 if str1[0] == str2[0] else 0
    for i in range(1, len(str1)):
        dp[i][0] = 1 if str1[i] == str2[0] else 0
    for j in range(1, len(str2)):
        dp[0][j] = 1 if str1[0] == str2[j] else 0

    for i in range(1, len(str1)):
        for j in range(1, len(str2)):
            if str1[i] == str2[j]:
                dp[i][j] = dp[i-1][j-1] + 1
            if dp[i][j] > max_l:
                max_l = dp[i][j]
                p = i + 1
    return str1[p - max_l: p], max_l

str1 = 'abc123cd3'
str2 = 'abc123abds'
print(f(str1, str2))
'''
'''
def f(str1, str2):
    l_max = 0
    p = 0
    dp = [[0 for i in range(len(str2) + 1)] for j in range(len(str1) + 1)]
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
                if l_max < dp[i][j]:
                    l_max = dp[i][j]
                    p = i
    return str1[p - l_max:p], l_max

str1 = 'abc123cd3'
str2 = 'abc123abds'
print(f(str1, str2))
'''
# 最长公共子序列
'''
def f(str1, str2):
    dic = [[0 for i in range(len(str2) + 1)] for j in range(len(str1) + 1)]
    dp = [[0 for i in range(len(str2) + 1)] for j in range(len(str1) + 1)]
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i-1] != str2[j-1]:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
                if dp[i-1][j] > dp[i][j-1]:
                    dic[i][j] = 1
                else:
                    dic[i][j] = 2
            else:
                dp[i][j] = max(dp[i][j], dp[i-1][j-1] + 1)
                dic[i][j] = 0
    m = len(str1)
    n = len(str2)
    res = ''
    while m > 0 and n > 0:
        if str1[m-1] == str2[n-1]:
            res += str1[m-1]
            m -= 1
            n -= 1
        else:
            if dic[m][n] == 1:
                m -= 1
            elif dic[m][n] == 2:
                n -= 1

    return dp[-1][-1], res[::-1]

str1 = 'abc123cad3'
str2 = 'abc123abds'
print(f(str1, str2))
'''














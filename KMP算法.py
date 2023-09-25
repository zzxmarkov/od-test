# 2022-04-10  21:40
'''
def func(s1, s2):
    max = 0
    i = 0
    while i < len(s1):
        m = i
        j = 0
        length = 0
        while (j < len(s2)) and (m < len(s1)):
            if s1[m] == s2[j]:
                m += 1
                j += 1
                length += 1
            else:
                if length == 0:
                    j += 1
                else:
                    break
        max = length if length >= max else max
        i += 1
    return max

s1 = '111112q123'
s2 = '45667112w123'
print(func(s1, s2))
'''
# KMP算法
'''

def nextList(alist):
    next_list = [0] * len(alist)
    if len(alist) == 1:
        next_list[0] = -1
        return next_list
    next_list[0] = -1
    next_list[1] = 0
    cur = 0
    i = 2
    while i < len(next_list):
        if alist[i-1] == alist[cur]:
            next_list[i] = cur + 1
            cur += 1
            i += 1
        elif cur > 0:
            cur = next_list[cur]
        else:
            next_list[i] = 0
            i += 1
    return next_list


def getIndexOf(s, m):
    if ((s == None) or (m == None) or (len(m) == 0) or (len(s) < len(m))):
        return -1
    i = 0
    j = 0
    next = nextList(m)
    while (i < len(s)) and (j < len(m)):
        if s[i] == m[j]:
            i += 1
            j += 1
        elif next[j] == -1:
            i += 1
        else:
            j = next[j]

    return (i - j) if j == len(m) else -1

s = 'abcdabcd'
m = 'dab'
print(getIndexOf(s, m))
'''
'''
def func(s):
    max = 0
    m = '#' + '#'.join(s) + '#'
    for i in range(len(m)):
        k = 1
        leng = 0
        while ((i + k) < len(m)) and ((i - k) >= 0):
            if m[i + k] == m[i - k]:
                leng += 2
                k += 1
            else:
                leng += 1
                break
        max = leng if leng >= max else max
        i += 1
    return max//2


s = 'abbacdddcacccca'
print(func(s))
'''
'''
def findRepeatNumber(li):
    my_dict = {}
    for i in range(len(li)):
        if li[i] not in my_dict:
            my_dict[li[i]] = 1
        else:
            my_dict[li[i]] += 1
    for j in my_dict:
        if my_dict[j] > 1:
            return j

li = [2, 3, 1, 0, 2, 5, 3]
print(findRepeatNumber(li))
'''
'''
def func(li):
    for i in range(len(li)):
        while li[i] != i:
            temp = li[i]
            if li[temp] != li[i]:
                li[temp], li[i] = li[i], li[temp]
            else:
                return temp
'''

# li = [4, 3, 1, 0, 2, 5, 3]
# for i in range(len(li)):
#     while li[i] != i:
#
#         li[i], li[li[i]] = li[li[i]], li[i]
# print(li[2])

'''
def func(s):
    new_s = '#' + '#'.join(s) + '#'
    li = [0] * len(new_s)
    c = -1
    r = -1
    l = 0
    for i in range(len(new_s)):
        if r > i:
            li[i] = min(r - i, li[2 * c - i])
        else:
            li[i] = 1

        while (i + li[i] < len(new_s) and i - li[i] > -1):
            if new_s[i + li[i]] == new_s[i - li[i]]:
                li[i] += 1

            else:
                break

        if i + li[i] > r:
            r = i + li[i]
            c = i

        l = max(l, li[i])
    return l - 1

s = 'abcdcbsbcdc'
print(func(s))
'''
'''
def func(li, w):
    res = []
    i = 0
    while i < (len(li) - w + 1):
        max_value = max(li[i:i + w])
        res.append(max_value)
        i += 1
    return res

li = [4, 3, 5, 4, 3, 3, 6, 7]
print(func(li, 3))
'''

def func(s):
    length = 0
    for i in range(len(s) - 1):
        l = 0
        j = i
        while j < len(s):
            n_s = s[i:j]
            if s[j] in n_s:
                break
            else:
                j += 1
                l += 1
        length = l if l >= length else length
    return length

s  = 'abcabcbb'
print(func(s))


a = [1, 3, 4]
print(a.index(3))























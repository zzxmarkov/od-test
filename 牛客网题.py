# 2022-05-13  19:19

# s = 'ABac 4'
# print(s.lower())
'''
def f(s):
    while len(s) > 8:
        print(s[:8])
        s = s[8:]
    s = s + (8 - len(s)) * '0'
    print(s)

s = 'abcdefgw'
f(s)
'''
'''
def f(s):
    s = list(s[::-1])
    res = 0
    dic = {'A':10, 'B':11, 'C':12, 'D':13, 'E':14, 'F':15}
    for i in range(len(s) - 2):
        if s[i] in dic:
            s[i] = dic[s[i]]
        res += (16 ** i) * int(s[i])
    return res

s = '0x2C'
print(f(s))
'''
'''
def f(x):
    mark = 1
    for i in range(2, int(x ** 0.5) +1):
        if x % i == 0:
            mark = 0
            print(i)
            y = x // i
            f(y)

    if mark == 1:
        print(x)

x = 180
f(x)
'''

# #一个整数x的质因子的求法：
# # step1:在2~x^0.5上,从2开始除x,如果能整除,记录下这个除数,然后用商去继续进行上述的操作,直到商为1；
# # step2:如果除不进，除数加一。如果一直加一，除数大于x^0.5,则说明x的质因子只有它本身。
# # 被除数 ÷ 除数=商···余数
# def FindPrimeNumber(num):
#     lst = []
#     i = 2    #从2开始除num
#     while num != 1:    #商不等1时
#         if num % i == 0:
#             lst.append(i)    #如果能整除,记录下这个除数i
#             num //= i    #更新num,num = 商
#         else:    #如果num除以i除不尽
#             if i>int(num**0.5):    #当i大于根号num时，说明num的质因子只有它本身,此时结束循环
#                 lst.append(num)
#                 break
#             else:
#                 i+=1    #除数i+1
#     for item in lst:
#         print(item,end=' ')
# if __name__=='__main__':
#     x = 180
#     FindPrimeNumber(x)

# 对字典排序，key必须是数字
# dic = {'1':5, '13':8, '9':12, '4':1}
# print(sorted(dic.items()))   # [('1', 5), ('13', 8), ('4', 1), ('9', 12)]
'''
s = list('YUANzhi1987')
dic = {'abc': 2, 'def': 3, 'ghi': 4, 'jkl': 5, 'mno': 6, 'pqrs': 7, 'tuv': 8, 'wxyz': 9}
for i in range(len(s)):
    if s[i] == 'Z':
        s[i] = 'a'
    elif s[i].isupper():
        s[i] = chr(ord(s[i].lower()) + 1)
    elif ord('a') <= ord(s[i]) <= ord('z'):
        for id, data in dic.items():
            if s[i] in id:
                s[i] = str(data)
s = ''.join(s)
print(s)
'''
'''
n = 7
a = n
res = 0
while a > 2:
    num = a // 3
    rem = a % 3
    a = num + rem
    if a <= 2:
        res += 1
    else:
        res += a

print(res)
'''
'''
s = 'aabcddd'
dic = {}
for i in s:
    if i not in dic:
        dic[i] = 1
    else:
        dic[i] += 1
c = min(dic.values())
for i in s:
    if dic[i] == c:
        s = s.replace(i, '')
print(s)
'''
'''
s = 'A Famous Saying: Much Ado About Nothing (2012/8).'
li = []
for i in s:
    if i.isalpha():
        li.append(i)
li.sort(key=lambda x: x.upper())
indx = 0
b = ''
for i in range(len(s)):
    if s[i].isalpha():
        b += li[indx]
        indx += 1
    else:
        b += s[i]
print(b)
'''
'''
s = list('A Famous Saying: Much Ado About Nothing (2012/8).')
ss = [False] * len(s)
li = []
for i in range(len(s)):
    if s[i].isalpha():
        li.append(s[i])
    else:
        ss[i] = s[i]
li.sort(key=lambda x: x.upper())

for i in range(len(s)):
    if s[i].isalpha():
        ss[i] = li[0]
        li.pop(0)
b = ''.join(ss)
print(b)
'''
'''
s = 'I am a student'

li = []


start = 0
end = 0
for i in range(1, len(s)):
    ss = ''
    if s[i].isalpha() and not s[i-1].isalpha():
        start = i
    if not s[i].isalpha() and s[i-1].isalpha():
        end = i
        ss += s[start:end]
        li.append(ss)
if s[-1].isalpha():
    li.append(s[start:])
print(li)
'''
'''
s = "i  lova   NLP"
letter_list = s.split()
space_list = []
_space = ""
word_start = 0
word_end = 0
for index in range(1,len(s)):
    if s[index] == " " and s[index-1] != " ":
        word_start = index
    elif s[index] != " " and s[index-1] == " ":
        word_end = index
        space_list.append(s[word_start:word_end])
print(space_list)
res = []
for i in range(1,len(space_list)+len(letter_list)+1):
    if i % 2 == 0:
        res.append(space_list.pop(0))
    else:
        res.append(letter_list.pop(-1))
res = "".join(res)
print(res)
'''
'''
li = []
for i in range(1, 5):
    li.append([0] * i)
a = 1
for i in range(4):
    for j in range(i + 1):
        li[i][j] = a
        a += 1
li2 = []
for i in range(4):
    s = []
    for j in li:
        if j:
            s.append(str(j.pop()))
    li2.append(' '.join(s))
print(li2)
'''
'''
w = [1, 2]
q = [2, 1]
res = [0]
for i in range(2):
    tmp = [w[i] * j for j in range(q[i] + 1)]
    res = list(set(a + b for a in tmp for b in res))
print(len(res))
'''
'''
s = 'zhangsan'
dic = {}
for i in s:
    if i in dic:
        dic[i] += 1
    else:
        dic[i] = 1

d = sorted(dic.items(), key=lambda x: x[1], reverse=True)
res = 0
a = 26
for i, j in d:
    res += a * j
    a -= 1
print(res)
'''
'''
n = 1000
result = 0
for x in range(1, n + 1):
    res = -x
    for i in range(1, int(x ** 0.5) + 1):
        if x % i == 0:
            m = x // i
            res += i + m
    if res == x:
        print(x)
'''
'''
def f(li):
    max1, max2, max3 = float('-inf'), float('-inf'), float('-inf')
    x, y, z = 0, 0, 0
    for i in range(1, len(li)):
        if li[i] > li[x]:
            x, y, z = i, x, y
        elif li[i] > li[y]:
            y, z = i, y
        elif li[i] > li[z]:
            z = i
    return nums[x], nums[y], nums[z]

nums=[3,2,1,4,5]
print(f(nums))
'''
'''
def f(nums):
    max_a = float('-inf')
    max_b = float('-inf')
    for i in nums:
        if i > max_a:
            max_b = max_a
            max_a = i
        elif i > max_b:
            max_b = i
    return max_a, max_b

nums=[3,2,1,4,5]
print(f(nums))
'''
'''
s = '0xA23A'
res = 0
dic = {'A':10, 'B':11, 'C':12, 'D':13, 'E':14, 'F':15}
s = s[2:][::-1]
for i in range(len(s)):
    if s[i] in dic:
        res += dic[s[i]] * (16 ** i)
    else:
        res += int(s[i]) * (16 ** i)
print(res)
'''
# matrix = [(7, 11), (6, 10), (5, 9)]
# matrix = list(zip(*matrix))[::-1]
# print(matrix)
# dict_one = {'name': 'John', 'last_name': 'Doe', 'job': 'Python Consultant'}
# dict_two = {'name': 'Jane', 'last_name': 'Doe', 'job': 'Community Manager'}
# s = list(zip(dict_one.items(), dict_two.items()))
# print(s) [(('name', 'John'), ('name', 'Jane')), \
#           (('last_name', 'Doe'), ('last_name', 'Doe')), (('job', 'Python Consultant'), ('job', 'Community Manager'))]
# 最长公共前缀
'''
def f(s):
    target = s[0]
    for i in range(1, len(s)):
        target = target[:min(len(target), len(s[i]))]
        for j in range(len(target)):
            if target[j] != s[i][j]:
                if j == 0:
                    return ''
                else:
                    target = target[: j]
                    break
    return target
'''
'''
def f(s):
    res = ''
    for i in list(zip(*s)):
        if len(set(i)) == 1:
            res += i[0]
        else:
            break
    return res


s = ['flower', 'flow', 'flight']
s1 = ['dog', 'racecar', 'car']
print(f(s))
'''
'''
def f(s):
    a, b = 0, 0
    res = []
    s = [i for i in s.split(';')]
    li = ['A', 'D', 'W', 'S']
    for i in s:
        if len(i) > 1 and (i[0] in li) and i[1:].isdigit():
            res.append(i)
    for j in res:
        if j[0] == 'A':
            a -= int(j[1:])
        elif j[0] == 'D':
            a += int(j[1:])
        elif j[0] == 'W':
            b += int(j[1:])
        else:
            b -= int(j[1:])
    return a, b
'''
'''
def f(s):
    s = s.split(';')
    dic = {}
    for i in s:
        if len(i) > 1 and i[0] in list('ASDW') and i[1:].isdigit():
            dic[i[0]] = dic.get(i[0], 0) + int(i[1:])
    x, y = (dic['D'] - dic['A']), (dic['W'] - dic['S'])
    return x, y

s = 'A10;S20;W10;D30;X;A1A;B10A11;;A10;'

print(f(s))
'''
'''
def f(s):
    a, b, c, d = 0, 0, 0, 0
    mark = True
    for i in s:
        if i.isdigit():
            a = 1
        elif i.isupper():
            b = 1
        elif i.islower():
            c = 1
        else:
            d = 1
    for i in range(len(s)):
        if s[i:i+3] in s[i+3:]:
            mark = False
    if len(s) > 8 and (a + b + c + d) >= 3 and mark:
        return 'OK'
    else:
        return 'NG'
'''
'''
s = '10.0.3.193'
s1 = int('167969729')
s = list(map(int, s.split('.')))
res = []
for i in s:
    r = ''
    while i > 0:
        m = i % 2
        i = i // 2
        r += str(m)
    while len(r) < 8:
        r += '0'
    res.append(r[::-1])
num = ''.join(res)
num = num[::-1]
result = 0
for i in range(len(num)):
    result += int(num[i]) * (2 ** i)

r1 = ''
while s1 > 0:
    m1 = s1 % 2
    s1 = s1 // 2
    r1 += str(m1)
re = []
while len(r1) > 8:
    re.append(r1[:8])
    r1 = r1[8:]
re.append(r1 + '0' * (8 - len(r1)))
r3 = []
r4 = 0
for i in re:
    r4 = 0
    for j in range(len(i)):
        r4 += (2 ** j) * int(i[j])
    r3.append(str(r4))
r3 = '.'.join(r3[::-1])
print(r3)
'''
'''
def f(s):
    s = s.strip()
    res = ''
    i = 0
    while i < len(s):
        if s[0] == '-' or s[0] == '+':
            res += s[0]
            s = s.replace(s[0], 'x')
            i += 1
        elif s[i].isdigit():
            res += s[i]
            i += 1
        else:
            break
    if len(res) == 0 or res == '+' or res == '-':
        return 0
    if res[0] == '+':
        res = res.replace('+', '')

    if int(res) > (2 ** 31) - 1:
        res = str((2 ** 31) - 1)
    if int(res) < -(2 ** 31):
        res = str(-(2 ** 31))
    return int(res)
s = '+-5'
print(f(s))
'''
'''
s = 'aeiaaioaaaaeiiiiouuuooaauuaeiu'
dic = {('a', 'e'), ('e', 'i'), ('i', 'o'), ('o', 'u'),
       ('a', 'a'), ('e', 'e'), ('i', 'i'), ('o', 'o'), ('u', 'u'),
       ('x', 'a'), ('e', 'a'), ('i', 'a'), ('o', 'a'), ('u', 'a')}
status = 'x'
ans = 0
c = 0
for i in s:
    if (status, i) in dic:
        if status != 'a' and i == 'a':
            c = 1
        else:
            c += 1
        status = i
    else:
        c = 0
        status = 'x'
    if status == 'u':
        ans = max(ans, c)
print(ans)
'''
'''
s = 'aeiaaioaaaaeiiiiouuuooaauuaeiu'
res = 0
l = 1
v = 1
for i in range(1, len(s)):
    if s[i] >= s[i-1]:
        l += 1
    if s[i] > s[i-1]:
        v += 1
    if s[i] < s[i-1]:
        l = 1
        v = 1
    if v == 5:
        res = max(res, l)
print(res)
'''
s = 'aeiaaioaaaaeiiiiouuuooaauuaeiu'
window = []
res = 0
c = set()
left, right = 0, 0
while right < len(s):
    if len(window) == 0 or s[right] >= window[-1]:
        window.append(s[right])
        c.add(s[right])
        if len(c) == 5:
            res = max(res, len(window))
    else:
        window = []
        c = set()
        left = right
        if s[right] == 'a':
            window.append(s[right])
            c.add(s[right])
    right += 1
print(res)



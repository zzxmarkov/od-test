# 2022-01-23  22:10
import random
import math
#1.两数之和
#暴力解法
"""
def twoSum(nums, target):

    for i in range(len(nums)):
        for j in range(i+1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
    else:
        return None

    sort_id = sorted(range(len(nums)), key=lambda k: nums[k])
    head = 0
    tail = len(nums) - 1
    sum_result = nums[sort_id[head]] + nums[sort_id[tail]]

    while sum_result != target:
        if sum_result < target:
            head += 1
        elif sum_result > target:
            tail -= 1

        sum_result = nums[sort_id[head]] + nums[sort_id[tail]]
    return [sort_id[head], sort_id[tail]]
"""


"""
    mydict = {}
    for i in range(len(nums)):
        remain = target - nums[i]
        if remain in mydict:
            return [i, mydict[remain]]
        else:
            mydict[nums[i]] = i

    else:
        return None
"""
"""
nums = [3, 2, 4]
target = 6
print(twoSum(nums, target))

"""

#头插
"""
class Node:
    def __init__(self, item):
        self.item = item
        self.next = None


def create_linklist(li):
    head = Node(li[0])
    for element in li[1:]:
        node = Node(element)
        node.next = head
        head = node

    return head


def print_linklist(lk):
    while lk:
        print(lk.item, end=',')
        lk = lk.next


lk = create_linklist([1, 2, 3])
print_linklist(lk)
"""
# 尾插
"""
class Node:
    def __init__(self, item):
        self.item = item
        self.next = None


def create_linklist(li):
    head = Node(li[0])
    tail = head
    for element in li[1:]:
        node = Node(element)
        tail.next = node
        tail = node

    return  head

def print_linklist(lk):
    while lk:
        print(lk.item, end=',')
        lk = lk.next


lk = create_linklist([1, 2, 3])
print_linklist(lk)
"""
# 两数之和
"""
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def addTwoNumbers(l1: ListNode, l2: ListNode):
    result = ListNode(0)
    r = result
    carry = 0

    while l1 or l2:
        if l1:
            x = l1.val
        else:
            x = 0
        if l2:
            y = l2.val
        else:
            y = 0

        r.next = ListNode((x + y + carry) % 10)
        carry = (x + y + carry) // 10
        r = r.next
        if l1 != None:
            l1 = l1.next
        if l2 != None:
            l2 = l2.next
    if carry == 1:
        r.next = ListNode(1)

    return result.next
"""

# s = [54, 26, 93, 17, 17, 31, 44, 55, 20]

# 选择排序
"""
def selected_sort(li):
    for i in range(len(li) - 1):
        pos = i
        for j in range(i + 1, len(li)):
            if li[j] < li[pos]:
                pos  = j
        temp = li[pos]
        li[pos] = li[i]
        li[i] = temp

    return li

print(selected_sort(s))
"""
# 冒泡排序
"""
def bubbled_sort(li):
    for j in range(len(li) - 1, 0, -1):
        i = 0
        while i < j:
            if li[i] > li[i + 1]:
                li[i], li[i + 1] = li[i + 1], li[i]
            i += 1
    return li

print(bubbled_sort(s))
"""
# 插入排序
'''
def inserted_sort(li):
    for i in range(1, len(li)):
        j = i
        while j > 0:
            if li[j] < li[j - 1]:
                li[j], li[j - 1] = li[j - 1], li[j]
                j -= 1
            else:
                break
    return li

print(inserted_sort(s))
'''
'''
def insert_sort(li):
    for i in range(1, len(li)):
        cur = li[i]
        j = i - 1
        while j >=0 and li[j] >= cur:
            li[j + 1] = li[j]
            j -= 1
        li[j + 1] = cur
    return li


print(insert_sort(s))
'''
'''
arr = [3, 2, -1, 6, 7, 2, -2]
def func(arr, l, r):
    li = []
    s = 0
    for i in arr:
        s += i
        li.append(s)
    return li[r] - li[l] + arr[l]
print(func(arr, 3, 6))
'''
'''
def f1():
    return int(random.random() * 17) + 3


def f2():
    s = f1()
    while s == 11:
        s = f1()
    if s > 11:
        return 1
    if s < 11:
        return 0

def f3():
    s = (f2() << 5) + (f2() << 4) + (f2() << 3) + (f2() << 2) + (f2() << 1) + (f2() << 0)
    while s > 40:
        s = (f2() << 5) + (f2() << 4) + (f2() << 3) + (f2() << 2) + (f2() << 1) + (f2() << 0)
    if s < 40:
        return s + 17
'''
'''
def f1():
    if random.random() < 0.84:
        return 0
    else:
        return 1

def f2():
    s1 = f1()
    s2 = f1()
    while s1 == s2:
        s1 = f1()
        s2 = f1()
    if (s1 == 1) and (s2 == 0):
        return 1
    if (s1 == 0) and (s2 == 1):
        return 0


count = 0
for i in range(1000000):
    if f2() == 0:
        count += 1

print(count/1000000)
print(f2())
'''
'''
def random_list(maxLen, maxValue):
    li = []
    for i in range(int(random.random() * maxLen)):
        li.append(int(random.random() * maxValue))
    return li


def isSorted(li):
    if len(li) < 2:
        return True
    maxV = li[0]
    for i in range(1, len(li)):
        if maxV > li[i]:
            return False
        maxV = max(li[i], maxV)
    return li

li = insert_sort(random_list(50, 1000))

print(isSorted(li))
'''

# li = [12, 11, 12, 15, 16, 22, 20, 18, 13, 15, 23, 34, 35]
# li = [3, 2, 1, 4]
'''
def func(li, num):
    first = 0
    last = len(li) - 1
    while first <= last:
        mid = (first + last) // 2
        if li[mid] == num:
            return mid
        if li[mid] < num:
            first = mid + 1
        if li[mid] > num:
            last = mid - 1
    return None
print(func(li, 116))
'''
'''
def func(li, num):
    first = 0
    last = len(li) - 1
    result = -1
    while first <= last:
        mid = (first + last) // 2
        if li[mid] >= num:
            result = mid
            last = mid - 1
        else:
            first = mid + 1
    return result

print(func(li, 54))
'''
'''
def random_list(maxLen, maxValue):
    s = int(random.random() * maxLen)
    li = []
    if s == 0:
        return -1
    if s == 1:
        li.append(int(random.random() * maxValue))
        return li
    else:
        li.append(int(random.random() * maxValue))
        for i in range(1, s):
            next = int(random.random() * maxValue)
            while next == li[i - 1]:
                next = int(random.random() * maxValue)
            li.append(next)

        return li


def func(li):
    if (li == -1) or (len(li) == 1):
        return -1
    if li[0] < li[1]:
        return 0
    if li[-1] < li[-2]:
        return len(li) - 1
    else:
        first = 0
        last = len(li) - 1
        while first < last - 1:
            mid = (first + last) // 2
            if li[mid] > li[mid - 1]:
                last = mid - 1
            if li[mid] < li[mid - 1]:
                if li[mid] < li[mid + 1]:
                    return mid
                else:
                    first = mid + 1
        if li[first] < li[last]:
            return first
        else:
            return last

li = random_list(10, 20)
print(li)
print(func(li))
'''
# 螺旋矩阵
'''
def f(matrix):
    res = []
    i, j = 0, 0
    pos = [[0, 1], [1, 0], [0, -1], [-1, 0]]
    pos_id = 0
    m, n = len(matrix), len(matrix[0])
    for _ in range(m * n):
        res.append(matrix[i][j])
        matrix[i][j] = 'x'
        if i + pos[pos_id][0] >= m or j + pos[pos_id][1] >= n \
            or j + pos[pos_id][1] < 0 or matrix[i+pos[pos_id][0]][j+pos[pos_id][1]] == 'x':
            pos_id = (pos_id + 1) % 4
        i += pos[pos_id][0]
        j += pos[pos_id][1]
    return res
'''
'''
matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
res = []
while matrix:
    res += matrix.pop(0)
    matrix = list(zip(*matrix))[::-1]
print(res)
'''
'''
n = 1221
x = n
y = 0
while x > y:
    rem = x % 10
    y = y * 10 + rem
    x = x // 10
print(x == y or x == y // 10)
'''
'''
s = "MCMXCIV"
r_dict = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
s_dict = {'IV': 4, 'IX': 9, 'XL': 40, 'XC': 90, 'CD': 400, 'CM': 900}
s_set = {'IV', 'IX', 'XL', 'XC', 'CD', 'CM'}
res = 0
for key in s_dict:
    if key in s:
        res += s_dict[key]
        s = s.replace(key, '')
for i in s:
    res += r_dict[i]
print(res)
'''







































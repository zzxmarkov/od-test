# -*- coding: utf-8 -*-
# @Time   : 2022/8/25 18:27

import re
class Node:
    def __init__(self, val = None):
        self.val = val
        self.next = None


def createLink(li):
    head = Node(li[0])
    p = head
    for element in li[1:]:
        node = Node(element)
        p.next = node
        p = node
    return head


def printLink(head):
    res = []
    while head:
        res.append(head.val)
        head = head.next
    return res


class Stack:
    def __init__(self):
        self.head = None

    def push(self, node):
        # node不能被改写，所以必须新建结点！！！
        n = Node(node)
        n.next = self.head
        self.head = n

    def pop(self):
        tmp = self.head.val
        self.head = self.head.next
        return tmp

    def isEmpty(self):
        return self.head == None

    def peek(self):
        return self.head.val


class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0, item)

    def dequeue(self):
        return self.items.pop()


class BinaryTree:
    def __init__(self, val):
        self.key = val
        self.left = None
        self.right = None

head = BinaryTree(5)
head.left = BinaryTree(3)
head.right = BinaryTree(8)
head.left.left = BinaryTree(2)
# head.left.left.left = BinaryTree(4)
head.left.right = BinaryTree(4)
head.right.left = BinaryTree(7)
head.right.right = BinaryTree(10)

head2 = BinaryTree(5)
head2.left = BinaryTree(3)
head2.right = BinaryTree(8)
head2.left.left = BinaryTree(2)
head2.left.left.left = BinaryTree(4)


# 2.两数之和
'''
def addTwoNumbers(l1, l2):
    head1 = l1
    head2 = l2
    carry = 0
    head = Node()
    p = head
    while head1 and head2:
        sum = carry + head1.val + head2.val
        carry = sum // 10
        p.next = Node(sum % 10)
        p = p.next
        head1 = head1.next
        head2 = head2.next
    while head1:
        sum = carry + head1.val
        carry = sum // 10
        p.next = Node(sum % 10)
        p = p.next
        head1 = head1.next
    while head2:
        sum = carry + head2.val
        carry = sum // 10
        p.next = Node(sum % 10)
        p = p.next
        head2 = head2.next
    if carry == 1:
        p.next = Node(1)
    return head.next


l1 = [9, 9, 9, 9, 9, 9, 9]
l2 = [9, 9, 9, 9]
head1 = createLink(l1)
head2 = createLink(l2)
head = addTwoNumbers(head1, head2)
print(printLink(head))
'''

# 167.有序数组的两数之和
'''
def twoSum(li, target):
    n = len(li)
    left = 0
    right = n - 1
    while left < right:
        if li[left] + li[right] == target:
            return [left+1, right+1]
        elif li[left] + li[right] > target:
            while left < right and li[right] == li[right - 1]:
                right -= 1
            right -= 1
        else:
            while left < right and li[left] == li[left + 1]:
                left += 1
            left += 1


li = [3, 24, 50, 79, 88, 150, 345]
target = 200
print(twoSum(li, target))
'''

'''
def twoSum(li, target):
    n = len(li)
    for i in range(n-1):
        left = i + 1
        right = n - 1
        while left <= right:
            mid = left + ((right - left) >> 1)
            if li[i] + li[mid] == target:
                return [i+1, mid+1]
            elif li[i] + li[mid] > target:
                right = mid - 1
            else:
                left = mid + 1


li = [3, 24, 50, 79, 88, 150, 345]
target = 200
print(twoSum(li, target))
'''

# JZ47. 礼物的最大礼物
'''
def maxValue(grid):
    m = len(grid)
    n = len(grid[0])
    dp = [0] * (n+1)
    for i in range(1, m+1):
        for j in range(1, n+1):
            dp[j] = max(dp[j], dp[j-1]) + grid[i-1][j-1]
    return dp[-1]


grid = [
  [1,3,1],
  [1,5,1],
  [4,2,1]
]
print(maxValue(grid))
'''

# 139.单词拆分
'''
def wordBreak(s, woedDict):
    n = len(s)
    dp = [False] * (n+1)
    dp[0] = True
    for i in range(n):
        for j in range(i+1, n+1):
            if dp[i] and s[i:j] in woedDict:
                dp[j] = True
    return dp[-1]


s = 'leetcode'
wordDict = ['leet', 'code']
print(wordBreak(s, wordDict))
'''

'''
def wordBreak(s, wordDict):
    n = len(s)
    dp = [False] * (n+1)
    # 必须设置dp[0] = True, 要是不的话，无法做到s]和s[j:i]都为True，也就是第一个无法匹配
    # dp[i]正好对应的是s[i+1]
    dp[0] = True
    for i in range(1, n+1):
        for word in wordDict:
            if dp[i] or (dp[i-len(word)] and s[i-len(word):i] == word):
                dp[i] = True
    return dp[-1]


s = 'leetcode'
wordDict = ['leet', 'code']
print(wordBreak(s, wordDict))
'''

'''
def wordBreak(s, wordDict):
    n = len(s)
    dp = [False] * (n+1)
    dp[0] = True
    for i in range(1, n+1):
        for j in range(i):
            if dp[j] and s[j:i] in wordDict:
                dp[i] = True
                break
    return dp[-1]


s = 'leetcode'
wordDict = ['leet', 'code']
print(wordBreak(s, wordDict))
'''

'''
# 2351. 第一个出现两次的字母
def repeatedCharacter(s):
    dic = {}
    for i in s:
        if i not in dic:
            dic[i] = 1
        else:
            return i
    return None

s = "abccbaacz"
print(repeatedCharacter(s))
'''

'''
# 1.两数之和
def twoSum(nums, target):
    dic = {}
    for i in range(len(nums)):
        if target - nums[i] not in dic:
            dic[nums[i]] = i
        else:
            return [dic[target - nums[i]], i]
    return None
'''

'''
# 2.两数相加
def addTwoNumbers(l1, l2):
    head = Node()
    p = head
    carry = 0
    while l1 and l2:
        sumNum = l1.val + l2.val + carry
        carry = sumNum // 10
        p.next = Node(sumNum % 10)
        p = p.next
        l1 = l1.next
        l2 = l2.next
    while l1:
        sumNum = l1.val + carry
        carry = sumNum // 10
        p.next = Node(sumNum % 10)
        p = p.next
        l1 = l1.next
    while l2:
        sumNum = l2.val + carry
        carry = sumNum // 10
        p.next = Node(sumNum % 10)
        p = p.next
        l2 = l2.next
    if carry != 0:
        p.next = Node(1)
    return head.next


l1 = [0, 9, 9]
l2 = [3, 9]
head1 = createLink(l1)
head2 = createLink(l2)
head = addTwoNumbers(head1, head2)


def f(head):
    res = []
    while head:
        res.append(head.val)
        head = head.next
    print(res)
f(head)
'''


'''
def findMinNum(arr):
    if arr is None or len(arr) <= 0:
        return
    if arr[0] >= 0:
        return arr[0]
    if arr[-1] <= 0:
        return arr[-1]
    mid = None
    absMin = None
    begin = 0
    end = len(arr) - 1
    while begin < end:
        mid = begin + (end - begin) >> 1
        if arr[mid] == 0:
            return 0
        elif arr[mid] > 0:
            if arr[mid - 1] > 0:
                end = mid - 1
            elif arr[mid - 1] == 0:
                return 0
            else:
                break
        else:

            if arr[mid + 1] < 0:
                begin = mid + 1
            elif arr[mid + 1] == 0:
                return 0
            else:
                break
    if (arr[mid] > 0):
        if arr[mid] < abs(arr[mid - 1]):
            absMin = arr[mid]
        else:
            absMin = arr[mid - 1]
    else:
        if abs(arr[mid]) < abs(arr[mid + 1]):
            absMin = arr[mid]
        else:
            absMin = arr[mid + 1]
    return absMin
'''


'''
# 有序列表找绝对值最小的数
def f(li):
    if li[0] > 0:
        return li[0]
    elif li[-1] < 0:
        return li[-1]
    else:
        start = 0
        end = len(li) - 1
        while start <= end:
            mid = start + ((end - start) >> 1)
            if li[mid] < 0:
                if li[mid + 1] > 0:
                    if abs(li[mid]) < li[mid + 1]:
                        return li[mid]
                    else:
                        return li[mid + 1]
                elif li[mid + 1] < 0:
                    start = mid + 1
                else:
                    return 0
            elif li[mid] > 0:
                if li[mid - 1] > 0:
                    end = mid - 1
                elif li[mid - 1] < 0:
                    if li[mid] < abs(li[mid - 1]):
                        return li[mid]
                    else:
                        return li[mid - 1]
                else:
                    return 0
            else:
                return 0


li = [2, 3, 5, 7, 9]
li1 = list(map(lambda x: -x, [2, 3, 5, 7, 9]))
s = [-5, -3, 2, 3, 4, 5, 6]
print(f(s))
'''


'''
# 21. 调整数组顺序使奇数位于偶数前面
def exchange(li):
    start, end = 0, len(li) - 1
    while start < end:
        if li[start] % 2 == 0:
            li[start], li[end] = li[end], li[start]
            end -= 1
        elif li[start] % 2 == 1:
            start += 1
    return li


li = [1, 2, 3, 4]
print(exchange(li))
'''













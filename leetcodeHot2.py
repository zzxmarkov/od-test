# -*- coding: utf-8 -*-
# @Time   : 2022/7/28 13:24
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

# JZ03. 数组中重复的数字
'''
def findRepeatNumber(nums):
    dic = set()
    for i in range(len(nums)):
        if nums[i] not in dic:
            dic.add(nums[i])
        else:
            return nums[i]
'''
'''
def findRepeatNumber(nums):
    for i in range(len(nums)):
        while nums[i] != i:
            if nums[nums[i]] == nums[i]:
                return nums[i]
            else:
                temp = nums[i]
                nums[i], nums[temp] = nums[temp], nums[i]


nums = [2, 3, 1, 2, 4]
print(findRepeatNumber(nums))
'''

# JZ05. 替换空格
'''
def replaceSpace(s):
    s = s.split(' ')
    return '%20'.join(s)


s = "We are happy."
print(replaceSpace(s))
'''
'''
def replaceSpace(s):
    sp = 0
    for i in range(len(s)):
        if s[i] == ' ':
            sp += 1
    res = [' '] * (len(s) + sp * 2)

    n1 = len(s) - 1
    n2 = len(res) - 1
    while n1 >= 0:
        if s[n1] != ' ':
            res[n2] = s[n1]
            n2 -= 1
        else:
            res[n2 - 2: n2 + 1] = '%20'
            n2 -= 3
        n1 -= 1
    return ''.join(res)

s = "We are happy."
print(replaceSpace(s))
'''

# JZ06. 从尾到头打印链表
'''
def reversePrint(head):
    s = []
    while head:
        s.append(head.val)
        head = head.next
    s = s[::-1]
    return s

li = [1, 3, 2]
head = createLink(li)
print(reversePrint(head))
'''
'''
def reversePrint(head):
    res = []
    process(head, res)
    return res
def process(head, res):
    if head == None:
        return res

    process(head.next, res)
    res.append(head.val)


li = [1, 3, 2]
head = createLink(li)
print(reversePrint(head))
'''
'''
def reversePrint(head):
    if head == None:
        return []
    return reversePrint(head.next) + [head.val]
'''

# JZ09. 用两个栈实现队列
'''
class CQueue:
    def __init__(self):
        self.a = []
        self.b = []

    def appendTail(self, value):
        self.a.append(value)

    def deleteHead(self):
        # 没有弹空，就继续弹
        if self.b:
            return self.b.pop()
        if self.a:
            return -1
        # 只有弹空的时候，才能往进加
        while self.a:
            self.b.append(self.a.pop())
        return self.b.pop()
'''

# JZ10-I 斐波那契数列
'''
def fib(n):
    dp = [0] * (n+1)
    if n == 0:
        return 0
    if n == 1:
        return 1
    dp[1] = 1
    for i in range(2, n+1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[-1]

print(fib(5))
'''
'''
def fib(n):
    if n == 0:
        return 0
    if n == 1:
        return 1
    a = 0
    b = 1
    for i in range(2, n+1):
        t = a + b
        a = b
        b = t
    return b

print(fib(5))
'''

# JZ11. 旋转数组的最小数字
'''
def minArray(numbers):
    for i in range(1, len(numbers)):
        if numbers[i] < numbers[i-1]:
            return numbers[i]
        return numbers[0]

numbers = [2, 2, 2, 0, 1]
print(minArray(numbers))
'''
'''
def minArray(numbers):
    left = 0
    right = len(numbers) - 1
    while left < right:
        mid = ((right - left) >> 1) + left
        if numbers[mid] < numbers[right]:
            right = mid
        elif numbers[mid] > numbers[right]:
            left = mid + 1
        else:
            right -= 1
    return numbers[left]


numbers = [3, 4, 5, 6, 1, 2]
print(minArray(numbers))
'''
# JZ15. 二进制中1的个数
'''
def hammingWeight(n):
    # 消掉最后一位的1：n & (n - 1)
    res = 0
    while n != 0:
        n = n & (n - 1)
        res += 1
    return res

n = 4294967293
print(hammingWeight(n))
'''

# JZ17. 打印从1到最大的n位数
'''
def printNumbers(n):
    s = 10 ** n - 1
    res = [i + 1 for i in range(s)]
    return res


n = 1
print(printNumbers(n))
'''

# 3.无重复字符的最长字串
# 本质就是找到left的位置
'''
def lengthOfLongestSubstring(s):
    dic = {}
    res = 0
    left = 0
    for i in range(len(s)):
        if s[i] not in dic:
            dic[s[i]] = i
        else:
            # 如果重复的在左指针左边，left不变；在左指针右边，left = dic[s[i]] + 1
            left = max(dic[s[i]] + 1, left)
            dic[s[i]] = i
        res = max(res, i - left + 1)
    return res


s = 'abba'
print(lengthOfLongestSubstring(s))
'''
'''
def lengthOfLongestSubstring(s):
    res = 0
    left = 0
    dic = set()
    for i in range(len(s)):
        while s[i] in dic:
            dic.remove(s[left])
            left += 1
        dic.add(s[i])
        res = max(res, i - left + 1)
    return res


s = 'abba'
print(lengthOfLongestSubstring(s))
'''

# 146.LRU缓存
'''
import collections

class LRUCache(collections.OrderedDict):
    
    def __init__(self, capacity):
        super().__init__()
        self.capacity = capacity
    
    
    def get(self, key):
        if key not in self:
            return -1
        
        self.move_to_end(key)
        return self[key]
    
    
    def put(self, key, value):
        self[key] = value
        self.move_to_end(key)
        if len(self) > self.capacity:
            self.popitem(last=False)
'''
'''
class DLinkedNode:
    def __init__(self, key=0, value=0):
        self.key = key
        self.value = value
        self.next = None
        self.prev = None


class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.size = 0
        self.dic = {}
        self.head = DLinkedNode()
        self.tail = DLinkedNode()
        self.head.next = self.tail
        self.tail.prev = self.head


    def addToHead(self, node):
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node


    def removeNode(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev


    def moveToHead(self, node):
        self. removeNode(node)
        self.addToHead(node)


    def get(self, key):
        if key not in self.dic:
            return -1

        node = self.dic[key]
        self.moveToHead(node)
        return node.value


    def put(self, key, value):
        if key not in self.dic:
            node = DLinkedNode(key, value)
            self.dic[key] = node
            self.addToHead(node)
            self.size += 1
            if self.size > self.capacity:
                node = self.tail.prev
                self.removeNode(node)
                self.dic.pop(node.key)
                self.size -= 1
        else:
            node = self.dic[key]
            node.value = value
            self.moveToHead(node)
'''
# 15.三数之和
'''
def threeSum(nums):
    n = len(nums)
    res = []
    if n < 3:
        return []

    nums.sort()
    for i in range(n - 1):
        if nums[i] > 0:
            break
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        left = i + 1
        right = n - 1
        while left < right:
            if nums[i] + nums[left] + nums[right] == 0:
                res.append([nums[i], nums[left], nums[right]])
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                left += 1
                right -= 1
            elif nums[i] + nums[left] + nums[right] > 0:
                right -= 1
            else:
                left += 1
    return res


nums = [-1, 0, 1, 2, -1, -4]
print(threeSum(nums))
'''

# 21. 合并两个有序链表
'''
def mergeTwoLists(list1, list2):
    res = Node()
    p = res
    if list1 == None:
        return list2
    if list2 == None:
        return list1

    while list1 and list2:
        if list1.val < list2.val:
            p.next = list1
            list1 = list1.next
        else:
            p.next = list2
            list2 = list2.next
        p = p.next
    if list1:
        p.next = list1
    else:
        p.next = list2
    return res.next


l1 = createLink([1, 2, 4])
l2 = createLink([1, 3, 4])
h = mergeTwoLists(l1, l2)
print(printLink(h))
'''
'''
def mergeTwoLists(list1, list2):
    if list1 == None:
        return list2
    if list2 == None:
        return list1

    if list1.val < list2.val:
        list1.next = mergeTwoLists(list1.next, list2)
        return list1
    else:
        list2.next = mergeTwoLists(list1, list2.next)
        return list2


l1 = createLink([1, 2, 4])
l2 = createLink([1, 3, 4])
h = mergeTwoLists(l1, l2)
print(printLink(h))
'''

# JZ58. 左旋转字符串
'''
def reverseLeftWords(s, n):
    return s[n:] + s[:n]
'''

'''
def reverseLeftWords(s, n):
    s = list(s)
    s[:n] = s[n-1::-1]
    s[n:] = s[:n-1:-1]
    s = s[::-1]
    return ''.join(s)
    
    
s = "abcdefg"
k = 2
print(reverseLeftWords(s, k))
'''

'''
def reverseLeftWords(s, n):
    li = list(s)
    end = len(s) - 1
    reverse(li, 0, n-1)
    reverse(li, n, end)
    reverse(li, 0, end)
    return ''.join(li)


def reverse(li, left, right):
    while left < right:
        li[left], li[right] = li[right], li[left]
        left += 1
        right -= 1
    return li


s = "abcdefg"
k = 2
print(reverseLeftWords(s, k))
'''

'''
def reverseLeftWords(s, n):
    res = ''
    m = len(s)
    for i in range(m):
        j = (i + n) % m
        res += s[j]
    return res


s = "abcdefg"
k = 2
print(reverseLeftWords(s, k))
'''

# JZ50.第一个只出现一次的字符
'''
def firstUniqChar(s):
    dic = {}
    for c in s:
        if c not in dic:
            dic[c] = 1
        else:
            dic[c] += 1
    for i in s:
        if dic[i] == 1:
            return i
    return ' '


s = 'abaccdeff'
print(firstUniqChar(s))
'''

'''
def firstUniqChar(s):
    d = []
    for c in s:
        if c not in d:
            d.append(c)
        else:
            d.remove(c)
    if d == []:
        return ' '
    return d[0]


s = 'abaccdeff'
print(firstUniqChar(s))
'''

# JZ58. 翻转单词顺序
'''
def reverseWords(s):
    s = s.strip().split()
    s = s[::-1]
    return ' '.join(s)


s = "a good   example"
print(reverseWords(s))
'''

'''
# 如果字符串不是空格，就放list。字符串是空格，但list[-1] != ' '，放进去个空格，这样可以保证，永远都是只有一个空格
s = "a good   example"
output = []
left = 0
right = len(s) - 1
while left <= right:
    if s[left] != ' ':
        output.append(s[left])
    elif output[-1] != ' ':
        output.append(s[left])
    left += 1
print(output)
'''

'''
def reverseWords(s):
    s = s.strip()
    n = len(s) - 1
    start = n
    end = n
    res = []
    while start >= 0:
        while start >= 0 and s[start] != " ":
            start -= 1
        res.append(s[start + 1:end + 1])
        while s[start] == ' ':
            start -= 1
        end = start
    return ' '.join(res)


s = "a good   example"
print(reverseWords(s))
'''

# 186. 翻转列表里的每个单词
'''
def reverseWords(s):
    n = len(s)
    start = end = 0
    while end < n:
        while end < n and s[end] != ' ':
            end += 1
        func(s, start, end-1)
        end += 1
        start = end
    func(s, 0, n-1)
    return s

def func(s, left, right):
    while left < right:
        s[left], s[right] = s[right], s[left]
        left += 1
        right -= 1
    return s


s = ["t", "h", "e", " ", "s", "k", "y", " ", "i", "s", " ", "b", "l", "u", "e"]
print(reverseWords(s))
'''

# JZ46.把数字翻译成字符串

# 就是青蛙跳台阶问题
'''
def translateNum(num):
    s = str(num)
    if len(s) < 2:
        return 1
    dp = [0] * len(s)
    dp[0] = 1
    if 9 < int(s[:2]) < 26:
        dp[1] = 2
    else:
        dp[1] = 1
    for i in range(2, len(s)):
        if 9 < int(s[i-1:i+1]) < 26:
            dp[i] = dp[i-1] + dp[i-2]
        else:
            dp[i] = dp[i-1]
    return dp[-1]


num = 1
print(translateNum(num))
'''

'''
def translateNum(num):
    if num < 10:
        return 1
    s = str(num)
    
    a = 1
    if 9 < int(s[:2]) < 26:
        b = 2
    else:
        b = 1
    for i in range(2, len(s)):
        if 9 < int(s[i-1:i+1]) < 26:
            temp = a + b
            a = b
            b = temp
        else:
            a = b
    return b


num = 12258
print(translateNum(num))
'''

# 62.不同路径
'''
def uniquePaths(m, n):
    dp = [[1] * n for j in range(m)]
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]
    return dp[-1][-1]

print(uniquePaths(3, 7))
'''


# JZ38. 字符串的全排列
'''
def permutation(s):
    res = []
    s = list(s)
    process(s, 0, len(s)-1, res)
    return res


def process(s, begin, end, res):
    if begin == end:
        return res.append(''.join(s))

    for i in range(begin, end+1):
        if i == begin or s[i] not in s[begin:i]:
            s[i], s[begin] = s[begin], s[i]
            process(s, begin+1, end, res)
            s[i], s[begin] = s[begin], s[i]


s = 'abc'
print(permutation(s))
'''

# JZ45.把数组排成最小的数
'''
def minNumber(nums):
    n = len(nums)
    nums = list(map(str, nums))
    for i in range(n-1, 0, -1):
        for j in range(i):
            if nums[j] + nums[j+1] > nums[j+1] + nums[j]:
                nums[j], nums[j+1] = nums[j+1], nums[j]
    return ''.join(nums)


nums = [3, 30, 34, 5, 9]
print(minNumber(nums))
'''

'''
def minNumber(nums):
    n = len(nums)
    nums = list(map(str, nums))
    for i in range(1, n):
        p = nums[i]
        pos = i
        while pos > 0 and (nums[pos-1] + p > p + nums[pos-1]):
            nums[pos] = nums[pos-1]
            pos -= 1
        nums[pos] = p

    return ''.join(nums)


nums = [3, 30, 34, 5, 9]
print(minNumber(nums))
'''
'''
import functools

def minNumber(nums):
    def fuc(a, b):
        if a + b > b + a:
            return 1
        elif a + b < b + a:
            return -1
        else:
            return 0

    nums = list(map(str, nums))
    nums.sort(key=functools.cmp_to_key(fuc))
    return ''.join(nums)


nums = [3, 30, 34, 5, 9]
print(minNumber(nums))
'''

# JZ48.最长不含重复字符的子字符串
'''
def lengthOfLongestSubstring(s):
    res = 0
    left = 0
    dic = {}
    for i in range(len(s)):
        if s[i] not in dic:
            dic[s[i]] = i
        else:
            left = max(left, dic[s[i]] + 1)
            dic[s[i]] = i
        res = max(res, i - left + 1)
    return res


s = 'pwwkew'
print(lengthOfLongestSubstring(s))
'''

'''
def lengthOfLongestSubstring(s):
    res = 0
    left = 0
    dic = set()
    for i in range(len(s)):
        while s[i] in dic:
            dic.remove(s[left])
            left += 1
        dic.add(s[i])
        res = max(res, i - left + 1)
    return res


s = 'pwwkew'
print(lengthOfLongestSubstring(s))
'''

# 8.字符串转换整数
'''
def myAtoi(s):
    res = 0
    dic = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}
    flag1 = False
    if s == None:
        return 0
    i = 0
    while i < len(s):
        if s[i] == ' ':
            i += 1
        else:
            break
    if s[i] == '-':
        flag1 = True
        i += 1
    elif s[i] == '+':
        i += 1

    while i < len(s):
        if s[i] in dic:
            res = 10 * res + int(s[i])
            i += 1
        else:
            break

    res = -res if flag1 else res
    if res > 2 ** 31 - 1:
        res = 2 ** 31 - 1
    if res < -(2 ** 31):
        res = -(2 ** 31)
    return res


s = "4193 with words"
print(myAtoi(s))
'''

# 5.最长回文字串

# 最快的算法
'''
def longestPalindrome(s):
    if s == None or len(s) == 0:
        return ''
    i = 0
    r = [0, 0]
    while i < len(s):
        i = findLongest(s, i, r)
        i += 1
    return s[r[0]:r[1]+1]

def findLongest(s, i, r):
    high = i
    while high < len(s) - 1 and s[high + 1] == s[i]:
        high += 1
    ans = high
    while i > 0 and high < len(s) - 1:
        if s[i - 1] == s[high + 1]:
            i -= 1
            high += 1
        else:
            break
    if high - i > r[1] - r[0]:
        r[1] = high
        r[0] = i
    return ans

s = 'abcddddc'
print(longestPalindrome(s))
'''

# 中心扩散法，最容易理解
'''
def longestPalindrome(s):
    n = len(s)
    res = 0
    low = 0
    high = 1
    for i in range(n):
        left = i - 1
        right = i + 1
        while left >= 0 and s[left] == s[i]:
            left -= 1
        while right < n and s[right] == s[i]:
            right += 1
        while left >= 0 and right < n and s[left] == s[right]:
            left -= 1
            right += 1
        if right - left - 1 > res:
            res = right - left - 1
            low = left + 1
            high = right
    return s[low:high]


s = 'babad'
print(longestPalindrome(s))
'''

# 动态规划
'''
def longestPalindrome(s):
    n = len(s)
    res = 1
    start = 0
    dp = [[False] * n for i in range(n)]
    for i in range(n):
        for j in range(i, -1, -1):
            if s[i] == s[j]:
                if i - j <= 2:
                    dp[i][j] = True
                else:
                    dp[i][j] = dp[i - 1][j + 1]
            if dp[i][j] and i - j + 1 > res:
                res = i - j + 1
                start = j
    return s[start: start+res]


s = "aacabdkacaa"
print(longestPalindrome(s))
'''

# 太巧了。。。。。
'''
def longestPalindrome(s):
    res = ''
    for i in range(len(s)):
        start = max(0, i - len(res) - 1)
        # 这就相当于前后各扩了一位，多了两个长度，如果它不是回文，那再判断s[1:]
        temp = s[start:i+1]
        if temp == temp[::-1]:
            res = temp
        else:
            temp = temp[1:]
            if temp == temp[::-1]:
                res = temp
    return res


s = "babad"
print(longestPalindrome(s))
'''

# 动态规划
'''
def longestPalindrome(s):
    n = len(s)
    res = 1
    start = 0
    dp = [[False] * n for i in range(n)]
    for i in range(n-1, -1, -1):
        for j in range(i, n):
            if s[i] == s[j]:
                if j - i <= 2:
                    dp[i][j] = True
                else:
                    dp[i][j] = dp[i+1][j-1]
            if dp[i][j] and j - i + 1 > res:
                res = j - i + 1
                start = i
    return s[start:start+res]
'''

# 266.回文排列
'''
def isPalindrome(s):
    n = len(s)
    dic = {}
    for i in range(n):
        if s[i] not in dic:
            dic[s[i]] = 1
        else:
            dic[s[i]] += 1
    res = 0
    for i in dic.values():
        if i % 2 == 1:
            res += 1
    if n % 2 == 0:
        if res == 0:
            return True
        else:
            return False
    if n % 2 == 1:
        if res == 1:
            return True
        else:
            return False


s = 'code'
print(isPalindrome(s))
'''

# 1143.最长公共子序列
'''
def longestCommonSubsequence(str1, str2):
    res = ''
    n1 = len(str1)
    n2 = len(str2)
    dp = [[0] * (n2 + 1) for i in range(n1 + 1)]
    dic = [[0] * (n2 + 1) for i in range(n1 + 1)]
    for i in range(1, n1 + 1):
        for j in range(1, n2 + 1):
            if str1[i - 1] != str2[j - 1]:
                dp[i][j] = max(dp[i][j - 1], dp[i - 1][j])
                if dp[i][j-1] > dp[i-1][j]:
                    dic[i][j] = 1
                else:
                    dic[i][j] = 2

            else:
                dp[i][j] = dp[i - 1][j - 1] + 1
                dic[i][j] = 0
    while n1 > 0 and n2 > 0:
        if str1[n1-1] == str2[n2-1]:
            res += str1[n1-1]
            n1 -= 1
            n2 -= 1
        else:
            if dic[n1][n2] == 1:
                n2 -= 1
            elif dic[n1][n2] == 2:
                n1 -= 1

    return dp[-1][-1], res[::-1]


str1 = 'abc123cad3'
str2 = 'abc123abds'
print(longestCommonSubsequence(str1, str2))
'''


# JZ2. 二进制加法

'''
def addBinary(a, b):
    i = len(a) - 1
    j = len(b) - 1
    carry = 0
    res = ''
    while i >= 0 and j >= 0:
        carry += int(a[i]) + int(b[j])
        res += str(carry % 2)
        carry //= 2
        i -= 1
        j -= 1
    while i >= 0:
        carry += int(a[i])
        res += str(carry % 2)
        carry //= 2
        i -= 1
    while j >= 0:
        carry += int(b[j])
        res += str(carry % 2)
        carry //= 2
        j -= 1
    if carry == 1:
        res += '1'
    return res[::-1]


a = '1010'
b = '1011'
print(addBinary(a, b))
'''


# 50.Pow(x, n)

'''
def myPow(x, n):
    if n >= 0:
        return quick(x, n)
    else:
        return 1.0 / quick(x, -n)


def quick(x, n):
    if n == 0:
        return 1.0
    y = quick(x, n // 2)
    if n % 2 == 0:
        return y * y
    else:
        return y * y * x


print(myPow(3, 5))
'''

'''
def myPow(x, n):
    i = abs(n)
    res = 1.0
    while i != 0:
        if i % 2 != 0:
            res *= x
        x *= x
        i //= 2
    return res if n >= 0 else 1.0 / res


print(myPow(2.00000, -2))
'''
'''
def myPow(x, n):
    if n >= 0:
        return process(x, n)
    else:
        return 1.0 / process(x, -n)

def process(x, n):
    if n == 0:
        return 1.0

    y = process(x, n // 2)
    if n % 2 != 0:
        return y * y * x
    else:
        return y * y


print(myPow(2.00000, 10))
'''

























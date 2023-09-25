# -*- coding: utf-8 -*-
# @Time    : 2022-06-21 15:44
import collections

class Node:
    def __init__(self, val):
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

# 1. 两数之和
'''
def twoSum(nums, target):
    m_dict = {}
    for i in range(len(nums)):
        if target - nums[i] not in m_dict:
            m_dict[nums[i]] = i
        else:
            return m_dict[target - nums[i]], i


nums = [2, 7, 11, 15]
target = 9
print(twoSum(nums, target))
'''
# 9.回文数
'''
def isPalindrome(x):

    x = str(x)
    start = 0
    last = len(x) - 1
    if len(x) == 1:
        return False
    while start < last:
        if x[start] == x[last]:
            start += 1
            last -= 1
        else:
            return False
    return True
    


    # return str(x) == str(x)[::-1]


    if x == 0:
        return True
    # 如果x是10的倍数的话，y永远都和x不相等，当x = 10时，最后x = 0, y = 1，那个判断条件会返回True
    if x < 0 or x % 10 == 0:
        return False
    # x > y，只需要判断一半，当len(x)是奇数，x == y // 10; 偶数时， x == y
    y = 0
    while x > y:
        y = y * 10 + x % 10
        x //= 10
    return x == y or x == y // 10

x = 10
print(isPalindrome(x))
'''

# 13.罗马数字转整数
'''
def romanToInt(s):
    
    res = 0
    r_dict = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    s_dict = {'IV': 4, 'IX': 9, 'XL': 40, 'XC': 90, 'CD': 400, 'CM': 900}
    for special_s in s_dict:
        if special_s in s:
            res += s_dict[special_s]
            s = s.replace(special_s, '')
    for i in s:
        res += r_dict[i]
    return res


    
    r_dict = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    r_list = [r_dict[i] for i in s]
    for i in range(len(r_list) - 1):
        if r_list[i] < r_list[i + 1]:
            r_list[i] = -r_list[i]
    return sum(r_list)


    r_dict = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    ans = 0
    for i in range(len(s) - 1):
        if r_dict[s[i]] >= r_dict[s[i + 1]]:
            ans += r_dict[s[i]]
        else:
            ans -= r_dict[s[i]]
    return ans + r_dict[s[-1]]


s = 'MCMXCIV'
print(romanToInt(s))
'''

# 20.有效的括号
'''
def isValid(s):
    while '()' in s or '[]' in s or '{}' in s:
        s = s.replace('()', '')
        s = s.replace('[]', '')
        s = s.replace('{}', '')
    return s == ''
'''
'''
def isValid(s):
    stack = []
    for i in range(len(s)):
        if s[i] == '(':
            stack.append(')')
        elif s[i] == '[':
            stack.append(']')
        elif s[i] == '{':
            stack.append('}')
        elif not stack or s[i] != stack[-1]:
            return False
        else:
            stack.pop()
    if stack:
        return False
    else:
        return True
'''
'''
def isValid(s):
    dic = {')': '(', ']': '[', '}': '{'}
    res = []
    for i in range(len(s)):
        if s[i] in dic and res:
            if dic[s[i]] == res[-1]:
                res.pop()
            else:
                return False
        else:
            res.append(s[i])
    return True if not res else False


s = "[({(())}[()])]"
# s = '()[]'
print(isValid(s))
'''
# 21.合并两个有序链表
class ListNode:
    def __init__(self, val):
        self.val = val
        self.next = None
'''
def mergeTwoLists(list1, list2):
    res = ListNode(None)
    pre = res
    while list1 and list2:
        if list1.val <= list2.val:
            pre.next = list1
            list1 = list1.next
        else:
            pre.next = list2
            list2 = list2.next
        pre = pre.next
    if list1:
        pre.next = list1
    else:
        pre.next = list2
    return res.next
'''
'''
def mergeTwoLists(head1, head2):
    if head1 == None:
        return head2
    if head2 == None:
        return head1

    if head1.val <= head2.val:
        head1.next = mergeTwoLists(head1.next, head2)
        return head1
    else:
        head2.next = mergeTwoLists(head1, head2.next)
        return head2

list1 = ListNode(1)
list1.next = ListNode(2)
list1.next.next = ListNode(4)
list2 = ListNode(1)
list2.next = ListNode(3)
list2.next.next = ListNode(4)
print(mergeTwoLists(list1, list2))
'''
# 136.只出现一次的数字
'''
def singleNumber(nums):
    dic = {}
    for i in nums:
        if i not in dic:
            dic[i] = 1
        else:
            dic[i] += 1
    for k, v in dic.items():
        if v == 1:
            return k
'''
'''
def singleNumber(nums):
    res = 0
    for i in nums:
        res = res ^ i
    return res

nums = [4, 1, 2, 1, 2]
print(singleNumber(nums))
'''
'''
# def head_creat(li):
#     head = ListNode(li[0])
#     for i in li[1:]:
#         node = ListNode(i)
#         node.next = head
#         head = node
#     return head
def tail_creat(li):
    head = ListNode(None)
    pre = head
    for i in li:
        node = ListNode(i)
        pre.next = node
        pre = node
    return head.next


def p_list(lk):
    while lk:
        print(lk.val, end=' ')
        lk = lk.next

li = [1, 2, 3, 4]
p_list(tail_creat(li))
'''

# 53.最大子数组和
'''
nums = [-2, 1]
res = float('-inf')
for i in range(len(nums)):
    for j in range(i, len(nums)):
        a = sum(nums[i: j + 1])
        if a >= res:
            res = a
print(res)
'''
'''
def maxSubArray(nums):
    res = nums[0]
    max = 0
    for i in range(len(nums)):
        if max >= 0:
            max += nums[i]
        else:
            max = nums[i]
        res = max if max >= res else res
    return res
'''
# 如果nums[i-1]小于0，那么就重新开始计算最大自序。
'''
def maxSubArray(nums):
    for i in range(1, len(nums)):
        nums[i] = nums[i] + max(nums[i - 1], 0)
    return max(nums)

nums = [-2,1,-3,4,-1,2,1,-5,4]
print(maxSubArray(nums))
'''
'''
# 动态规划
def maxSubArray(nums):
    dp = [0] * len(nums)
    dp[0] = nums[0]
    for i in range(1, len(nums)):
        if dp[i - 1] < 0:
            dp[i] = nums[i]
        else:
            dp[i] = nums[i] + dp[i - 1]
    return max(dp)

nums = [-2,1,-3,4,-1,2,1,-5,4]
print(maxSubArray(nums))
'''

# 461.汉明距离
'''
def f(x):
    s = ''
    while x > 0:
        s += str(x % 2)
        x = x // 2
    return s
def hammingDistance(x, y):
    s1 = f(x)
    s2 = f(y)
    while len(s1) > len(s2):
        s2 += '0'
    while len(s2) > len(s1):
        s1 += '0'
    res1 = s1[::-1]
    res2 = s2[::-1]
    res = 0
    for i in range(len(res1)):
        if res1[i] != res2[i]:
            res += 1
    return res
'''
'''
def hammingDistance(x, y):
    z = x ^ y
    res = 0
    while z > 0:
        z = z & (z - 1)
        res += 1
    return res
        
x = 4
y = 1
print(hammingDistance(x, y))
'''
# 338.比特位计数
'''
def countBits(n):
    res = []
    for i in range(n + 1):
        num = 0
        while i != 0:
            num += i & 1
            i = i >> 1
        res.append(num)
    return res
'''
'''
def countBites(n):
    res = [0] * (n+1)
    for i in range(1, n + 1):
        # i & (i-1)可以把i最后一个1消掉，所以i中1的个数等于，i & (i-1)中1的个数+1
         
        res[i] = res[i & (i-1)] + 1
    return res

n = 5
print(countBites(n))
'''
'''
def countBites(n):
    res = [0] * (n+1)
    for i in range(n+1):
        res[i] = res[i >> 1] + (i & 1)
    return res
'''
'''
def counBites(n):
    dp = [0] * (n+1)
    for i in range(1, n+1):
        if i % 2 == 0:
            dp[i] = dp[i // 2]
        else:
            dp[i] = dp[i-1] + 1
    return dp

print(counBites(5))
'''
# 70.爬楼梯，跳台阶
# 不能用递归，会超时
'''
def climbStairs(n):
    if n == 1:
        return 1
    elif n == 2:
        return 2
    return climbStairs(n - 1) + climbStairs(n - 2)
'''
'''
# 类似于链表一样，依次传下去。[a, b, temp]，然后a更新为b，b更新为temp
def climbStairs(n):
    if n == 1:
        return 1
    if n == 2:
        return 2
    a = 1
    b = 2
    for i in range(3, n+1):
        temp = a + b
        a = b
        b = temp
    return b
'''
'''
def climbStairs(n):
    dp = [0] * (n+1)
    if n <= 2:
        return n
    dp[1], dp[2] = 1, 2
    for i in range(3, n+1):
        # 两种情况：1.第一步上1个，剩下i-1；
        # 2.第一步上2个，剩下i-2
        dp[i] = dp[i-1] + dp[i-2]
    return dp[-1]


print(climbStairs(44))
'''
# 206.反转链表
'''
def reverseList(head):
    pre = None
    while head:
        nextNode = head.next
        head.next = pre
        pre = head
        head = nextNode
    return pre
'''
# 递归法
'''
def reverseList(head):
    return r(None, head)


def r(pre, head):
    if head == None:
        return pre
    
    temp = head.next
    head.next = pre
    return r(head, temp)
'''
# 283.移动零

'''
def moveZeros(nums):
    i = 0
    j = 0
    for i in range(len(nums)):
        if nums[i] != 0:
            nums[j] = nums[i]
            j += 1
    while j < len(nums):
        nums[j] = 0
        j += 1
    return nums
'''
'''
def moveZeros(nums):
    nums.sort(key=bool, reverse=True)
    return nums
'''
# 双指针
'''
def moveZeros(nums):
    left = right = 0
    while right < len(nums):
        if nums[right]:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
        right += 1
    return nums

nums = [0, 0, 0, 31, 12]
print(moveZeros(nums))
'''
# 二叉树中序遍历
'''
def inorderTravesal(head):
    # 一定要单独写个函数，把res隔开，不然每次递归都会把res改成[]
    res = []
    def inorder(head):
        if head == None:
            return
        inorder(head.left)
        res.append(head.key)
        inorder(head.right)
    inorder(head)
    return res

print(inorderTravesal(head))
'''
'''
def inorderTravesal(head):
    s = []
    res = []
    while s or head:
        if head:
            s.append(head)
            head = head.left
        else:
            head = s.pop()
            res.append(head.key)
            head = head.right
    return res

print(inorderTravesal(head))
'''
# 104. 二叉树的最大深度
'''
def maxDepth(head):
    q = []
    q.insert(0, head)
    curLevel = 1
    leveMap = {head: 1}
    while q:
        head = q.pop()
        if leveMap[head] != curLevel:
            curLevel += 1
        if head.left:
            leveMap[head.left] = curLevel + 1
            q.insert(0, head.left)
        if head.right:
            leveMap[head.right] = curLevel + 1
            q.insert(0, head.right)
    return curLevel

print(maxDepth(head))
'''
'''
def maxDepth(head):
    if head == None:
        return 0
    return max(maxDepth(head.left), maxDepth(head.right)) + 1

print(maxDepth(head))
'''
'''
def maxDepth(head):
    q = []
    q.append(head)
    res = 0
    while q:
        res += 1
        for i in range(len(q)):
            h = q.pop(0)
            if h.left:
                q.append(h.left)
            if h.right:
                q.append(h.right)
    return res

print(maxDepth(head))
'''
# 448. 找到所有数组中消失的数字
'''
def findDisappearedNumbers(li):
    res = []
    s = set(li)
    for i in range(1, len(li) + 1):
        if i not in s:
            res.append(i)
    return res


li = [4, 3, 2, 7, 8, 2, 3, 1]
print(findDisappearedNumbers(li))
'''
'''
def findDisappearedNumbers(li):
    res = []
    li2 = [i + 1 for i in range(len(li))]
    for i in range(len(li)):
        if li[i] in li2:
            li2[li[i] - 1] = -li2[li[i] - 1]
    for i in li2:
        if i > 0:
            res.append(i)
    return res

li = [4, 3, 2, 7, 8, 2, 3, 1]
print(findDisappearedNumbers(li))
'''
'''
def findDisappearedNumbers(li):
    res = []
    for i in range(len(li)):
        if li[abs(li[i]) - 1] > 0:
            li[abs(li[i]) - 1] = -li[abs(li[i]) - 1]
    for i in range(len(li)):
        if li[i] > 0:
            res.append(i + 1)
    return res

li = [4, 3, 2, 7, 8, 2, 3, 1]
print(findDisappearedNumbers(li))
'''
# 101.对称二叉树
'''
def isSymmetric(root):
    if root == None:
        return True
    return dfs(root.left, root.right)


def dfs(left, right):
    if left == None and right == None:
        return True
    if left == None or right == None or left.key != right.key:
        return False
    return dfs(left.left, right.right) and dfs(left.right, right.left)

print(isSymmetric(head))
'''
'''
def isSymmetric(head):
    q = []
    q.append(head.left)
    q.append(head.right)
    while q:
        n1 = q.pop(0)
        n2 = q.pop(0)
        if n1 == None and n2 == None:
            continue
        if n1 == None or n2 == None or n1.key != n2.key:
            return False

        q.append(n1.left)
        q.append(n2.right)
        q.append(n1.right)
        q.append(n2.left)
    return True

print(isSymmetric(head))
'''
# 121.买卖股票的最佳时机
'''
def maxProfit(prices):
    res = float('-inf')
    for i in range(len(prices) - 1):
        pre = prices[i]
        for j in range(i + 1, len(prices)):
            res = max(prices[j] - pre, res)
    res = 0 if res <= 0 else res
    return res

prices = [7, 6, 4, 3, 1]
print(maxProfit(prices))
'''
# 前i天的最大收益 = max{前i-1天的最大收益，第i天的价格-前i-1天最小价格}
'''
def maxProfit(prices):
    dp = [0]
    if len(dp) == 0:
        return 0
    temp = prices[0]
    for i in range(1, len(prices)):
        temp = min(temp, prices[i-1])
        dp.append(max(dp[i-1], prices[i] - temp))
    return max(dp)

prices = [7, 6, 4, 3, 1]
print(maxProfit(prices))
'''
'''
def maxProfit(prices):
    if len(prices) == 0:
        return 0
    res = 0
    temp = prices[0]
    for i in range(1, len(prices)):
        res = max(prices[i] - temp, res)
        temp = min(temp, prices[i])
    return res


prices = [7, 6, 4, 3, 1]
print(maxProfit(prices))
'''
# 234.回文链表
'''
def isPalindrome(head):
    s = []
    p = head
    while p:
       s.append(p)
       p = p.next
    while s:
        if head.val == s.pop().val:
            head = head.next
        else:
            return False
    return True

li = [1, 1, 2, 1]
head = createLink(li)
print(isPalindrome(head))
'''
'''
def isPalindrome(head):
    slow = head.next
    fast = head
    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next
    s = []
    while slow:
        s.append(slow)
        slow = slow.next
    while s:
        if head.val == s.pop().val:
            head = head.next
        else:
            return False
    return True

li = [1, 2, 2, 1]
head = createLink(li)
print(isPalindrome(head))
'''
# 226.翻转二叉树
# 前序
'''
def invertTree(head):
    if head == None:
        return
    
    rightNode = head.right
    head.right = invertTree(head.left)
    head.left = invertTree(rightNode)
    return head
'''
# 中序
'''
def invertTree(head):
    if head == None:
        return 
    
    invertTree(head.left)
    rightNode = head.right
    head.right = head.left
    head.left = rightNode
    invertTree(head.left)
    return head
'''
# 后序
'''
def invertTree(head):
    if head == None:
        return 
    
    leftNode = invertTree(head.left)
    rightNode = invertTree(head.right)
    head.right = leftNode
    head.left = rightNode
    return head
'''
# 层序遍历
'''
def invertTree(head):
    q = [head]
    while q:
        n = q.pop(0)
        rightNode = n.right
        n.right = n.left
        n.left = rightNode
        if n.left:
            q.append(n.left)
        if n.right:
            q.append(n.right)
    return head
'''
# 169.多数元素
'''
def majorityElement(nums):
    dic = {}
    for i in range(len(nums)):
        if nums[i] not in dic:
            dic[nums[i]] = 1
        else:
            dic[nums[i]] += 1
    dic2 = sorted(dic.items(), key=lambda x: x[1])
    return dic2[-1][0]


nums = [2, 2, 1, 1, 1, 2, 2]
print(majorityElement(nums))
'''
'''
def majorityElement(nums):
    dic = {}
    for i in range(len(nums)):
        if nums[i] not in dic:
            dic[nums[i]] = 1
        else:
            dic[nums[i]] += 1
            if dic[nums[i]] > len(nums) // 2:
                return nums[i]
    # for key in dic.keys():
    #     if dic[key] > len(nums) // 2:
    #         return key

nums = [6, 5, 5]
print(majorityElement(nums))
'''
'''
# 从第一个数开始，count=1， 遇到相同的加1，不同的减1，减到0时，重新计数
def majorityElement(nums):
    res = nums[0]
    count = 1
    for i in range(1, len(nums)):
        if nums[i] != res:
            count -= 1
            if count == -1:
                res = nums[i]
                count = 1
        else:
            count += 1
    return res

nums = [6, 5, 5]
print(majorityElement(nums))
'''
'''
def majorityElement(nums):
    res = nums[0]
    c = 1
    for i in range(1, len(nums)):
        if res == nums[i]:
            c += 1
        else:
            c -= 1
            if c == 0:
                c = 1
                res = nums[i]
    return res

nums = [6, 5, 5]
print(majorityElement(nums))
'''
# 617.合并二叉树
'''
def mergeTrees(head, head1):
    if head == None:
        return head1
    if head1 == None:
        return head

    head.key = head.key + head1.key
    head.left = mergeTrees(head.left, head1.left)
    head.right = mergeTrees(head.right, head1.right)
    return head

print(mergeTrees(head,head2))
'''
# 不修改head,head1的方法
'''
def mergeTrees(head, head1):
    if head == None:
        return head1
    if head1 == None:
        return head
    
    root = BinaryTree(head.key + head1.key)
    root.left = mergeTrees(head.left, head1.left)
    root.right = mergeTrees(head.right, head1.right)
    return root
'''
'''
def mergeTrees(head, head1):
    if head == None:
        return head1
    if head1 == None:
        return head

    q = [head, head1]
    while q:
        n1 = q.pop(0)
        n2 = q.pop(0)
        n1.key = n1.key + n2.key
        if n1.left and n2.left:
            q.append(n1.left)
            q.append(n2.left)
        if n1.right and n2.right:
            q.append(n1.rigt)
            q.append(n2.right)
        if n1.left == None and n2.left:
            n1.left = n2.left
        if n1.right == None and n2.right:
            n1.right = n2.right
    return head
'''





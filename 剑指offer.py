# -*- coding: utf-8 -*-
# @Time   : 2023/1/13 20:58

class Node:
    def __init__(self, val):
        self.val = val
        self.next = None


def creatLink(li):
    head = Node(li[0])
    p = head
    for i in li[1:]:
        p.next = Node(i)
        p = Node(i)
    return head


def printLink(head):
    li = []
    while head:
        li.append(head.val)
        head = head.next
    return li


class Stack:
    def __init__(self):
        self.head = None

    def push(self, node):
        n = Node(node)
        n.next = self.head
        self.head = n

    def pop(self):
        temp = self.head.val
        self.head = self.head.next
        return temp

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
        self.leftChild = None
        self.rightChild = None


# 03. 数组中重复的数字
'''
def findRepeatNumber(nums):
    dic = {}
    for i in nums:
        if i not in dic:
            dic[i] = 1
        else:
            return i
    return None
'''


'''
def findRepeatNumber(nums):
    for i in range(len(nums)):
        while nums[i] != i:
            if nums[i] == nums[nums[i]]:
                return nums[i]
            else:
                temp = nums[i]
                nums[i], nums[temp] = nums[temp], nums[i]


nums = [2, 3, 1, 0, 2, 5, 3]
print(findRepeatNumber(nums))
'''


# 05. 替换空格
'''
def replaceSpace(s):
    sp = 0
    lenth = len(s)
    for i in range(lenth):
        if s[i] == ' ':
            sp += 1

    res = list(s)
    res.extend([' '] * sp * 2)

    n1 = lenth - 1
    n2 = len(res) - 1
    while n1 >= 0:
        if s[n1] == ' ':
            res[n2-2: n2+1] = '%20'
            n2 -= 3
        else:
            res[n2] = res[n1]
            n2 -= 1
        n1 -= 1
    return ''.join(res)
'''

'''
# 53 - I.在排序数组中查找数字

def search(nums, target):
    left = 0
    right = len(nums) - 1
    while left <= right:
        mid = left + ((right - left) >> 1)
        if left == right and nums[left] != target:
            return 0
        if nums[mid] > target:
            right = mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            if nums[left] != target:
                left += 1
            elif nums[right] != target:
                right -= 1
            else:
                break
    return right - left + 1


s = [5, 7, 7, 8, 8, 10]
print(search(s, 6))
'''


# 06. 从尾到头打印链表

def reversePrint(head):
    res = []
    while head:
        res.append(head.val)
        head = head.next
    return res[::-1]




















# 2022-04-07  21:07
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
        self.leftChild = None
        self.rightChild = None

'''
s = dict()
s['zuo'] = '我'
s['you'] = '你'
a = {}
print(s.pop('you'))
b = 12
c = 12
print(b == c)
'''
# 反转链表
'''
class Node():
    def __init__(self, item):
        self.item = item
        self.next = None
        self.prev = None


node1 = Node(1)
node1.next = Node(2)
node1.next.next = Node(3)

def reverseLinkedList(head):
    pre = None
    next = None
    while head != None:
        next = head.next
        head.next = pre
        pre = head
        head = next
    return pre.next.next.item
print(reverseLinkedList(node1))
'''

'''
def reverseDoubleLinkedList(head):
    pre = None
    next = None
    while head != None:
        next = head.next
        head.next = pre
        head.prev = next
        pre = head
        head = next
    return pre
'''
# 单链表实现队列
'''
class Node():
    def __init__(self, value):
        self.value = value
        self.next = None


class Queue():
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0


    def size(self):
        return self.size


    def is_empty(self):
        return self.head is None


    def dequeue(self, data):
        cur = Node(data)
        if self.tail == None:
            self.tail = cur
            self.head = cur
        else:
            self.tail.next = cur
            self.tail = cur
        self.size += 1


    def poll(self):
        ans = None
        if self.head == None:
            return None
        else:
            ans = self.head.value
            self.head = self.head.next
            self.size -= 1
        return ans


    def peek(self):
        if self.head == None:
            self.tail = None
        else:
            return self.head.value


s = Queue()
s.dequeue(5)
s.dequeue(4)
s.dequeue(3)
s.dequeue(2)
s.dequeue(1)
s.poll()
print(s.poll())
'''
# 单链表实现栈
'''
class Node():
    def __init__(self, data):
        self.data = data
        self.next = None


class Stack():
    def __init__(self):
        self.head = None
        self.size = 0


    def size(self):
        return self.size


    def push(self, value):
        cur = Node(value)
        if self.head != None:
            cur.next = self.head
            self.head = cur
        if self.head == None:
            self.head = cur
        self.size += 1


    def pop(self):
        ans = None
        if self.head == None:
            return None
        if self.head != None:
            ans = self.head.data
            self.head = self.head.next
            self.size -= 1
        return ans

s = Stack()
s.push(5)
s.push(4)
s.push(3)
s.push(2)
s.push(1)

print(s.pop())
'''
# 双链表实现双端队列
'''
class Node():
    def __int__(self, data):
        self.data = data
        self.next = None
        self.prev = None


class Queue():
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0


    def size(self):
        return self.size


    def pushHead(self, value):
        cur = Node(value)
        if self.head != None:
            cur.next = self.head
            self.head.prev = cur
            self.head = cur
        else:
            self.head = cur
            self.tail = cur
        self.size += 1


    def pushTail(self, value):
        cur = Node(value)
        if self.head != None:
            self.tail.next = cur
            cur.prev = self.tail
            self.tail = cur
        else:
            self.head = cur
            self.tail = cur
        self.size += 1


    def pollHead(self):
        ans = None
        if self.head == None:
            return None
        ans = self.head.data
        if self.head == self.tail:
            self.head = None
            self.tail = None
        else:
            self.head = self.head.next
            self.head.prev = None
        self.size -= 1
        return ans


    def pollTail(self):
        ans = None
        if self.head == None:
            return None
        ans = self.head.data
        if self.head == self.tail:
            self.head = None
            self.tail = None
        else:
            self.tail = self.tail.prev
            self.tail.next = None
        self.size -= 1
        return ans
'''
'''
class Node():
    def __init__(self, data):
        self.data = data
        self.next = None


def getKGroupEnd(start, k):
    i = 1
    while (i < k) and (start != None):
        start = start.next
        i += 1
    return start


def reverse(start, end):
    pre = None
    next = None
    cur = start
    end = end.next
    while cur != end:
        next = cur.next
        cur.next = pre
        pre = cur
        cur = next
    start.next = end


def reverseKGroupLinkList(head, k):
    start = head
    end = getKGroupEnd(start, k)
    if end == None:
        return head
    # 凑齐第一组
    head = end
    reverse(start, end)
    lastEnd = start
    while lastEnd.next != None:
        start = lastEnd.next
        end = getKGroupEnd(start, k)
        if end == None:
            return head
        else:
            reverse(start, end)
            lastEnd.next = end
            lastEnd = start
    return head


head = Node(1)
head.next = Node(2)
head.next.next = Node(3)
head.next.next.next = Node(4)
head.next.next.next.next = Node(5)
head.next.next.next.next.next = Node(6)

print(reverseKGroupLinkList(head, 4).data)

'''
# 两数之和
'''
class Node():
    def __init__(self, data):
        self.data = data
        self.next = None


def sumOfTwoLinkList(head1, head2):
    carry = 0
    result = Node(0)
    r = result
    if head1 == None:
        return head2.data
    if head2 == None:
        return head1.data
    while (head1 != None) and (head2 != None) :
        r.next = Node((head1.data + head2.data + carry) % 10)
        carry = (head1.data + head2.data + carry) // 10
        r = r.next
        head1 = head1.next
        head2 = head2.next
    while head1 != None:
        r.next = Node((head1.data + carry) % 10)
        carry = (head1.data + carry) // 10
        r = r.next
        head1 = head1.next
    while head2 != None:
        r.next = Node((head2.data + carry) % 10)
        carry = (head2.data + carry) // 10
        r = r.next
        head2 = head2.next
    if carry != 0:
        r.next = Node(1)


    return result.next
'''
'''
class ListNode():
    def __init__(self, val):
        self.val = val
        self.next = None


def listNodeLength(node):
    length = 0
    while node != None:
        length += 1
        node = node.next
    return length


def sumOfTwoLinkList(l1, l2):
    len1 = listNodeLength(l1)
    len2 = listNodeLength(l2)
    long = l1 if len1 >= len2 else l2
    s = l2 if len1 >= len2 else l1

    curL = long
    curS = s
    last = curL
    carry = 0
    while curS != None:
        num = curL.val + curS.val + carry
        curL.val = num % 10
        carry = num // 10
        last = curL
        curL = curL.next
        curS = curS.next
    while curL != None:
        num = curL.val+ carry
        curL.val = num % 10
        carry = num // 10
        last = curL
        curL = curL.next
    if carry != 0:
        last.next = ListNode(1)

    return long


l1 = ListNode(2)
l1.next = ListNode(5)
l1.next.next = ListNode(3)
l2 = ListNode(4)
l2.next = ListNode(3)
print(sumOfTwoLinkList(l1, l2).next.val)
'''

# 合并两个有序链表


'''
def mergeTwoLists(head1, head2):
    if (head1 == None) or (head2 == None):
        return head1 if head1 == None else head2

    head = head1 if head1.val <= head2.val else head2
    cur1 = head.next
    cur2 = head2 if head == head1 else head1
    pre = head
    while (cur1 != None) and (cur2 != None):
        if cur1.val <= cur2.val:
            pre.next = cur1
            cur1 = cur1.next
        else:
            pre.next = cur2
            cur2 = cur2.next
        pre = pre.next
    pre.next = cur1 if cur1 != None else cur2
    return head

# head1 = ListNode(1)
# head1.next = ListNode(5)
# head1.next.next = ListNode(7)
#
# head2 = ListNode(2)
# head2.next = ListNode(3)
# head2.next.next = ListNode(8)
# head2.next.next.next = ListNode(9)
head1 = ListNode([1, 2, 4])
head2 = ListNode([1, 3, 4])

print(mergeTwoLists(head1, head2))
'''
'''
def f(li):
    res = 0
    for i in li:
        res = res ^ i
    rightOne = res & (~res + 1)
    res2 = 0
    for i in li:
        if rightOne & i == 0:
            res2 = res2 ^ i
    res3 = res ^ res2
    return res3, res2


li = [21, 21, 32, 33, 34, 34, 33, 32, 21, 34]
print(f(li))
'''
'''
def f(li, target):
    first = 0
    last = len(li) - 1
    while first <= last:
        m = (first + last) // 2
        if target < li[m]:
            last = m - 1
        elif target > li[m]:
            first = m + 1
        elif target == li[m]:
            return m

li = [2, 3, 7, 9, 11]
print(f(li, 3))
'''
'''
li = [1, 1, 1, 3, 3, 3, 3, 3, 4, 4, 4]
target = 3
first = 0
last = len(li) - 1
res = -1
while first <= last:
    m = (first + last) // 2
    if target <= li[m]:
        res = m
        last = m - 1
    else:
        first = m + 1
print(res)
'''
'''
def getMix(li):
    return process(li, 0, len(li) - 1)


def process(li, L, R):
    if L == R:
        return li[L]

    mid = L + ((R - L) >> 1)
    left = process(li, L, mid)
    right = process(li, mid + 1, R)
    res = max(left, right)
    return res

li = [3, 2, 5, 6, 7, 4]
print(getMix(li))
'''
# 小和问题
'''
def mergeSorted(li):
    return process(li, 0, len(li) - 1)

def process(li, L, R):
    if L == R:
        return 0

    mid = L + ((R - L) >> 1)
    return process(li, L, mid) + \
           process(li, mid + 1, R) + \
           merge(li, L, mid, R)

def merge(li, l, mid, r):
    res = 0
    m = [0] * (r - l + 1)
    i = l
    j = mid + 1
    k = 0
    while i <= mid and j <= r:
        if li[i] < li[j]:
            res += (r - j + 1) * li[i]
            m[k] = li[i]
            i += 1
        else:
            res += 0
            m[k] = li[j]
            j += 1
        k += 1
    while i <= mid:
        m[k] = li[i]
        k += 1
        i += 1
    while j <= r:
        m[k] = li[j]
        j += 1
        k += 1
    for i in range(len(m)):
        li[l + i] = m[i]
    return res

li = [3, 2, 5, 6, 7, 4]
print(mergeSorted(li))
'''
# 逆序对
'''
def f(li):
    return process(li, 0, len(li) - 1)


def process(li, L, R):
    if L == R:
        return 0

    mid = L + ((R - L) >> 1)
    return process(li, L, mid) + \
           process(li, mid + 1, R) + \
           merge(li, L, mid, R)


def merge(li, l, m, r):
    res = 0
    help = [0] * (r - l + 1)
    k = 0
    i = l
    j = m + 1
    while i <= m and j <= r:
        if li[i] > li[j]:
           res += r - j + 1
           help[k] = li[i]
           i += 1
        else:
            res += 0
            help[k] = li[j]
            j += 1
        k += 1
    while i <= m:
        help[k] = li[i]
        k += 1
        i += 1
    while j <= r:
        help[k] = li[j]
        k += 1
        j += 1
    for i in range(len(help)):
        li[l + i] = help[i]
    return res

li = [3, 5, 2, 7, 4, 1]
print(f(li))
'''
'''
def f(li):
    l = 0
    r = 0
    while r < len(li):
        if li[r] <= 3:
            li[l], li[r] = li[r], li[l]
            l += 1
        r += 1
    return li

li = [3, 5, 2, 7, 4, 1]
print(f(li))
'''
# 三指针
'''
def f(li, num):
    left = 0
    right = len(li) - 1
    i = 0
    while i < len(li) and i <= right:
        if li[i] < num:
            li[left], li[i] = li[i], li[left]
            i += 1
            left += 1
        elif li[i] == num:
            i += 1
        elif li[i] > num:
            li[i], li[right] = li[right], li[i]
            right -= 1
    return li

li = [3, 5, 2, 7, 4, 1, 5]
print(f(li, 5))
'''
# 二分
'''
def binarySearch(li, num):
    i = 0
    j = len(li) - 1
    while i <= j:
        mid = (i + j) >> 1
        if li[mid] < num:
            i = mid + 1
        elif li[mid] > num:
            j = mid - 1
        else:
            return mid

li = [1, 2, 4, 5, 7]
print(binarySearch(li, 7))
'''

'''
def reverseNode(head):
    pre = None
    while head:
        nextNode = head.next
        head.next = pre
        pre = head
        head = nextNode

    return pre

li = [1, 2, 4, 5, 7]
head = createLink(li)
head1 = reverseNode(head)
print(printLink(head1))
'''
'''
def mergeLink(head1, head2):
    head = Node(None)
    pre = head
    if head1 == None:
        return head2
    if head2 == None:
        return head1

    while head1 and head2:
        if head1.val <= head2.val:
            pre.next = head1
            head1 = head1.next
        else:
            pre.next = head2
            head2 = head2.next
        pre = pre.next

    if head1:
        pre.next = head1
    else:
        pre.next = head2
    return head.next

li1 = [1, 3, 4, 4, 6]
li2 = [2, 3, 5, 5, 5, 7]
head1 = createLink(li1)
head2 = createLink(li2)
head = mergeLink(head1, head2)
print(printLink(head))
'''




'''
def isPalindrome(head):
    p = head
    s = Stack()
    while p:
        s.push(p)
        p = p.next
    print(printLink(head))
    while head:
        if head.val == s.pop().val:
            head = head.next
        else:
            return False
    return True
'''
'''
def isPalindrome(head):
    if head == None or head.next == None:
        return True
    cur = head
    right = head.next
    while cur.next and cur.next.next:
        cur = cur.next.next
        right = right.next
    s = Stack()
    while right:
        s.push(right)
        right = right.next
    while not s.isEmpty():
        if s.pop().val == head.val:
            head = head.next
        else:
            return False
    return True

li = [1, 2, 3, 3, 2, 1]
head = createLink(li)
print(isPalindrome(head))
'''
'''
def f(head):
    d = {}
    cur = head
    while cur:
        d[cur] = Node(cur.val)
        cur = cur.next
    cur = head
    while cur:
        # cur：老，d[cur]:新
        d[cur].next = d[cur.next]
        d[cur].rand = d[cur.rand]
        cur = cur.next
    return d[head]
'''
'''
def noLoop(h1, h2):
    head1 = h1
    head2 = h2
    n = 0
    while head1.next:
        n += 1
        head1 = head1.next
    while head2.next:
        n -= 1
        head2 = head2.next
    if head1 == head2:
        if n > 0:
            for i in range(n):
                h1 = h1.next
        else:
            for i in range(-n):
                h2 = h2.next
    while h1 != h2:
        h1 = h1.next
        h2 = h2.next
'''
'''
def bothLoop(head1, head2, loop1, loop2):
    if loop1 == loop2:
        n = 0
        h1 = head1
        h2 = head2
        while h1 != loop1:
            n += 1
            h1 = h1.next
        while h2 != loop2:
            n -= 1
            h2 = h2.next
        cur1 = head1
        cur2 = head2
        if n > 0:
            while n != 0:
                cur1 = cur1.next
                n -= 1
        else:
            while n != 0:
                cur2 = cur2.next
                n += 1
        while cur1 != cur2:
            cur1 = cur1.next
            cur2 = cur2.next
        return cur1

    else:
        p = loop1.next
        while p != loop1:
            if p == loop2:
                return loop2
            p = p.next
        return None
'''
'''
def getLoop(head):
    if head == None or head.next == None or head.next.next == None:
        return None
    slow = head.next
    fast = head.next.next
    while slow != fast:
        if fast.next == None and fast.next.next == None:
            return None
        slow = slow.next
        fast = fast.next.next
    fast = head
    while fast != slow:
        fast = fast.next
        slow = slow.next
    return fast
'''
# 二叉树
head = BinaryTree(5)
head.leftChild = BinaryTree(3)
head.rightChild = BinaryTree(8)
head.leftChild.leftChild = BinaryTree(2)
# head.leftChild.leftChild.leftChild = BinaryTree(4)
head.leftChild.rightChild = BinaryTree(4)
head.rightChild.leftChild = BinaryTree(7)
head.rightChild.rightChild = BinaryTree(10)
'''
def OrderRecur(head):
    if head == None:
        return
    print(head.val, end=' ')
    OrderRecur(head.left)
    print(head.val, end=' ')
    OrderRecur(head.right)
    print(head.val, end=' ')
'''

'''
def preOrderRecur(head):
    s = Stack()
    s.push(head)
    while not s.isEmpty():
        head = s.pop()
        print(head.key, end=' ')
        if head.rightChild:
            s.push(head.rightChild)
        if head.leftChild:
            s.push(head.leftChild)
'''
'''
def postOrderRecur(head):
    s = Stack()
    s1 = Stack()
    s.push(head)
    while not s.isEmpty():
        head = s.pop()
        s1.push(head)
        if head.leftChild:
            s.push(head.leftChild)
        if head.rightChild:
            s.push(head.rightChild)
    while not s1.isEmpty():
        head = s1.pop()
        print(head.key, end=' ')
'''
'''
def inOrderRecur(head):
    s = Stack()
    while (not s.isEmpty()) or head:
        if head:
            s.push(head)
            head = head.leftChild
        else:
            head = s.pop()
            print(head.key, end=' ')
            head = head.rightChild
'''
'''
def f(head):
    q = Queue()
    q.enqueue(head)
    while not q.isEmpty():
        head = q.dequeue()
        print(head.key, end=' ')
        if head.leftChild:
            q.enqueue(head.leftChild)
        if head.rightChild:
            q.enqueue(head.rightChild)
'''
'''
def getMaxWidth(head):
    q = Queue()
    q.enqueue(head)
    levelMap = {head: 1}
    curLevel = 1
    curNode = 0
    res = -1
    while not q.isEmpty():
        head = q.dequeue()
        if levelMap[head] == curLevel:
            curNode += 1
        else:
            res = max(res, curNode)
            curLevel += 1
            curNode = 1
        if head.leftChild:
            levelMap[head.leftChild] = levelMap[head] + 1
            q.enqueue(head.leftChild)
        if head.rightChild:
            levelMap[head.rightChild] = levelMap[head] + 1
            q.enqueue(head.rightChild)
    return max(res, curNode)
    

print(getMaxWidth(head))
'''
'''
def getMaxWith(head):
    q = Queue()
    q.enqueue(head)
    levelMap = {head: 1}
    curLevel = 1
    curNode = 0
    res = -1
    while not q.isEmpty():
        head = q.dequeue()
        if levelMap[head] == curLevel:
            curNode += 1
        else:
            res = max(res, curNode)
            curLevel += 1
            curNode = 1
        if head.leftChild:
            levelMap[head.leftChild] = levelMap[head] + 1
            q.enqueue(head.leftChild)
        if head.rightChild:
            levelMap[head.rightChild] = levelMap[head] + 1
            q.enqueue(head.rightChild)
    res = max(res, curNode)
    return res

print(getMaxWith(head))
'''
# 判断是否为搜索二叉树
'''
def BST(head):
    s = Stack()
    p = float('-inf')
    while (not s.isEmpty()) or head:
        if head:
            s.push(head)
            head = head.leftChild
        else:
            head = s.pop()
            if head.key <= p:
                return False
            else:
                p = head.key
            head = head.rightChild
    return True

print(BST(head))
'''
'''
def BST(head):
    p = float('-inf')
    if head == None:
        return True
    s = BST(head.leftChild)
    if not s:
        return False
    if head.key <= p:
        return False
    else:
        return BST(head.rightChild)

print(BST(head))
'''
'''
# 判断是否是完全二叉树
def CBT(head):
    q = Queue()
    q.enqueue(head)
    # left 是否存在少孩子的情况
    left = False
    while not q.isEmpty():
        head = q.dequeue()
        l = head.leftChild
        r = head.rightChild
        if l == None and r != None:
            return False
        if left and (l != None or r != None):
            return False
        if l:
            q.enqueue(l)
        if r:
            q.enqueue(r)
        if l == None or r == None:
            left = True
    return True

print(CBT(head))
'''
# 判断是否为满二叉树
'''
def deep(head):
    q = Queue()
    q.enqueue(head)
    curLevel = 1
    curNode = 0
    levelMap = {head: 1}
    while not q.isEmpty():
        head = q.dequeue()
        if levelMap[head] == curLevel:
            curNode += 1
        else:
            curLevel += 1
            curNode += 1
        if head.leftChild:
            levelMap[head.leftChild] = levelMap[head] + 1
            q.enqueue(head.leftChild)
        if head.rightChild:
            levelMap[head.rightChild] = levelMap[head] + 1
            q.enqueue(head.rightChild)
    return curNode == 2 ** curLevel - 1

print(deep(head))
'''
# 判断是否为平衡二叉树
'''
def isBalanced(head):
    return process(head).isBalanced


class ReturnType:
    def __init__(self, isB, hei):
        self.isBalanced = isB
        self.height = hei


def process(head):
    if head == None:
        return ReturnType(True, 0)

    leftData = process(head.leftChild)

    rightData = process(head.rightChild)

    height = max(leftData.height, rightData.height) + 1
    isBalanced = leftData.isBalanced and rightData.isBalanced and (abs(leftData.height - rightData.height) < 2)
    return ReturnType(isBalanced, height)

print(isBalanced(head))
'''
# 搜索二叉树
'''
def BST(head):
    return process(head).isB


class Info:
    def __init__(self, minV, maxV, isB):
        self.minV = minV
        self.maxV = maxV
        self.isB = isB

def process(head):
    if head == None:
        return None
    leftData = process(head.leftChild)
    rightData = process(head.rightChild)
    minV = head.key
    maxV = head.key
    if leftData:
        minV = min(minV, leftData.minV)
        maxV = max(maxV, leftData.maxV)
    if rightData:
        minV = min(minV, rightData.minV)
        maxV = max(maxV, rightData.maxV)
    isB = True
    if leftData and ((leftData.maxV > head.key) or not leftData.isB):
        isB = False
    if rightData and ((rightData.minV < head.key) or not rightData.isB):
        isB = False
    return Info(minV, maxV, isB)

print(BST(head))
'''
'''
# 满二叉树
class Info:
    def __init__(self, num, height):
        self.num = num
        self.height = height


def DBT(head):
    return process(head).num == 2 ** process(head).height - 1


def process(head):
    if head == None:
        return Info(0, 0)

    leftData = process(head.leftChild)
    rightData = process(head.rightChild)

    height = max(leftData.height, rightData.height) + 1
    num = leftData.num + rightData.num + 1
    return Info(num, height)

print(DBT(head))
'''
# 二叉树的序列化
'''
def preOrder(head):
    s = Stack()
    s.push(head)
    res = ''
    while not s.isEmpty():
        head = s.pop()
        res += str(head.key) + '_'
        if head.leftChild == None:
            res += '#_'
        if head.rightChild == None:
            res += '#_'
        if head.rightChild:
            s.push(head.rightChild)
        if head.leftChild:
            s.push(head.leftChild)
    return res
print(preOrder(head))
'''
'''
def preOrder(head):
    if head == None:
        return '#_'
    
    res = str(head.key) + '_'
    res += preOrder(head.leftChild)
    res += preOrder(head.rightChild)
    return res


preS = preOrder(head)
print(preS)

def f(s):
    s = s.split('_')
    q = Queue()
    for i in s:
        q.enqueue(i)
    return process(q)

def process(q):
    v = q.dequeue()
    if v == '#':
        return None

    head = BinaryTree(v)
    head.leftChild = process(q)
    head.rightChild = process(q)
    return head


print(f(preS))
'''
def inorder(head):
    res = []
    inodre(head, res)
    return res
def inodre(head, res):
    if head == None:
        return
    inodre(head.leftChild, res)
    res.append(head.key)
    inodre(head.rightChild, res)


print(inorder(head))






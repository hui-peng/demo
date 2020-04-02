import math
import numpy as np
array = [1,4,6,9,7,11,54,4,28,36,47,52,19,26,35,0]
array_1 = [1, 3, 5, 8, 1, 4, 9]
array_2 = [3, 34, 4, 12, 5, 2]
graph = {
    "A": ["B", "C"],
    "B": ["A", "C", "D"],
    "C": ["A", "B", "D", "E"],
    "D": ["B", "C", "E", "F"],
    "E": ["C", "D"],
    "F": ["D"]
}


def bubble_sort(array):
    """冒泡排序实现"""
    for i in range(0, len(array) - 1):
        for j in range(0, len(array) - i - 1):
            if array[j] > array[j + 1]:
                array[j], array[j + 1] = array[j + 1], array[j]
    return array


def select_sort(array):
    """选择排序实现"""
    for i in range(0, len(array) - 1):
        for j in range(i + 1, len(array)):
            if array[j] < array[i]:
                array[j], array[i] = array[i], array[j]
    return array


def insert_sort(array):
    """插入排序实现"""
    for i in range(1, len(array)):
        preindex = i - 1
        current = array[i]
        while preindex >= 0 and current < array[preindex]:
            array[preindex + 1] = array[preindex]
            preindex -= 1
        array[preindex + 1] = current
        # print(array)
    return array


def insert_sort2(array):
    """改进版插入排序"""
    for i in range(1, len(array)):
        # preindex = i - 1
        while i >= 1 and array[i] < array[i - 1]:
            array[i], array[i - 1] = array[i - 1], array[i]
            i -= 1
    return array


def shell_sort(array):
    """希尔排序实现"""
    gap = len(array) // 2
    while gap >= 1:
        for i in range(gap, len(array)):
            while i >= gap and array[i] < array[i - gap]:
                array[i], array[i - gap] = array[i - gap], array[i]
                i -= gap
        print(array)
        gap = gap // 2
    return array


def merge_sort(array):
    """归并排序实现"""
    mid = math.floor(len(array) // 2)
    if len(array) < 2:
        return array
    return merge(merge_sort(array[0:mid]), merge_sort(array[mid:]))


def merge(left, right):
    array = []
    while left and right:
        if left[0] <= right[0]:
            array.append(left.pop(0))
        else:
            array.append(right.pop(0))
    while left:
        array.append(left.pop(0))
    while right:
        array.append(right.pop(0))
    return array


def quick_sort(array, left = None, right = None):
    """快速排序实现"""
    left = 0 if not isinstance(left, (int, float)) else left
    right = len(array) - 1 if not isinstance(right,(int, float)) else right
    if left < right:
        pivot = partition(array, left, right)
        quick_sort(array, left, pivot - 1)
        quick_sort(array, pivot + 1, right)
    return array


def partition(array, left, right):
    pivot = left
    tmp = array[pivot]
    for i in range(left, right):
        if tmp > array[i + 1]:
            array[pivot + 1], array[i + 1] = array[i + 1], array[pivot + 1]
            pivot += 1
    array[left], array[pivot] = array[pivot], array[left]
    return pivot


def count_sort(array, maxvalue):
    bucketlen = maxvalue + 1
    bucket = [0] * bucketlen
    index = 0
    for i in range(len(array)):
        tmp = array[i]
        bucket[tmp] += 1
    for j in range(bucketlen):
        while bucket[j]:
            array[index] = j
            index += 1
            bucket[j] -= 1
    return array


def heap_sort(array):
    """堆排序实现"""
    n = len(array)
    for i in range(math.floor(n / 2), -1, -1):
        heapify(array, n, i)
    for j in range(n - 1, -1, -1):
        array[0], array[j] = array[j], array[0]
        heapify(array, j, 0)
    return array


def heapify(array, n, i):
    c1 = 2 * i + 1
    c2 = 2 * i + 2
    max = i
    if c1 < n and array[c1] > array[max]:
        max = c1
    if c2 <n and array[c2] > array[max]:
        max = c2
    if max != i:
        array[i], array[max] = array[max], array[i]
        heapify(array, n, max)
    return array


def bfs(graph, s):
    queue = []
    queue.append(s)
    seen = set()
    seen.add(s)
    parent = {s: None}

    while (len(queue) > 0):
        vertex = queue.pop(0)
        nodes = graph[vertex]
        for w in nodes:
            if w not in seen:
                queue.append(w)
                seen.add(w)
                parent[w] = vertex
        print(vertex)
    return parent


def dfs(graph, s):
    stack = []
    stack.append(s)
    seen = set()
    seen.add(s)
    while (len(stack) > 0):
        vertex = stack.pop()
        nodes = graph[vertex]
        for w in nodes:
            if w not in seen:
                stack.append(w)
                seen.add(w)
        print(vertex)


def rec_opt(array, i):
    """递归实现"""
    if i == 0:
        return array[0]
    elif i == 1:
        return max(array[0], array[1])
    else:
        A = rec_opt(array, i - 2) + array[i]
        B = rec_opt(array, i - 1)
        return max(A, B)


def dp_opt(array):
    opt = np.zeros(len(array))
    opt[0] = array[0]
    opt[1] = max(array[0], array[1])
    for i in range(2, len(array)):
        A = opt[i - 2] + array[i]
        B = opt[i - 1]
        opt[i] = max(A, B)
    return opt[len(array) - 1]


def rec_subset(array, i, s):
    """递归实现"""
    if s == 0:
        return True
    elif i == 0:
        return array[0] == s
    elif array[i] > s:
        return rec_subset(array, i - 1, s)
    else:
        A = rec_subset(array, i - 1, s - array[i])
        B = rec_subset(array, i - 1, s)
        return A or B


def dp_subset(array, s):
    subset = np.zeros((len(array), s + 1,), dtype=bool)
    subset[:, 0] = True
    subset[0, :] = False
    subset[0, array[0]] = True
    for i in range(1, len(array)):
        for j in range(1, s + 1):
            if array[i] > j:
                subset[i, j] = subset[i - 1, j]
            else:
                A = subset[i - 1, j - array[i]]
                B = subset[i - 1, j]
                subset[i, j] = A or B
    r, c = subset.shape
    return subset[r - 1, c - 1]


def isAnagram(s, t):
    state_1 = [0 for i in range(26)]
    state_2 = [0 for i in range(26)]
    slen = len(s)
    tlen = len(t)
    for j in range(slen):
        index = ord(s[j]) - ord('a')
        state_1[index] += 1
    for k in range(tlen):
        index = ord(t[k]) - ord('a')
        state_2[index] += 1
    for l in range(26):
        if state_1[l] != state_2[l]:
            return False
    return True


def majority(array):
    stack = []
    top = -1
    for i in range(len(array)):
        if top == -1:
            stack.append(array[i])
            top += 1
        elif stack[top] == array[i]:
            stack.append(array[i])
            top += 1
        else:
            top -= 1
            stack.pop(-1)
        print(stack)
    return stack[0]


def majority_pro(array):
    count = 0
    condi = 0
    for i in range(len(array)):
        if count == 0:
            condi = array[i]
            count += 1
        elif condi == array[i]:
            count += 1
        else:
            count -= 1
    return condi


def miss_num(array):
    count = [0 for i in range(len(array) + 1)]
    for j in range(len(array)):
        id = array[j]
        count[id] = 1
    for k in range(len(count)):
        if count[k] == 0:
            return k


def miss_num_pro(array):
    sum = (len(array) * (len(array) + 1)) / 2
    for i in range(len(array)):
        sum -= array[i]
    return sum


"""python异或"""
print(5 ^ 6 ^ 5 == 6)


def fib(n):
    if n < 1:
        return 0
    dp = [0 for i in range(n + 1)]
    return helper(dp, n)


def helper(dp, n):
    if n == 1 or n == 2:
        return 1
    if dp[n] != 0:
        return dp[n]
    dp[n] = helper(dp, n - 1) + helper(dp, n - 2)
    return dp[n]


print(fib(4))


"""
算法工具模块

这个模块提供了常用的算法实现，包括斐波那契数列计算和二分查找。

主要功能:
- 计算斐波那契数列的第n项
- 在有序数组中进行二分查找

作者: Claude Assistant
版本: 1.0.0
"""

from typing import List, Optional


def fib(n: int) -> int:
    """
    计算斐波那契数列的第n项。
    
    斐波那契数列是一个经典的数学序列，其中每个数字是前两个数字的和。
    序列从0和1开始：0, 1, 1, 2, 3, 5, 8, 13, ...
    
    参数:
        n (int): 要计算的斐波那契数列的项数，必须是非负整数
        
    返回:
        int: 斐波那契数列的第n项的值
        
    异常:
        ValueError: 当n为负数时抛出
        
    示例:
        >>> fib(0)
        0
        >>> fib(1)
        1
        >>> fib(5)
        5
        >>> fib(10)
        55
    """
    if n < 0:
        raise ValueError("n必须是非负整数")

    if n <= 1:
        return n

    dp = [0] * (n + 1)
    dp[1] = 1

    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]

    return dp[n]


def binary_search(arr: List[int], target: int) -> int:
    """
    在有序数组中执行二分查找算法。
    
    二分查找是一种高效的搜索算法，时间复杂度为O(log n)。
    该算法要求输入数组必须是有序的（升序排列）。
    
    参数:
        arr (List[int]): 要搜索的有序整数数组，必须按升序排列
        target (int): 要查找的目标值
        
    返回:
        int: 目标值在数组中的索引，如果未找到则返回-1
        
    异常:
        ValueError: 当输入数组为空时抛出
        
    示例:
        >>> binary_search([1, 3, 5, 7, 9], 5)
        2
        >>> binary_search([1, 3, 5, 7, 9], 2)
        -1
        >>> binary_search([2, 4, 6, 8, 10], 8)
        3
        >>> binary_search([1], 1)
        0
    """
    if not arr:
        raise ValueError("输入数组不能为空")
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2

        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1

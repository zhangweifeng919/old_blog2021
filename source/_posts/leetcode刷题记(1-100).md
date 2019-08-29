---
title: leetcode刷题记(1-100)
date: 2019-05-09 14:40:53
tags:
- leetcode
- 算法
categories:
- 算法刷题笔记
description: leetcode 题库标号顺序（1-100）
---
## 两数之和 （简单）

给定一个整数数组 nums 和一个目标值 target，请你在该数组中找出和为目标值的那 两个 整数，并返回他们的数组下标。
你可以假设每种输入只会对应一个答案。但是，你不能重复利用这个数组中同样的元素。

示例：
```
给定 nums = [2, 7, 11, 15], target = 9
因为 nums[0] + nums[1] = 2 + 7 = 9
所以返回 [0, 1]
```
代码：
```PYTHON
from typing import List
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        d={}
        for i in range(len(nums)):
            b=target-nums[i]
            if b in d:
                return [d[b],i]
            else:
                d[nums[i]]=i

if __name__ == '__main__':
    s=Solution()
    r=s.twoSum(nums=[7, 2, 11, 15], target=9)
    print(r)

```
## 两数相加 （简单）
给出两个 非空 的链表用来表示两个非负的整数。其中，它们各自的位数是按照 逆序 的方式存储的，并且它们的每个节点只能存储 一位 数字。
如果，我们将这两个数相加起来，则会返回一个新的链表来表示它们的和。
您可以假设除了数字 0 之外，这两个数都不会以 0 开头。
示例：
```
输入：(2 -> 4 -> 3) + (5 -> 6 -> 4)
输出：7 -> 0 -> 8
原因：342 + 465 = 807
```
代码：
```PYTHON
# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    def addTwoNumbers(self,l1: ListNode, l2: ListNode) -> ListNode:
        ac = False
        head = ListNode(0)
        pre = head
        while True:
            a = 0
            if l1 is None and l2 is None:
                if ac:
                    tmpNode = ListNode(1)
                    pre.next = tmpNode
                break
            if l1 is not None:
                a += l1.val
                l1 = l1.next
            if l2 is not None:
                a += l2.val
                l2 = l2.next
            if ac:
                a += 1
            if a >= 10:
                ac = True
                a %= 10
            else:
                ac = False
            tmpNode = ListNode(a)
            pre.next = tmpNode
            pre = tmpNode
        return head.next


if __name__ == '__main__':
    a = ListNode(2)
    b = ListNode(4)
    c = ListNode(3)
    a.next = b
    b.next = c
    d = ListNode(5)
    e = ListNode(6)
    f = ListNode(4)
    d.next = e
    e.next = f
    g = Solution().addTwoNumbers(a, d)
    while g is not None:
        print(g.val)
        g = g.next

```
## 无重复字符的最长子串 （中等）
给定一个字符串，请你找出其中不含有重复字符的 最长子串 的长度。
示例：
```
输入: "pwwkew"
输出: 3
解释: 因为无重复字符的最长子串是 "wke"，所以其长度为 3。
     请注意，你的答案必须是 子串 的长度，"pwke" 是一个子序列，不是子串。
```
代码：
```PYTHON
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        i = 0
        m = 0
        ls = []
        for c in s:
            if c not in ls:
                ls.append(c)
                if len(ls) > m:
                    m = len(ls)
            else:
                index = ls.index(c)
                ls = ls[index+1:]
                ls.append(c)

        return m


if __name__ == '__main__':
    s = Solution()
    print(s.lengthOfLongestSubstring("dvdf"))
```

## 寻找两个有序数组的中位数 （困难）
给定两个大小为 m 和 n 的有序数组 nums1 和 nums2。
请你找出这两个有序数组的中位数，并且要求算法的时间复杂度为 O(log(m + n))。
你可以假设 nums1 和 nums2 不会同时为空
示例1：
```
nums1 = [1, 3]
nums2 = [2]

则中位数是 2.0
```
示例2：
```
nums1 = [1, 2]
nums2 = [3, 4]

则中位数是 (2 + 3)/2 = 2.5
```
代码：
```PYTHON
from typing import List


def findMedianSortedArrays_K(nums1: List[int], nums2: List[int], k: int) -> float:
    l1 = len(nums1)
    l2 = len(nums2)
    if l1 > l2:
        tmp= l1
        l1=l2
        l2=tmp
        tmp=nums1
        nums1=nums2
        nums2=tmp
    if l1 == 0:
        return nums2[k - 1]
    if k == 1:
        return min(nums1[0], nums2[0])
    m1 = k // 2 - 1
    m2 = k // 2 - 1
    if m1 >= l1:
        m1 = l1 - 1
    if nums1[m1] < nums2[m2]:
        nums1 = nums1[m1 + 1:]
        nums2 = nums2[:k - m1 - 1]
        return findMedianSortedArrays_K(nums1, nums2, k - m1 - 1)
    else:
        nums2 = nums2[m2 + 1:]
        nums1 = nums1[:k - m1 - 1]
        return findMedianSortedArrays_K(nums1, nums2, k - m2 - 1)

class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        l1 = len(nums1)
        l2 = len(nums2)
        l = l1 + l2
        if l % 2 == 1:
            return findMedianSortedArrays_K(nums1, nums2, l // 2 + 1) * 1.0
        else:
            a = findMedianSortedArrays_K(nums1, nums2, l // 2 + 1)
            b = findMedianSortedArrays_K(nums1, nums2, l // 2)
            return (a + b) / 2


if __name__ == '__main__':
    s = Solution()
    b = s.findMedianSortedArrays(nums1=[2,3,4,5,6], nums2=[1])
    print(b)

```
## 最长回文子串 （中等）
给定一个字符串 s，找到 s 中最长的回文子串。你可以假设 s 的最大长度为 1000。
示例：
```
输入: "babad"
输出: "bab"
注意: "aba" 也是一个有效答案。
```
代码：
```PYTHON
class Solution:
    def longestPalindrome(self, s: str) -> str:
        s = "$" + "$".join(s) + "$"
        p = 0
        mx = 0
        ls = {}
        mxindex = 0
        for cur in range(len(s)):
            i = cur - 1
            j = cur + 1
            if cur < mx:
                l = p + p - cur
                if mx - cur > ls[l]:
                    ls[cur] = ls[l]
                    continue
                if mx - cur <= ls[l]:
                    i = cur + cur - mx
                    j = mx
            while i >= 0 and j < len(s):
                if s[i] == s[j]:
                    i -= 1
                    j += 1
                else:
                    break
            p = cur
            mx = j
            ls[cur] = j - cur
            if ls[cur] > ls[mxindex]:
                mxindex = cur
        s = s[mxindex - ls[mxindex] + 1:mxindex + ls[mxindex]]
        return s.replace('$', '')


if __name__ == '__main__':
    so = Solution()
    a = so.longestPalindrome("cbbd")
    print(a)

```
## Z字符变换 （中等）
将一个给定字符串根据给定的行数，以从上往下、从左到右进行 Z 字形排列。

比如输入字符串为 "LEETCODEISHIRING" 行数为 3 时，排列如下：
```
L   C   I   R
E T O E S I I G
E   D   H   N
```
之后，你的输出需要从左往右逐行读取，产生出一个新的字符串，比如："LCIRETOESIIGEDHN"。

示例：
```
输入: s = "LEETCODEISHIRING", numRows = 3
输出: "LCIRETOESIIGEDHN"
```
代码：
```PYTHON
class Solution:
    def convert(self, s: str, numRows: int) -> str:
        r = ''
        if numRows == 1:
            return s
        k = 0
        while k < len(s):
            r += s[k]
            k = k + numRows + numRows - 2

        for i in range(1, numRows - 1):
            k = i
            while k < len(s):
                r += s[k]
                k = k + numRows + numRows - i - i - 2
                if k < len(s):
                    r += s[k]
                    k = k + i + i
        w = numRows - 1
        while w < len(s):
            r += s[w]
            w = w + numRows + numRows - 2
        return r


if __name__ == '__main__':
    s = Solution()
    a = s.convert("0123456789", 4)
    print(a)

```
## 整数反转 （简单）
给出一个 32 位的有符号整数，你需要将这个整数中每位上的数字进行反转。
```
输入: 123
输出: 321

输入: -123
输出: -321

输入: 120
输出: 21
```
注意:

假设我们的环境只能存储得下 32 位的有符号整数，则其数值范围为 [−2^31,  2^31 − 1]。请根据这个假设，如果反转后整数溢出那么就返回 0。

代码：
```python
class Solution:
    def reverse(self, x: int) -> int:
        m = 2147483647
        a = 0
        fu = x < 0
        b = abs(x)
        while True:
            if b > 0:
                c = b % 10
                if a >= 214748365:
                    return 0
                a = a * 10 + c 
                b = b // 10
            else:
                break
        if fu:
            return -1 * a
        else:
            return a


if __name__ == '__main__':
    s = Solution()
    print(s.reverse(-123))

```

## 字符串转换为整数 （中等）

请你来实现一个 atoi 函数，使其能将字符串转换成整数。

首先，该函数会根据需要丢弃无用的开头空格字符，直到寻找到第一个非空格的字符为止。

当我们寻找到的第一个非空字符为正或者负号时，则将该符号与之后面尽可能多的连续数字组合起来，作为该整数的正负号；假如第一个非空字符是数字，则直接将其与之后连续的数字字符组合起来，形成整数。

该字符串除了有效的整数部分之后也可能会存在多余的字符，这些字符可以被忽略，它们对于函数不应该造成影响。

注意：假如该字符串中的第一个非空格字符不是一个有效整数字符、字符串为空或字符串仅包含空白字符时，则你的函数不需要进行转换。

在任何情况下，若函数不能进行有效的转换时，请返回 0。

说明：

假设我们的环境只能存储 32 位大小的有符号整数，那么其数值范围为 [−231,  231 − 1]。如果数值超过这个范围，qing返回  INT_MAX (231 − 1) 或 INT_MIN (−231) 。
示例 1:
```
输入: "42"
输出: 42
```
```
示例 2:

输入: "   -42"
输出: -42
解释: 第一个非空白字符为 '-', 它是一个负号。
     我们尽可能将负号与后面所有连续出现的数字组合起来，最后得到 -42 。
```
```
示例 3:

输入: "4193 with words"
输出: 4193
解释: 转换截止于数字 '3' ，因为它的下一个字符不为数字。
```
代码：
```PYTHON
class Solution:
    def myAtoi(self, str: str) -> int:

        myMin = -2147483648
        myMax = 2147483647
        str = str.lstrip()
        begin = 0
        fu = False
        r = 0
        if len(str) == 0:
            return 0
        if str[0] == '-':
            fu = True
            begin = 1
        elif str[0] == '+':
            begin = 1

        for i in range(begin, len(str)):
            if '0' <= str[i] <= '9':
                if r >= 214748365:
                    if fu:
                        return myMin
                    else:
                        return myMax
                if fu and r == 214748364 and str[i] >= '8':
                    return myMin
                if not fu and  r == 214748364 and str[i] >= '7':
                    return myMax
                b = int(str[i]) - int('0')
                r = r * 10 + b
            else:
                break
        if fu:
            return 0 - r
        else:
            return r


if __name__ == '__main__':
    s = Solution()
    print(s.myAtoi("-2147483648"))

```

## 回文数 （简单）
判断一个整数是否是回文数。回文数是指正序（从左向右）和倒序（从右向左）读都是一样的整数。
示例：
示例 1:
```
输入: 121
输出: true
示例 2:
```
```
输入: -121
输出: false
解释: 从左向右读, 为 -121 。 从右向左读, 为 121- 。因此它不是一个回文数。
```
代码：
```PYTHON
class Solution:
    def isPalindrome(self, x: int) -> bool:
        if x < 0:
            return false
        t = x
        y = 0
        while t > 0:
            y = y * 10 + t % 10
            t = t // 10
        if x == y:
            return True
        else:
            return False


if __name__ == '__main__':
    s = Solution()
    print(s.isPalindrome(121))
```
## 正则表达式匹配 （困难）
给定一个字符串 (s) 和一个字符模式 (p)。实现支持 '.' 和 '*' 的正则表达式匹配。
```
'.' 匹配任意单个字符。
'*' 匹配零个或多个前面的元素。
```
匹配应该覆盖整个字符串 (s) ，而不是部分字符串。
说明:
s 可能为空，且只包含从 a-z 的小写字母。
p 可能为空，且只包含从 a-z 的小写字母，以及字符 . 和 *。
示例 1:
```
输入:
s = "aa"
p = "a"
输出: false
解释: "a" 无法匹配 "aa" 整个字符串。
```
示例2：
```
输入:
s = "aa"
p = "a*"
输出: true
解释: '*' 代表可匹配零个或多个前面的元素, 即可以匹配 'a' 。因此, 重复 'a' 一次, 字符串可变为 "aa"。
```
示例3：
```
输入:
s = "ab"
p = ".*"
输出: true
解释: ".*" 表示可匹配零个或多个('*')任意字符('.')。
```
代码：
```PYTHON
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        p_len = len(p)
        s_len = len(s)
        dp = [[False] * (p_len + 1) for _ in range(s_len + 1)]
        dp[0][0] = True  # dp[i][j] 表示s[:i]与p[:j]是否匹配
        for j in range(2, len(p) + 1):
            dp[0][j] = p[j - 1] == '*' and dp[0][j - 2]

        for i in range(len(s)):
            for j in range(len(p)):
                if s[i] == p[j] or p[j] == '.':
                    dp[i + 1][j + 1] = dp[i][j]
                elif p[j] == '*':
                    if p[j - 1] == s[i] or p[j - 1] == '.':
                        dp[i + 1][j + 1] = dp[i][j] or dp[i][j - 1] or dp[i][j + 1] or dp[i + 1][j - 1]
                    else:
                        dp[i + 1][j + 1] = dp[i + 1][j - 1]
                else:
                    dp[i + 1][j + 1] = False
        return dp[len(s)][len(p)]


if __name__ == '__main__':
    s = Solution()
    print(s.isMatch("aab", "c*a*b"))
```
## 盛最多水的容器 （中等）
给定 n 个非负整数 a1，a2，...，an，每个数代表坐标中的一个点 (i, ai) 。在坐标内画 n 条垂直线，垂直线 i 的两个端点分别为 (i, ai) 和 (i, 0)。找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。
说明：你不能倾斜容器，且 n 的值至少为 2。
![11盛最多水的容器](/images/leetcode/11盛最多水的容器.png)
示例:
```
输入: [1,8,6,2,5,4,8,3,7]
输出: 49
```
代码：
```PYTHON
from typing import List

class Solution:
    def maxArea(self, height: List[int]) -> int:
        i = 0
        j = len(height) - 1
        m = 0
        while i < j:
            a = min(height[i], height[j])
            d = j - i
            t = a * d
            if t > m:
                m = t
            if height[i] < height[j] and i < j:
                while height[i] <= a:
                    i += 1
            else:
                while height[j] <= a and i < j:
                    j -= 1
        return m


if __name__ == '__main__':
    s = Solution()
    r = s.maxArea([1, 8, 6, 2, 5, 4, 8, 3, 7])
    print(r)

```
## 整数转罗马数字 （中等）
罗马数字包含以下七种字符： I， V， X， L，C，D 和 M。
```
字符          数值
I             1
V             5
X             10
L             50
C             100
D             500
M             1000
```
例如， 罗马数字 2 写做 II ，即为两个并列的 1。12 写做 XII ，即为 X + II 。 27 写做  XXVII, 即为 XX + V + II 。
通常情况下，罗马数字中小的数字在大的数字的右边。但也存在特例，例如 4 不写做 IIII，而是 IV。数字 1 在数字 5 的左边，所表示的数等于大数 5 减小数 1 得到的数值 4 。同样地，数字 9 表示为 IX。这个特殊的规则只适用于以下六种情况：
* I 可以放在 V (5) 和 X (10) 的左边，来表示 4 和 9。
* X 可以放在 L (50) 和 C (100) 的左边，来表示 40 和 90。 
* C 可以放在 D (500) 和 M (1000) 的左边，来表示 400 和 900。

给定一个整数，将其转为罗马数字。输入确保在 1 到 3999 的范围内。
示例 1:
```
输入: 3
输出: "III"
```
示例 2:
```
输入: 4
输出: "IV"
```
示例 3:
```
输入: 9
输出: "IX"
```
示例 4:
```
输入: 58
输出: "LVIII"
解释: L = 50, V = 5, III = 3.
```
示例 5:
```
输入: 1994
输出: "MCMXCIV"
解释: M = 1000, CM = 900, XC = 90, IV = 4.
```
代码：
```
class Solution:
    def intToRoman(self, num: int) -> str:
        d = {}
        d[1] = 'I'
        d[2] = 'II'
        d[3] = 'III'
        d[4] = 'IV'
        d[5] = 'V'
        d[6] = 'VI'
        d[7] = 'VII'
        d[8] = 'VIII'
        d[9] = 'IX'
        d[10] = 'X'
        d[20] = 'XX'
        d[30] = 'XXX'
        d[40] = 'XL'
        d[50] = 'L'
        d[60] = 'LX'
        d[70] = 'LXX'
        d[80] = 'LXXX'
        d[90] = 'XC'
        d[100] = 'C'
        d[200] = 'CC'
        d[300] = 'CCC'
        d[400] = 'CD'
        d[500] = 'D'
        d[600] = 'DC'
        d[700] = 'DCC'
        d[800] = 'DCCC'
        d[900] = 'CM'
        d[1000] = 'M'
        d[2000] = 'MM'
        d[3000] = 'MMM'
        if num in d:
            return d[num]

        r = ""
        l = [1000, 100, 10, 1]
        for i in l:
            a = num // i
            if a > 0:
                r = r + d[a * i]
                num = num % i
        return r


if __name__ == '__main__':
    s = Solution()
    r = s.intToRoman(6)
    print(r)

```
## 罗马数字转整数 (简单)
题意如上题基本一致
代码：
```PYTHON
class Solution:
    def romanToInt(self, s: str) -> int:

        i = 0
        r = 0
        while i < len(s):
            if s[i] == 'M':
                r += 1000
                if i > 0 and s[i - 1] == 'C':
                    r -= 200
            elif s[i] == 'D':
                r += 500
                if i > 0 and s[i - 1] == 'C':
                    r -= 200
            elif s[i] == 'C':
                r += 100
                if i > 0 and s[i - 1] == 'X':
                    r -= 20
            elif s[i] == 'L':
                r += 50
                if i > 0 and s[i - 1] == 'X':
                    r -= 20
            elif s[i] == 'X':
                r += 10
                if i > 0 and s[i - 1] == 'I':
                    r -= 2
            elif s[i] == 'V':
                r += 5
                if i > 0 and s[i - 1] == 'I':
                    r -= 2
            elif s[i] == 'I':
                r += 1
            i += 1
        return r


if __name__ == '__main__':
    s = Solution()
    a = s.romanToInt("MCMXCIV")
    print(a)

```
## 最长公共前缀 （简单）
编写一个函数来查找字符串数组中的最长公共前缀。

如果不存在公共前缀，返回空字符串 ""。
示例 1:
```
输入: ["flower","flow","flight"]
输出: "fl"
```
示例 2:
```
输入: ["dog","racecar","car"]
输出: ""
解释: 输入不存在公共前缀。
```
说明:
所有输入只包含小写字母 a-z 。
代码：
```PYTHON
from typing import List


class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        c = ""
        i = 0
        find = False
        while True:
            if len(strs) > 0 and len(strs[0]) > i:
                com = strs[0][i]
            else:
                break
            for s in strs:
                if len(s) <= i or s[i] != com:
                    find = True
            if find:
                break
            else:
                c += com
            i += 1
        return c[:i]


if __name__ == '__main__':
    s = Solution()
    a = s.longestCommonPrefix(["dog", "racecar", "car"])
    print(a)
```
## 三数之和 （中等）
给定一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？找出所有满足条件且不重复的三元组。

注意：答案中不可以包含重复的三元组。
```
例如, 给定数组 nums = [-1, 0, 1, 2, -1, -4]，

满足要求的三元组集合为：
[
  [-1, 0, 1],
  [-1, -1, 2]
]
```
代码：
```PYTHON
from typing import List


class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums = sorted(nums)
        l = len(nums) - 1
        mylist = []
        cur = 0
        while cur < l - 1:
            i = cur + 1
            j = l
            while i < j:
                r = nums[cur] + nums[i] + nums[j]
                if r == 0:
                    tmp = [nums[cur], nums[i], nums[j]]
                    mylist.append(tmp)
                    t = i
                    while nums[i] == nums[t] and i < j:
                        i += 1
                elif r < 0:
                    i += 1
                else:
                    j -= 1
            w = cur
            while nums[w] == nums[cur] and cur < l - 1:
                cur += 1
        return mylist


if __name__ == '__main__':
    s = Solution()
    a = s.threeSum([0,0,0,0])
    print(a)
```
## 最接近的三数之和
给定一个包括 n 个整数的数组 nums 和 一个目标值 target。找出 nums 中的三个整数，使得它们的和与 target 最接近。返回这三个数的和。假定每组输入只存在唯一答案。
```
例如，给定数组 nums = [-1，2，1，-4], 和 target = 1.

与 target 最接近的三个数的和为 2. (-1 + 2 + 1 = 2).
```
代码：
```PYTHON
from typing import List


class Solution:
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        nums = sorted(nums)
        k = 0
        i = k + 1
        j = len(nums) - 1
        a = 100000000
        c = 0
        while k < len(nums) - 2:
            while i < j:
                t = nums[k] + nums[i] + nums[j]
                if t == target:
                    return target
                elif t > target:
                    j -= 1
                elif t < target:
                    i += 1
                if abs(t - target) < a:
                    c = t
                    a = abs(t - target)
            while k + 1 < len(nums) - 1 and nums[k] == nums[k + 1]:
                k += 1
            k += 1
            i = k + 1
            j = len(nums) - 1
        return c


if __name__ == '__main__':
    s = Solution()
    r = s.threeSumClosest([0, 0, 0], 1)
    print(r)

```
## 电话号码的字母组合
给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合。
给出数字到字母的映射如下（与电话按键相同）。注意 1 不对应任何字母。
![图片](/images/leetcode/17_telephone_keypad.png)
```
输入："23"
输出：["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"].
说明:
尽管上面的答案是按字典序排列的，但是你可以任意选择答案输出的顺序。
```

代码：
```PYTHON
class Solution(object):
    m = {
        '2':'abc', '3':'def', '4':'ghi',
        '5':'jkl', '6':'mno', '7':'pqrs',
        '8':'tuv', '9':'wxyz'
    }
    def dfs(self, i, digits, ans, tmp):
        if i == len(digits):
            ans.append(''.join(tmp))
            return
        for ch in self.m[digits[i]]:
            tmp.append(ch)
            self.dfs(i+1, digits, ans, tmp)
            tmp.pop()
            
    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """
        if not digits:
            return []
        ans = []
        tmp = []
        self.dfs(0, digits, ans, tmp)
        return ans
```
## 四数之和
给定一个包含 n 个整数的数组 nums 和一个目标值 target，判断 nums 中是否存在四个元素 a，b，c 和 d ，使得 a + b + c + d 的值与 target 相等？找出所有满足条件且不重复的四元组。
注意：
答案中不可以包含重复的四元组。
示例：
```PYTHON
给定数组 nums = [1, 0, -1, 0, -2, 2]，和 target = 0。

满足要求的四元组集合为：
[
  [-1,  0, 0, 1],
  [-2, -1, 1, 2],
  [-2,  0, 0, 2]
]
```
代码：
```PYTHON
class Solution:
    def fourSum(self,nums: List[int], target: int) -> List[List[int]]:
        nums=sorted(nums)
        print(nums)
        result=[]
        if len(nums)<4:
            return result
        first=0  
        length=len(nums)
        while first< length-3:
            second=first+1
            while second < length-2:
                t1=target-nums[first]-nums[second]
                third=second+1
                fourth=length-1
                while third<fourth:
                    tmp=nums[third]+nums[fourth]
                    if t1==tmp:
                         result.append([nums[first],nums[second],nums[third],nums[fourth]])
                         lt=nums[third]
                         third+=1
                         while nums[third]==lt and third<fourth:
                             third+=1
                    elif t1>tmp:
                        third+=1
                    else:
                        fourth-=1
                ls=nums[second]
                second+=1
                while ls==nums[second] and second<length-2:
                    second+=1
            lf=nums[first]
            first+=1
            while lf==nums[first] and first<length-3:
                first+=1
        return result

```


## 删除链表的倒数第N个节点
给定一个链表，删除链表的倒数第 n 个节点，并且返回链表的头结点。

示例：
```
给定一个链表: 1->2->3->4->5, 和 n = 2.

当删除了倒数第二个节点后，链表变为 1->2->3->5.
说明：

给定的 n 保证是有效的。
```
代码：
```PYTHON
class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        i=head
        j=head
        while n>0:
            i=i.next
            n-=1
        if i==None:
            head=j.next
            return head
        while i.next != None:
            i=i.next
            j=j.next
        j.next=j.next.next
        return head
```
## 有效括号
给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串，判断字符串是否有效。

有效字符串需满足：

左括号必须用相同类型的右括号闭合。
左括号必须以正确的顺序闭合。
注意空字符串可被认为是有效字符串。

代码：
```PYTHON
class Solution:
    def isValid(self, s: str) -> bool:
        stack=[]
        for i in range(len(s)):
            if s[i] in ['(','{','[']:
                stack.append(s[i])
                continue
            if len(stack)<1:
                return False
            if s[i]==')' and stack[-1]=='(':
                stack.pop()
                continue
            if s[i]=='}' and stack[-1]=='{':
                stack.pop()
                continue
            if s[i]==']' and stack[-1]=='[':
                stack.pop()
                continue
            
            return False
        if len(stack)>0:
            return False
        else:
            return True

```
## 合并两个有序列表
将两个有序链表合并为一个新的有序链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。 
示例：
```
输入：1->2->4, 1->3->4
输出：1->1->2->3->4->4
```
代码：
```PYTHON
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        head=ListNode(0)
        i=head
        while True:
            if l1==None and l2==None:
                break
            if l1!=None and l2 !=None:
                if l1.val>l2.val:
                    tmp=ListNode(l2.val)
                    l2=l2.next
                else:
                    tmp=ListNode(l1.val)
                    l1=l1.next
                i.next=tmp
                i=i.next
                continue
                
            if l1==None:
                i.next=l2
                break
            if l2==None:
                i.next=l1
                break
        return head.next

```
## 括号生成
给出 n 代表生成括号的对数，请你写出一个函数，使其能够生成所有可能的并且有效的括号组合。
例如，给出 n = 3，生成结果为：
```
[
  "((()))",
  "(()())",
  "(())()",
  "()(())",
  "()()()"
]
```
代码：
```PYTHON
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        
        result=[]
        self.f(0,0,n,"",result)
        return result
        
        
    def f(self,opened,closed,m,tmp,result):
        if opened==closed and closed==m:
            result.append(tmp)
            return
        if opened>closed:
            self.f(opened,closed+1,m,tmp+')',result)
        if opened<m:
            self.f(opened+1,closed,m,tmp+'(',result)
            
```
## 合并K个排序链表
合并 k 个排序链表，返回合并后的排序链表。请分析和描述算法的复杂度。
示例：
```
输入:
[
  1->4->5,
  1->3->4,
  2->6
]
输出: 1->1->2->3->4->4->5->6
```
代码：
```PYTHON
class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        if len(lists)<1:
            return None
        if len(lists)<2:
            return lists[0]
        
        if len(lists)==2:
            return self.mergeTwo(lists[0],lists[1])
        length=len(lists)
        a=self.mergeKLists(lists[0:length//2])
        b=self.mergeKLists(lists[length//2:])
        return self.mergeTwo(a,b)
    def mergeTwo(self,a,b):
        head=ListNode(0)
        iNode=head
        while True:
            if a==None:
                iNode.next=b
                break
            if b==None:
                iNode.next=a
                break
            if a.val>b.val:
                iNode.next=ListNode(b.val)
                b=b.next
            else:
                iNode.next=ListNode(a.val)
                a=a.next
            iNode=iNode.next
        return head.next
                
```
## 两两交换链表中的节点
给定一个链表，两两交换其中相邻的节点，并返回交换后的链表。
你不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。
示例：
```
给定 1->2->3->4, 你应该返回 2->1->4->3.
```
代码：
```PYTHON
class Solution:
    def swapPairs(self, head: ListNode) -> ListNode:
        if head==None or head.next==None:
            return head
        iNode=head
        pre=ListNode(0)
        pre.next=iNode
        head=pre
        while iNode!=None and iNode.next !=None:
            nex=iNode.next
            pre.next=nex
            iNode.next=nex.next
            nex.next=iNode
            pre=iNode
            iNode=iNode.next
        return head.next
```
## K个一组翻转链表
给你一个链表，每 k 个节点一组进行翻转，请你返回翻转后的链表。
k 是一个正整数，它的值小于或等于链表的长度。
如果节点总数不是 k 的整数倍，那么请将最后剩余的节点保持原有顺序。
示例：
```
给定这个链表：1->2->3->4->5
当 k = 2 时，应当返回: 2->1->4->3->5
当 k = 3 时，应当返回: 3->2->1->4->5
说明 :
你的算法只能使用常数的额外空间。
你不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。
```
代码：
```PYTHON
class Solution:
    def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
        if head==None or head.next==None or k==1:
            return head
        pre=None
        start=head
        tail=head
        while tail!=None:
            for i in range(k-1):
                tail=tail.next
                if tail==None:
                    if pre!=None:
                        pre.next=start
                    return head
            
            next_start=tail.next
            tail.next=None
            restart=self.reverseList(start)
            if pre==None:
                head=restart
            else:
                pre.next=restart
            pre=start
            start=next_start
            tail=start
    
        return head
    def reverseList(self,head):
        pre=head
        c=head
        tail=head
        
        while c!=None:
            nex=c.next
            c.next=pre
            pre=c
            c=nex
        tail.next=None
        return pre
```
## 删除排序数组中的重复项
给定一个排序数组，你需要在原地删除重复出现的元素，使得每个元素只出现一次，返回移除后数组的新长度。
不要使用额外的数组空间，你必须在原地修改输入数组并在使用 O(1) 额外空间的条件下完成。
示例：
```
给定数组 nums = [1,1,2], 
函数应该返回新的长度 2, 并且原数组 nums 的前两个元素被修改为 1, 2。 
你不需要考虑数组中超出新长度后面的元素。

给定 nums = [0,0,1,1,1,2,2,3,3,4],
函数应该返回新的长度 5, 并且原数组 nums 的前五个元素被修改为 0, 1, 2, 3, 4。
你不需要考虑数组中超出新长度后面的元素。
```
代码：
```PYTHON
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        i=0
        j=1
        while j<len(nums):
            if nums[i]==nums[j]:
                j+=1
            else:
                i+=1
                nums[i]=nums[j]
                j+=1
        return i+1
```
## 移除元素
给定一个数组 nums 和一个值 val，你需要原地移除所有数值等于 val 的元素，返回移除后数组的新长度。
不要使用额外的数组空间，你必须在原地修改输入数组并在使用 O(1) 额外空间的条件下完成。
元素的顺序可以改变。你不需要考虑数组中超出新长度后面的元素。
示例：
```
给定 nums = [3,2,2,3], val = 3,
函数应该返回新的长度 2, 并且 nums 中的前两个元素均为 2。
你不需要考虑数组中超出新长度后面的元素。
给定 nums = [0,1,2,2,3,0,4,2], val = 2,
函数应该返回新的长度 5, 并且 nums 中的前五个元素为 0, 1, 3, 0, 4。
注意这五个元素可为任意顺序。
```
代码：
```PYTHON
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        i=0
        j=len(nums)-1
        while i<=j:
            if nums[i]==val:
                nums[i]=nums[j]
                j-=1
            else:
                i+=1
            
        return i
```
## 实现str.Str()
实现 strStr() 函数。
给定一个 haystack 字符串和一个 needle 字符串，在 haystack 字符串中找出 needle 字符串出现的第一个位置 (从0开始)。如果不存在，则返回  -1。
示例：
```
输入: haystack = "hello", needle = "ll"
输出: 2

输入: haystack = "aaaaa", needle = "bba"
输出: -1
```
说明:
当 needle 是空字符串时，我们应当返回什么值呢？这是一个在面试中很好的问题。
对于本题而言，当 needle 是空字符串时我们应当返回 0 。这与C语言的 strstr() 以及 Java的 indexOf() 定义相符。
代码：
```PYTHON
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        a=len(needle)
        b=len(haystack)
        if a<1:
            return 0
        if b<1:
            return -1
        next=self.getNext(needle)
        i=0
        j=0
        index=-1
        while i<b and j<a:
            if j==-1 or haystack[i]==needle[j]:
                i+=1
                j+=1
            else:
                j=next[j]
        if j==a:
            index=i-j
        
        return index
        
    def getNext(self,template):
        l=len(template)
        next=[0]*l
        next[0]=-1
        k=-1
        j=0
        while j<l-1:
            if k==-1 or template[k]==template[j]:
                k+=1
                j+=1
                next[j]=k
            else:
                k=next[k]
        return next
        
        
```
## 未完待续。。。

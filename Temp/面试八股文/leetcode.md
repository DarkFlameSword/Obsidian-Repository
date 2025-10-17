---
date: 2025-10-16
author:
  - Siyuan Liu
tags:
  - 八股
---
![[Pasted image 20251017150145.png]]
# 排序类（Sort）

**基础知识：**
[快速排序](https://zhida.zhihu.com/search?content_id=165894837&content_type=Article&match_order=1&q=%E5%BF%AB%E9%80%9F%E6%8E%92%E5%BA%8F&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NjA3Njg2MzUsInEiOiLlv6vpgJ_mjpLluo8iLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxNjU4OTQ4MzcsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.MhI7CmuobuuO6J1luE-QjaiyQa84mmY15T6LJKuMPbA&zhida_source=entity)（Quick Sort）， [归并排序](https://zhida.zhihu.com/search?content_id=165894837&content_type=Article&match_order=1&q=%E5%BD%92%E5%B9%B6%E6%8E%92%E5%BA%8F&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NjA3Njg2MzUsInEiOiLlvZLlubbmjpLluo8iLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxNjU4OTQ4MzcsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.b9h4g3rCxQWGlblORyO0b-_p2erFAfe_xEdiq1aIknE&zhida_source=entity)（Merge Sort）， 堆排序的原理与代码实现。需要能讲明白代码中每一行的目的

快速排序时间复杂度平均状态下O（NlogN），空间复杂度O（1），归并排序最坏情况下时间复杂度O（NlogN），空间复杂度O（N）

**入门题目：**
- Leetcode 148. Sort List
- Leetcode 56. Merge Intervals
- Leetcode 27. Remove elements

**进阶题目：**
- Leetcode 179. Largest Number
- Leetcode 75. Sort Colors
- ==Leetcode 215. Kth Largest Element （可以用[堆](https://zhida.zhihu.com/search?content_id=165894837&content_type=Article&match_order=1&q=%E5%A0%86&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NjA3Njg2MzUsInEiOiLloIYiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxNjU4OTQ4MzcsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.zSvBWMiTnXGgZlYgZ6jhRdT1IITSu2AX2BewZrv8atY&zhida_source=entity)的解法替代）==
- Leetcode 4. Median of Two Sorted Arrays

注意：后两题是与快速排序非常相似的快速选择（Quick Select）算法，面试中很常考

---
# 链表类（Linked List）

**基础知识：**
- [链表](https://zhida.zhihu.com/search?content_id=165894837&content_type=Article&match_order=1&q=%E9%93%BE%E8%A1%A8&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NjA3Njg2MzUsInEiOiLpk77ooagiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxNjU4OTQ4MzcsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.WkxsNtjX3SuVFPEgfVyVqAdo84Hmo1BIDuhS8ZhUCgQ&zhida_source=entity)如何实现，如何遍历链表。链表可以保证头部尾部插入删除操作都是O（1），查找任意元素位置O（N）

**基础题目：**
- ==Leetcode 206. Reverse Linked List==
- Leetcode 876. Middle of the Linked List

注意：快慢指针和链表反转几乎是所有链表类问题的基础，尤其是反转链表，代码很短，建议直接背熟。

**进阶题目:**
- Leetcode 25. K 个一组翻转链表
- Leetcode 21. 合并两个有序链表
- Leetcode 160. Intersection of Two Linked Lists
- Leetcode 141. Linked List Cycle (Linked List Cycle II)
- Leetcode 92. Reverse Linked List II
- Leetcode 328. Odd Even Linked List

---
# 哈希表（Hashmap、Hashset）
**基础知识：**
[哈希表](https://zhida.zhihu.com/search?content_id=165894837&content_type=Article&match_order=1&q=%E5%93%88%E5%B8%8C%E8%A1%A8&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NjA3Njg2MzUsInEiOiLlk4jluIzooagiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxNjU4OTQ4MzcsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.PTOvW6FWBQHuCA5O_B6ORvfRc2y8hBgR1J_jHD7t7mo&zhida_source=entity)

**基础题目：**
- Leetcode 1. Two Sum
- ==Leetcode 146. LRU Cache (Python中可以使用OrderedDict来代替)==
- Leetcode 128. Longest Consecutive Sequence
- Leetcode 73. Set Matrix Zeroes
- Leetcode 380. Insert Delete GetRandom O(1)
- Leetcode 49. Group Anagrams
- Leetcode 350. Intersection of Two Arrays II
- Leetcode 299. Bulls and Cows
- Leetcode 348 Design Tic-Tac-Toe

---
# 字符串

## 字符串反转类
**基础知识：**

**基础题目：**
- Leetcode 344. 反转字符串
- Leetcode 541. 反转字符串 II
- Leetcode 557. 反转字符串中的单词 III
- Leetcode 7. 整数反转
- Leetcode 151. 反转字符串中的单词

## 回文串类
**基础题目：**
- Leetcode 125. 验证回文串
- Leetcode 647. 回文子串
- Leetcode 5. 最长回文子串
- Leetcode 516. 最长回文子序列

## 字符操作类
**基础题目：**
- Leetcode 415. 字符串相加
- Leetcode 43. 字符串相乘
- Leetcode 387. 字符串中的第一个唯一字符
- Leetcode 242. 有效的字母异位词
- Leetcode 434. 字符串中的单词数
- Leetcode 6. Z 字形变换

## 子串匹配类
**基础题目：**
- Leetcode 28. 找出字符串中第一个匹配项的下标
- Leetcode 14. 最长公共前缀
- Leetcode 459. 重复的子字符串

## 括号匹配类
**基础题目：**
- Leetcode 20. 有效的括号
- Leetcode 1021. 删除最外层的括号
- Leetcode 32. 最长有效括号

---
# 二分法（Binary Search）

**基础知识：**
[二分法](https://zhida.zhihu.com/search?content_id=165894837&content_type=Article&match_order=1&q=%E4%BA%8C%E5%88%86%E6%B3%95&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NjA3Njg2MzUsInEiOiLkuozliIbms5UiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxNjU4OTQ4MzcsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.LlfEl5XDpCgoEbVbaQB2yEVeL93gYseBdeHUdmZcmGc&zhida_source=entity)是用来解法基本模板，时间复杂度logN；常见的二分法题目可以分为两大类，显式与隐式，即是否能从字面上一眼看出二分法的特点：要查找的数据是否可以分为两部分，前半部分为X，后半部分为O

## 显式二分法
- Leetcode 34. Find First and Last Position of Element in Sorted Array
- Leetcode 33. Search in Rotated Sorted Array
- Leetcode 1095. Find in Mountain Array
- Leetcode 162. Find Peak Element
- Leetcode 278. First Bad Version
- Leetcode 74. Search a 2D Matrix
- Leetcode 240. Search a 2D Matrix II

## 隐式二分法
- Leetcode 69. Sqrt(x)
- Leetcode 540. Single Element in a Sorted Array
- Leetcode 644. Maximum Average Subarray II
- Leetcode 528. Random Pick with Weight
- Leetcode 1300. Sum of Mutated Array Closest to Target
- Leetcode 1060. Missing Element in Sorted Array
- Leetcode 1062. Longest Repeating Substring
- Leetcode 1891. Cutting Ribbons
- Leetcode 410. Split Array Largest Sum (与1891类似)

---
# 双指针（2 Pointer）
**基础知识：**
常见双指针算法分为三类，同向（即两个指针都相同一个方向移动），背向（两个指针从相同或者相邻的位置出发，背向移动直到其中一根指针到达边界为止），相向（两个指针从两边出发一起向中间移动直到两个指针相遇）

## 背向双指针
基本上全是回文串的题
- Leetcode 409. Longest Palindrome
- Leetcode 125. Valid Palindrome (I、II)
- Leetcode 5. Longest Palindromic Substring
- Leetcode 647. Palindromic Substrings

## 相向双指针
以two sum为基础的一系列题

- Leetcode 1. Two Sum (这里使用的是先排序的双指针算法，不同于hashmap做法)
- Leetcode 167. Two Sum II - Input array is sorted
- Leetcode 15. 3Sum
- Leetcode 18. 4Sum
- Leetcode 454. 4Sum II
- Leetcode 277. Find the Celebrity
- Leetcode 11. Container With Most Water
- Leetcode 186 Reverse Words in a String II
- Leetcode 125. 验证回文串


## 同向双指针
个人觉得最难的一类题，可以参考下这里 [TimothyL：Leetcode 同向双指针/滑动窗口类代码模板](https://zhuanlan.zhihu.com/p/390570255)

- Leetcode 283. Move Zeroes
- Leetcode 26. Remove Duplicate Numbers in Array
- Leetcode 395. Longest Substring with At Least K Repeating Characters
- Leetcode 340. Longest Substring with At Most K Distinct Characters
- Leetcode 424. Longest Repeating Character Replacement
- Leetcode 76. Minimum Window Substring
- ==Leetcode 3. Longest Substring Without Repeating Characters==
- Leetcode 1004 Max Consecutive Ones III
- Leetcode 1658 Minimum Operations to Reduce X to Zero

---
# 堆（Heap）
**基础知识：**
[栈](https://zhida.zhihu.com/search?content_id=165894837&content_type=Article&match_order=1&q=%E6%A0%88&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NjA3Njg2MzUsInEiOiLmoIgiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxNjU4OTQ4MzcsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.Ebt8haz9_zJVSwaQTUVZ8aDniMjMZDVOEbMey6o_zPk&zhida_source=entity)（Stack）、（Queue or Priority Queue）、类（Hashmap、Hashset)

**基础题目：**
- Leetcode 973. K Closest Points
- Leetcode 347. Top k Largest Elements
- Leetcode 23. Merge K Sorted Lists
- Leetcode 264. Ugly Number II
- Leetcode 1086. High Five
- Leetcode 88. Merge Sorted Arrays
- Leetcode 692. Top K Frequent Words
- Leetcode 378. Kth Smallest Element in a Sorted Matrix
- Leetcode 295. Find Median from Data Stream （标准解法是双heap，但是SortedDict会非常容易）
- Leetcode 767. Reorganize String
- Leetcode 1438. Longest Continuous Subarray With Absolute Diff Less Than or Equal to Limit (这个题用单调双端队列、TreeMap、双heap都可以)
- Leetcode 895. Maximum Frequency Stack

---
# 栈（Stack）
**基础知识：**
[栈](https://zhida.zhihu.com/search?content_id=165894837&content_type=Article&match_order=1&q=%E6%A0%88&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NjA3Njg2MzUsInEiOiLmoIgiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxNjU4OTQ4MzcsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.Ebt8haz9_zJVSwaQTUVZ8aDniMjMZDVOEbMey6o_zPk&zhida_source=entity)

**基础题目：**
- Leetcode 739 Daily Temperatures
- Leetcode 503 Next Greater Element II
- Leetcode 901 Online Stock Span
- Leetcode 85 Maximum Rectangle  
- Leetcode 84 Largest Rectangle in Histogram  
- Leetcode 907 Sum of Subarray Minimums (与84类似)  
- Leetcode 239 Sliding Window Maximum （唯一的单调队列题）

---
# 队列（Queue）
**基础知识：**
[队列](https://zhida.zhihu.com/search?content_id=165894837&content_type=Article&match_order=1&q=%E9%98%9F%E5%88%97&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NjA3Njg2MzUsInEiOiLpmJ_liJciLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxNjU4OTQ4MzcsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.m-7MZ9oqc-0JeR4L-zzsxlmC-J7UCyEi2UtBMyk-BJo&zhida_source=entity)

**基础题目：**
- Leetcode 225. Implement Stack using Queues
- Leetcode 346. Moving Average from Data Stream
- Leetcode 281. Zigzag Iterator
- Leetcode 1429. First Unique Number
- Leetcode 54. Spiral Matrix
- Leetcode 362. Design Hit Counter


---

# 宽度优先搜索（BFS）
面试中最常考的

**基础知识：**
常见的[BFS](https://zhida.zhihu.com/search?content_id=165894837&content_type=Article&match_order=1&q=%E5%AE%BD%E5%BA%A6%E4%BC%98%E5%85%88%E6%90%9C%E7%B4%A2&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NjA3Njg2MzUsInEiOiLlrr3luqbkvJjlhYjmkJzntKIiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxNjU4OTQ4MzcsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.CINR5GDdLbIyfgnQV1ES05GzRwVVP8xIugN7i4Jiz-E&zhida_source=entity)用来解决什么问题？(1) 简单图（有向无向皆可）的最短路径长度，注意是长度而不是具体的路径（2）拓扑排序 （3） 遍历一个图（或者树）

BFS基本模板（需要记录层数或者不需要记录层数）

多数情况下时间复杂度空间复杂度都是O（N+M），N为节点个数，M为边的个数

## 基于树的BFS
不需要专门一个set来记录访问过的节点

- Leetcode 102 Binary Tree Level Order Traversal
- Leetcode 103 Binary Tree Zigzag Level Order Traversal
- Leetcode 297 Serialize and Deserialize Binary Tree （很好的BFS和双指针结合的题）
- Leetcode 314 Binary Tree Vertical Order Traversal

## 基于图的BFS
一般需要一个set来记录访问过的节点

- Leetcode 200. Number of Islands
- Leetcode 133. Clone Graph
- Leetcode 127. Word Ladder
- Leetcode 490. The Maze
- Leetcode 323. Connected Component in Undirected Graph
- Leetcode 130. Surrounded Regions
- Leetcode 752. Open the Lock
- Leetcode 815. Bus Routes
- Leetcode 1091. Shortest Path in Binary Matrix
- Leetcode 542. 01 Matrix
- Leetcode 1293. Shortest Path in a Grid with Obstacles Elimination
- Leetcode 417. Pacific Atlantic Water Flow

## 拓扑排序
[https://zh.wikipedia.org/wiki/%E6%8B%93%E6%92%B2%E6%8E%92%E5%BA%8F](https://link.zhihu.com/?target=https%3A//zh.wikipedia.org/wiki/%25E6%258B%2593%25E6%2592%25B2%25E6%258E%2592%25E5%25BA%258F)

- Leetcode 207 Course Schedule （I, II）
- Leetcode 444 Sequence Reconstruction
- Leetcode 269 Alien Dictionary
- Leetcode 310 Minimum Height Trees
- Leetcode 366 Find Leaves of Binary Tree

---
# 深度优先搜索（DFS）
面试中最常考的（分类的稍微有点粗糙了，没有细分出回溯/分治来，准备找个时间给每个DFS的题标记下是哪种DFS）

基础知识：
常见的[DFS](https://zhida.zhihu.com/search?content_id=165894837&content_type=Article&match_order=1&q=%E6%B7%B1%E5%BA%A6%E4%BC%98%E5%85%88%E6%90%9C%E7%B4%A2&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NjA3Njg2MzUsInEiOiLmt7HluqbkvJjlhYjmkJzntKIiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxNjU4OTQ4MzcsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.Gd11DkMiGVejjeHpRrmw0VdQfQaLi7D5Qhdrhr6TL_Q&zhida_source=entity)用来解决什么问题？(1) 图中（有向无向皆可）的符合某种特征（比如最长）的路径以及长度（2）排列组合（3） 遍历一个图（或者树）（4）找出图或者树中符合题目要求的全部方案

DFS基本模板（需要记录路径，不需要返回值 and 不需要记录路径，但需要记录某些特征的返回值）

除了遍历之外多数情况下时间复杂度是指数级别，一般是O(方案数×找到每个方案的时间复杂度)

递归题目都可以用非递归迭代的方法写，但一般实现起来非常麻烦

## 基于树的DFS
需要记住递归写前序中序后序遍历二叉树的模板

- Leetcode 543 Diameter of Binary Tree (分治)
- Leetcode 124 Binary Tree Maximum Path Sum (分治)
- Leetcode 226 Invert Binary Tree (分治)
- Leetcode 101 Symmetric Tree (回溯 or 分治)
- Leetcode 951 Flip Equivalent Binary Trees (分治)
- Leetcode 236 Lowest Common Ancestor of a Binary Tree (相似题：235、1650) (回溯 or 分治)
- Leetcode 105 Construct Binary Tree from Preorder and Inorder Traversal (分治)
- Leetcode 104 Maximum Depth of Binary Tree (回溯 or 分治)
- Leetcode 987 Vertical Order Traversal of a Binary Tree
- Leetcode 1485 Clone Binary Tree With Random Pointer
- Leetcode 572 Subtree of Another Tree (分治)
- Leetcode 863 All Nodes Distance K in Binary Tree
- Leetcode 1110 Delete Nodes And Return Forest (分治)




## 基于图的DFS
和BFS一样一般需要一个set来记录访问过的节点，避免重复访问造成死循环; Word XXX 系列面试中非常常见，例如word break，word ladder，word pattern，word search。

- Leetcode 341 Flatten Nested List Iterator (339 364)
- Leetcode 394 Decode String
- Leetcode 51 N-Queens (I II基本相同)
- Leetcode 291 Word Pattern II (I为简单的Hashmap题)
- Leetcode 126 Word Ladder II （I为BFS题目）
- Leetcode 93 Restore IP Addresses
- Leetcode 22 Generate Parentheses
- Leetcode 856 Score of Parentheses
- Leetcode 301 Remove Invalid Parentheses
- Leetcode 37 Sodoku Solver
- Leetcode 212 Word Search II （I, II）
- Leetcode 1087 Brace Expansion
- Leetcode 399 Evaluate Division
- Leetcode 1274 Number of Ships in a Rectangle
- Leetcode 1376 Time Needed to Inform All Employees
- Leetcode 694 Number of Distinct Islands
- Leetcode 131 Palindrome Partitioning

## 基于排列组合的DFS
其实与图类DFS方法一致，但是排列组合的特征更明显

- Leetcode 17 Letter Combinations of a Phone Number
- Leetcode 39 Combination Sum（I, II, III相似， IV为动态规划题目）
- Leetcode 78 Subsets （I, II 重点在于如何去重）
- Leetcode 46 Permutation (I, II 重点在于如何去重)
- Leetcode 77 Combinations (I, II 重点在于如何去重)
- Leetcode 698 Partition to K Equal Sum Subsets
- Leetcode 526 Beautiful Arrangement (similar to 46)

## 记忆化搜索（DFS + Memoization Search）
算是用递归的方式实现动态规划，递归每次返回时同时记录下已访问过的节点特征，避免重复访问同一个节点，可以有效的把指数级别的DFS时间复杂度降为多项式级别; 注意这一类的DFS必须在最后有返回值（分治法），不可以用回溯法; for循环的dp题目都可以用记忆化搜索的方式写，但是不是所有的记忆化搜索题目都可以用for循环的dp方式写。

- Leetcode 139 Word Break II
- Leetcode 72 Edit Distance
- Leetcode 377 Combination Sum IV
- Leetcode 1235 [Maximum Profit in Job Scheduling](https://link.zhihu.com/?target=https%3A//leetcode.com/explore/item/3950)
- Leetcode 1335 Minimum Difficulty of a Job Schedule
- Leetcode 1216 Valid Palindrome III
- Leetcode 97 Interleaving String
- Leetcode 472 Concatenated Words
- Leetcode 403 Frog Jump
- Leetcode 329 Longest Increasing Path in a Matrix


# 二叉树
**基础知识：**


**常见题目：**
- Leetcode 144. 二叉树的前序遍历
- Leetcode 94. 二叉树的中序遍历
- Leetcode 145. 二叉树的后序遍历
- Leetcode 102. 二叉树的层序遍历
- Leetcode 107. 二叉树的层序遍历 II
- Leetcode 236. 二叉树的最近公共祖先

## 二叉搜索树（BST）
[BST](https://zhida.zhihu.com/search?content_id=165894837&content_type=Article&match_order=1&q=%E4%BA%8C%E5%8F%89%E6%90%9C%E7%B4%A2%E6%A0%91&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NjA3Njg2MzUsInEiOiLkuozlj4nmkJzntKLmoJEiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxNjU4OTQ4MzcsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.cxvs3cylHAikDgSeFOwxNFTem5f6k4yjK_FmytjvCXk&zhida_source=entity)特征：中序遍历为单调递增的二叉树，换句话说，根节点的值比左子树任意节点值都大，比右子树任意节点值都小，增删查改均为O（h）复杂度，h为树的高度；注意不是所有的BST题目都需要递归，有的题目只需要while循环即可

- Leetcode 230 Kth Smallest element in a BST
- Leetcode 98 Validate Binary Search Tree
- Leetcode 270 Cloest Binary Search Tree Value
- Leetcode 235 Lowest Common Ancestor of a Binary Search Tree
- Leetcode 669 Trim a Binary Search Tree (分治)
- Leetcode 700 Search in a Binary Search Tree
- Leetcode 108 Convert Sorted Array to Binary Search Tree (分治)
- Leetcode 333 Largest BST Subtree (与98类似) (分治)
- Leetcode 285 Inorder Successor in BST (I, II)

---
# 回溯
**基础知识：**
[回溯算法](https://zhida.zhihu.com/search?content_id=239052399&content_type=Article&match_order=1&q=%E5%9B%9E%E6%BA%AF%E7%AE%97%E6%B3%95&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NjA4NDQ4NjIsInEiOiLlm57muq_nrpfms5UiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyMzkwNTIzOTksImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.3qwvAWIiw7OC6T7bhlNcF-QZPcxO-Hr6LHbXlJrkzzU&zhida_source=entity)


**常见题目：**
- Leetcode 46. 全排列
- Leetcode 47. 全排列 II
- Leetcode 78. 子集
- Leetcode 39. 组合总和
- Leetcode 40. 组合总和 II
- Leetcode 77. 组合
- Leetcode 306. 累加数
- Leetcode 17. 电话号码的字母组合
# 动态规划
**基础知识：**
[动态规划](https://zhida.zhihu.com/search?content_id=239052399&content_type=Article&match_order=1&q=%E5%8A%A8%E6%80%81%E8%A7%84%E5%88%92&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NjA4NDQ4NjIsInEiOiLliqjmgIHop4TliJIiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyMzkwNTIzOTksImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.E97vjcLo3VDU7YjD49vP0foRQws_QOtZB2P6N5DMcng&zhida_source=entity)

**常见题目：**
- Leetcode 53. 最大子数组和
- Leetcode 5. 最长回文子串
- Leetcode 121. 买卖股票的最佳时机
- Leetcode 300. 最长递增子序列
- Leetcode 72. 编辑距离
- Leetcode 1143. 最长公共子序列
- Leetcode 322. 零钱兑换
- Leetcode 718. 最长重复子数组
- Leetcode 139. 单词拆分
- Leetcode 198. 打家劫舍
- Leetcode 518. 零钱兑换 II
- Leetcode 91. 解码方法
- Leetcode 213. 打家劫舍 II
- Leetcode 279. 完全平方数


---
# 前缀和（Prefix Sum）
**基础知识：**
前缀和本质上是在一个list当中，用O（N）的时间提前算好从第0个数字到第i个数字之和，在后续使用中可以在O（1）时间内计算出第i到第j个数字之和，一般很少单独作为一道题出现，而是很多题目中的用到的一个小技巧

**常见题目：**
- Leetcode 53 Maximum Subarray
- Leetcode 1423 Maximum Points You Can Obtain from Cards
- Leetcode 1031 Maximum Sum of Two Non-Overlapping Subarrays
- Leetcode 523 Continuous Subarray Sum
- Leetcode 304 Range Sum Query 2D - Immutable


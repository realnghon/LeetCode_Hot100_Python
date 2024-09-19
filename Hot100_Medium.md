
### [49. 字母异位词分组](https://leetcode.cn/problems/group-anagrams/description/?envType=study-plan-v2&envId=top-100-liked)

#### 题目描述
给你一个字符串数组，请你将 **字母异位词** 组合在一起。可以按任意顺序返回结果列表。

**字母异位词** 是由重新排列源单词的所有字母得到的一个新单词。

示例：

```
输入：strs = `["eat", "tea", "tan", "ate", "nat", "bat"]`
输出：[["bat"],["nat","tan"],["ate","eat","tea"]]
```

#### 核心思路
```
可以通过哈希表存储由相同字母组成的单词列表。

1. 创建一个空的哈希表。
2. 对于 `strs` 中的每一个单词：
   - 将其按字典序排序，得到 `key`。
   - 若 `key` 不在哈希表中，则将 `key` 加入哈希表，并初始化对应的值为包含该未排序单词的列表。
   - 若 `key` 已在哈希表中，将未排序单词添加到哈希表中 `key` 对应的列表中。

这样就能高效地将相同字母组成的单词分组。
```

#### 代码
```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
		# 创建一个defaultdict对象，默认值为空列表
        hashmap = defaultdict(list)
        
        for word in strs:
            # 对每个单词进行排序，作为哈希表的key
            sortedWord = "".join(sorted(word))
            # 将未排序的单词加入到对应的key的列表中
            hashmap[sortedWord].append(word)
        
        # 直接返回哈希表中所有value的列表
        return list(hashmap.values())
```


### [128. 最长连续序列](https://leetcode.cn/problems/longest-consecutive-sequence/description/?envType=study-plan-v2&envId=top-100-liked)

#### 题目描述
给定一个未排序的整数数组 `nums` ，找出数字连续的最长序列（不要求序列元素在原数组中连续）的长度。

请你设计并实现时间复杂度为 `O(n)` 的算法解决此问题。

示例：

```
输入：nums = [100,4,200,1,3,2]
输出：4
解释：最长数字连续序列是 [1, 2, 3, 4]。它的长度为 4。
```

#### 核心思路
```
这个问题要求时间复杂度为O(n)，可以使用哈希表（字典）来实现。

1. 创建一个哈希表：
    - 用来存储每个数对应的连续序列的长度。
2. 遍历数组： 
    - 对于每个数，如果它已经在哈希表中，直接跳过。
    - 如果是新数：
        - 检查这个数的左右相邻数是否在哈希表中，取出它们对应的连续序列长度 left 和 right。
        - 计算当前数的连续序列长度：cur_length = left + right + 1。
        - 更新最大长度 max_length。
        - 更新当前数及其连续序列两端点的长度为 cur_length。
3. 返回最大长度：
    - 在遍历结束后，即可得到最长的连续序列长度。
```
#### 代码

```python
class Solution:
    def longestConsecutive(self, nums):
        if not nums:
            # 如果输入列表为空，直接返回0
            return 0
        
        # 创建一个哈希表，用于记录每个数所在序列的长度
        num_dict = {}
        max_length = 0

        # 遍历数组中的每个数
        for num in nums:
            if num in num_dict:
                # 如果当前数已经在哈希表中，跳过（避免重复计算）
                continue
            
            # 获取当前数左边和右边相邻数的连续序列长度
            left = num_dict.get(num - 1, 0)
            right = num_dict.get(num + 1, 0)

            # 当前数所在连续序列的总长度
            cur_length = left + right + 1
            
            # 更新最大长度
            max_length = max(max_length, cur_length)
            
            # 更新当前数及其连续序列两端点的长度信息
            num_dict[num] = cur_length
            num_dict[num - left] = cur_length
            num_dict[num + right] = cur_length

        # 返回最长的连续序列长度
        return max_length
```

### [11. 盛最多水的容器](https://leetcode.cn/problems/container-with-most-water/description/?envType=study-plan-v2&envId=top-100-liked)

#### 题目描述
给定一个长度为 `n` 的整数数组 `height` 。有 `n` 条垂线，第 `i` 条线的两个端点是 `(i, 0)` 和 `(i, height[i])` 。

找出其中的两条线，使得它们与 `x` 轴共同构成的容器可以容纳最多的水。

返回容器可以储存的最大水量。

说明：你不能倾斜容器。

示例：

![](https://aliyun-lc-upload.oss-cn-hangzhou.aliyuncs.com/aliyun-lc-upload/uploads/2018/07/25/question_11.jpg)

```
输入：[1,8,6,2,5,4,8,3,7]
输出：49 
解释：图中垂直线代表输入数组 [1,8,6,2,5,4,8,3,7]。在此情况下，容器能够容纳水（表示为蓝色部分）的最大值为 49。
```
#### 核心思路
```
采用双指针的方法实现。

1. 初始化指针和变量：
    - start 指向 height 数组的最左端。
    - end 指向 height 数组的最右端。
    - max_volume 初始化为0。
2. 循环计算盛水容积，并更新指针：
    - 当 start < end 时，重复以下步骤：
        1. 计算当前容积： volume = min(height[start], height[end]) * (end - start)
        2. 更新最大容积： 若 volume > max_volume，则 max_volume = volume
        3. 移动较小值对应的指针：
            - 若 height[start] 小于 height[end]，则 start += 1
            - 否则，end -= 1
3. 循环结束时，max_volume 即为最大容积。
```

#### 代码
```python
class Solution:
    def maxArea(self, height: List[int]) -> int:
        start, end = 0, len(height) - 1
        max_volume = 0

        while start < end:
            volume = min(height[start], height[end]) * (end - start)
            max_volume = max(max_volume, volume)

            if height[start] < height[end]:
                start += 1
            else:
                end -= 1

        return max_volume
```

### [15. 三数之和](https://leetcode.cn/problems/3sum/description/?envType=study-plan-v2&envId=top-100-liked)
#### 题目描述

给你一个整数数组 `nums` ，判断是否存在三元组 `[nums[i], nums[j], nums[k]]` 满足 `i != j`、`i != k` 且 `j != k` ，同时还满足 `nums[i] + nums[j] + nums[k] == 0` 。请你返回所有和为 `0` 且不重复的三元组。

注意：答案中不可以包含重复的三元组。

示例：

```
输入：nums = [-1,0,1,2,-1,-4]
输出：[[-1,-1,2],[-1,0,1]]
解释：
nums[0] + nums[1] + nums[2] = (-1) + 0 + 1 = 0 。
nums[1] + nums[2] + nums[4] = 0 + 1 + (-1) = 0 。
nums[0] + nums[3] + nums[4] = (-1) + 2 + (-1) = 0 。
不同的三元组是 [-1,0,1] 和 [-1,-1,2] 。
注意，输出的顺序和三元组的顺序并不重要。
```

#### 核心思路
```
1. 特殊情况处理：如果数组为 `null` 或者数组长度小于 `3`，直接返回空结果。
2. 对数组进行排序。
3. 遍历排序后的数组：
   - 如果当前数 nums[i] 大于 0，因为数组已经排序，所以后面的数都大于 0，无法找到三个数的和为 0，因此直接返回空列表。
   - 如果当前数与前一个数相同，跳过以避免重复解。
   - 设定两个指针，左指针 L 初始化为 i+1，右指针 R 初始化为数组末尾。
   - 在 L < R 的情况下，计算 nums[i] + nums[L] + nums[R] 的和：
     - 如果和为 0，记录这个三元组，并移动 L 和 R 指针，跳过重复元素。
     - 如果和大于 0，说明 nums[R] 太大，右指针左移。
     - 如果和小于 0，说明 nums[L] 太小，左指针右移。

复杂度分析：
- 时间复杂度：排序需要 O(nlog n)，遍历数组和双指针查找的时间复杂度为 O(n^2)，总体复杂度为 O(n^2)
- 空间复杂度：O(1)
```
#### 代码
```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        if len(nums) < 3: return []
        nums.sort()
        result = []
        
        for i in range(len(nums)):
            if nums[i] > 0: break
            if i > 0 and nums[i] == nums[i - 1]: continue
            
            L, R = i + 1, len(nums) - 1
            while L < R:
                total = nums[i] + nums[L] + nums[R]
                if total == 0:
                    result.append([nums[i], nums[L], nums[R]])
                    while L < R and nums[L] == nums[L + 1]:
                        L += 1
                    while L < R and nums[R] == nums[R - 1]:
                        R -= 1
                    L += 1
                    R -= 1
                elif total < 0:
                    L += 1
                else:
                    R -= 1
        
        return result
```

### [3. 无重复字符的最长子串](https://leetcode.cn/problems/longest-substring-without-repeating-characters/description/?envType=study-plan-v2&envId=top-100-liked)

#### 题目描述
给定一个字符串 `s` ，请你找出其中不含有重复字符的 **最长子串** 的长度。

示例：
```
输入: s = "abcabcbb"
输出: 3 
解释: 因为无重复字符的最长子串是 `"abc"`，所以其长度为 3。
```

#### 核心思路
```
目标是找到字符串中最长的无重复字符的子串。

1. 初始化：
    - 我们定义两个指针 start 和 end，用于表示滑动窗口的起始和结束位置。
    - 一个哈希集合（HashSet）用来存储当前窗口内的字符，从而帮助我们检测是否有重复字符。
2. 滑动窗口的操作： 
    - 扩展窗口：将 end 指针向右移动，即增大窗口的范围。
    - 检查重复：每次移动 end 后，我们检查新加入窗口的字符 s[end]。
        - 如果 s[end] 已经存在于集合中，说明出现了重复字符，此时需要缩小窗口。
        - 为了缩小窗口，我们移动 start 指针，直到没有重复字符为止。在移动 start 的过程中，需要将移出窗口的字符从集合中删除。
3. 更新最大长度：
    - 在窗口内没有重复字符时，计算当前窗口的长度（end - start + 1），并更新记录的最大长度。
4. 继续上述过程，直到 end 指针遍历完整个字符串。

复杂度分析：
	- 时间复杂度为 O(n)，因为每个字符在最坏情况下只会被访问两次（一次被加入集合，一次被移出集合）。
```
#### 代码
```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        if not s:
            return 0
        
        # 初始化指针和结果变量
        start, max_length = 0, 0
        char_index_map = {}  # 用于存储字符及其最后出现的位置

        for end in range(len(s)):
            if s[end] in char_index_map and char_index_map[s[end]] >= start:
                # 如果当前字符已经存在于字典中且其索引在start之后或等于start，说明有重复字符，
                # 需要移动start指针，以排除重复字符
                start = char_index_map[s[end]] + 1
            
            # 更新字符的最新索引位置
            char_index_map[s[end]] = end
            
            # 更新最长无重复子串的长度
            max_length = max(max_length, end - start + 1)

        return max_length
```

### [438. 找到字符串中所有字母异位词](https://leetcode.cn/problems/find-all-anagrams-in-a-string/description/?envType=study-plan-v2&envId=top-100-liked)

#### 题目描述
给定两个字符串 `s` 和 `p`，找到 `s` 中所有 `p` 的 **异位词** 的子串，返回这些子串的起始索引。不考虑答案输出的顺序。

**异位词** 指由相同字母重排列形成的字符串（包括相同的字符串）。

示例：

```
输入：s = "cbaebabacd", p = "abc"
输出：[0,6]
解释：
起始索引等于 0 的子串是 "cba", 它是 "abc" 的异位词。
起始索引等于 6 的子串是 "bac", 它是 "abc" 的异位词。
```

#### 核心思路
```
【方法一：滑动窗口 + 计数器】
1. 初始化计数器： 初始化 p_count 和 s_count 各需要 O(n) 时间，其中 n 是字符串 p 的长度。
2. 滑动窗口过程：
	- 每次移动窗口时，添加新字符和移除老字符的操作都是 O(1) 时间复杂度。
	- 比较两个字典是否相等（s_count == p_count）的时间复杂度是 O(1)，因为字典的键数固定为 26（即字符集大小）。

因此，这种方法的总时间复杂度为 O(m + (m-n+1) * 1) = O(m) ，其中 m 是字符串 s 的长度。

【方法二：滑动窗口 + 排序】
1. 初始化排序： 对字符串 p 进行排序需要 O(n log n) 时间。
2. 滑动窗口过程：
	- 每次移动窗口时，对滑动窗口内的字符串进行排序，时间复杂度是 O(n log n)，其中 n 是窗口大小（即 p 的长度）。
	- 比较两个排序后的字符串的时间复杂度是 O(n)。

因此，这种方法的总时间复杂度为 O((m-n+1) * (n log n + n)) = O((m-n+1) * n log n)，其中 m 是字符串 s 的长度。
```

#### 代码
【方法一：滑动窗口 + 计数器】
```python
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        # 初始化 p 的字符计数器和滑动窗口的字符计数器
        p_count = Counter(p)
        s_count = Counter(s[:len(p)-1])
        
        result = []
        # 遍历字符串 s，从索引 len(p)-1 到 len(s)-1
        for i in range(len(p)-1, len(s)):
            # 将新的字符加入当前滑动窗口的计数器中
            s_count[s[i]] += 1
            
            # 如果当前窗口的字符计数器与 p 的字符计数器相同，则记录起始索引
            if s_count == p_count:
                result.append(i - len(p) + 1)
            
            # 移除当前窗口左侧即将滑出字符的计数
            s_count[s[i - len(p) + 1]] -= 1
            # 如果某个字符的计数变为零，将其从计数器中删除
            if s_count[s[i - len(p) + 1]] == 0:
                del s_count[s[i - len(p) + 1]]
        
        return result
```

### [560. 和为 K 的子数组](https://leetcode.cn/problems/subarray-sum-equals-k/description/?envType=study-plan-v2&envId=top-100-liked)
#### 题目描述
给你一个整数数组 `nums` 和一个整数 `k` ，请你统计并返回该数组中和为 `k` 的子数组的个数。

子数组是数组中元素的连续非空序列。

示例：

```
输入：nums = [1,1,1], k = 2
输出：2
```
#### 核心思路
```
首先，我们了解问题要求是找到和为 k 的子数组。这意味着我们需要计算很多子数组的和。如果使用暴力方法逐个计算所有可能的子数组和，将会导致时间复杂度非常高，达到 O(n^2) 或更高。在大型数据集上，这种方法效率极低。

1. 前缀和（Prefix Sum）
为了解决上述问题，可以引入前缀和的概念。前缀和是从数组的起点到当前位置的元素总和。有了前缀和，任意子数组的和可以通过两个前缀和之差快速计算出来：如果我们知道从起点到第 j 个位置的前缀和 prefix_sum_j 和从起点到第 i 个位置的前缀和 prefix_sum_i，那么从 i+1 到 j 的子数组的和为 prefix_sum_j - prefix_sum_i。

2. 哈希表优化查找
为了快速查找某个前缀和是否存在，以及其出现次数，我们可以使用哈希表。当遍历数组时，我们记录每个前缀和及其出现的次数。这样我们就能快速判断之前是否有某个前缀和使得当前前缀和减去它等于 k。

解题步骤的推导
1. 初始化：
	- prefix_sum = 0: 表示初始的前缀和。
	- prefix_sum_count = {0: 1}: 初始化哈希表，表示前缀和为0的情况出现一次。这一步很重要，它处理了当从数组开头到某个位置子数组和恰好为 k 的情况。

2. 遍历数组：
	- 对每个元素，更新当前前缀和 prefix_sum。
	- 检查 prefix_sum - k 是否在 prefix_sum_count 中：如果在，说明存在之前的一个前缀和，使得这段区间的和为 k，于是增加计数器。
	- 将当前前缀和加入或更新到 prefix_sum_count 中。
```

#### 代码
```python
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        prefix_sum = 0  # 初始化前缀和
        prefix_sum_count = {0: 1}  # 初始化哈希表，包含前缀和为0的情况
        count = 0  # 初始化计数器
        
        for num in nums:
            prefix_sum += num  # 更新当前前缀和
            count += prefix_sum_count.get(prefix_sum - k, 0)  # 如果存在符合条件的前缀和，则增加计数
            prefix_sum_count[prefix_sum] = prefix_sum_count.get(prefix_sum, 0) + 1  # 更新当前前缀和出现次数
                
        return count  # 返回计数结果
```

#### 题目描述

#### 核心思路

#### 代码
#### 题目描述

#### 核心思路

#### 代码
#### 题目描述

#### 核心思路

#### 代码
#### 题目描述

#### 核心思路

#### 代码
#### 题目描述

#### 核心思路

#### 代码
#### 题目描述

#### 核心思路

#### 代码
#### 题目描述

#### 核心思路

#### 代码
#### 题目描述

#### 核心思路

#### 代码
#### 题目描述

#### 核心思路

#### 代码
#### 题目描述

#### 核心思路

#### 代码
#### 题目描述

#### 核心思路

#### 代码
#### 题目描述

#### 核心思路

#### 代码
#### 题目描述

#### 核心思路

#### 代码
#### 题目描述

#### 核心思路

#### 代码
#### 题目描述

#### 核心思路

#### 代码
#### 题目描述

#### 核心思路

#### 代码
#### 题目描述

#### 核心思路

#### 代码
#### 题目描述

#### 核心思路

#### 代码
#### 题目描述

#### 核心思路

#### 代码
#### 题目描述

#### 核心思路

#### 代码
#### 题目描述

#### 核心思路

#### 代码
#### 题目描述

#### 核心思路

#### 代码
#### 题目描述

#### 核心思路

#### 代码
#### 题目描述

#### 核心思路

#### 代码
#### 题目描述

#### 核心思路

#### 代码
#### 题目描述

#### 核心思路

#### 代码
#### 题目描述

#### 核心思路

#### 代码
#### 题目描述

#### 核心思路

#### 代码
#### 题目描述

#### 核心思路

#### 代码
#### 题目描述

#### 核心思路

#### 代码
#### 题目描述

#### 核心思路

#### 代码
#### 题目描述

#### 核心思路

#### 代码
#### 题目描述

#### 核心思路

#### 代码
#### 题目描述

#### 核心思路

#### 代码
#### 题目描述

#### 核心思路

#### 代码
#### 题目描述

#### 核心思路

#### 代码
#### 题目描述

#### 核心思路

#### 代码
#### 题目描述

#### 核心思路

#### 代码
#### 题目描述

#### 核心思路

#### 代码
#### 题目描述

#### 核心思路

#### 代码
#### 题目描述

#### 核心思路

#### 代码
#### 题目描述

#### 核心思路

#### 代码
#### 题目描述

#### 核心思路

#### 代码
#### 题目描述

#### 核心思路

#### 代码
#### 题目描述

#### 核心思路

#### 代码
#### 题目描述

#### 核心思路

#### 代码
#### 题目描述

#### 核心思路

#### 代码
#### 题目描述

#### 核心思路

#### 代码
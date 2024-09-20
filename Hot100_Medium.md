
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

### [53. 最大子数组和](https://leetcode.cn/problems/maximum-subarray/description/?envType=study-plan-v2&envId=top-100-liked)
#### 题目描述
给你一个整数数组 `nums` ，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

**子数组**是数组中的一个连续部分。

示例：

```
输入：nums = [-2,1,-3,4,-1,2,1,-5,4]
输出：6
解释：连续子数组 [4,-1,2,1] 的和最大，为 6 。
```
#### 核心思路
```
动态规划思路（Kadane算法）

Kadane算法是一种线性时间复杂度O(n)的动态规划算法，用来解决最大子数组和问题。该算法的核心在于通过迭代数组，计算以每个位置结尾的子数组的最大和，并实时更新全局最大和。

1. 初始条件：
    - 使用一个变量 current_sum 来表示当前子数组的和。
    - 使用另一个变量 max_sum 来记录找到的最大和。
    - 初始化 current_sum 和 max_sum 为数组的第一个元素，因为单独一个元素也是一个子数组。
2. 迭代数组： 
    - 从第二个元素开始遍历数组。
    - 对于每个元素 nums[i]，current_sum = max(nums[i], current_sum + nums[i])。
        - 如果 current_sum + nums[i] 比 nums[i] 大，则意味着延续前面的子数组是有益的。
        - 否则，从当前元素重新开始一个新的子数组。
    - 更新全局最大和：max_sum = max(max_sum, current_sum)，这样可以确保 max_sum 永远是我们迄今为止找到的最大子数组和。
3. 返回结果：
    - 最后，max_sum 就是所求的具有最大和的连续子数组的和。
```
#### 代码
```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        # 初始化当前子数组和和最大子数组和为第一个元素
        current_sum = max_sum = nums[0]
        
        # 从第二个元素开始遍历数组
        for num in nums[1:]:
            # 当前子数组和的计算
            current_sum = max(num, current_sum + num)
            # 更新全局最大和
            max_sum = max(max_sum, current_sum)
        
        return max_sum
```

### [56. 合并区间](https://leetcode.cn/problems/merge-intervals/description/?envType=study-plan-v2&envId=top-100-liked)
#### 题目描述
以数组 `intervals` 表示若干个区间的集合，其中单个区间为 `intervals[i] = [starti, endi]` 。请你合并所有重叠的区间，并返回一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间。

示例：

```
输入：intervals = [[1,3],[2,6],[8,10],[15,18]]
输出：[[1,6],[8,10],[15,18]]
解释：区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].
```

#### 核心思路
```
1. 排序：首先按照每个区间的起始值对intervals进行排序。
2. 初始化结果列表：创建一个空列表merged用于存储最终的合并结果。
3. 遍历区间：
    - 如果merged为空或者当前区间与merged的最后一个区间不重叠，将当前区间加入merged。
    - 否则，合并当前区间与merged的最后一个区间。
```
#### 代码
```python
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        # 按起始值排序区间
        intervals.sort(key=lambda x: x[0])
        merged_intervals = []
        
        for current in intervals:
            if not merged_intervals or merged_intervals[-1][1] < current[0]:
                merged_intervals.append(current)
            else:
                merged_intervals[-1][1] = max(merged_intervals[-1][1], current[1])
        
        return merged_intervals
```

### [189. 轮转数组](https://leetcode.cn/problems/rotate-array/?envType=study-plan-v2&envId=top-100-liked)
#### 题目描述

给定一个整数数组 `nums`，将数组中的元素向右轮转 `k` 个位置，其中 `k` 是非负数。

示例：

```
输入：nums = [1,2,3,4,5,6,7], k = 3
输出：[5,6,7,1,2,3,4]
解释：
向右轮转 1 步: [7,1,2,3,4,5,6]
向右轮转 2 步: [6,7,1,2,3,4,5]
向右轮转 3 步: [5,6,7,1,2,3,4]
```

#### 核心思路
```
【方法一】

1. 计算有效的旋转步数： 通过 k = k % n，确保旋转步数不超过数组长度。
2. 切片操作重新排列数组：
    - 使用 nums[-k:] 获取数组末尾 k 个元素。
    - 使用 nums[:-k] 获取数组前面部分。
    - 将这两个部分拼接起来形成新的数组顺序，并赋值给原数组 nums[:]。

- 时间复杂度：(O(n))。切片操作需要遍历整个数组，因此时间复杂度为 (O(n))。
- 空间复杂度：(O(n))。虽然没有显式地使用额外的数据结构，但切片操作会产生临时数组，占用额外的空间。

【方法二】

1. 整体翻转数组：将整个数组完全反转。
2. 翻转前 k 个元素：将前 k 个元素再次反转。
3. 翻转后 n-k 个元素：将后 n-k 个元素再次反转。

- 时间复杂度：(O(n))。每次翻转操作需要遍历部分或者整个数组，总共三次翻转，因此时间复杂度为 (O(n))。
- 空间复杂度：(O(1))。只使用了常数个额外空间用于变量存储，没有使用额外的数据结构。
```

#### 代码
【方法一】
```python
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        n = len(nums)
        k = k % n  # 防止 k 超过数组长度
        nums[:] = nums[-k:] + nums[:-k]
```

【方法二】
```python
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        n = len(nums)
        k = k % n  # 防止 k 超过数组长度
        
        def reverse(start: int, end: int) -> None:
            while start < end:
                nums[start], nums[end] = nums[end], nums[start]
                start += 1
                end -= 1

        reverse(0, n - 1)
        reverse(0, k - 1)
        reverse(k, n - 1)
```

### [238. 除自身以外数组的乘积](https://leetcode.cn/problems/product-of-array-except-self/description/?envType=study-plan-v2&envId=top-100-liked)
#### 题目描述

给你一个整数数组 `nums`，返回 数组 `answer` ，其中 `answer[i]` 等于 `nums` 中除 `nums[i]` 之外其余各元素的乘积 。

题目数据 **保证** 数组 `nums`之中任意元素的全部前缀元素和后缀的乘积都在  **32 位** 整数范围内。

请不要使用除法，且在 `O(n)` 时间复杂度内完成此题。

示例：

```
输入：nums = [1,2,3,4]
输出：[24,12,8,6]
```

#### 核心思路
```
我们需要找到一个不包含nums[i]的数组乘积，并且要求时间复杂度为O(n)且不使用除法。可以通过前缀积和后缀积的方法来实现。

1. 计算前缀积：创建一个与输入数组等长的结果数组answer。首先将answer初始化为全1，然后遍历一遍数组，将每个位置的值替换为该位置之前所有元素的乘积。
2. 计算后缀积：在同一个结果数组中从后向前遍历，同时维护一个变量suffix_product表示从当前元素到最后一个元素的乘积。将answer中的每个元素与suffix_product相乘，然后更新suffix_product。

通过上述方法，我们可以在一次遍历中完成前缀积的计算，另一遍遍历中完成后缀积的计算，从而满足O(n)的时间复杂度要求。同时，借助结果数组本身来存储计算结果，实现了O(1)的空间复杂度（不包括输出数组）。
```
#### 代码
```python
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        n = len(nums)
        answer = [1] * n  # 初始化结果数组，全1
        # 计算前缀积并存储在 answer 中
        for i in range(1, n):
            answer[i] = answer[i - 1] * nums[i - 1]
            
        suffix_product = 1  # 初始化后缀积为1
        # 计算后缀积，并直接更新 answer
        for i in range(n - 1, -1, -1):  # 从后往前遍历数组
            answer[i] *= suffix_product  # 将当前后缀积乘以 answer 中对应位置的值
            suffix_product *= nums[i]  # 更新后缀积
        
        return answer
```

### [73. 矩阵置零](https://leetcode.cn/problems/set-matrix-zeroes/description/?envType=study-plan-v2&envId=top-100-liked)

#### 题目描述
给定一个 m×n 的矩阵，如果一个元素为 **0** ，则将其所在行和列的所有元素都设为 **0** 。请使用原地算法。

示例：

![](https://assets.leetcode.com/uploads/2020/08/17/mat1.jpg)

```
输入：matrix = [[1,1,1],[1,0,1],[1,1,1]]
输出：[[1,0,1],[0,0,0],[1,0,1]]
```
#### 核心思路
```
要解决这个问题，可以使用矩阵的第一行和第一列作为标记，来记录哪些行和哪些列需要被设为0。这样可以避免使用额外的空间。

1. 检查第一行和第一列是否有0：首先检查矩阵的第一行和第一列中是否有0，因为这两个位置会被用来标记其他行和列。
    
2. 使用第一行和第一列作为标记：遍历矩阵的其余部分（从第二行和第二列开始），如果某个元素是0，就将该元素所在的行的第一个元素和列的第一个元素设为0，作为标记。
    
3. 根据标记设置0：再次遍历矩阵（从第二行和第二列开始），如果某个元素所在的行的第一个元素或列的第一个元素是0，就将该元素设为0。
    
4. 处理第一行和第一列：最后，根据第一步中记录的信息，决定是否将第一行和第一列全部设为0。
```

#### 代码
```python
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        m, n = len(matrix), len(matrix[0])
        
        # 检查第一行是否有0
        first_row_has_zero = any(matrix[0][j] == 0 for j in range(n))
        # 检查第一列是否有0
        first_col_has_zero = any(matrix[i][0] == 0 for i in range(m))
        
        # 使用第一行和第一列作为标记
        for i in range(1, m):
            for j in range(1, n):
                if matrix[i][j] == 0:
                    matrix[i][0] = 0  # 标记第i行需要变为0
                    matrix[0][j] = 0  # 标记第j列需要变为0
        
        # 根据标记设置0
        for i in range(1, m):
            for j in range(1, n):
                if matrix[i][0] == 0 or matrix[0][j] == 0:
                    matrix[i][j] = 0
        
        # 处理第一行，如果第一行原来存在0，则将第一行全部变为0
        if first_row_has_zero:
            for j in range(n):
                matrix[0][j] = 0
        
        # 处理第一列，如果第一列原来存在0，则将第一列全部变为0
        if first_col_has_zero:
            for i in range(m):
                matrix[i][0] = 0
```

### [54. 螺旋矩阵](https://leetcode.cn/problems/spiral-matrix/description/?envType=study-plan-v2&envId=top-100-liked)
#### 题目描述
给你一个 `m` 行 `n` 列的矩阵 `matrix` ，请按照 **顺时针螺旋顺序** ，返回矩阵中的所有元素。

示例：

![](https://assets.leetcode.com/uploads/2020/11/13/spiral1.jpg)

```
输入：matrix = [[1,2,3],[4,5,6],[7,8,9]]
输出：[1,2,3,6,9,8,7,4,5]
```
#### 核心思路
```
这道题可以通过模拟螺旋遍历的方式来解决。核心思想是按照顺时针方向逐步缩小遍历的范围，直到遍历完整个矩阵。

1. 初始化边界：定义四个边界变量 top, bottom, left, right，分别表示当前未遍历的矩阵的上、下、左、右边界。

2. 循环遍历：
	- 从左到右：遍历 left 到 right 的顶部行，然后将 top 下移一行。
	- 从上到下：遍历 top 到 bottom 的右侧列，然后将 right 左移一列。
	- 从右到左：遍历 right 到 left 的底部行，然后将 bottom 上移一行。
	- 从下到上：遍历 bottom 到 top 的左侧列，然后将 left 右移一列。

3. 终止条件：当 top 超过 bottom 或 left 超过 right 时，遍历结束。

通过这种方式，可以确保每个元素只被访问一次，并且按照顺时针螺旋顺序返回所有元素。
```
#### 代码
```python
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        if not matrix or not matrix[0]:
            return []
        
        m, n = len(matrix), len(matrix[0])
        top, bottom, left, right = 0, m - 1, 0, n - 1
        result = []
        
        while top <= bottom and left <= right:
            # 从左到右遍历顶部行
            for i in range(left, right + 1):
                result.append(matrix[top][i])
            top += 1  # 顶部行已遍历，下移一行
            
            # 从上到下遍历右侧列
            for i in range(top, bottom + 1):
                result.append(matrix[i][right])
            right -= 1  # 右侧列已遍历，左移一列
            
            # 检查是否还有剩余的行和列
            if top <= bottom:
                # 从右到左遍历底部行
                for i in range(right, left - 1, -1):
                    result.append(matrix[bottom][i])
                bottom -= 1  # 底部行已遍历，上移一行
            
            if left <= right:
                # 从下到上遍历左侧列
                for i in range(bottom, top - 1, -1):
                    result.append(matrix[i][left])
                left += 1  # 左侧列已遍历，右移一列
        
        return result
```

### [48. 旋转图像](https://leetcode.cn/problems/rotate-image/description/?envType=study-plan-v2&envId=top-100-liked)
#### 题目描述
给定一个 _n_ × _n_ 的二维矩阵 `matrix` 表示一个图像。请你将图像顺时针旋转 90 度。

你必须在 **原地** 旋转图像，这意味着你需要直接修改输入的二维矩阵。**请不要** 使用另一个矩阵来旋转图像。

示例：

![](https://assets.leetcode.com/uploads/2020/08/28/mat1.jpg)

```
输入：matrix = [[1,2,3],[4,5,6],[7,8,9]]
输出：[[7,4,1],[8,5,2],[9,6,3]]
```
#### 核心思路
```
1. 确定层数：
    - 对于一个 n × n 的矩阵，旋转层数为 n // 2

2. 逐层旋转：
    - 每次选择一个层，从 top 行开始，按顺时针方向交换四个角上的元素
3. 具体元素交换：  
    对于每个需要旋转的层和边界：
    - matrix[top][i] 表示当前层的上边元素，存储到临时变量 temp 中。
    - 左边元素 matrix[bottom - offset][top] 移到上边。
    - 下边元素 matrix[bottom][bottom - offset] 移到左边。
    - 右边元素 matrix[i][bottom] 移到下边。
    - 最后将存储的 temp 移到右边。
```
#### 代码
```python
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        n = len(matrix)
        
        # 逐层旋转
        for layer in range(n // 2):
            # 当前层的边界
            top, bottom = layer, n - 1 - layer
            
            for i in range(top, bottom):
                # 计算偏移量
                offset = i - top
                
                # 顺时针旋转四个位置
                
                # 保存上边元素
                temp = matrix[top][i]
                # 左 -> 上
                matrix[top][i] = matrix[bottom - offset][top]
                # 下 -> 左
                matrix[bottom - offset][top] = matrix[bottom][bottom - offset]
                # 右 -> 下
                matrix[bottom][bottom - offset] = matrix[i][bottom]
                # 上 -> 右
                matrix[i][bottom] = temp

```

### [240. 搜索二维矩阵 II](https://leetcode.cn/problems/search-a-2d-matrix-ii/description/?envType=study-plan-v2&envId=top-100-liked)
#### 题目描述
编写一个高效的算法来搜索 `m × n` 矩阵 `matrix` 中的一个目标值 `target` 。该矩阵具有以下特性：

- 每行的元素从左到右升序排列。
- 每列的元素从上到下升序排列。

示例：

![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2020/11/25/searchgrid2.jpg)

```
输入：matrix = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], target = 5
输出：true
```
#### 核心思路
```
从矩阵的右上角开始搜索，这样每一步都可以通过比较大小来确定下一步的方向。详细步骤如下：

1. 从右上角开始：我们选择从矩阵的右上角（即第一行的最后一个元素）开始。这个位置有一个特殊的性质：
    - 如果当前元素比目标值 target 大，我们可以往左移动，因为同一行左边的元素更小。
    - 如果当前元素比目标值 target 小，我们可以向下移动，因为同一列下面的元素更大。

2. 不断缩小搜索空间：根据上面的逻辑，我们每次都可以抛弃一整行或一整列，因此每次比较都有效地缩小了搜索范围。
    
3. 终止条件：如果找到了目标值，返回 True。如果搜索越界（即行列索引超出矩阵范围），则返回 False，表示未找到目标值。
```
#### 代码
```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        # 获取矩阵的行数和列数
        if not matrix or not matrix[0]:
            return False
        rows, cols = len(matrix), len(matrix[0])
        
        # 从右上角开始
        row, col = 0, cols - 1
        
        while row < rows and col >= 0:
            if matrix[row][col] == target:
                return True
            elif matrix[row][col] > target:
                # 如果当前元素比目标大，向左移动
                col -= 1
            else:
                # 如果当前元素比目标小，向下移动
                row += 1
        
        return False
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
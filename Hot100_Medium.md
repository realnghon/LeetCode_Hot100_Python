
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

### [142. 环形链表 II](https://leetcode.cn/problems/linked-list-cycle-ii/description/?envType=study-plan-v2&envId=top-100-liked)
#### 题目描述
给定一个链表的头节点  `head` ，返回链表开始入环的第一个节点。 如果链表无环，则返回 `null`。

如果链表中有某个节点，可以通过连续跟踪 `next` 指针再次到达，则链表中存在环。 为了表示给定链表中的环，评测系统内部使用整数 `pos` 来表示链表尾连接到链表中的位置（**索引从 0 开始**）。如果 `pos` 是 `-1`，则在该链表中没有环。**注意：`pos` 不作为参数进行传递**，仅仅是为了标识链表的实际情况。

**不允许修改** 链表。

示例：

![](https://assets.leetcode.com/uploads/2018/12/07/circularlinkedlist.png)
```
输入：head = [3,2,0,-4], pos = 1
输出：返回索引为 1 的链表节点
解释：链表中有一个环，其尾部连接到第二个节点。
```
#### 核心思路
```
快慢指针用来检测链表中是否存在环。一旦快慢指针相遇，说明链表中存在环。接下来的关键是如何找到环的起始节点。

1. 检测环：
    - 使用两个指针 slow 和 fast，slow 每次移动一步，fast 每次移动两步。
    - 如果 fast 和 slow 相遇，则说明链表中存在环。

2. 找到环的起始节点： 
    - 当 fast 和 slow 相遇时，将其中一个指针（例如 slow）重置到链表的头部。
    - 然后，两个指针都每次移动一步，直到它们再次相遇。这个相遇点就是环的起始节点。

原理：

假设链表的头部到环的起始节点的距离为 a，环的长度为 b。当 slow 和 fast 相遇时，slow 走了 a + b * k 步，fast 走了 a + b * m 步，其中 m > k。因为 fast 的速度是 slow 的两倍，所以有： [ 2(a + b * k) = a + b * m ] 简化后得到： [ a = (m - 2k) * b ] 这说明 a 是环长度 b 的整数倍。因此，当 slow 从头开始，fast 从相遇点开始，每次移动一步时，它们会在环的起始节点相遇。
```
#### 代码
```python
class Solution:
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head or not head.next:
            return None

        slow = head
        fast = head

        # 检测是否有环
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                break
        else:
            return None

        # 找到环的起始节点
        slow = head
        while slow != fast:
            slow = slow.next
            fast = fast.next

        return slow
```
### [2. 两数相加](https://leetcode.cn/problems/add-two-numbers/description/?envType=study-plan-v2&envId=top-100-liked)
#### 题目描述
给你两个 **非空** 的链表，表示两个非负的整数。它们每位数字都是按照 **逆序** 的方式存储的，并且每个节点只能存储 **一位** 数字。

请你将两个数相加，并以相同形式返回一个表示和的链表。

你可以假设除了数字 0 之外，这两个数都不会以 0 开头。

示例：

![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2021/01/02/addtwonumber1.jpg)

```
输入：l1 = [2,4,3], l2 = [5,6,4]
输出：[7,0,8]
解释：342 + 465 = 807
```
#### 核心思路
```
1. 初始化结果链表和进位变量：
	- 创建一个虚拟头节点 dummy，用于简化边界条件处理。
	- 初始化进位变量 carry 为 0。

2. 遍历两个链表：
	- 使用两个指针 p1 和 p2 分别指向链表 L1 和 L2 的头节点。
	- 创建一个指针 current 指向 dummy，用于构建结果链表。

3. 逐位相加：
	- 在 p1 或 p2 不为空时，执行以下操作：
		- 计算当前位的和 sum = (p1.val if p1 else 0) + (p2.val if p2 else 0) + carry。
		- 更新进位 carry = sum // 10。
		- 创建一个新节点，节点值为 sum % 10，并将其连接到结果链表的末尾。
		- 移动 current 指针到新节点。
		- 如果 p1 不为空，移动 p1 指针到下一个节点。
		- 如果 p2 不为空，移动 p2 指针到下一个节点。

4. 处理剩余的进位：
	- 如果遍历完两个链表后，carry 仍不为 0，则在结果链表的末尾新增一个节点，节点值为 carry。

5. 返回结果链表：
	- 返回 dummy.next，即结果链表的头节点。
```
#### 代码
```python
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        # 创建虚拟头节点，用于简化链表操作
        dummy = ListNode(0)
        current = dummy
        carry = 0

        # 遍历两个链表，直到两者都为空
        while l1 or l2:
            # 计算当前位的和，考虑进位
            sum_val = (l1.val if l1 else 0) + (l2.val if l2 else 0) + carry
            carry = sum_val // 10  # 更新进位
            current.next = ListNode(sum_val % 10)  # 创建新节点，存储当前位的值
            current = current.next  # 移动 current 指针到新节点

            # 移动 l1 和 l2 指针到下一个节点
            if l1:
                l1 = l1.next
            if l2:
                l2 = l2.next

        # 如果最后有进位，需要在结果链表的末尾新增一个节点
        if carry > 0:
            current.next = ListNode(carry)

        # 返回结果链表的头节点，即虚拟头节点的下一个节点
        return dummy.next
```
### [19. 删除链表的倒数第 N 个结点](https://leetcode.cn/problems/remove-nth-node-from-end-of-list/description/?envType=study-plan-v2&envId=top-100-liked)
#### 题目描述
给你一个链表，删除链表的倒数第 `n` 个结点，并且返回链表的头结点。

示例：

![](https://assets.leetcode.com/uploads/2020/10/03/remove_ex1.jpg)

```
输入：head = [1,2,3,4,5], n = 2
输出：[1,2,3,5]
```
#### 核心思路
```
使用快慢指针解决删除链表的倒数第 n 个结点的问题是一个非常高效的方法。这里的关键在于如何让两个指针在链表中保持一定的距离，从而当一个指针到达链表末尾时，另一个指针正好指向需要删除的结点的前一个结点。

1. 初始化指针： 
    - 创建一个虚拟头结点 dummy，它的 next 指向原链表的头结点 head。这样做可以简化边界条件的处理，尤其是当需要删除的结点是头结点时。
    - 初始化两个指针 fast 和 slow，都指向 dummy。

2. 移动快指针：
    - 先移动 fast 指针 n 步。这样 fast 和 slow 之间的距离就是 n。

3. 同时移动两个指针：
    - 当 fast 指针到达链表末尾时（即 fast.next 为 None），slow 指针正好指向需要删除的结点的前一个结点。
    - 这是因为 fast 和 slow 之间的距离始终保持为 n，所以当 fast 到达末尾时，slow 就在倒数第 n 个结点的前一个位置。

4. 删除结点：
    - 修改 slow 的 next 指针，使其跳过需要删除的结点，即 slow.next = slow.next.next。

5. 返回结果：
    - 返回 dummy.next，即新的链表头结点。
```
#### 代码
```python
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        # 创建虚拟头结点，简化边界条件处理
        dummy = ListNode(0, head)
        
        # 初始化快慢指针
        fast = slow = dummy
        
        # 移动快指针 n 步
        for _ in range(n):
            fast = fast.next
        
        # 同时移动两个指针，直到快指针到达链表末尾
        while fast.next:
            fast = fast.next
            slow = slow.next
        
        # 删除结点
        slow.next = slow.next.next
        
        # 返回新的头结点
        return dummy.next
```
### [24. 两两交换链表中的节点](https://leetcode.cn/problems/swap-nodes-in-pairs/description/?envType=study-plan-v2&envId=top-100-liked)
#### 题目描述
给你一个链表，两两交换其中相邻的节点，并返回交换后链表的头节点。你必须在不修改节点内部的值的情况下完成本题（即，只能进行节点交换）。

示例：

![](https://assets.leetcode.com/uploads/2020/10/03/swap_ex1.jpg)

```
输入：head = [1,2,3,4]
输出：[2,1,4,3]
```
#### 核心思路
```
解决这个问题的一个有效方法是使用迭代，通过维护几个指针来交换链表中的节点。

1. 创建一个虚拟头节点
	- 为了简化边界情况的处理，可以在原链表的头部添加一个虚拟头节点（dummy node），这样可以方便地处理原链表头部的节点交换。

2. 初始化指针
	- prev 指向虚拟头节点，用于记录当前需要交换的两个节点的前一个节点。
	- current 指向链表的头节点，即需要交换的第一个节点。
	- next 指向当前节点的下一个节点，即需要交换的第二个节点。

3. 迭代交换节点
	在循环中，每次交换两个相邻的节点，并更新指针：
	- 将 prev 的 next 指向 next（即第二个节点）。
	- 将 current 的 next 指向 next.next（即第二个节点的下一个节点）。
	- 将 next 的 next 指向 current（即将第二个节点的 next 指向第一个节点）。
	- 更新 prev 和 current 指针，以便处理下一对节点。

4. 返回新的头节点
	- 当所有节点都交换完成后，返回虚拟头节点的 next，即新的链表头节点。
```
#### 代码
```python
class Solution:
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # 创建虚拟头节点，方便处理链表头部的交换
        dummy = ListNode(0)
        dummy.next = head
        prev = dummy
        
        while head and head.next:
            # 定义需要交换的两个节点
            first = head
            second = head.next
            
            # 交换节点
            # 1. 将 prev 的 next 指向 second
            prev.next = second
            # 2. 将 first 的 next 指向 second 的 next
            first.next = second.next
            # 3. 将 second 的 next 指向 first
            second.next = first
            
            # 更新指针，准备处理下一对节点
            prev = first
            head = first.next
        
        # 返回新的链表头节点
        return dummy.next
```
### [138. 随机链表的复制](https://leetcode.cn/problems/copy-list-with-random-pointer/description/?envType=study-plan-v2&envId=top-100-liked)
#### 题目描述
给你一个长度为 `n` 的链表，每个节点包含一个额外增加的随机指针 `random` ，该指针可以指向链表中的任何节点或空节点。

构造这个链表的 **[深拷贝](https://baike.baidu.com/item/%E6%B7%B1%E6%8B%B7%E8%B4%9D/22785317?fr=aladdin)**。 深拷贝应该正好由 `n` 个 **全新** 节点组成，其中每个新节点的值都设为其对应的原节点的值。新节点的 `next` 指针和 `random` 指针也都应指向复制链表中的新节点，并使原链表和复制链表中的这些指针能够表示相同的链表状态。**复制链表中的指针都不应指向原链表中的节点** 。

例如，如果原链表中有 `X` 和 `Y` 两个节点，其中 `X.random --> Y` 。那么在复制链表中对应的两个节点 `x` 和 `y` ，同样有 `x.random --> y` 。

返回复制链表的头节点。

用一个由 `n` 个节点组成的链表来表示输入/输出中的链表。每个节点用一个 `[val, random_index]` 表示：

- `val`：一个表示 `Node.val` 的整数。
- `random_index`：随机指针指向的节点索引（范围从 `0` 到 `n-1`）；如果不指向任何节点，则为  `null` 。

你的代码 **只** 接受原链表的头节点 `head` 作为传入参数。

示例：
![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2020/01/09/e1.png)
```
输入：head = [[7,null],[13,0],[11,4],[10,2],[1,0]]
输出：[[7,null],[13,0],[11,4],[10,2],[1,0]]
```
#### 核心思路
```
1. 复制每个节点并插入到原节点后面：
    - 首先遍历原链表，对于每个节点，创建一个新的节点，并将其插入到当前节点和下一个节点之间。这样，原链表 A -> B -> C 变成 A -> A' -> B -> B' -> C -> C'，其中 A', B', C' 是新复制的节点。

2. 设置新节点的随机指针：
    - 再次遍历链表，这次设置每个新节点的 random 指针。由于新节点紧跟在原节点后面，原节点的 random 指针指向的节点后面就是新节点的 random 指针应该指向的节点。例如，如果原节点 A 的 random 指针指向 B，那么新节点 A' 的 random 指针应该指向 B'。

3. 分离两个链表：
    - 最后，我们需要将原链表和新链表分开。遍历链表，将新节点从原链表中分离出来，恢复原链表的结构，同时构建新链表。
```
#### 代码
```python
class Solution:
    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        if not head:
            return None

        # Step 1: 创建新节点并插入到原节点后面
        curr = head
        while curr:
            # 创建新节点，并将其值设为当前节点的值
            new_node = Node(curr.val, curr.next, None)
            # 将新节点插入到当前节点和下一个节点之间
            curr.next = new_node
            # 移动到下一个原节点
            curr = new_node.next

        # Step 2: 设置新节点的随机指针
        curr = head
        while curr:
            # 如果当前节点有随机指针
            if curr.random:
                # 新节点的随机指针应该指向原节点随机指针对应的新节点
                curr.next.random = curr.random.next
            # 移动到下一个原节点
            curr = curr.next.next

        # Step 3: 分离两个链表
        old_head = head
        new_head = head.next
        curr_old = old_head
        curr_new = new_head

        while curr_old:
            # 恢复原链表的结构
            curr_old.next = curr_old.next.next if curr_old.next else None
            # 构建新链表的结构
            curr_new.next = curr_new.next.next if curr_new.next else None
            # 移动到下一个原节点
            curr_old = curr_old.next
            # 移动到新链表的下一个新节点
            curr_new = curr_new.next

        return new_head
```
### [148. 排序链表](https://leetcode.cn/problems/sort-list/description/?envType=study-plan-v2&envId=top-100-liked)
#### 题目描述
给你链表的头结点 `head` ，请将其按 **升序** 排列并返回 **排序后的链表** 。

示例：
![](https://assets.leetcode.com/uploads/2020/09/14/sort_list_1.jpg)
```
输入：head = [4,2,1,3]
输出：[1,2,3,4]
```
#### 核心思路
```
归并排序（Merge Sort）的时间复杂度为 O(nlog n)，空间复杂度为 O(1)（如果使用自底向上的方法）。

1. 找到链表的中间节点：使用快慢指针法，快指针每次移动两步，慢指针每次移动一步，当快指针到达链表末尾时，慢指针正好在中间。
2. 分割链表：将链表从中间节点分成两个子链表。
3. 递归排序：对两个子链表分别进行归并排序。
4. 合并排序后的子链表：将两个有序的子链表合并成一个有序的链表。
```
#### 代码
```python
class Solution:
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head or not head.next:
            return head
        
        # 找到链表的中间节点
        def find_mid(head: ListNode) -> ListNode:
            slow, fast = head, head.next
            while fast and fast.next:
                slow = slow.next
                fast = fast.next.next
            return slow
        
        # 合并两个有序链表
        def merge(l1: ListNode, l2: ListNode) -> ListNode:
            dummy = ListNode()
            current = dummy
            while l1 and l2:
                if l1.val < l2.val:
                    current.next = l1
                    l1 = l1.next
                else:
                    current.next = l2
                    l2 = l2.next
                current = current.next
            if l1:
                current.next = l1
            if l2:
                current.next = l2
            return dummy.next
        
        # 找到中间节点并分割链表
        mid = find_mid(head)
        right = mid.next
        mid.next = None
        left = head
        
        # 递归排序
        left = self.sortList(left)
        right = self.sortList(right)
        
        # 合并排序后的子链表
        return merge(left, right)
```
### [146. LRU 缓存](https://leetcode.cn/problems/lru-cache/description/?envType=study-plan-v2&envId=top-100-liked)
#### 题目描述
请你设计并实现一个满足  [LRU (最近最少使用) 缓存](https://baike.baidu.com/item/LRU) 约束的数据结构。

实现 `LRUCache` 类：

- `LRUCache(int capacity)` 以 **正整数** 作为容量 `capacity` 初始化 LRU 缓存
- `int get(int key)` 如果关键字 `key` 存在于缓存中，则返回关键字的值，否则返回 `-1` 。
- `void put(int key, int value)` 如果关键字 `key` 已经存在，则变更其数据值 `value` ；如果不存在，则向缓存中插入该组 `key-value` 。如果插入操作导致关键字数量超过 `capacity` ，则应该 **逐出** 最久未使用的关键字。

函数 `get` 和 `put` 必须以 `O(1)` 的平均时间复杂度运行。

示例：

```
输入
["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
[[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]

输出
[null, null, null, 1, null, -1, null, -1, 3, 4]

解释
LRUCache lRUCache = new LRUCache(2);
lRUCache.put(1, 1); // 缓存是 {1=1}
lRUCache.put(2, 2); // 缓存是 {1=1, 2=2}
lRUCache.get(1);    // 返回 1
lRUCache.put(3, 3); // 该操作会使得关键字 2 作废，缓存是 {1=1, 3=3}
lRUCache.get(2);    // 返回 -1 (未找到)
lRUCache.put(4, 4); // 该操作会使得关键字 1 作废，缓存是 {4=4, 3=3}
lRUCache.get(1);    // 返回 -1 (未找到)
lRUCache.get(3);    // 返回 3
lRUCache.get(4);    // 返回 4
```
#### 核心思路
```
LRU（最近最少使用）缓存是一种常见的缓存策略，用于管理有限的内存资源。当缓存满时，它会优先淘汰最近最少使用的数据项。这个题目要求你设计一个数据结构来实现 LRUCache 类，该类需要支持 get 和 put 操作，并且这些操作的时间复杂度都是 O(1)。

1. 双向链表 + 哈希表    
    - 双向链表：用于维护缓存中的键值对顺序，最近使用的节点放在链表的头部，最久未使用的节点放在链表的尾部。
    - 哈希表：用于快速查找节点，哈希表的键是缓存的键，值是对应的链表节点。

2. 操作细节
    - get 操作：
        - 如果键存在，返回对应的值，并将该节点移动到链表的头部（表示最近使用）。
        - 如果键不存在，返回 -1。
    - put 操作：
        - 如果键已存在，更新其值，并将该节点移动到链表的头部。
        - 如果键不存在，插入新的键值对到链表的头部。
        - 如果插入后缓存超过容量，移除链表尾部的节点（最久未使用的节点）。
```
#### 代码
```python
class LRUCache:
    def __init__(self, capacity: int):
        # 使用 OrderedDict 来维护键值对的顺序
        self.data = OrderedDict()
        # 初始化缓存的容量
        self.capacity = capacity

    def get(self, key: int) -> int:
        # 如果键存在于缓存中
        if key in self.data:
            # 将该键值对移动到有序字典的末尾（表示最近使用）
            self.data.move_to_end(key)
            # 返回对应的值
            return self.data[key]
        # 如果键不存在于缓存中，返回 -1
        return -1

    def put(self, key: int, value: int) -> None:
        # 如果键已经存在于缓存中
        if key in self.data:
            # 更新该键的值
            self.data[key] = value
        else:
            # 如果缓存已满
            if len(self.data) >= self.capacity:
                # 移除最久未使用的键值对（有序字典的头部）
                self.data.popitem(last=False)
            # 插入新的键值对
            self.data[key] = value
        # 将该键值对移动到有序字典的末尾（表示最近使用）
        self.data.move_to_end(key)
```
### [102. 二叉树的层序遍历](https://leetcode.cn/problems/binary-tree-level-order-traversal/description/?envType=study-plan-v2&envId=top-100-liked)
#### 题目描述
给你二叉树的根节点 `root` ，返回其节点值的 **层序遍历** 。 （即逐层地，从左到右访问所有节点）。

示例：
![](https://assets.leetcode.com/uploads/2021/02/19/tree1.jpg)
```
输入：root = [3,9,20,null,null,15,7]
输出：[[3],[9,20],[15,7]]
```
#### 核心思路
```
1. 初始化： 
    - 创建一个结果列表 result，用于存储每一层的节点值。
    - 创建一个双端队列 queue，并将根节点加入队列。

2. 遍历队列：
    - 当队列不为空时，进行以下步骤：
        - 记录当前层的节点数 level_size。
        - 创建一个空列表 current_level，用于存储当前层的节点值。
        - 遍历当前层的所有节点：
            - 从队列中取出一个节点 node。
            - 将 node 的值加入 current_level。
            - 如果 node 有左子节点，将左子节点加入队列。
            - 如果 node 有右子节点，将右子节点加入队列。
        - 将 current_level 加入 result。

3. 返回结果：
    - 遍历结束后，返回 result。
```
#### 代码
```python
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []  # 如果根节点为空，直接返回空列表
        
        result = []  # 用于存储最终的层序遍历结果
        queue = deque([root])  # 初始化队列，将根节点加入队列
        
        while queue:
            level_size = len(queue)  # 记录当前层的节点数
            current_level = []  # 用于存储当前层的节点值
            
            for _ in range(level_size):
                node = queue.popleft()  # 从队列中取出一个节点
                current_level.append(node.val)  # 将节点值加入当前层的列表
                
                if node.left:
                    queue.append(node.left)  # 如果节点有左子节点，将左子节点加入队列
                if node.right:
                    queue.append(node.right)  # 如果节点有右子节点，将右子节点加入队列
            
            result.append(current_level)  # 将当前层的节点值列表加入结果列表
        
        return result  # 返回最终的层序遍历结果
```
### [98. 验证二叉搜索树](https://leetcode.cn/problems/validate-binary-search-tree/description/?envType=study-plan-v2&envId=top-100-liked)
#### 题目描述
给你一个二叉树的根节点 `root` ，判断其是否是一个有效的二叉搜索树。

**有效** 二叉搜索树定义如下：

- 节点的左子树只包含 **小于** 当前节点的数。
- 节点的右子树只包含 **大于** 当前节点的数。
- 所有左子树和右子树自身必须也是二叉搜索树。

示例：
![](https://assets.leetcode.com/uploads/2020/12/01/tree1.jpg)
```
输入：root = [2,1,3]
输出：true
```
#### 核心思路
```
验证一个二叉树是否为二叉搜索树（BST）的关键在于确保每个节点的值都在一个特定的范围内。具体来说：
	- 对于每个节点，其左子树的所有节点值必须小于该节点的值。
	- 对于每个节点，其右子树的所有节点值必须大于该节点的值。

可以通过递归的方法来实现这一点，同时传递一个范围给每个递归调用，确保当前节点的值在这个范围内。

1. 定义一个辅助函数：
	- isValidBST，它接受三个参数：当前节点 node、当前节点值的下限 min_val 和上限 max_val。

2. 递归终止条件：
    - 如果当前节点为空，返回 True，因为空树是有效的BST。
    - 如果当前节点的值不在 min_val 和 max_val 之间，返回 False。

3. 递归调用：
    - 递归检查左子树，左子树的值必须小于当前节点的值，所以传递 min_val 和 node.val 作为新的范围。
    - 递归检查右子树，右子树的值必须大于当前节点的值，所以传递 node.val 和 max_val 作为新的范围。

4. **返回结果**：只有当左右子树都是有效的BST时，当前节点才是有效的BST。
```
#### 代码
```python
class Solution:
	def isValidBST(root: TreeNode) -> bool:
	    def helper(node, min_val=float('-inf'), max_val=float('inf')):
	        # 如果当前节点为空，返回 True，因为空树是有效的BST
	        if not node:
	            return True
	        
	        # 检查当前节点的值是否在指定的范围内
	        if not (min_val < node.val < max_val):
	            return False
	        
	        # 递归检查左子树
	        # 左子树的所有节点值必须小于当前节点的值
	        left_valid = helper(node.left, min_val, node.val)
	        
	        # 递归检查右子树
	        # 右子树的所有节点值必须大于当前节点的值
	        right_valid = helper(node.right, node.val, max_val)
	        
	        # 只有当左右子树都满足条件时，当前节点才是有效的BST
	        return left_valid and right_valid
	    
	    # 初始调用，范围是负无穷到正无穷
	    return helper(root)
```
### [230. 二叉搜索树中第 K 小的元素](https://leetcode.cn/problems/kth-smallest-element-in-a-bst/description/?envType=study-plan-v2&envId=top-100-liked)
#### 题目描述
给定一个二叉搜索树的根节点 `root` ，和一个整数 `k` ，请你设计一个算法查找其中第 `k` 小的元素（从 1 开始计数）。

示例：
![](https://assets.leetcode.com/uploads/2021/01/28/kthtree1.jpg)
```
输入：root = [3,1,4,null,2], k = 1
输出：1
```
#### 核心思路
```
在二叉搜索树（BST）中，左子树的所有节点值都小于根节点的值，右子树的所有节点值都大于根节点的值。利用这一性质，可以通过中序遍历来按升序访问所有节点。

1. 中序遍历：中序遍历二叉搜索树会得到一个有序列表（从小到大）。
2. 提前终止：在遍历到第 k 个节点时，可以直接返回该节点的值，而不需要遍历完整棵树。
```
#### 代码
```python
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        stack = []
        current = root
        
        while current or stack:
            # 先遍历左子树，将所有左子节点压入栈中
            while current:
                stack.append(current)
                current = current.left
            
            # 从栈中弹出一个节点，访问该节点
            current = stack.pop()
            k -= 1  # 每访问一个节点，k 减 1
            
            # 如果 k 为 0，说明当前节点是第 k 小的元素，直接返回
            if k == 0:
                return current.val
            
            # 继续遍历右子树
            current = current.right
```
### [199. 二叉树的右视图](https://leetcode.cn/problems/binary-tree-right-side-view/description/?envType=study-plan-v2&envId=top-100-liked)
#### 题目描述
给定一个二叉树的 **根节点** `root`，想象自己站在它的右侧，按照从顶部到底部的顺序，返回从右侧所能看到的节点值。

示例：
![](https://assets.leetcode.com/uploads/2021/02/14/tree.jpg)
```
输入：[1,2,3,null,5,null,4]
输出：[1,3,4]
```
#### 核心思路
```
我们需要逐层遍历二叉树，记录每层最右侧的节点，在每一层中，我们只关心最右侧的节点。

1. 初始化：
    - 创建一个队列，将根节点加入队列。
    - 创建一个结果列表，用于存储从右侧看到的节点值。

2. 层次遍历：
    - 使用一个 while 循环，只要队列不为空，就继续处理。
    - 在每次循环开始时，记录当前队列的长度，这个长度表示当前层的节点数。

3. 处理当前层：
    - 使用一个 for 循环，遍历当前层的所有节点。
    - 在每次迭代中，从队列中取出一个节点。
    - 如果是当前层的最后一个节点，将其值加入结果列表。
    - 将当前节点的左子节点和右子节点依次加入队列。

4. 返回结果：
    - 最终，结果列表中存储的就是从右侧看到的节点值。
```
#### 代码
```python
class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []  # 如果根节点为空，返回空列表
        
        result = []  # 用于存储从右侧看到的节点值
        queue = deque([root])  # 初始化队列，将根节点加入队列
        
        while queue:
            # 当前层的节点数
            level_length = len(queue)
            
            for i in range(level_length):
                # 从队列中取出一个节点
                node = queue.popleft()
                
                # 如果是当前层的最后一个节点，将其值加入结果列表
                if i == level_length - 1:
                    result.append(node.val)
                
                # 将当前节点的子节点加入队列
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        
        return result  # 返回从右侧看到的节点值
```
### [114. 二叉树展开为链表](https://leetcode.cn/problems/flatten-binary-tree-to-linked-list/description/?envType=study-plan-v2&envId=top-100-liked)
#### 题目描述
给你二叉树的根结点 `root` ，请你将它展开为一个单链表：
- 展开后的单链表应该同样使用 `TreeNode` ，其中 `right` 子指针指向链表中下一个结点，而左子指针始终为 `null` 。
- 展开后的单链表应该与二叉树 [**先序遍历**](https://baike.baidu.com/item/%E5%85%88%E5%BA%8F%E9%81%8D%E5%8E%86/6442839?fr=aladdin) 顺序相同。

示例：
![](https://assets.leetcode.com/uploads/2021/01/14/flaten.jpg)
```
输入：root = [1,2,5,3,4,null,6]
输出：[1,null,2,null,3,null,4,null,5,null,6]
```
#### 核心思路
```
将二叉树展开为单链表的问题可以通过递归或迭代的方法来解决。最优的解题思路通常是利用后序遍历的思想，因为我们需要先处理子节点再处理父节点。

1. 后序遍历：由于我们需要先处理子节点再处理父节点，所以使用后序遍历（左-右-根）的思想。
2. 调整指针：在遍历过程中，将当前节点的右子树接到左子树的最右节点的右子树上，然后将左子树变为右子树，左子树置为空。
3. 递归调用：对每个节点都进行上述操作，直到遍历完整棵树。
```
#### 代码
```python
class Solution:
    def flatten(self, root: Optional[TreeNode]) -> None:
        if not root:
            return
        
        # 先递归展开左右子树
        self.flatten(root.left)
        self.flatten(root.right)
        
        # 如果有左子树
        if root.left:
            # 保存当前的右子树
            right_subtree = root.right
            
            # 将左子树移动到右子树的位置
            root.right = root.left
            root.left = None  # 清空左子树
            
            # 找到右子树的最右节点
            current = root
            while current.right:
                current = current.right
            
            # 将保存的原右子树接到最右节点
            current.right = right_subtree
```
### [105. 从前序与中序遍历序列构造二叉树](https://leetcode.cn/problems/construct-binary-tree-from-preorder-and-inorder-traversal/description/?envType=study-plan-v2&envId=top-100-liked)
#### 题目描述
给定两个整数数组 `preorder` 和 `inorder` ，其中 `preorder` 是二叉树的**先序遍历**， `inorder` 是同一棵树的**中序遍历**，请构造二叉树并返回其根节点。

示例：

![](https://assets.leetcode.com/uploads/2021/02/19/tree.jpg)

```
输入：preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
输出：[3,9,20,null,null,15,7]
```
#### 核心思路
```
理解先序和中序遍历的特点：
    - 先序遍历的第一个元素是树的根节点。
    - 中序遍历中，根节点左边的部分是左子树，右边的部分是右子树。

递归构建二叉树：
    - 使用先序遍历的第一个元素创建根节点。
    - 在中序遍历中找到这个根节点的位置，这样可以确定左子树和右子树的范围。
    - 递归地构建左子树和右子树。

具体实现：    
    - 定义一个递归函数 buildTree，该函数接受先序遍历和中序遍历的子数组范围作为参数。
    - 在每次递归调用中，使用先序遍历的第一个元素创建根节点，并在中序遍历中找到该根节点的位置。
    - 根据根节点的位置，划分出左子树和右子树的范围，并递归构建左子树和右子树。
```
#### 代码
```python
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        # 如果输入的先序或中序遍历为空，则返回 None
        if not preorder or not inorder:
            return None
        
        # 先序遍历的第一个元素是根节点
        root_val = preorder[0]
        root = TreeNode(root_val)
        
        # 在中序遍历中找到根节点的位置
        root_index = inorder.index(root_val)
        
        # 递归构建左子树
        # 先序遍历中，根节点后的前 `root_index` 个元素是左子树的先序遍历
        # 中序遍历中，根节点前的 `root_index` 个元素是左子树的中序遍历
        root.left = self.buildTree(preorder[1:1 + root_index], inorder[:root_index])
        
        # 递归构建右子树
        # 先序遍历中，从 `1 + root_index` 到末尾的元素是右子树的先序遍历
        # 中序遍历中，从 `root_index + 1` 到末尾的元素是右子树的中序遍历
        root.right = self.buildTree(preorder[1 + root_index:], inorder[root_index + 1:])
        
        # 返回构建的根节点
        return root
```
### [437. 路径总和 III](https://leetcode.cn/problems/path-sum-iii/description/?envType=study-plan-v2&envId=top-100-liked)
#### 题目描述
给定一个二叉树的根节点 `root` ，和一个整数 `targetSum` ，求该二叉树里节点值之和等于 `targetSum` 的 **路径** 的数目。

**路径** 不需要从根节点开始，也不需要在叶子节点结束，但是路径方向必须是向下的（只能从父节点到子节点）。

示例：

![](https://assets.leetcode.com/uploads/2021/04/09/pathsum3-1-tree.jpg)

```
输入：root = [10,5,-3,3,2,null,11,3,-2,null,1], targetSum = 8
输出：3
解释：和等于 8 的路径有 3 条，如图所示。
```
#### 核心思路
```
这个问题可以使用递归结合前缀和的方式来求解。下面是详细步骤：

1. 前缀和的概念：
	- 前缀和是指从树的根节点到当前节点的路径上的所有节点的值之和。
	- 我们使用一个哈希表（字典）来记录遍历过程中所有前缀和出现的次数。通过计算前缀和，我们可以快速查找某段路径的和是否等于 targetSum。

2. 核心思路：
	- 遍历树时，我们要检查以当前节点为终点，有没有一条路径的和等于 targetSum。
	- 我们可以通过计算当前路径和 currSum，再检查哈希表中是否有一个前缀和（即 currSum - targetSum）存在。
	- 如果存在，说明从该前缀和之后的部分路径，其和就是 targetSum。

3. 具体步骤
	- 用一个递归函数遍历二叉树，每到一个节点：
	    1. 计算当前节点到根节点的路径和 currSum。
	    2. 在哈希表中查找是否有 currSum - targetSum 的前缀和，存在说明从某段路径和等于 targetSum。
	    4. 将当前路径和 currSum 加入哈希表，并将其出现次数增加1。
	    5. 递归地遍历该节点的左子树和右子树。
	    6. 返回时，将哈希表中当前路径和 currSum 的计数减1，避免影响其他路径。
```
#### 代码
```python
class Solution:
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> int:
        def dfs(node, currSum):
            if not node:
                return 0
            
            # 更新当前路径和
            currSum += node.val
            
            # 计算从某个祖先节点到当前节点的路径和是否等于 targetSum
            # 即 (currSum - targetSum) 是否在 prefix_sum_count 中存在
            count = prefix_sum_count.get(currSum - targetSum, 0)
            
            # 更新前缀和字典，记录当前路径和出现的次数
            prefix_sum_count[currSum] = prefix_sum_count.get(currSum, 0) + 1
            
            # 递归遍历左子树和右子树
            count += dfs(node.left, currSum)
            count += dfs(node.right, currSum)
            
            # 回溯：移除当前节点对应的路径和，以便不影响其他分支的路径计数
            prefix_sum_count[currSum] -= 1
            
            return count
        
        # 前缀和哈希表，记录每个路径和出现的次数
        # 初始值为 {0: 1}，表示从根节点到当前节点的路径和为0的路径有一条（即空路径）
        prefix_sum_count = {0: 1}
        
        return dfs(root, 0)
```
### [236. 二叉树的最近公共祖先](https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-tree/description/?envType=study-plan-v2&envId=top-100-liked)
#### 题目描述
给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。

[百度百科](https://baike.baidu.com/item/%E6%9C%80%E8%BF%91%E5%85%AC%E5%85%B1%E7%A5%96%E5%85%88/8918834?fr=aladdin)中最近公共祖先的定义为：“对于有根树 T 的两个节点 p、q，最近公共祖先表示为一个节点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（**一个节点也可以是它自己的祖先**）。”

**示例 1：**

![](https://assets.leetcode.com/uploads/2018/12/14/binarytree.png)

```
输入：root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
输出：3
解释：节点 5 和节点 1 的最近公共祖先是节点 3 。
```
#### 核心思路
```
要找到二叉树中节点 p 和 q 的最近公共祖先（LCA）。有几个关键点：
	1. 如果当前节点是 p 或 q，那么它就是一个祖先节点。
	2. 如果 p 和 q 分别位于当前节点的左右子树，那么当前节点就是它们的最近公共祖先。
	3. 如果两个节点都位于左子树或都位于右子树，则继续在对应子树中寻找。

1. 递归基：  
    - 如果当前节点是 None，直接返回 None；如果当前节点是 p 或 q，直接返回当前节点。
    
2. 递归查找左右子树：  
    - 在左子树和右子树递归查找 p 和 q。
    
3. 返回结果： 
    - 如果左右子树都找到值，说明 p 和 q 分别在左右子树，当前节点是最近公共祖先。
    - 如果只有左子树找到结果，返回左子树结果。
    - 如果只有右子树找到结果，返回右子树结果。
```
#### 代码
```python
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        
        # 如果当前节点为空，说明已经到达叶子节点的底部，返回 None
        # 如果当前节点是 p 或 q，说明我们找到了其中一个节点，返回当前节点
        if not root or root == p or root == q:
            return root
        
        # 在左子树中递归查找 p 和 q，返回在左子树中的结果
        left = self.lowestCommonAncestor(root.left, p, q)
        
        # 在右子树中递归查找 p 和 q，返回在右子树中的结果
        right = self.lowestCommonAncestor(root.right, p, q)
        
        # 如果左子树和右子树都找到了 p 或 q，那么当前节点 root 就是最近公共祖先
        # 因为 p 和 q 分别在当前节点的左右子树中
        if left and right:
            return root
        
        # 如果只有左子树找到结果（right 为 None），说明 p 和 q 都在左子树中
        # 返回左子树的结果
        # 如果只有右子树找到结果（left 为 None），说明 p 和 q 都在右子树中
        # 返回右子树的结果
        return left if left else right
```
### [200. 岛屿数量](https://leetcode.cn/problems/number-of-islands/description/?envType=study-plan-v2&envId=top-100-liked)
#### 题目描述
给你一个由 `1`（陆地）和 `0`（水）组成的的二维网格，请你计算网格中岛屿的数量。

岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。

此外，你可以假设该网格的四条边均被水包围。

示例：
```
输入：grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
输出：1
```
#### 核心思路
```
使用DFS处理岛屿数量问题时更加直观和简洁。

1. 遍历网格：我们需要遍历整个网格中的每一个单元格。
2. 遇到陆地时：如果遇到一个值为 '1' 的单元格，表示这是一个岛屿的一部分。
3. 标记岛屿：使用DFS从这个单元格开始，将所有与之相连的陆地标记为 '0'，以避免重复计数。
4. 计数岛屿：每次启动DFS时，表示发现了一个新的岛屿，因此计数器加一。
```
#### 代码
```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        # 如果网格为空，直接返回0
        if not grid:
            return 0

        # 获取网格的行数和列数
        rows, cols = len(grid), len(grid[0])
        count = 0

        def dfs(r: int, c: int):
            """
            使用深度优先搜索 (DFS) 标记从 (r, c) 开始的所有相连的陆地。
            """
            # 检查边界条件，如果超出边界或者当前单元格是水，则返回
            if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] == '0':
                return
            
            # 将当前单元格标记为已访问（即变为 '0'）
            grid[r][c] = '0'
            
            # 递归访问上下左右四个方向
            dfs(r + 1, c)  # 下
            dfs(r - 1, c)  # 上
            dfs(r, c + 1)  # 右
            dfs(r, c - 1)  # 左

        # 遍历整个网格
        for r in range(rows):
            for c in range(cols):
                # 如果当前单元格是陆地（'1'），则发现了一个新的岛屿
                if grid[r][c] == '1':
                    count += 1
                    # 使用DFS标记整个岛屿
                    dfs(r, c)

        return count
```
### [994. 腐烂的橘子](https://leetcode.cn/problems/rotting-oranges/description/?envType=study-plan-v2&envId=top-100-liked)
#### 题目描述
在给定的 `m x n` 网格 `grid` 中，每个单元格可以有以下三个值之一：

- 值 `0` 代表空单元格；
- 值 `1` 代表新鲜橘子；
- 值 `2` 代表腐烂的橘子。

每分钟，腐烂的橘子 **周围 4 个方向上相邻** 的新鲜橘子都会腐烂。

返回直到单元格中没有新鲜橘子为止所必须经过的最小分钟数。如果不可能，返回 `-1`。

示例：
![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2019/02/16/oranges.png)
```
输入：grid = [[2,1,1],[1,1,0],[0,1,1]]
输出：4
```
#### 核心思路
```
这道题是一道经典的 “多源最短路径” 问题，可以通过广度优先搜索（BFS）来解决。

1. 初始化队列：首先遍历整个网格，找到所有初始时腐烂的橘子，并将它们的位置加入到队列中。同时，统计新鲜橘子的数量。
2. 广度优先搜索（BFS）：使用队列进行BFS，每次处理队列中的所有腐烂橘子，将它们周围的新鲜橘子变为腐烂橘子，并将新腐烂的橘子加入队列。每处理完一轮，时间增加1分钟。
3. 检查结果：当队列为空时，检查是否还有新鲜橘子。如果没有新鲜橘子，返回经过的分钟数；如果有，返回-1。
```
#### 代码
```python
class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        if not grid:
            return -1

        rows, cols = len(grid), len(grid[0])
        fresh_count = 0
        queue = deque()

        # 初始化队列和新鲜橘子计数
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 2:
                    queue.append((r, c))  # 将所有初始腐烂的橘子加入队列
                elif grid[r][c] == 1:
                    fresh_count += 1  # 统计新鲜橘子的数量

        # 定义四个方向：上下左右
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        minutes_passed = 0  # 记录时间

        # 广度优先搜索
        while queue and fresh_count > 0:
            minutes_passed += 1  # 每处理完一轮，时间增加1分钟
            for _ in range(len(queue)):
                r, c = queue.popleft()  # 从队列中取出一个腐烂的橘子
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc  # 计算相邻的单元格位置
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1:
                        # 如果相邻单元格是新鲜橘子，将其变为腐烂橘子
                        grid[nr][nc] = 2
                        fresh_count -= 1  # 新鲜橘子数量减少1
                        queue.append((nr, nc))  # 将新腐烂的橘子加入队列

        # 检查是否还有新鲜橘子
        if fresh_count == 0:
            return minutes_passed  # 如果没有新鲜橘子，返回经过的分钟数
        else:
            return -1  # 如果还有新鲜橘子，返回-1
```
### [207. 课程表](https://leetcode.cn/problems/course-schedule/description/?envType=study-plan-v2&envId=top-100-liked)
#### 题目描述
你这个学期必须选修 `numCourses` 门课程，记为 `0` 到 `numCourses - 1` 。

在选修某些课程之前需要一些先修课程。 先修课程按数组 `prerequisites` 给出，其中 `prerequisites[i] = [ai, bi]` ，表示如果要学习课程 `ai` 则 **必须** 先学习课程  `bi` 。

- 例如，先修课程对 `[0, 1]` 表示：想要学习课程 `0` ，你需要先完成课程 `1` 。

请你判断是否可能完成所有课程的学习？如果可以，返回 `true` ；否则，返回 `false` 。

示例：
```
输入：numCourses = 2, prerequisites = [[1,0]]
输出：true
解释：总共有 2 门课程。学习课程 1 之前，你需要完成课程 0 。这是可能的。
```
#### 核心思路
```
这个问题的核心是判断是否有依赖冲突。你可以把课程想象成一系列任务，部分任务必须先完成其他任务才能开始。如果出现循环依赖，就意味着有些任务永远无法完成（例如任务 A 依赖任务 B，B 又依赖 A），所以问题的关键就是要检查这些循环依赖是否存在。

1. 建立依赖关系：
   - 假设有很多课程，每门课可能需要先修另一门课。我们可以用一个图来表示这些课程之间的依赖关系。
   - 图中的每个节点就是一门课，边则表示“先修关系”（从一门课指向另一门先修课）。

2. 找到没有依赖的课程：
   - 如果一门课不需要任何先修课（没有依赖），我们可以直接开始学这门课。

3. 依次完成这些课程：
   - 我们先学完没有依赖的课程，然后看看哪些课程现在已经可以学了（它们之前依赖的课程已经完成了）。
   - 依次重复这个过程，直到所有课程都学完，或者发现有些课程永远学不了（说明有循环依赖）。

4. 检查是否有循环依赖：
   - 如果我们可以把所有课程都顺利完成，那就没有循环依赖，可以学完所有课程。
   - 如果有些课程永远无法学完，那就有循环依赖，无法完成所有课程。
```
#### 代码
```python
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        # 1. 初始化邻接表和入度表
        # 邻接表（graph）：记录每门课的后续课，即哪些课依赖于这门课
        # 入度表（in_degree）：记录每门课有多少门前置课（依赖的课程）
        graph = {i: [] for i in range(numCourses)}  # 每门课的后续课程列表
        in_degree = {i: 0 for i in range(numCourses)}  # 每门课的前置课个数

        # 2. 构建图结构和入度表
        # 遍历所有的前置课程关系 [a, b]，表示要先学课程 b 再学课程 a
        for course, prereq in prerequisites:
            graph[prereq].append(course)  # 课程 b 指向课程 a，表示 a 依赖 b
            in_degree[course] += 1  # 课程 a 的前置课程数加 1

        # 3. 找到所有没有前置课程的课程（即入度为 0 的课程）
        queue = deque([course for course in in_degree if in_degree[course] == 0])

        # 记录已经学完的课程数量
        completed_courses = 0

        # 4. 进行广度优先搜索（BFS）
        while queue:
            current_course = queue.popleft()  # 从队列中取出一门可以学习的课程
            completed_courses += 1  # 这门课算作已经学完

            # 对这门课的后续课程进行处理
            for next_course in graph[current_course]:
                in_degree[next_course] -= 1  # 该后续课的前置课程少了一门
                if in_degree[next_course] == 0:  # 如果该课的前置课程已经全部学完
                    queue.append(next_course)  # 将其加入队列

        # 5. 判断是否所有课程都学完了
        return completed_courses == numCourses  # 如果已学完的课程数量等于总课程数，则返回 True
```
### [208. 实现 Trie (前缀树)](https://leetcode.cn/problems/implement-trie-prefix-tree/description/?envType=study-plan-v2&envId=top-100-liked)
#### 题目描述
**[Trie](https://baike.baidu.com/item/%E5%AD%97%E5%85%B8%E6%A0%91/9825209?fr=aladdin)**（发音类似 "try"）或者说 **前缀树** 是一种树形数据结构，用于高效地存储和检索字符串数据集中的键。这一数据结构有相当多的应用情景，例如自动补全和拼写检查。

请你实现 Trie 类：

- `Trie()` 初始化前缀树对象。
- `void insert(String word)` 向前缀树中插入字符串 `word` 。
- `boolean search(String word)` 如果字符串 `word` 在前缀树中，返回 `true`（即，在检索之前已经插入）；否则，返回 `false` 。
- `boolean startsWith(String prefix)` 如果之前已经插入的字符串 `word` 的前缀之一为 `prefix` ，返回 `true` ；否则，返回 `false` 。

示例：
```
输入：
["Trie", "insert", "search", "search", "startsWith", "insert", "search"]
[[], ["apple"], ["apple"], ["app"], ["app"], ["app"], ["app"]]
输出：
[null, null, true, false, true, null, true]

解释：
Trie trie = new Trie();
trie.insert("apple");
trie.search("apple");   // 返回 True
trie.search("app");     // 返回 False
trie.startsWith("app"); // 返回 True
trie.insert("app");
trie.search("app");     // 返回 True
```
#### 核心思路
```
1. Trie（字典树）简介：
    - Trie 是一种树形结构，专门用于高效存储和查找字符串集合，常用于字典中的单词搜索。
    - 每个节点代表一个字符，路径代表某个单词的前缀。
    - 最常见的操作有插入一个单词、搜索某个单词是否存在、以及判断某个前缀是否存在。

2. __init__ 方法： 
    - 初始化字典树的根节点。用一个字典（dict）来存储，根节点是空字典 {}。

3. insert 方法：
    - 逐个字符插入到字典树中，从根节点开始。如果当前字符已经存在，则移动到下一节点；否则就创建一个新的字典节点。
    - 在单词的最后一个字符插入后，添加一个特殊标记符（#），表示该节点代表一个完整单词的结束。

4. search 方法：
    - 用于搜索某个单词是否在 Trie 中存在。
    - 逐个字符从根节点开始查找，如果某个字符不在节点中，直接返回 False，表示没有这个单词。
    - 如果能遍历完整个单词，并且最后一个字符的节点有结束标记符（#），说明这个单词存在。

5. startsWith 方法：
    - 检查是否存在以某个前缀开头的单词。
    - 逐个字符检查，如果当前字符在 Trie 中存在，就继续；否则返回 False，表示没有该前缀。
    - 如果可以遍历完整个前缀，返回 True，表示存在这个前缀。
```
#### 代码
```python
class Trie:
    def __init__(self):
        # 初始化一个空的字典树根节点，根节点本身也是一个字典
        self.root = {}

    def insert(self, word: str) -> None:
        # 从根节点开始逐个插入字符
        node = self.root
        for char in word:
            # 如果当前字符不在当前节点中，添加这个字符
            if char not in node:
                node[char] = {}
            node = node[char]
        # 插入完成后，在末尾标记此位置为单词结束
        node['#'] = True  # '#' 是结束标记符，表示这个路径为一个完整单词

    def search(self, word: str) -> bool:
        # 从根节点开始搜索字符
        node = self.root
        for char in word:
            # 如果字符不在节点中，则表示字典树中不存在该单词
            if char not in node:
                return False
            node = node[char]
        # 判断最后的节点是否有单词结束标记
        return '#' in node

    def startsWith(self, prefix: str) -> bool:
        # 从根节点开始逐个字符检查前缀
        node = self.root
        for char in prefix:
            # 如果前缀中的字符不在字典树中，返回 False
            if char not in node:
                return False
            node = node[char]
        # 如果可以走完前缀，说明存在该前缀
        return True
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
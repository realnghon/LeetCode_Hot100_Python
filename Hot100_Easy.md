
### [1. 两数之和](https://leetcode.cn/problems/two-sum/description/?envType=study-plan-v2&envId=top-100-liked)

^ef28b1

#### 题目描述
给定一个整数数组 `nums` 和一个整数目标值 `target`，请你在该数组中找出 **和为目标值** _`target`_  的那 **两个** 整数，并返回它们的数组下标。
你可以假设每种输入只会对应一个答案，并且你不能使用两次相同的元素。
你可以按任意顺序返回答案。

示例：
```
输入：nums = [2,7,11,15], target = 9
输出：[0,1]
解释：因为 nums[0] + nums[1] == 9 ，返回 [0, 1] 
```

#### 核心思路
该题目可以通过暴力枚举的方法计算出满足`nums[i]+nums[j]=target`的`i`和`j`，但是时间复杂度过高，因此采用哈希表的方法实现。
遍历nums数组中的每一个元素，若`target-nums[i]`不存在于哈希表中，则将`num[i]`作为key，`i`作为value插入哈希表。【可以避免nums中相同元素相加为target】
时间复杂度：由于哈希表查询的时间复杂度为$O(1)$，因此，题目的复杂度取决于遍历nums数组，所以为$O(n)$

#### 代码
```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        hashmap={}
        #查询target-nums[i]是否存在于hashmap中，若不存在，则将nums[i]插入hashmap
        for i in range(len(nums)):
            if target-nums[i] in hashmap:
                return [i,hashmap[target-nums[i]]]
            hashmap[nums[i]]=i

```


### [283. 移动零](https://leetcode.cn/problems/move-zeroes/description/?envType=study-plan-v2&envId=top-100-liked)

^06faf6

#### 题目描述
给定一个数组 `nums`，编写一个函数将所有 `0` 移动到数组的末尾，同时保持非零元素的相对顺序。
**请注意** ，必须在不复制数组的情况下原地对数组进行操作。

示例
```
输入: nums = [0,1,0,3,12]
输出: [1,3,12,0,0]
```

#### 核心思路
采用类似快排的划分思想，以非0数为基准，左边为非零数，右边为0，通过`zeroindex`记录第一个为0的元素的下标。然后循环遍历nums数组，当`nums[i]!=0`时，`nums[i]`和`nums[zeroindex]`进行交换，然后`zeroindex++`
时间复杂度：O(n)，空间复杂度：O(1)

#### 代码
```python
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        zeroindex=-1
        for i in range(len(nums)):
            if nums[i]==0 and zeroindex==-1:
                zeroindex=i
            elif nums[i]!=0 and zeroindex!=-1:
                nums[zeroindex],nums[i]=nums[i],nums[zeroindex]
                zeroindex+=1
        return nums
```
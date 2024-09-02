
### [1. 两数之和](https://leetcode.cn/problems/two-sum/description/?envType=study-plan-v2&envId=top-100-liked)

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

### [160. 相交链表](https://leetcode.cn/problems/intersection-of-two-linked-lists/description/?envType=study-plan-v2&envId=top-100-liked)

#### 题目描述
给你两个单链表的头节点 `headA` 和 `headB` ，请你找出并返回两个单链表相交的起始节点。如果两个链表不存在相交节点，返回 `null` 。
图示两个链表在节点 `c1` 开始相交，题目数据 **保证** 整个链式结构中不存在环。

![相交链表](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/14/160_statement.png)

**注意**，函数返回结果后，链表必须 **保持其原始结构** 。


示例：
![示例](https://assets.leetcode.com/uploads/2021/03/05/160_example_1_1.png)

```
输入：intersectVal = 8, listA = [4,1,8,4,5], listB = [5,6,1,8,4,5], skipA = 2, skipB = 3
输出：Intersected at '8'
解释：相交节点的值为8（注意，如果两个链表相交则不能为0）。
从各自的表头开始算起，链表 A 为 [4,1,8,4,5]，链表 B 为 [5,6,1,8,4,5]。
在 A 中，相交节点前有 2 个节点；在 B 中，相交节点前有 3 个节点。
请注意相交节点的值不为 1，因为在链表 A 和链表 B 之中值为 1 的节点 (A 中第二个节点和 B 中第三个节点) 是不同的节点。换句话说，它们在内存中指向两个不同的位置，而链表 A 和链表 B 中值为 8 的节点 (A 中第三个节点，B 中第四个节点) 在内存中指向相同的位置。
```
#### 核心思路
```
假设两个链表分别为 A 和 B，并且它们在某一点相交。设 A 的长度为 m，B 的长度为 n，交点之前的部分长度分别为 a 和 b，交点之后的部分长度为 c。

- 如果两个链表没有交点，那么 indexA 和 indexB 最终都会到达 None，从而退出循环。
- 如果有交点，由于两个指针都会遍历完自己的链表后再遍历对方的链表，因此它们会在交点处相遇。
```

#### 代码

```python
class Solution: 
	def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]: 
		indexA, indexB = headA, headB 
		while indexA != indexB: 
			indexA = indexA.next if indexA else headB 
			indexB = indexB.next if indexB else headA 
		return indexA
```

### [206. 反转链表](https://leetcode.cn/problems/reverse-linked-list/description/?envType=study-plan-v2&envId=top-100-liked)

#### 题目描述
给你单链表的头节点 `head` ，请你反转链表，并返回反转后的链表。

示例：
![](https://assets.leetcode.com/uploads/2021/02/19/rev1ex1.jpg)

```
输入：head = [1,2,3,4,5]
输出：[5,4,3,2,1]
```

#### 核心思路
```
反转链表的基本思路是遍历链表，并将每个节点的 next 指针从指向它的下一个节点改为指向前一个节点。为了做到这一点，我们需要维护三个指针：

- prev：指向当前节点的前一个节点。
- current：指向当前节点。
- next_node：指向当前节点的下一个节点。
```

#### 代码
```python
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head:
            return None
        # 初始化前驱节点和当前节点
        prev = None
        current = head
        # 遍历链表，反转指针
        while current:
            next_node = current.next
            # 反转指针
            current.next = prev
            # 移动前驱节点和当前节点
            prev = current
            current = next_node
        
        # 返回新的头节点（即原链表的最后一个节点）
        return prev
```


### [234. 回文链表](https://leetcode.cn/problems/palindrome-linked-list/description/?envType=study-plan-v2&envId=top-100-liked)

#### 题目描述
给你一个单链表的头节点 `head` ，请你判断该链表是否为回文链表。如果是，返回 `true` ；否则，返回 `false` 。

示例：

![](https://assets.leetcode.com/uploads/2021/03/03/pal1linked-list.jpg)

```
输入：head = [1,2,2,1]
输出：true
```

#### 核心思路
方法一：链表转化为数组
```
可以遍历链表，并将其中的元素存储在数组中，然后使用首尾双指针判断数组是否为回文数组。
时间复杂度：O(n)，空间复杂度：O(n)
```
【方法二】
```
为了降低空间复杂度，改变链表结构，将链表后半部分及进行逆序，然后和链表前半部分的数据进行比较，判断是否为回文。

步骤一：使用快慢指针寻找到链表中间位置。初始化慢指针slow和快指针fast为链表头指针head，slow每次移动一位，fast每次移动两位，最终在fast为空时，slow指针指向链表的中间位置，将链表分为了两个部分。在接下来的步骤中，会对这两个部分是否为回文进行判断。
步骤二：以slow为头节点，将slow指向的后半部分指针进行逆序排序。【参考反转链表】
步骤三：将反转后的链表和以head节点为首的链表前半部分按照次序进行比较，若出现不相等的值，则返回False；反之最终返回True
```

#### 代码
```python
class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        if not head or not head.next:
            return True
        # 步骤一：使用快慢指针寻找到链表中间位置
        slow, fast = head, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        # 步骤二：对于链表后半部分进行反转
        def reverseLink(head):
            prev = None
            current = head
            while current:
                next_node = current.next
                current.next = prev
                prev = current
                current = next_node
            return prev
        slow = reverseLink(slow)
        # 步骤三：判断是否为回文链表
        while slow:
            if slow.val != head.val:
                return False
            slow, head = slow.next, head.next
        return True
```
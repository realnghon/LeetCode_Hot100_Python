
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
时间复杂度：由于哈希表查询的时间复杂度为 $O(1)$，因此，题目的复杂度取决于遍历nums数组，所以为 $O(n)$

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

### [141. 环形链表](https://leetcode.cn/problems/linked-list-cycle/description/?envType=study-plan-v2&envId=top-100-liked)
#### 题目描述
给你一个链表的头节点 `head` ，判断链表中是否有环。

如果链表中有某个节点，可以通过连续跟踪 `next` 指针再次到达，则链表中存在环。 为了表示给定链表中的环，评测系统内部使用整数 `pos` 来表示链表尾连接到链表中的位置（索引从 0 开始）。**注意：`pos` 不作为参数进行传递** 。仅仅是为了标识链表的实际情况。

_如果链表中存在环_ ，则返回 `true` 。 否则，返回 `false` 。

示例：

![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/07/circularlinkedlist.png)

```
输入：head = [3,2,0,-4], pos = 1
输出：true
解释：链表中有一个环，其尾部连接到第二个节点。
```

#### 核心思路
```
采用快慢指针的方式，设置slow和fast指针：

初始化：slow=fast=head
循环：当slow，fast和fast.next均不为None时，令slow=slow.next,fast=fast.next.next，若slow=fast，则存在环形链表；若slow和fast中存在None，则不存在环形链表。
```

#### 代码
```python
class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        if not head:
            return False
        slow, fast=head, head
        while slow and fast and fast.next:
            slow, fast=slow.next, fast.next.next
            if slow==fast:
                return True    
        return False
```

### [21. 合并两个有序链表](https://leetcode.cn/problems/merge-two-sorted-lists/description/?envType=study-plan-v2&envId=top-100-liked)
#### 题目描述
将两个升序链表合并为一个新的 **升序** 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。 

示例：

![](https://assets.leetcode.com/uploads/2020/10/03/merge_ex1.jpg)

```
输入：l1 = [1,2,4], l2 = [1,3,4]
输出：[1,1,2,3,4,4]
```

#### 核心思路
```
1. 创建哑节点：使用dummy作为合并后链表的头节点的前驱，简化头部处理。
2. 初始化当前指针：用current指向当前合并链表的最后一个节点。
3. 迭代合并：比较list1和list2的值，选择较小的节点接入合并链表，并移动相应的指针。
4. 处理剩余部分：将未遍历完的链表直接接入合并链表的末尾。
5. 返回结果：返回dummy.next，即合并后的链表头节点。
```

#### 代码
```python
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        # 创建一个哑节点作为合并后链表的头节点的前驱节点
        dummy = ListNode(-1)
        current = dummy
        
        # 当两个链表都不为空时，进行迭代合并
        while list1 and list2:
            if list1.val <= list2.val:
                # 如果list1的值较小或相等，将其节点接入合并链表
                current.next = list1
                list1 = list1.next
            else:
                # 如果list2的值较小，将其节点接入合并链表
                current.next = list2
                list2 = list2.next
            # 移动当前指针到合并链表的最后一个节点
            current = current.next
        
        # 将未遍历完的链表直接接入合并链表的末尾
        current.next = list1 if list1 else list2
        
        # 返回合并后的链表头节点
        return dummy.next
```

### [94. 二叉树的中序遍历](https://leetcode.cn/problems/binary-tree-inorder-traversal/description/?envType=study-plan-v2&envId=top-100-liked)

#### 题目描述

给定一个二叉树的根节点 `root` ，返回 _它的 **中序** 遍历_ 。

示例：

![](https://assets.leetcode.com/uploads/2020/09/15/inorder_1.jpg)

```
输入：root = [1,null,2,3]
输出：[1,3,2]
```
#### 核心思路
【方法一：递归】
```
中序遍历的顺序是先遍历左子树，然后访问根节点，最后遍历右子树。

1. 主方法：
   - 初始化一个空列表 result 用于存储遍历结果。
   - 调用递归辅助方法 inorder，传入根节点和结果列表。
   - 返回结果列表。

2. 递归辅助方法 inorder：
   - 检查当前节点是否为空，如果为空则返回。
   - 递归遍历当前节点的左子树。
   - 将当前节点的值添加到结果列表中。
   - 递归遍历当前节点的右子树。
   - 返回结果列表。
```
【方法二：栈】
```
1. 初始化一个栈和一个结果列表。
2. 使用一个循环，只要当前节点或栈不为空，就继续遍历。
3. 在循环中，将当前节点的所有左子节点压入栈中，直到没有左子节点为止。
4. 弹出栈顶节点，访问该节点并将它的值加入结果列表。
5. 将当前节点更新为刚访问节点的右子节点，继续遍历右子树。
```

#### 代码
【方法一：递归】
```python
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        result = []  # 初始化一个空列表来存储遍历结果
        self.inorder(root, result)  # 调用辅助递归函数进行中序遍历
        return result  # 返回遍历结果列表
    
    def inorder(self, root, result):
        if not root:  # 如果当前节点为空，直接返回
            return
        self.inorder(root.left, result)  # 递归遍历左子树
        result.append(root.val)  # 将当前节点的值加入结果列表
        self.inorder(root.right, result)  # 递归遍历右子树
        return result  # 返回结果列表（虽然这个返回值在主方法中没有用到）
```
【方法二：栈】
```python
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        stack, result = [], []  # 初始化栈和结果列表
        
        while root or stack:  # 当根节点不为空或栈不为空时循环
            while root:  # 遍历到当前子树的最左节点
                stack.append(root)  # 将当前节点入栈
                root = root.left  # 移动到左子节点
            
            root = stack.pop()  # 弹出栈顶节点（最左节点）
            result.append(root.val)  # 将弹出节点的值加入结果列表
            root = root.right  # 移动到右子节点
        
        return result  # 返回结果列表
```

### [104. 二叉树的最大深度](https://leetcode.cn/problems/maximum-depth-of-binary-tree/description/?envType=study-plan-v2&envId=top-100-liked)

#### 题目描述
给定一个二叉树 `root` ，返回其最大深度。
二叉树的 **最大深度** 是指从根节点到最远叶子节点的最长路径上的节点数。

示例：

![](https://assets.leetcode.com/uploads/2020/11/26/tmp-tree.jpg)

```
输入：root = [3,9,20,null,null,15,7]
输出：3
```

#### 核心思路
```
【方法一：递归】
终止条件：如果当前节点（root）为空，说明我们已经到达了树的末端，这时候返回高度为0。
递归拆解：对每个节点，二叉树的最大深度等于其左子树和右子树中较大的那个深度再加上1。

【方法二：层序遍历】
初始检查：如果根节点为空，则直接返回深度0。
初始化：使用一个队列来进行广度优先搜索（BFS），将根节点放入队列，并初始化最大深度为1。
BFS遍历：
  - 每次从队列中取出当前层的所有节点，检查它们的左子节点和右子节点，如果存在则加入队列。
  - 每处理完一层后，增加最大深度计数器。
结束条件：当队列为空时，表示已遍历完所有节点，返回最大深度。
```
#### 代码
【方法一：递归】
```python
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        return max(self.maxDepth(root.left)+1,self.maxDepth(root.right)+1)
```

【方法二：广度优先搜索】
```python
class Solution:
    # 使用广度优先搜索 (BFS) 计算最大深度
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0  # 空树的深度为0
        
        maxDepth = 1  # 初始最大深度为1（至少有根节点）
        queue = deque()  # 创建一个队列来进行BFS遍历
        queue.append(root)  # 将根节点添加到队列中
        
        while len(queue):  # 当队列不为空时，继续遍历
            length = len(queue)  # 获取当前层的节点数量
            for i in range(length):
                head = queue.popleft()  # 弹出当前层的第一个节点
                if head.left is not None:  # 如果左子节点存在，加入队列
                    queue.append(head.left)
                if head.right is not None:  # 如果右子节点存在，加入队列
                    queue.append(head.right)
            if len(queue) == 0:  # 如果队列为空，说明已遍历完所有节点
                break
            maxDepth += 1  # 每处理完一层，最大深度加1
        
        return maxDepth  # 返回最终的最大深度
```

### [226. 翻转二叉树](https://leetcode.cn/problems/invert-binary-tree/description/?envType=study-plan-v2&envId=top-100-liked)

#### 题目描述
给你一棵二叉树的根节点 `root` ，翻转这棵二叉树，并返回其根节点。

示例：
![](https://assets.leetcode.com/uploads/2021/03/14/invert1-tree.jpg)

```
输入：root = [4,2,7,1,3,6,9]
输出：[4,7,2,9,6,3,1]
```

#### 核心思路
```
【方法一：DFS（递归）】
- 检查根节点是否为空：如果根节点为空，则直接返回，因为空树不需要翻转。
- 递归翻转左右子树：递归调用翻转函数，先翻转左子树，再翻转右子树。确保每棵子树都被正确翻转。
- 交换左右子节点：在递归处理完左右子树之后，将当前节点的左子节点和右子节点进行交换。
- 返回翻转后的根节点：最后返回翻转后的根节点，以保证整棵树从上到下都是翻转后的结构。

【方法二：BFS（队列）】

构造队列queue，对于root根节点，若root为空，则直接返回root；若root不为空，则将其入队。

在队列不为空时，开始循环弹出队列队首元素。head=queue.pop(）。然后交换head
的左右节点，若head的左右节点不为空，则将其入队。

最后，返回root。
```

#### 代码
【方法一：DFS（递归）】
```python
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        def exchange(root):
            if not root:
                return 
            root.left,root.right=root.right,root.left
            exchange(root.left)
            exchange(root.right)
        exchange(root)
        return root
```
【方法二：BFS（队列）】
```python
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root:
            return root        
        # 初始化队列并将根节点入队
        queue = deque([root])
        # 当队列不为空时，进行广度优先遍历
        while queue:
            # 弹出队首元素
            head = queue.popleft()
            # 交换左右子节点
            head.left, head.right = head.right, head.left
            # 将非空的左右子节点分别入队
            if head.left:
                queue.append(head.left)
            if head.right:
                queue.append(head.right)
        # 返回反转后的树的根节点
        return root
```

### [543. 二叉树的直径](https://leetcode.cn/problems/diameter-of-binary-tree/description/?envType=study-plan-v2&envId=top-100-liked)

#### 题目描述
给你一棵二叉树的根节点，返回该树的 **直径** 。
二叉树的 **直径** 是指树中任意两个节点之间最长路径的 **长度** 。这条路径可能经过也可能不经过根节点 `root` 。
两节点之间路径的 **长度** 由它们之间边数表示。

示例：
![](https://assets.leetcode.com/uploads/2021/03/06/diamtree.jpg)

```
输入：root = [1,2,3,4,5]
输出：3
解释：3 ，取路径 [4,2,1,3] 或 [5,2,1,3] 的长度。
```

#### 核心思路
```
1. 在每个节点计算通过该节点的左子树深度和右子树深度。
2. 计算通过该节点的路径长度（左子树深度 + 右子树深度）。
3. 更新全局最大直径。
4. 返回当前节点的深度（即 max(左子树深度, 右子树深度) + 1）。
```

#### 代码
```python
class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        maxDiameter = 0
        
        def depth(node: TreeNode) -> int:
            nonlocal maxDiameter
            if not node:
                return 0
            leftDepth = depth(node.left)
            rightDepth = depth(node.right)
            # 更新最大直径
            maxDiameter = max(maxDiameter, leftDepth + rightDepth)
            
            # 返回该节点的深度
            return max(leftDepth, rightDepth) + 1
        
        depth(root)
        return maxDiameter
```

### [108. 将有序数组转换为二叉搜索树](https://leetcode.cn/problems/convert-sorted-array-to-binary-search-tree/description/?envType=study-plan-v2&envId=top-100-liked)

#### 题目描述
给你一个整数数组 `nums` ，其中元素已经按 **升序** 排列，请你将其转换为一棵 平衡 二叉搜索树。

> 平衡二叉树是一种特殊的二叉搜索树，其中任意节点的左右子树高度差不超过1。它可以保证在最坏情况下的查找、插入和删除操作的时间复杂度都是 $O(\log n)$ 级别。
> 平衡二叉树常用的实现方式有红黑树、AVL树、Treap等。

示例：
![](https://assets.leetcode.com/uploads/2021/02/18/btree1.jpg)
```
输入：nums = [-10,-3,0,5,9]
输出：[0,-3,9,-10,null,5]
解释：[0,-10,5,null,-3,null,9] 也将被视为正确答案
```
#### 核心思路
```
1. 选择中间元素作为根节点：升序数组的中间元素自然地成为根节点，因为它可以将数组分成两部分，左边部分的所有元素都小于它，右边部分的所有元素都大于它。
2. 递归构建左右子树：
  - 对于根节点的左子树，从数组的开始位置到中间位置-1的部分，递归执行相同的过程。
  - 对于根节点的右子树，从中间位置+1到数组的结束位置的部分，递归执行相同的过程。
3. 停止条件：当子数组为空时，返回None。
```

#### 代码
```python
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        # 辅助函数，将子数组转换为BST
        def helper(left: int, right: int) -> Optional[TreeNode]:
            if left > right:
                return None
            # 选择中间元素作为根节点
            mid = (left + right) // 2
            root = TreeNode(nums[mid])
            # 递归构建左子树和右子树
            root.left = helper(left, mid - 1)
            root.right = helper(mid + 1, right)
            return root        
        return helper(0, len(nums) - 1)
```


### [35. 搜索插入位置](https://leetcode.cn/problems/search-insert-position/description/?envType=study-plan-v2&envId=top-100-liked)

#### 题目描述
给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。
请必须使用时间复杂度为 `O(log n)` 的算法。

示例：

```
输入：nums = [1,3,5,6], target = 5
输出：2

输入：nums = [1,3,5,6], target = 7
输出：4
```

#### 核心思路
```
1. 初始化：设置两个指针 left 和 right 分别指向数组的起始和结束位置。
2. 二分查找：
    - 计算中间索引 mid = left + (right - left) // 2。
    - 如果 nums[mid] 等于目标值 target，返回 mid。
    - 如果 nums[mid] 大于目标值，更新 right 为 mid - 1。
    - 如果 nums[mid] 小于目标值，更新 left 为 mid + 1。
3. 返回插入位置：如果没有找到目标值，循环结束后 left 即为目标值将要被插入的位置。
```

#### 代码
```python
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums) - 1      
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] > target:
                right = mid - 1
            else:
                left = mid + 1
        return left
```


### [20. 有效的括号](https://leetcode.cn/problems/valid-parentheses/description/?envType=study-plan-v2&envId=top-100-liked)

#### 题目描述
给定一个只包括 `'('`，`')'`，`'{'`，`'}'`，`'['`，`']'` 的字符串 `s` ，判断字符串是否有效。

有效字符串需满足：

1. 左括号必须用相同类型的右括号闭合。
2. 左括号必须以正确的顺序闭合。
3. 每个右括号都有一个对应的相同类型的左括号。

示例：

```
输入：s = "()"

输出：true

输入：s = "()[]{}"

输出：true
```

#### 核心思路
```
遍历s字符串：
- 当s[i]为左括号时，将s[i]压栈
- 当s[i]为右括号时，分为两种情况：
	- 若栈为空，则栈顶元素无法弹出与s[i]匹配，返回False；
	- 若栈不为空，则弹出栈顶元素与s[i]匹配。
		- 当栈顶元素与s[i]匹配时，则继续遍历字符串中下一个元素；
		- 当栈顶元素和s[i]不匹配时，则返回False

- 时间复杂度：由于压栈和弹栈操作的时间复杂度均为O(1)，因此本题的时间复杂度取决于对字符串的遍历，所以为O(n)
```

#### 代码
```python
class Solution:
    def isValid(self, s: str) -> bool:
        stack=[]
        bracketsMatch={"(":1,"[":2,"{":3,"}":4,"]":5,")":6}
        for i in range(len(s)):
            #若为左括号，则入栈
            if bracketsMatch[s[i]]<=3:
                stack.append(s[i])
            else:#若为右括号
                #首先，若栈此时为空，则return false
                if len(stack)==0:
                    return False
                else:
                    if bracketsMatch[s[i]]+bracketsMatch[stack.pop()]!=7:
                        return False
        return True if len(stack)==0 else False
```

### [121. 买卖股票的最佳时机](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock/description/?envType=study-plan-v2&envId=top-100-liked)

#### 题目描述

给定一个数组 `prices` ，它的第 `i` 个元素 `prices[i]` 表示一支给定股票第 `i` 天的价格。

你只能选择 **某一天** 买入这只股票，并选择在 **未来的某一个不同的日子** 卖出该股票。设计一个算法来计算你所能获取的最大利润。

返回你可以从这笔交易中获取的最大利润。如果你不能获取任何利润，返回 `0` 。

示例：
```
输入：[7,1,5,3,6,4]
输出：5
解释：在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。
     注意利润不能是 7-1 = 6, 因为卖出价格需要大于买入价格；同时，你不能在买入前卖出股票。
```

#### 核心思路

```
【贪心算法】

1. 保持最低价格：记录到目前为止的最低股价 `min_price`。
2. 计算最大利润：
   - 对于每一天的股价，计算如果在这一天卖出股票的利润，即 `prices[i] - min_price`。
   - 不断更新最大利润 `max_profit`。

总结：通过维护一个最低买入价和当前最大利润，每天更新这两个数值来计算最大可能利润。
```

#### 代码

```python
class Solution(object):
    def maxProfit(self, prices):
        if len(prices) == 1:
            return 0
        min_price = prices[0]
        max_profit = 0
        for i in range(1, len(prices)):
            min_price = min(min_price, prices[i])
            max_profit = max(max_profit, prices[i] - min_price)
        
        return max_profit
```

### [70. 爬楼梯](https://leetcode.cn/problems/climbing-stairs/description/?envType=study-plan-v2&envId=top-100-liked)

#### 题目描述

假设你正在爬楼梯。需要 `n` 阶你才能到达楼顶。

每次你可以爬 `1` 或 `2` 个台阶。你有多少种不同的方法可以爬到楼顶呢？

示例：
```
输入：n = 2
输出：2
解释：有两种方法可以爬到楼顶。
1. 1 阶 + 1 阶
2. 2 阶
```

#### 核心思路

```

1. 定义状态：用数组 dp[i] 表示爬到第 i 个台阶的方法数。
2. 初始化：
    - 爬 0 个台阶只有一种方法（即不动），所以 dp[0] = 1。
    - 爬 1 个台阶也只有一种方法（直接爬上去），所以 dp[1] = 1。
3. 状态转移方程：
    - 要爬到第 i 个台阶，可以从第 i-1 个台阶爬 1 步到达，也可以从第 i-2 个台阶爬 2 步到达。
    - 因此，dp[i] = dp[i-1] + dp[i-2]。
4. 时间复杂度：
    - 由于每个状态只依赖前两个状态，因此时间复杂度为 O(n)。
```

#### 代码
```python
class Solution:
    def climbStairs(self, n: int) -> int:
        dp = [0] * (n + 1)
        dp[0], dp[1] = 1, 1
        for i in range(2, n + 1):
            dp[i] = dp[i - 1] + dp[i - 2]
        return dp[n]
```

### [118. 杨辉三角](https://leetcode.cn/problems/pascals-triangle/description/?envType=study-plan-v2&envId=top-100-liked)

#### 题目描述
给定一个非负整数 `numRows`，生成「杨辉三角」的前 `numRows` 行。

在「杨辉三角」中，每个数是它左上方和右上方的数的和。

![](https://pic.leetcode-cn.com/1626927345-DZmfxB-PascalTriangleAnimated2.gif)

示例：

```
输入: numRows = 5
输出: [[1],[1,1],[1,2,1],[1,3,3,1],[1,4,6,4,1]]
```

#### 核心思路

```
1. 初始化：
    - 创建一个长度为 numRows 的二维列表 triangle。
    - 每行 i 都初始化为有 i+1 个元素的列表，其中第一个和最后一个元素都设为1，因为杨辉三角的两侧边界上的元素都是1。
2. 填充中间部分： 
    - 从第三行开始（即索引为2），对每一行的中间元素进行动态规划计算。
    - 对于 triangle[i][j]，它等于上一行的两个元素之和：triangle[i-1][j-1] + triangle[i-1][j]。
3. 返回结果： 
    - 最后返回整个 triangle。
```

#### 代码
```python
class Solution:
    def generate(self, numRows: int) -> List[List[int]]:
        if numRows <= 0:
            return []
        # Initialize the first row
        triangle = [[1]]
        
        for i in range(1, numRows):
            row = [1] * (i + 1)  # First step: initialize current row with 1s
            for j in range(1, i):  # Second step: fill the middle elements
                row[j] = triangle[i-1][j-1] + triangle[i-1][j]
            triangle.append(row)

        return triangle
```


### [136. 只出现一次的数字](https://leetcode.cn/problems/single-number/description/?envType=study-plan-v2&envId=top-100-liked)

#### 题目描述

给你一个 **非空** 整数数组 `nums` ，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。

你必须设计并实现线性时间复杂度的算法来解决此问题，且该算法只使用常量额外空间。

示例：

```
输入：nums = [2,2,1]
输出：1
```

#### 核心思路
```
异或运算的性质
1. 任何数与0异或为其本身： (a \oplus 0 = a)
2. 任何数与其自身异或为0： (a \oplus a = 0)

利用异或运算性质，我们可以在O(n)时间复杂度和O(1)空间复杂度内找到只出现一次的数字。
因为对于任意两个相同的数，异或后结果为0。而0与任何数异或结果为该数本身。所以，所有成对出现的数在异或后都抵消为0，最终剩下的就是那个只出现一次的数。

具体步骤如下：
1. 初始化变量result为0。
2. 遍历数组，对每个元素执行异或运算并更新result。
3. 最终，result的值即为只出现一次的那个数字。
```

#### 代码
```python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        result = 0
        for num in nums:
            result ^= num
        return result
```

### [169. 多数元素](https://leetcode.cn/problems/majority-element/description/?envType=study-plan-v2&envId=top-100-liked)

#### 题目描述

给定一个大小为 `n` 的数组 `nums` ，返回其中的多数元素。多数元素是指在数组中出现次数 **大于** `⌊ n/2 ⌋` 的元素。

你可以假设数组是非空的，并且给定的数组总是存在多数元素。

示例：

```
输入：nums = [3,2,3]
输出：3
```

#### 核心思路
```
【方法一：哈希表】

1. 记录次数：
    - 使用一个哈希表 counts 来记录每个元素出现的次数。
    - 遍历数组 nums，对每个元素进行统计。
	    - 如果元素已经在哈希表中，则其对应的计数器加1；
	    - 如果不在，则初始化该元素的计数器为1。
2. 查找多数元素：
    - 遍历哈希表中的所有键值对，找到出现次数大于 [n/2] 的元素，并返回该元素。

时间复杂度：O(n)，因为我们需要遍历数组一次来填充哈希表，然后再遍历哈希表一次来找到多数元素。
空间复杂度：O(n)，因为最坏情况下，需要存储数组中所有不同的元素的计数。

【方法二：摩尔投票】

1. 候选者和计数器： 
    - 维护一个变量 candidate 用于跟踪当前的候选多数元素，以及一个 count 计数器来表示 candidate 在目前遇到的元素中的净出现次数。
2. 遍历数组： 
    - 如果 count 为 0，说明我们需要更换候选者，将当前元素设为新的候选者，并将 count 设为 1。
    - 如果当前元素等于 candidate，则 count 加 1。
    - 如果当前元素不等于 candidate，则 count 减 1。
3. 最终候选者：
    - 遍历完成后，candidate 所指的元素就是数组的多数元素。

时间复杂度：O(n)，因为我们只需要一次遍历数组。
空间复杂度：O(1)，只使用了常数级别的额外空间。
```

#### 代码
【方法一：哈希表】
```python
class Solution:
	def majorityElement(self, nums: List[int]) -> int:
	    counts = {}
	    for num in nums:
	        if num in counts:
	            counts[num] += 1
	        else:
	            counts[num] = 1

	    for key, value in counts.items():
	        if value > len(nums) // 2:
	            return key
```

【方法二：摩尔投票】
```python
class Solution:
	def majorityElement(self, nums: List[int]) -> int:
	    candidate = None
	    count = 0
	
	    for num in nums:
	        if count == 0:
	            candidate = num
	        count += (1 if num == candidate else -1)
	
	    return candidate
```

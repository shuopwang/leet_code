---
description: 菜鸟刷编程题
---

# Leetcode 刷题

子序列和子字符串有区别，子字符串（子串）必须连续

最小生成树：在联通网的所有生成树中，所有边的代价和最小的生成树，称为最小生成树。每次选最小代价的边加入树中。

python相关问题

```text
E(x) = sum(x . P(x))
Var(x)=sum(x1-E(x)^2,x2-E(x)^2,...(xn-E(x)^2)).1/n
```

两个大集合求交集。将一个集合存到hashset中，另外一个也存到hash中如果发生冲突，说明有交集+1

装饰器：封装一个函数，令函数在执行前后，执行几个相同的任务。装饰器封装一个函数，用一定的方式修改它的行为。需要装饰器的函数，只需要在定义的上方加上一个@装饰器函数 就可以了。在装饰器内部加上@wraps\(\)，可以不改写被装饰的函数的名字和注释文档。装饰器还可以有装饰器方法和装饰器类

with使用对资源进行访问的场合，确保即使发生异常也能执行必要的清理操作，释放资源。

生成器：就是用于迭代操作的对象，不是一次性加载所有数据，而是延迟计算。把列表的\[\]改成（），一边循环一边计算，用next可以得到生成器的下一个值，还是可以用for来进行迭代的，遇到yield返回，再次执行从上次返回的yield继续执行。

列表推导式:\[表达式 for 变量 in 列表\]／\[表达式 for variable in list if condition\]

字典的update方法更新字典的值

==比较的是值大小是否相同，is就是对象地址是否相同

cmp对python2，先比长度，再比key值，最后比value

operator.eq

list用长度可变的数组实现，开始时分配一定量的内存空间

dictionary：hashtable实现

动态规划与分治法的区别：分治法是把原问题分解乘一系列的子问题。递归求解各个子问题，然后将子问题的结果合并成原问题的答案

动态规划：适合各个子问题包括公共的问题，也就是子问题重叠的适合比较好。动态规划只需要求解一次，而分治可能需要重复求解子问题。找到子问题的最优解之后，逐步推广。

贪心在找到子问题的最优解之后，每次选择当前最优的情况

中缀表达式:正常的表达式，后缀表达式：操作符在操作的数字之后。

中缀边后缀:1.数字则直接加入list。2。遇到运算符则比较与栈顶元素的优先级，如果优先级小于等于栈顶元素则将栈顶元素出栈，直至栈顶元素优先级小于新的运算符。3.若已经没有数字则考虑将运算符号依次pop

二叉树：每个节点最多含有两个子树的树称为二叉树

完全二叉树：除了最后一层，其余各层的节点数目均已达到最大值。且最后一层节点从左到右紧密排列。

满二叉树：所有叶子结点都在最底层的完全二叉树。2^k-1个总节点

二叉搜索树：左&lt;根&lt;右

平衡二叉树：左右子树也是平衡树且深度差为1

#### python 倒序遍历数组

```text
for x in range(len(array)-1,-1,-1):
           print array[x]
```

#### 在用python刷题的时候切记用tmp\[:\]来添加,li\[:\] 是生成一个 list 完全拷贝的一个简写

**生成二维数组：**

```text
graph=[[0 for  in range(cols)] for  in range(rows)]
```

#### 1.two sum

**15 3 sum一样的套路**

利用hash map存储target-A\[i\] 做为key，index做为value，若遍历数组时A\[i\]出现在hash map中则说明有对应的值，返回i, hashmap\(key\)

第二种做法，利用两个指针左右控制，如果之和等于target，那么就向下走，遇到重复就skip，防止出现重复的方法。 

3sum

```text
for:
    i=k
    j=len(nums)-1
    while i<j:#skip the reprete. find the sum equals the -nums[i]
```

#### 3sum closest

思路和3sum是一样的。在while里面计算k，i，j之和如果和之差小于diff那么就更新，如果大于target说明right太大，right-=1，如果小于target说明left太小，left+=1

#### 3. Longest Substring Without Repeating Characters

既然要求只是子串没有重复字符，那么只要利用两个变量来维护就可以了，lenth, substring，若有重复字符，则substring重新置空，并添加那个字符，比较length和当前子串长度大小

#### 5. Longest Palindromic Substring

最长回文串问题。采用DP求解。回文串的意思是当字符串为奇数时，中间元素唯一两边元素相同，偶数是则两边全部相同。考虑将字符串看为一个二维数组DP。若DP\[i\]\[j\]=True,则说明stirng A\[i\] = A\[j\]。显然这样一个二维数组时对角元素DP\[i\]\[i\]均为True，且若相邻字符相同则有DP\[i\]\[i+1\]=True。在判断其余元素时，则只有当s\[i\] == s\[j\] && dp\[i + 1\]\[j - 1\] 时才有有可能时回文

```text
for j in range(n): #up to the element j
    for i in range(0, j-1):# 0 to j-1
        if s[i] == s[j] and DP[i+1][j-1]:
            DP[i][j] = True
```

**6. ZigZag Conversion**

这题考验的是，朋友你是不是一个机智的小少年，考虑两个边界条件，如果移动至第一行或者最后一行时，则需要交换到下一个column中，移动到最后一行则相当于row-=1，移动到了第一行则考虑row+=1

```text
for i in range(len(s)):
     res[row]+=s[i]
     row+=direction
     if row == numRows-1 or row==0:
         direction *= -1
```

#### 9. Palindrome Number

只需要检验反转之后的number是否等于原number即可

#### 10. Regular Expression Matching

用二维数组DP来存储截止到i，j的情况

```text
for i in xrange(0, m+1):
    for j in xrange(1, n+1):
        if p[j-1] == '*':
        #当p第j个字符是*，那么有两种情况
        #第一种情况，用了*号，那么就判断前面字符是否相同或者p为. aa and a*，此时i=2，j=3，
        #此时相当于匹配上了s的上一个字符DP[i-1][j]
        #第二种情况*但是没有用，就看DP[i][j-2]
            dp[i][j] = dp[i][j-2] or ( i>0 and (s[i-1] == p[j-2] or p[j-2] == '.') and dp[i-1][j])
        else:
        #当向上看一位字符s，p相同时，或者当其中一个是.时满足关系
        #DP[i][j]=DP[i-1][j-1]
            dp[i][j] = i>0 and dp[i-1][j-1] and (s[i-1] == p[j-1] or p[j-1] == '.')
return dp[m][n]
```

#### 11. Container With Most Water

用两个指针来控制i，j

```text
res=max(res,min(arr[i],arr[j])*(j-i))
if arr[i]>arr[j]:#则向左寻找有没有更高的
    j-=1
else:#向右寻找有没有更高的
    i+=1
```

#### 14. Longest Common Prefix

无脑查找即可

#### 在使用回溯的时候切记有一个pop的过程

#### 17. Letter Combinations of a Phone Number\#所有可能组合时不需要添加任何限制条件. 可以有111，112，113

#### 39. Combination Sum\#递归push入栈， 当这条路径走完全之后需要pop出去，这样之后在遍历时仍可以利用这个节点，可以重复使用同一个字符. 

example: A\[2,3,4\] target 8 valid\_ans: \[2,2,2,2\]\[2,2,4\]

```text
for (int i = start; i < candidates.size(); ++i)
     pass
```

#### 40. Combination Sum II\#此时由于我们的结果中每次遍历的答案顺序前后不重要，顺序不同也被认为时同一种解法，所以可以用range \(begin,len\(s\) \)用来避免重复使用同一个字符，此外为了防止1，1，2的重复出现，添加num\[i\]!=num\[i-1\]的判断,当然要求i&gt;start。

#### 77. Combinations：这个就是完全的组合了，k代表组合需要的位数

#### 78. Subsets:一样使用递归求解，dfs每一层代表是否使用其中某个数字

**90. SubsetsII**

39-78：共同特点就是各个子答案的长度不相同，采用dfs遍历的时候，每层考虑的是是否使用其中一个节点，所以在循环时的写的为\(start,end,i++\)每次遍历的时候start都会有变化, start=i+1 not start+=1

#### 46. Permutations\#此题要求各个字母不重复的全排列在乎顺序情况。此题虽然也是不重复出现的排列，但是字母出现的前后顺序意味着字符串的不同，所以在使用dfs时，因为字母可能有重复出现的情况，所以需要考虑用一个额外的数组来看这个节点是否已经遍历过，若还没遍历则继续走。

如果题目解法对字母出现顺序有要求则考虑用。全排列时visited复位为0是为了考虑这个点在其他时候可能还会用到。

```text
for i in arr:
    if visited[i]==1:
        continue
    visited[i]=1
    dfs()
    visited[i]=0
```

#### 47. Permutations II: 在46的基础上加入判断

#### nums\[i\]==nums\[i-1\]:continue

当需要排列的长度都统一时，可以直接用for i in arr

```text
if i>0 and nums[i]>nums[i-1] and visited[i-1]==0:
   continue
```

#### 131. Palindrome Partitioning: 添加判断只有子串是回文的情况才加到栈内

这些求组合的，需要遍历图的，大概都可以直接通过dfs的遍历来做。dfs的套路在于将节点存储在一个stack里，后进先出。core part在如下：

```text
def dfs():
    if start>end:
        append(tmp)
    for num in range(start,end):
        #could addd some condition
        if meet the condition:
            tmp.append(num)
            dfs(tmp,parameter)
            tmp.pop()
```

#### 129. Sum Root to Leaf Numbers

树的遍历，做dfs遍历这个树，记录路径的值，然后转换求和。树形采用递归的时候，请想清楚究竟这个function的递归目的是啥，不要盲目直接for d in dd

```text
def dfs():
    if not node:#节点本身为none
        return
    do something..
    if not node.left and not.node.right:
        return #为叶子节点
    dfs(...)
```

**130. Surrounded Regions以及第200题**

这些是关于图上用dfs做遍历的情况，只要在边界上找到了‘0‘进行遍历处理，先进行边界处理，如果这个这个‘0‘点已经被遍历过或者该点为1则不处理。上下左右四处跑递归起来。在图上的遍历不需要复位visited因为这个点被遍历之后我们将不再考虑再次遍历

```text
def dfs():
    if ...边界情况:
        pass
    do somthing:
    dfs(...)
    dfs(...)
    dfs(...)
```

**79. Word Search**

这个其实也可以看作是一个dfs求解问题，首先遍历全图的基础找到第一个word的首字母，然后就相当于在坐标\(i,j\)开始找这样一条路径了。用dfs会快一点，只要其中有一个条件到达末尾，即可中止。上下左右四处找。这样时没有错，但是要记得要增加一个visited数组来防止重复遍历路径，在visited同时上下左右找完，需要reset visited数组，因为此时这个节点又可以遍历了。

#### 51. N-Queens  N皇后问题，最重要的其实就是在判断没放一步棋子的时候是不是有效的。用一个数组长度为n记录A\[i\]=j第i行第j列放棋子

```text
def valid(self, col, n, record):
        for i in range(n):
        #检查第n行前摆放棋子的情况。
            if (record[i] == col) or (abs(record[i] - col) == abs(n - i)):
            #检查第col是否在之前的棋子摆放中出现，或者他们处于斜对角的位置
            #满足一个则证明不是一个合法的情况
                    return False
        return True
```

#### 19. Remove Nth Node From End of List

剑指offer上有一样的题，和那题一样，用两个指针来控制，一个先前进n步，之后再让第二个指针同步前进，直至第一个指针到达尾部。最后判断一下这个n和linked list的长度关系

#### 20. Valid Parentheses

stack; 遇到左括号push，右括号pop

#### 21. Merge Two Sorted Lists

#### 23. Merge k Sorted Lists

合并k个，也是要在合并2个的基础上做呐，先将k个转为两半，l1,l2 合并l1 和l2即可，在这样的情况下， 相当于递归的合并l1 和l2。这样写的都感觉有点像快排了。

```text
def part(lists,l,r)
if len(lists)>2:
    mid=(len(lists))/2
    l1=part(lists[:mid],l,mid-1)
    l2=part(lists[mid:],mid,r)

def merge_2():
curr=head=listnode(0)
while t1 and t2:
    compare element
    pass
if t1:
    curr.next=t1
else
    pass
return head.next
```

#### 4. Median of Two Sorted Arrays

首先应该明确，中位数的概念，当数组长度为偶数时，中位数是指中间两个数的平均数，当数组长度为奇数时，中位数为中间个数的值。时间复杂度要求log\(m+n\)首先想到用二分查找

所以，当两个数组长度之和是一个偶数时，则通过一个funciton求出第\(m+n\)/2个数和\(m+n\)/2+1这两个数，中位数就是这个function返回数的平均值。同理奇数时，实际上只需要需要function第k个数，k为\(m+n\)/2

核心在于两个已排序数组如何找到其中第k个元素

```text
def findK(array1,array2,k):
    m=len(array1),n=len(array2)
    
    if m>n:findK(array2,array1,k)
    if m=0:return array2[k-1]
    
    if k=1: return min(array1[0],array2[0])
    
    i=min(m,k/2)
    j=min(n,k/2)
    if array1[i-1]>array2[j-1]:
        #array2 [0:j] will smaller than half
        findk(array1,array2[j:n],k-j)
    else:
        findk(array1[i:m],array2,k-i)
```

#### 22. Generate Parentheses. 

```text
#递归，觉得想成一个dfs也未尝不可，相当于只有这两个可选的节点
#for direction in [left,right]
left,right为目前剩下未使用的括号数目，tmp为一个临时变量用以存储零时的括号情况
def gene(left,right,res,tmp)
if left>right:return
if left=0 and right=0: res.append(tmp)#说明已经合法的搞完了
if left>0:
    gene(left-1,right,res,tmp+'(')
if right>0:
    gene(left,right-1,res,tmp+')')
```

#### 27 remove dup element:

```text
c=0
for i in range(len()):
    if arr[i]!=val:
        arr[c]=arr[i]
        i==1
```

#### 31. Next Permutation

这题现在才看懂，是这样的题，比如给你n个数，随意组合之后形成数字k，给出同样有这n个数组成的比k大的第一个数。如果是已经到达最高位，则返回最小。

如果从末尾往前看，数字逐渐变大，到A时才减小的，然后我们再从后往前找第一个比A大的数字，是B，那么我们交换A和B，再把此时B后面的所有数字转置一下即可

```text
for i in range(m-2,-1,-1):
    if arr[i+1]>arr[i]:
        for j in range(m-1,i,-1):
            if arr[j]>arr[i]:
                 break
         swap(nums[i],nums[j])
         nums[i+1:]=sorted(nums)
nums[:] = nums[::-1]#反转list，就是一个sorted的过程
#nums.sort()
```

#### 32. Longest Valid Parentheses

求极值问题一般想到DP或Greedy

一维的DP数组。DP\[i\]代表截止到string\[i\]时最长的valid parenthese。

```text
if s[i]=='(':
    left+=1
    continue
else:
     if s[i]==')' and left>0:
        DP[i+1]=DP[i]+2#对于嵌套的括号来说，现在的右括号相当于多了2
        DP[i+1]+=DP[i+1-DP[i+1]]#对于不是嵌套的来说，现在合法，加上之前合法的长达
        longest=max(longest,DP[i+1])
        left-=1
```

#### 33. Search in Rotated Sorted Array

用两个指针low和high来控制位置，这里采用二分查找的方法。这里既然是rotated sorted array那么就有可能这个顺序不是完全排好的。如果中间的数小于最右边的数，则右半段是有序的，若中间数大于最右边数，则左半段是有序的，我们只要在有序的半段里用首尾两个数组来判断目标值是否在这一区域内，这样就可以确定保留哪半边了。

```text
while low<high:
    mid=(low+high)/2
    if arr[mid]=target:return mid
    if arr[low]<arr[mid]:
        if arr[low]<target<arr[mid]:
              high=mid-1
        else:
              low=mid+1
    else:
        if arr[mid]<target<arr[high]:
            low=mid+1
        else:
            high=mid-1
```

#### 34. Find First and Last Position of Element in Sorted Array

同样是已经排好序的数组，这个时候需要找到某个值在数组中的位置范围。照例用二分查找。

```text
while low<high:
   mid=(low+high)/2
   if arr[mid]==target: find_range()
   elif <: low=mid+1
   elif >: high=mid-1

find_range():
   i,j control the value between mid, find the those index euqal mid
```

#### 38. Count and Say

相当于给你这样的规律，让你推导出直至n时数字该是啥样，就只要count重复次数和num，然后写下来即可

#### 48. Rotate Image

90度旋转图像，用于数据增强, 

180度只需要对面旋转即可

270在得到90度的图像之后对面旋转

```text
#对角线旋转
for i in range(m):
    for j in range(n-i):
        tmp=matrix[i][j]
        matrix[i][j]=matrix[m-1-j][n-1-i]
        matrix[m-1-j][n-1-i]=tmp    
#对面旋转
for i in range(int(m/2)):
    for j in range(n):
        tmp=matrix[i][j]
        matrix[i][j]=matrix[m-i-1][j]
        matrix[m-i-1][j]=tmp     
```

#### 53. Maximum Subarray

用两个量来控制local, global。采用DP的思想，local代表局部的最大值，global全局最大值。

```text
local=max(A[i],local+A[i])
global=max(local,global)
```

#### 55. Jump Game

贪心法，遍历全部arr，用一个值保存最远可到达点。若最远可到达点已大于最后则返回True

#### 45. Jump Game II

用两个变量reach and curr来控制，reach代表上次最远可到达的位置，curr代表此前最远可到达的位置，当遍历到reach时用reach=curr，step+=1

#### 62. Unique Paths \#有终点和起点的时候，就不要用bfs或者dfs，考虑使用二维的DP数组来看有多少的遍历情况。

#### 63. Unique Paths II

采用DP的思想，二维数组DP，用于存储到这个DP\[i\]\[j\]有多少种可能性。考虑，每个点可能来自两个方向。

```text
if i > 0 and j > 0:
    dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
elif i > 0:
    dp[i][j] = dp[i - 1][j]
elif j > 0:
    dp[i][j] = dp[i][j - 1]
```

#### 72. Edit Distance

编辑距离，编辑的能力在于对字符串的增删改，这道题也是采用DP的思想来做。一个二维的数组dp，其中dp\[i\]\[j\]表示从word1的前i个字符转换到word2的前j个字符所需要的步骤。

```text
if word1[i]=word2[j]:
    DP[i][j]=DP[i-1][j-1]
else:
    #要不然就是两个字符串i，j之前相同，到i，j不同则只需要看DP[i-1][j-1]
    #or DP[i-1][j]只需要改word1上的一个字符
    #or DP[i][j-1]只需要改word2上的一个字符
    #选择三者中最小的
    DP[i][j]=min(DP[i-1][j-1],min(DP[i-1][j],DP[i][j-1]))+1
```

#### 82. Remove Duplicates from Sorted List II

#### 83. Remove Duplicates from Sorted List

这两道题都是链表题，83对相同值的linked list只保留一次，82则是只要有重复就丢掉。幸运的事，两个链表都是已经排好序的链表，所以只要这么做。需要不再有重复的节点就考虑在循环的内部再构建一个循环把下面所有和它相等的直接跳过

```text
while cur:
    while cur.val=cur.next.val:
        cur=cur.next
    pre.next=cur
```

#### 88. Merge Sorted Array

合并两个有序数组

```text
while(m or n):
    get the bigger number
    own_index==1
    A[index]=bigger
    index+=1
```

#### 94. Binary Tree Inorder Traversal: 二叉树的中序遍历

先序，中序，后序遍历其实就是代表遍历时根元素的位置，先序：根左右，中序：左根右，后序：左右根

```text
def dfs():
    if not node:
        return None
    #in order
    dfs(node.left)
    res.append(node.val)
    dfs(node.right)
```

**95. Unique Binary Search Trees II**

#### 96. Unique Binary Search Trees\#如果直接用回溯法会造成超时，换个思路想，可以考虑用DP，DP\[i\]=DP\[j\]+DP\[i-j-1\]

#### 98. Validate Binary Search Tree \#根据binary search tree的特质来做这道题，严格要求根节点的值大于左节点，小于右节点即可。对于各层节点也采取这样的策略。

二叉查找树的特点是：一个节点的左子节点的关键字值小于这个节点，右子节点的关键字值大于或等于这个父节点。

此题在于给定一个整数n，求可以生成的二叉查找树

```text
def dfs(start,end):
    if start>end:
        return None
    for i in range(start,end):
        root=node(i)
        for left in dfs(start,i)
            for right in dfs(i,end)
                root.left=left
                root.right=right
                subtree.append(root)
        return subtree
```

**98. valid Binary Search Tree**

binary search tree: left &lt; root &lt; right

```text
if node.val > min and node.val <max:
return dfs(node.left,min,node.val)and dfs(node.right,node.val,max)
```

左子树节点val要小于root的值，

#### 102. Binary Tree Level Order Traversal

#### 103. Binary Tree Zigzag Level Order Traversal\#与102的大体上的套路时一样的，区别在于深度不同是添加方式倒转一下即可

bfs求解。bfs的是将树节点存储在队列里，先进先出。只要队列里还有值就继续进行遍历。

```text
queue=[root]
while queue:
    new_queue=[]
    for node in queue:
        if node.left:
            new_queue.append(left)
        if right:
            append(right)
    queue=new_queue
```

#### 105. Construct Binary Tree from Preorder and Inorder Traversal 先序和中序

这道题应该和下一题异曲同工，先序遍历头节点是根，控制一下顺序就好

**106. Construct Binary Tree from Inorder and Postorder Traversal 中序和后序**

根据后序性质，根节点在最后，那么只要找到根节点大小，则可在中序array中定位出left和right两部分子树，后序和中序左右子树长度相同，递归调用

```text
val=post[right]
for i in range(il,ir+1):
    if val=inorder[i]:break
node.left=function(inorder,il,i-1,post,left,left+i-il-1)
node.right=function(inorder,i+1,ir,post,left+i-il,il-1)
```

#### 108. Convert Sorted Array to Binary Search Tree

已经排过顺序的数组，中间的num肯定就是bst的中间节点。根据性质。

#### 109. Convert Sorted List to Binary Search Tree

思路与上一题一致，但是链表找中值的方法，需要掌握

```text
slow
fast
while fast and fast.next:
    fast=fast.next.next
    slow=slow.next
slow is the mid of the linked list
```

#### 111. Minimum Depth of Binary Tree

```text
left_depth=function(root.left)+1
right_depth=function(root.right)+1
if not right or not left:
    return max(left,right)
return min(left,right)
```

#### 112. Path Sum

```text
def function(node,sum):
    if not node:return False
    if node.val=sum reutrn True
    left=function(node.left,sum-node.val)
    right=fucntion(node.right,sum-node.val)
    reutrn left or right
```

#### 113. Path Sum II

需要打印相应路径时考虑用回溯法

```text
res=[]
def bt(res,tmp,node,sum):
    if not node:reutrn 
    if node.val=sum and left right are empty:
        res.append(tmp[:])
    tmp.append(node.val)
    bt(res,tmp,node.left,sum-node.val)
    bt(res,tmp,node.right,sum-node.val)
    tmp.pop()
```

#### 120. Triangle

给一个triangle，找到这样一条自顶向下的最短路径之和。求极值问题就考虑用DP,建立这样一个二维数组，每个点代表到该点的最短路径之和。最后只需要找到最后一层最小的即可

```text
for i in range(row):
    for j in range(column):
        DP[i][j]=min(DP[i-1][j],DP[i-1][j-1])+triangle[i][j]
```

#### 121. Best Time to Buy and Sell Stock

只允许买卖一次

```text
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        local_profit = 0
        global_profit = 0
        n = len(prices)
        for i in range(0, n - 1):
            local_profit = max(local_profit + prices[i + 1] - prices[i], 0)
            #print('local_profit : ', local_profit)
            global_profit = max(local_profit, global_profit)
            #print('global_profit: ', global_profit)
        return global_profit
```

#### 122. Best Time to Buy and Sell Stock II

可以多次买卖。那么利润最大就是只要当前比之后小，那么我们就吃进

#### 123. Best Time to Buy and Sell Stock III

最多两次，总利润最大。采用DP。DP\[i\]\[j\]:截止到第i项进行j次交易时的最大利润

#### 125. Valid Palindrome

检验是否为回文，采用两个指针的方法，left和right分别控制向右和左移动，如果不一样则证明不是回文

#### 139. Word Break

给定字符串和字典，判断是否可以分成字典中的词。

```text
DP[i]:截止到第i位是否满足条件
for i in range(len(s)):
    for word in dictionary:
        if i>len(word) and DP[i-len(word)] and s[i-len(word):i]==word:
            DP[i]=True
            break
```

#### 141. Linked List Cycle

判断链表里是否有环。两个指针，快慢指针。一个走两步一个走一步，如果相遇那么就说明有环

```text
fast=slow=head
while fast.next and fast.next.next:
    fast=fast.next.next
    slow=slow.next
    if fast==slow:
        return True
```

#### 142. Linked List Cycle IIarr

找到有环之后，令其中一个从头开始，两个指针相遇的位置就是第一个环头

#### 153. Find Minimum in Rotated Sorted Array

二分查找。旋转数组最重要的是记得，如果中间的数小于最右边的数，则右半段是有序的，若中间数大于最右边数，则左半段是有序的

```text
low=0
high=len(arr)-1
mid=(low+high)/2
```

#### 576. Out of Boundary Paths

动态规划问题。当N比较小的时候，BFS也可以求解。每次取决遇上一轮运行的情况。运行超过边界说明，需要累加走这一步之前的DP状态数目。DP\[i\]\[j\]代表上一轮在节点i，j的可以有的路径数目。走到本轮的时候，考虑是否超过边界，超过则累加走这一步之前的DP状态数目，未超过则给本轮的tmp+=DP

```text
for step in range(N):
    cur=DP
    DP=[[0 for _ in range(n)] for _ in range(m)]
    for x in range(m):
        for y in range(n):
            for dx,dy in [(1,0),(0,1),(-1,0),(0,-1)]:
                new_x=x+dx
                new_y=y+dy
                if 0<=new_x<m and 0<=new_y<n :
                    DP[x+dx][y+dy]=(cur[x][y]+DP[x+dx][y+dy])%(10**9+7)
                else:
                    ans=(ans+cur[x][y])%(10**9+7)
```

#### 688. Knight Probability in Chessboard

与上题的DP建立过程一致，DP\[i\]\[j\]均保证直到上一轮在节点i,j可以有的所有路径数目.最后需要遍历全表，求和除以总共的概率情况

注意如果在牛客网上刷题，遇到没有限定输入行数的情况，那么就请用：

```text
import sys
for line in sys.stdin:
    line=line.split()
#########################
#如果遇到输入整数
########################
m=int(raw_input())
########################
#整型数组
#######################
arr=map(int,raw_input().split())
#######################
#知道长度的数组输入
###################
steps = [[int(s) for s in raw_input().split()] for i in range(k)]

```

## 剑指offer

1.二维数组的查找。数组本是就是一个递增的

```text
rows = len(array) - 1
cols= len(array[0]) - 1
i = rows
j = 0
        while j<=cols and i>=0:
            if target<array[i][j]:
                i -= 1
            elif target>array[i][j]:
                j += 1
            else:
                return True
        return False
```

14.倒数链表输出第k个值

用两个指针来控制，保持两个指针的距离为k，先让第一个指针走k步之后再开始同时走两个指针。

```text
p=head
q=head
count=0
k=k
a=k
while p:
    p=p.next
    count++
    if count>k:
        q=q.next
return count < k ? null : q;
```

2.反转链表

拿节点，该节点的next应该是之前已经倒序完成的linkedlist,倒序完成的linkedlist即为该节点～

```text
newhead=None
while head:
      tmp = head.next
      head.next = newhead
      newhead = head
      head = tmp
```

4.重建二叉树

已知一棵树的中序和前序遍历重建二叉树。leetcode原题，利用性质，先序遍历第一个节点必然是根

```text
def rebuild(post,pl,pr,mid,ml,mr):
    if pr<pl or ml>mr:
        return None
    root_val=post[pl]
    for i in range(ml,mr+1):
        mid[i]=root_val:break
    root.left=rebuild(post,pl+1,pl+i-ml,mid,ml,i)
    root.right=rebuild(post,pl+i-ml+1,pr,i,mr)
```

5.用两个栈实现队列

```text
push:l1.append()
pop:l2.append(l1.pop())#整体转移令先进元素在栈顶
l2.pop()
l1.append(l2.pop())
```

6.旋转数组中的最小值

即找到第一个波底元素

7.找出第n项Fibonacci

用动态规划

```text
DP[n]=DP[n-1]+DP[n-2]
```

8.跳台阶问题 和第七题是一样的做法

```text
DP[n]=DP[n-1]+DP[n-2]
```

9.跳台阶，一口气可以跳n阶台阶

  
关于本题，前提是n个台阶会有一次n阶的跳法。分析如下:

f\(1\) = 1

f\(2\) = f\(2-1\) + f\(2-2\)         //f\(2-2\) 表示2阶一次跳2阶的次数。

f\(3\) = f\(3-1\) + f\(3-2\) + f\(3-3\) 

...

f\(n\) = f\(n-1\) + f\(n-2\) + f\(n-3\) + ... + f\(n-\(n-1\)\) + f\(n-n\) 

推导出f\(n\)=2\*f\(n-1\)

10. 矩阵覆盖，依然是DP求解

11。求整数的二进制数里有几个1

```text
while n!=0:
    count++
    n=n&(n-1)
```

12. 数的整数次方

相当于一个累乘的操作，注意当幂是负数的情况。

13.整数数组调整奇偶顺序，奇数字在前偶数在后

```text
i=0
while i <len(arr):
    while i<len(arr) and arr[i] %2==1:#找完一个序列的奇数
        i+=1
    j=i+1
    while j<len(arr) and arr[j]%2==0:
        j+=1
    if j<len(arr):
        tmp=arr[j]
        t=j
        while t>i:
            arr[t]=arr[t-1]
            t-=1
        arr[i]=tmp
    else:
        break
```

15合并两个有序链表

```text
tmp=new=ListNode(0)
merge
return new.next
```

16树的子结构

子结构不是子树。顶部调用时，考虑子结构有可能本身就和原本相同，或者是树的左子树或者是右子树

```text
if not tree_b:
    return True
if not tree_a or tree_a.val!=tree_b.val:
    return False
return subs(tree_a.left,tree_b.left) and subs(right,right)
```

17树的镜像

```text
right,left=left,right
mirror(right)
mirror(left)
```

18顺时针打印数组

19判断栈的入栈和出栈顺序问题,相当于用一个辅助栈不停加入元素，每次加入之后比对一下当前辅助栈的栈顶元素和弹栈元素是否相等，如果一直相等那么就一直pop这个元素。最后比较是否栈为空

```text
stack
for num in arr:
    stack.append(num)
    while stack and stack[-1]=pop[index]:
        index+=1
        stack.pop()
```

20二叉搜索树变有序列表

考虑二叉搜索树的性质，中序遍历之后即可得到有序数组。

```text
if left:
    while left.right:
        left=left.right
    root.left,left.right=left,root
same for the right
都有序了之后将root得到其第一位开始位置
```

21数组中出现次数最多的数

```text
count=0
n=arr[0]
for num in arr:
    if num==n:count+=1
    else:count-=1
    if count==0:n=num,count=1
回头做检测是否超过1／2
```

22数组的最小的k个数

利用快排中partition的思想

```text
def partition(arr, begin, end):
    pivot = arr[begin]
    i = begin
    j = end
    while i < j:
        while i < j and arr[j] >= pivot:
            j -= 1
        arr[i] = arr[j]
        while i < j and arr[i] < pivot:
            i += 1
        arr[j] = arr[i]
    arr[i] = pivot
    return i
```

23数组组合最小的数

递归求解所有组合然后min

24丑数

给2，3，5分别都有一个进行的索引。新的丑数必然是

```text
new=min(list[index2]*2,list[index3]*3,list[index5]*5)
用了哪个索引就把对应索引的引用计数加1
```

25逆序对

26两个链表的第一个公共节点。

计算链表长度，让长度长的先走长出来的长度，然后再挨个比较

27平衡二叉树：

计算左右子树深度，如果深度绝对值大于1那么就不是，如果==0则分别判断左子树内部和右子树内部是不是平衡二叉树。

28和为s的连续正数

```text
for i in range(0,arr/2+1):
    for j in range(i,arr/2+2):
        tmp=(i+j)(j-i+1)/2
```

29找到和为s乘积最小的两个数字

a+b=S,a和b越远乘积越小，而一头一尾两个指针往内靠近的方法找到的就是乘积最小的情况。如果是乘积最大的情况就是一直找到两个指针重合，每次找到一个就将之前返回的结果向量清空然后更新为新找到的。

30二叉树的下一个中序遍历的节点

给定了一个二叉树的节点，找到下一个中序遍历的节点。如果它有右子树，那么下一个中序遍历节点就是右子树的最左节点。如果它本身是父节点的最左子树，那么下一个节点就是父节点

31  
  
`所谓序列化指的是遍历二叉树为字符串；所谓反序列化指的是依据字符串重新构造成二叉树。    依据前序遍历序列来序列化二叉树，因为前序遍历序列是从根结点开始的。当在遍历二叉树时碰到Null指针时，这些Null指针被序列化为一个特殊的字符“#”。    另外，结点之间的数值用逗号隔开`

有序数组求并集：

两个index，

```text
a[i]==b[j]:i++,j++
a[i]>b[j]:j++
a[i]<b[j]:i++
```

3.顺子

考虑计算大小王的数目，两种情况，出现相同数字，则直接gg，第二种大小王不够导致不能是顺子

```text
count(zeros)
number[i]+count+1>=number[i+1]:
    count-=number[i+1]-number[i]-1
```


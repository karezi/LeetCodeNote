

# 算法笔记

> 来源：LeetCode&牛客网&剑指Offer

------

## [剪绳子](https://www.nowcoder.com/practice/57d85990ba5b440ab888fc72b0751bf8)

> dp、数学

### 解法1-dp

```java
public class Solution {
    public int cutRope(int target) {
        // dp[n] = max(dp[i]*dp[n-i]) n-[0,target]
        if (target == 2)
            return 1;
        else if (target == 3)
            return 2;
        else if (target == 4)
            return 4;
        int[] dp = new int[target + 1];
        dp[1] = 1;
        dp[2] = 2;
        dp[3] = 3;
        for (int i = 4; i <= target; ++i) {
            int max = 0;
            for (int j = 1; j <= i / 2; ++j) {
                int temp = dp[j] * dp[i - j];
                if (temp > max)
                    max = temp;
            }
            dp[i] = max;
        }
        return dp[target];
    }
}
```

### 解法2-数学

```java
public class Solution {
    public int cutRope(int target) {
        int res = 1;
        while(true) {
            if (target == 4) {
                res *= 4;
                break;
            } else if (target == 3) {
                res *= 3;
                break;
            } else if (target == 2) {
                res *= 2;
                break;
            } else {
                res *= 3;
                target -= 3;
            }
        }
        return res;
    }
}
```



## [机器人的运动范围](https://www.nowcoder.com/practice/6e5207314b5241fb83f2329e89fdecc8)

> 回溯法

```java
public class Solution {
    public int movingCount(int threshold, int rows, int cols) {
        boolean[][] visited = new boolean[rows][cols];
        return judge(threshold, rows, cols, 0, 0, visited);
    }
   	
    private int judge(int threshold, int rows, int cols, int i, int j, boolean[][] visited) {
      	// 超出限制
        if (i < 0 || j < 0 || i >= rows || j >= cols || visited[i][j] || bitSum(i) + bitSum(j) > threshold) {
            return 0;
        }
      	// 标记访问
        visited[i][j] = true;
      	// 上下左右试探
        return 1 + judge(threshold, rows, cols, i - 1, j, visited)
                 + judge(threshold, rows, cols, i, j - 1, visited)
                 + judge(threshold, rows, cols, i + 1, j, visited)
                 + judge(threshold, rows, cols, i, j + 1, visited);
    }
    // 计算每位和
    private int bitSum(int n) {
        int sum = 0;
        while(n > 0) {
            sum += n % 10;
            n /= 10;
        }
        return sum;
    }
}
```

## 矩阵中的路径

> 回溯法

```java
public class Solution {
    public boolean hasPath(char[] matrix, int rows, int cols, char[] str) {
        boolean[] visited = new boolean[matrix.length];
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                if (judge(matrix, rows, cols, str, i, j, 0, visited)) {
                    return true;
                }
            }
        }
        return false;
    }

    private boolean judge(char[] matrix, int rows, int cols, char[] str, int i, int j, int k, boolean[] visited) {
        int pos = i * cols + j;
        if (i < 0 || j < 0 || i >= rows || j >= cols || visited[pos] || matrix[pos] != str[k])
            return false;
        // 最后一个
        if (k == str.length - 1)
            return true;
        // 非最后一个，且上下左右满足条件
        visited[pos] = true;
        if (judge(matrix, rows, cols, str, i - 1, j, k + 1, visited) 
              || judge(matrix, rows, cols, str, i, j - 1, k + 1, visited)
              || judge(matrix, rows, cols, str, i + 1, j, k + 1, visited)
              || judge(matrix, rows, cols, str, i, j + 1, k + 1, visited)) {
            return true;
        }
        // 不满足，释放节点
        visited[pos] = false;
        return false;
    }

}
```

## [斐波那契数列](https://www.nowcoder.com/practice/c6c7742f5ba7442aada113136ddea0c3)

> 递推

```java
public class Solution {
    public int Fibonacci(int n) {
        int res = 0;
        if (n == 0)
            return 0;
        if (n == 1)
            return 1;
        int last1 = 0;
        int last2 = 1;
        for (int i = 2; i <= n; ++i) {
            res = last1 + last2;
            last1 = last2;
            last2 = res;
        }
        return res;
    }
}
```

## [变态跳台阶](https://www.nowcoder.com/practice/22243d016f6b47f2a6928b4313c85387)

> 递推

```java
public class Solution {
    public int JumpFloorII(int target) {
        if (target == 1)
            return 1;
        int res = 1;
        for (int i = 1; i < target; ++i) {
            res = 2 * res;
        }
        return res;
    }
}
```

## 206.翻转链表

> 双指针迭代/递归

### 解法1：双指针迭代

空间复杂度O(1)，时间复杂度O(n)

```java
public class ListNode {
    int val;
    ListNode next;
    ListNode(int x) { val = x };
}
class Solution {
		public ListNode reverseList(ListNode head) {
				ListNode curr = head;
				ListNode pre = null;
				while(curr != null) {
						ListNode temp = curr.next;
						curr.next = pre;
						pre = curr;
						curr = temp;
				}
				return pre;
		}
}
```

### 解法2：递归

空间复杂度O(n)，时间复杂度O(n)

```java
public class ListNode {
    int val;
    ListNode next;
    ListNode(int x) { val = x };
}
class Solution {
		public ListNode reverseList(ListNode head) {
				while(head == null || head.next == null)
          	return head;
      	ListNode p = reverseList(head.next);
      	head.next.next = head;
      	head.next = null;
      	return p;
		}
}
```

## 31.下一个排列

> 一遍扫描

```java
class Solution {
    public void nextPermutation(int[] nums) {
        int i = nums.length - 2;
        while(i >= 0 && nums[i + 1] <= nums[i]) i--; // 从后往前找到第一个小大
        if (i >= 0) {
            int j = nums.length - 1;
            while(j >= i && nums[j] <= nums[i]) j--; // 从后往前找到第一个大于i
            swap(nums, i, j);
        }
        reverse(nums, i + 1, nums.length - 1);
    }

    private void reverse(int[] nums, int l, int r) {
        while(l < r) {
            swap(nums, l, r);
            l++;
            r--;
        }
    }

    private void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
}
```

## 33.搜索旋转排序数组

> BS

时间复杂度O(logN)，空间复杂度O(1)

```java
class Solution {
    public int search(int[] nums, int target) {
        int l = 0, r = nums.length - 1;
        while(l <= r) {
            int mid = (l + r) / 2;
            if (target == nums[mid]) return mid;
            if (target > nums[mid]) {
                if (nums[l] > nums[mid]) {
                    // 右侧有序
                    if (target == nums[l]) return l;
                    if (target == nums[r]) return r;
                    if (target > nums[l]) {
                        // 必然在左侧
                        r = mid - 1;
                    } else {
                        // 必然在右侧
                        l = mid + 1;
                    }
                } else {
                    // 左侧有序，必然在右侧
                    l = mid + 1;
                }
            } else {
                if (nums[l] > nums[mid]) {
                    // 右侧有序，必然在左侧
                    r = mid - 1;
                } else {
                    // 左侧有序
                    if (target == nums[l]) return l;
                    if (target == nums[r]) return r;
                    if (target > nums[l]) {
                        // 必然在左侧
                        r = mid - 1;
                    } else {
                        // 必然在右侧
                        l = mid + 1;
                    }
                }
            }
        }
        return -1;
    }
}
```

## 70.爬楼梯

> dp、递归、循环数组、数学都可以

```java
import java.util.*;

class Solution {
    public int climbStairs(int n) {
        if (n == 1)
            return 1;
        else if (n == 2)
            return 2;
        else {
            int[] res = new int[2];
            for (int i = 3; i <= n; ++i) {
                res = last1 + last2;
                last1 = last2;
                last2 = res;
            }
            return res;
        }
    }
}
```

## [剑指 Offer 09. 用两个栈实现队列](https://leetcode-cn.com/problems/yong-liang-ge-zhan-shi-xian-dui-lie-lcof/)

> 队列、栈

```java
class CQueue {
    private Stack<Integer> stackPush;
    private Stack<Integer> stackPop;

    public CQueue() {
        stackPop = new Stack<>();
        stackPush = new Stack<>();
    }
    
    public void appendTail(int value) {
        stackPush.push(value);
    }
    
    public int deleteHead() {
        if (stackPop.isEmpty()) {
            while(!stackPush.isEmpty()) {
                stackPop.push(stackPush.pop());
            }
        }
        return stackPop.isEmpty() ? -1 : stackPop.pop();
    }
}

/**
 * Your CQueue object will be instantiated and called as such:
 * CQueue obj = new CQueue();
 * obj.appendTail(value);
 * int param_2 = obj.deleteHead();
 */
```

## [121. 买卖股票的最佳时机](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/)

> dp

```java
class Solution {
    public int maxProfit(int[] prices) {
        // dp[i][0] = max{dp[i-1][0], dp[i-1][1] + price[i]}
        // dp[i][1] = max{dp[i-1][1], - price[i]}
        int n = prices.length;
        int dp_i_0 = 0, dp_i_1 = Integer.MIN_VALUE;
        for (int i = 0; i < n; ++i) {
            dp_i_0 = Math.max(dp_i_0, dp_i_1 + prices[i]);
            dp_i_1 = Math.max(dp_i_1, -prices[i]);
        }
        return dp_i_0;
    }
}
```

## [122. 买卖股票的最佳时机 II](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/)

> dp

```java
class Solution {
    public int maxProfit(int[] prices) {
        // dp[i][0] = max{dp[i-1][0], dp[i-1][1]+prices[i]}
        // dp[i][1] = max{dp[i-1][1], dp[i-1][0]-prices[i]}
        int n = prices.length;
        int dp_i_0 = 0, dp_i_1 = Integer.MIN_VALUE;
        for (int i = 0; i < n; ++i) {
            int temp = dp_i_0;
            dp_i_0 = Math.max(dp_i_0, dp_i_1 + prices[i]);
            dp_i_1 = Math.max(dp_i_1, temp - prices[i]);
        }
        return dp_i_0;
    }
}
```

## [714. 买卖股票的最佳时机含手续费](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/)

> dp

```java
class Solution {
    public int maxProfit(int[] prices, int fee) {
        // 定义dp[i][0]为第i天不持有最大收益，dp[i][1]为第i天持有最大收益
        // dp[i + 1][0] = max(dp[i][0], dp[i][1] + prices[i + 1] - fee)
        // dp[i + 1][1] = max(dp[i][1], dp[i][0] - prices[i + 1])
        // 初始化：dp[1][0]= 0, dp[1][1]=prices[0]-fee
        // 求：dp[n][0]
        if (prices.length == 0)
            return 0;
        int dp_i_0 = 0, dp_i_1 = - prices[0];
        for (int i = 0; i < prices.length; ++i) {
            int tmp = dp_i_0;
            dp_i_0 = Math.max(dp_i_0, dp_i_1 + prices[i] - fee);
            dp_i_1 = Math.max(dp_i_1, tmp - prices[i]);
        }
        return dp_i_0;
    }
}
```



## [42. 接雨水](https://leetcode-cn.com/problems/trapping-rain-water/)

> 双指针

```java
class Solution {
    public int trap(int[] height) {
        int res = 0;
        int l = 0;
        int r = height.length - 1;
        int max_l = 0;
        int max_r = 0;
        while(l <= r) {
            if (max_l < max_r) {
                // 左最大<右最大，以左计算
                if (max_l > height[l]) {
                    // 左最大>左当前，累积
                    res += max_l - height[l];
                } else {
                    // 左最大<=左当前，更新最大值
                    max_l = height[l];
                }
                l++;
            } else {
                // 左最大>=右最大，以右计算
                if (max_r > height[r]) {
                    // 右最大>右当前，累积
                    res += max_r - height[r];
                } else {
                    // 右最大<=右当前，更新最大值
                    max_r = height[r];
                }
                r--;
            }
        }
        return res;
    }
}
```

## [162. 寻找峰值](https://leetcode-cn.com/problems/find-peak-element/)

> BS

时间复杂度：O(logN)

```java
class Solution {
    public int findPeakElement(int[] nums) {
        int l = 0, r = nums.length - 1;
        while(l < r) {
            int mid = (l + r) / 2;
            if (nums[mid] < nums[mid + 1]) {
                l = mid + 1;
            } else {
                r = mid;
            }
        }
        return l;
    }
}
```

## [1299. 将每个元素替换为右侧最大元素](https://leetcode-cn.com/problems/replace-elements-with-greatest-element-on-right-side/)

> 逆序遍历

```java
class Solution {
    public int[] replaceElements(int[] arr) {
        int n = arr.length;
        int[] max = new int[n];
        max[n - 1] = -1;
        if (n >= 2) {
            // 逆序遍历
            for (int i = n - 2; i >= 0; --i) {
                // 更新max数组：max(右侧的右侧最大值,右侧的值)
                max[i] = Math.max(max[i + 1], arr[i + 1]);
            }
        }
        return max;
    }
}
```

## [139. 单词拆分](https://leetcode-cn.com/problems/word-break/)

> dp

```java
class Solution {
    public boolean wordBreak(String s, List<String> wordDict) {
        // dp[i] = dp[j] && check(s[j, i])
        int maxw = 0; // 只需要遍历到最长单词即可
        // 用HashSet效率高
        Set<String> wordSet = new HashSet<>();
        for (String word: wordDict) {
            wordSet.add(word);
            maxw = Math.max(maxw, word.length());
        }
        boolean[] dp = new boolean[s.length() + 1];
        dp[0] = true;
        for (int i = 1; i <= s.length(); ++i) {
            for (int j = i; j >= 0 && i - j <= maxw; --j) {
                // 可以分割即可，可以提前终止
                if (dp[j] && wordSet.contains(s.substring(j, i))) {
                    dp[i] = true;
                    break;
                }
            }
        }
        return dp[s.length()];
    }
}
```

## [155. 最小栈](https://leetcode-cn.com/problems/min-stack/)

> 辅助栈

```java
class MinStack {
    private Stack<Integer> stack;
    private Stack<Integer> minStack;

    /** initialize your data structure here. */
    public MinStack() {
        stack = new Stack<>();
        minStack = new Stack<>();
        minStack.push(Integer.MAX_VALUE);
    }
    
    public void push(int x) {
        stack.push(x);
        minStack.push(Math.min(x, minStack.peek()));
    }
    
    public void pop() {
        stack.pop();
        minStack.pop();
    }
    
    public int top() {
        return stack.peek();
    }
    
    public int getMin() {
        return minStack.peek();
    }
}

/**
 * Your MinStack object will be instantiated and called as such:
 * MinStack obj = new MinStack();
 * obj.push(x);
 * obj.pop();
 * int param_3 = obj.top();
 * int param_4 = obj.getMin();
 */
```

## [108. 将有序数组转换为二叉搜索树](https://leetcode-cn.com/problems/convert-sorted-array-to-binary-search-tree/)

> 二分+递归

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public TreeNode sortedArrayToBST(int[] nums) {
        return helper(nums, 0, nums.length - 1);
    }

    private TreeNode helper(int[] nums, int left, int right) {
        if (left > right) {
            return null;
        }
        int mid = (left + right) / 2;
        TreeNode root = new TreeNode(nums[mid]);
        root.right = helper(nums, mid + 1, right);
        root.left = helper(nums, left, mid - 1);
        return root;
    }
}
```

## [160. 相交链表](https://leetcode-cn.com/problems/intersection-of-two-linked-lists/)

> 巧妙方法

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) {
 *         val = x;
 *         next = null;
 *     }
 * }
 */
public class Solution {
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        ListNode a = headA, b = headB;
        while(a != b) {
            a = a == null ? headB : a.next;
            b = b == null ? headA : b.next; 
        }
        return a;
    }
}
```

## [67. 二进制求和](https://leetcode-cn.com/problems/add-binary/)

> 模拟计算

```java
class Solution {
    public String addBinary(String a, String b) {
        int carry = 0;
        StringBuilder sb = new StringBuilder();
        for (int i = a.length() -1, j = b.length() - 1; i >= 0 || j >= 0; i--, j--) {
            int it = i >= 0 ? a.charAt(i) - '0' : 0;
            int jt = j >= 0 ? b.charAt(j) - '0' : 0;
            int sum = it + jt + carry;
            sb.append("" + sum % 2);
            carry = sum / 2;
        }
        if (carry != 0) {
            sb.append("1");
        }
        return sb.reverse().toString();
    }
}
```

## [110. 平衡二叉树](https://leetcode-cn.com/problems/balanced-binary-tree/)

> 递归

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public boolean isBalanced(TreeNode root) {
        return helper(root) != -1;
    }

    private int helper(TreeNode root) {
        if (root == null)
            return 0;
        int l = helper(root.left);
        int r = helper(root.right);
        if (l == -1 || r == -1)
            return -1;
        return Math.abs(l - r) < 2 ? Math.max(l, r) + 1 : -1;
    }
}
```

## [83. 删除排序链表中的重复元素](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list/)

> 双指针法

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public ListNode deleteDuplicates(ListNode head) {
        if (head == null)
            return head;
        ListNode pre = head.next;
        ListNode cur = head;
        while(pre != null) {
            while (pre != null && pre.val == cur.val) {
                pre = pre.next;
            }
            cur.next = pre;
            if (pre == null)
                break;
            else {
                cur = pre;
                pre = pre.next;
            }
        }
        return head;
    }
}
```

> 直接法

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public ListNode deleteDuplicates(ListNode head) {
        ListNode cur = head;
        while(cur != null && cur.next != null) {
            if (cur.next.val == cur.val) {
        	      cur.next = cur.next.next; // 跳过一个
            } else {
                cur = cur.next; // 直到不相等才下一个
            }
        }
        return head;
    }
}
```

## [543. 二叉树的直径](https://leetcode-cn.com/problems/diameter-of-binary-tree/)

> dfs、二叉树遍历、递归、树的高度；有陷阱

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    int max = 0;

    public int diameterOfBinaryTree(TreeNode root) {
        dfs(root);
        return max;
    }

    private int dfs(TreeNode root) {
        if (root == null)
            return 0;
        int l = dfs(root.left);
        int r = dfs(root.right);
        max = Math.max(max, l + r);
        return Math.max(l, r) + 1;
    }
}
```

## [415. 字符串相加](https://leetcode-cn.com/problems/add-strings/)

> 双指针模拟计算

```java
class Solution {
    public String addStrings(String num1, String num2) {
        StringBuilder sb = new StringBuilder();
        int ca = 0;
        for (int i = num1.length() - 1, j = num2.length() - 1; i >= 0 || j >= 0; --i, --j) {
            int one = i >= 0 ? num1.charAt(i) - '0' : 0;
            int two = j >= 0 ? num2.charAt(j) - '0' : 0;
            int bitSum = (one + two + ca) % 10;
            sb.append(bitSum);
            ca = (one + two + ca) / 10;
        }
        if (ca != 0)
            sb.append("1");
        return sb.reverse().toString();
    }
}
```

## [112. 路径总和](https://leetcode-cn.com/problems/path-sum/)

> 递归

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public boolean hasPathSum(TreeNode root, int sum) {
        if (root == null)
            return false;
        sum -= root.val;
        if (root.left == null && root.right == null) // 是子节点，观察是否扣减为0了
            return sum == 0;
        return hasPathSum(root.left, sum) || hasPathSum(root.right, sum);
    }
}
```

## [141. 环形链表](https://leetcode-cn.com/problems/linked-list-cycle/)

> 快慢指针

```java
/**
 * Definition for singly-linked list.
 * class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) {
 *         val = x;
 *         next = null;
 *     }
 * }
 */
public class Solution {
    public boolean hasCycle(ListNode head) {
        if (head == null || head.next == null)
            return false;
        ListNode fast = head.next;
        ListNode slow = head;
        while(fast != slow) {
            if (fast == null || fast.next == null)
                return false;
            fast = fast.next.next;
            slow = slow.next;
        }
        return true;
    }
}
```

## [88. 合并两个有序数组](https://leetcode-cn.com/problems/merge-sorted-array/)

> 双指针

```java
class Solution {
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        int p = m + n - 1;
        int p1 = m - 1;
        int p2 = n - 1;
        while(p2 >= 0) {
            if (p1 >= 0 && nums1[p1] > nums2[p2]) {
                nums1[p--] = nums1[p1--];
            } else {
                nums1[p--] = nums2[p2--];
            }
        }
    }
}
```

## [剑指 Offer 61. 扑克牌中的顺子](https://leetcode-cn.com/problems/bu-ke-pai-zhong-de-shun-zi-lcof/)

> 模拟计算

```JAVA
class Solution {
    public boolean isStraight(int[] nums) {
        Arrays.sort(nums);
        int zeroCount = 0;
        int gapCount = 0;
        int temp = 0;
        for (int i = 0; i < 5; ++i) {
            if (nums[i] == 0)
                zeroCount++;
            else if (nums[i] == temp) {
                return false;
            } else {
                if (temp != 0) 
                    gapCount += nums[i] - temp - 1;
                temp = nums[i];
            }
        }
        System.out.println(gapCount);
        System.out.println(zeroCount);
        if (gapCount - zeroCount <= 0)
            return true;
        else
            return false;
    }
}
```

## [704. 二分查找](https://leetcode-cn.com/problems/binary-search/)

> BS

```java
int binary_search(int[] nums, int target) {
    int left = 0, right = nums.length - 1; 
    while(left <= right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] < target) {
            left = mid + 1;
        } else if (nums[mid] > target) {
            right = mid - 1; 
        } else if(nums[mid] == target) {
            // 直接返回
            return mid;
        }
    }
    // 直接返回
    return -1;
}

int left_bound(int[] nums, int target) {
    int left = 0, right = nums.length - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] < target) {
            left = mid + 1;
        } else if (nums[mid] > target) {
            right = mid - 1;
        } else if (nums[mid] == target) {
            // 别返回，收缩左侧边界
            right = mid - 1;
        }
    }
    // 最后要检查 left 越界的情况
    if (left >= nums.length || nums[left] != target)
        return -1;
    return left;
}


int right_bound(int[] nums, int target) {
    int left = 0, right = nums.length - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] < target) {
            left = mid + 1;
        } else if (nums[mid] > target) {
            right = mid - 1;
        } else if (nums[mid] == target) {
            // 别返回，收缩右侧边界
            left = mid + 1;
        }
    }
    // 最后要检查 right 越界的情况
    if (right < 0 || nums[right] != target)
        return -1;
    return right;
}
```

## [34. 在排序数组中查找元素的第一个和最后一个位置](https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/)

> BS

```java
class Solution {
    public int[] searchRange(int[] nums, int target) {
        int[] res = new int[2];
        res[0] = findBoundary(nums, target, 0);
        res[1] = findBoundary(nums, target, 1);
        return res;
    }

    int findBoundary(int[] nums, int target, int direction) {
        int left = 0, right = nums.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] < target) {
                // 更新left
                left = mid + 1;
            } else if (nums[mid] > target) {
                // 更新right
                right = mid - 1;
            } else if (nums[mid] == target) {
                if (direction == 0) {
                    // 找左边界，收缩右边界
                    right = mid - 1;
                } else {
                    // 找右边界，收缩左边界
                    left = mid + 1;
                }
            }
        }
        if (direction == 0) {
            if (left >= nums.length || nums[left] != target)
                return -1;
            return left;
        } else {
            if (right < 0 || nums[right] != target)
                return -1;
            return right;
        }
    }
}
```

## [76. 最小覆盖子串](https://leetcode-cn.com/problems/minimum-window-substring/)

> 滑动窗口

```java
class Solution {
    public String minWindow(String s, String t) {
        int left = 0, right = 0; // 滑动窗口前后指针
        int min = Integer.MAX_VALUE; // 最小子串的长度
        int start = 0, end = 0; // 最小子串的左右位置
        int count = 0; // 相同字符的个数

        Map<Character, Integer> tMap = new HashMap<>(); // target串的字符计数（目标）
        Map<Character, Integer> sMap = new HashMap<>(); // source串的字符计数（窗口）

        // 初始化target串的字符计数
        for (int i = 0; i < t.length(); ++i) {
            tMap.put(t.charAt(i), tMap.getOrDefault(t.charAt(i), 0) + 1);
        }

        while (right < s.length()) {
            char c = s.charAt(right);
            // 更新窗口状态
            if (tMap.containsKey(c)) { // 是所求字符
                sMap.put(c, sMap.getOrDefault(c, 0) + 1); // 存字符进窗口
                if (tMap.get(c).compareTo(sMap.get(c)) == 0) { // 看是不是该字符达标
                    count++;
                }
            }
            right++; // 右滑动扩大
            while (count == tMap.size()) {
                // 满足条件，更新最值
                if (min > right - left) {
                    end = right;
                    start = left;
                    min = right - left;
                }
                char d = s.charAt(left);
                // 更新窗口状态
                if (tMap.containsKey(d)) {
                    sMap.put(d, sMap.get(d) - 1);
                    if (tMap.get(d) > sMap.get(d)) {
                        count--;
                    }
                }
                left++; //左滑动缩小
            }
        }
        return min == Integer.MIN_VALUE ? "" : s.substring(start, end);
    }
}
```

## [829. 连续整数求和](https://leetcode-cn.com/problems/consecutive-numbers-sum/)

> 数学

```java
class Solution {
    public int consecutiveNumbersSum(int N) {
        int ans = 0;
        int upper = (int)Math.sqrt(2 * N);
        for (int k = 1; k <= upper; ++k) {
            if ((2 * N) % k == 0) {
                int tmp = 2 * N / k - k + 1;
                if (tmp % 2 == 0) {
                    ans++;
                }
            }
        }
        return ans;
    }
}
```

## 2.两数相加

> 双指针

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        int carry = 0;
        ListNode p1 = l1;
        ListNode p2 = l2;
        int sum0 = p1.val + p2.val;
        carry = sum0 > 9 ? 1 : 0;
        ListNode p = new ListNode(sum0 > 9 ? sum0 - 10 : sum0);
        ListNode cur = p;
        p1 = p1.next;
        p2 = p2.next;
        while(p1 != null || p2 != null) {
            int l1v = 0;
            int l2v = 0;
            if (p1 != null)
                l1v = p1.val;
            if (p2 != null)
                l2v = p2.val;
            int sum = l1v + l2v + carry;
            carry = sum > 9 ? 1 : 0;
            ListNode newNode = new ListNode(sum > 9 ? sum - 10 : sum);
            cur.next = newNode;
            cur = cur.next;
            if (p1 != null)
                p1 = p1.next;
            if (p2 != null)
                p2 = p2.next;
        }
        // 最后一位可能的进位
        if (carry == 1) {
            cur.next = new ListNode(1);
        }
        return p;
    }
```

## [102. 二叉树的层序遍历](https://leetcode-cn.com/problems/binary-tree-level-order-traversal/)

> BFS

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        Queue<TreeNode> q = new LinkedList<>();
        q.offer(root);
        while (!q.isEmpty()) {
            int size = q.size();
            List<Integer> levelNum = new ArrayList<>();
            for (int i = 0; i < size`; ++i) {
                TreeNode cur = q.peek();
                q.poll();
                if (cur == null)
                    continue;
                levelNum.add(cur.val);
                q.offer(cur.left);
                q.offer(cur.right);
            }
            if (!levelNum.isEmpty())
                res.add(levelNum);
        }
        return res;
    }
}
```

## [38. 外观数列](https://leetcode-cn.com/problems/count-and-say/)

> 递归数组

```java
class Solution {
    public String countAndSay(int n) {
        if (n == 1) {
            return "1";
        } else if (n == 2) {
            return "11";
        }
        String lastStr = countAndSay(n - 1);
        char[] lastCharList = lastStr.toCharArray();
        StringBuffer result = new StringBuffer();
        int len = lastCharList.length;
        int count = 1;
        for (int i = 0; i < len - 1; ++i) {
            if (lastCharList[i] == lastCharList[i + 1]) {
                count++;
            } else {
                result.append(count + "").append(lastCharList[i]);
                count = 1;
            }
        }
        result.append(count).append(lastCharList[len - 1]);
        return result.toString();
    }
}
```

## [1. 两数之12和](https://leetcode-cn.com/problems/two-sum/)

> 哈希（可优化）

 ```java
class Solution {
    public int[] twoSum(int[] nums, int target) {
        int[] res = new int[2];
        res[0] = -1;
        res[1] = -1;
        Map<Integer, Integer> hmap = new HashMap<>();
        for (int i = 0; i < nums.length; ++i) {
            if (hmap.containsKey(target - nums[i])) {
                res[0] = hmap.get(target - nums[i]);
                res[1] = i;
                return res;
            } else {
                hmap.put(nums[i], i);
            }
        }
        return res;
    }
}
 ```

## [3. 无重复字符的最长子串](https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/)

> 滑动窗口（可优化）

```java
class Solution {
    public int lengthOfLongestSubstring(String s) {
        int left = 0, right = 0;
        int max = 0;
        int count = 0; // 不重复计数值
        Set<Character> hset = new HashSet<>();
        while (right < s.length()) {
            if (hset.contains(s.charAt(right))) { // hash里包含快指针字符，有重复，收缩左指针
                while (left < right) {
                    if (s.charAt(left) == s.charAt(right)) { // 重复时候推出
                        hset.remove(s.charAt(left));
                        left++;
                        count--;
                        break;
                    }
                    hset.remove(s.charAt(left));
                    left++;
                    count--;
                }
            } else {
                hset.add(s.charAt(right));
                right++;
                count++;
                max = Math.max(max, count);
            }
        }
        return max;
    }
}
```

## [4. 寻找两个正序数组的中位数](https://leetcode-cn.com/problems/median-of-two-sorted-arrays/)

> 二分，双100%

```java
class Solution {
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int m = nums1.length, n = nums2.length;
        if (m > n) {
            return findMedianSortedArrays(nums2, nums1); // 强行变成左小右大
        }
        int leftNum = (m + n + 1) / 2; // 划分后左数组的大小
        int rightNum = m + n - leftNum; // 划分后右数组的大小
        // 二分查找合理划分点
        int left = 0, right = m;
        while (left <= right) {
            int mid = (left + right + 1) / 2;
            int mLeftNum = leftNum - mid;
            int mLeftMax =  mid - 1 >= 0 ? nums1[mid - 1] : -1;
            int nLeftMax = mLeftNum - 1 >=0 ? nums2[mLeftNum - 1]: -1;
            int leftMax = Math.max(mLeftMax, nLeftMax); // 左集合最大
            int mRightMin = mid < m ? nums1[mid] : Integer.MAX_VALUE;
            int nRightMin = mLeftNum < n ? nums2[mLeftNum] : Integer.MAX_VALUE;
            int rightMin = Math.min(mRightMin, nRightMin); // 右集合最小
            if (leftMax <= rightMin) { // 满足条件
                return (m + n) % 2 == 0 ? (leftMax + rightMin) * 0.5 : leftMax;
            } else { // 下一个划分
                if (mLeftMax > nLeftMax) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            }
        }
        return -1;
    }
}
```

## [7. 整数反转](https://leetcode-cn.com/problems/reverse-integer/)

> 数学

```java
class Solution {
    public int reverse(int x) {
        if (x < 0) {
            if (-x < 0) // 预防-2147483648
                return 0;
            return -reverse(-x); //强行改为正数
        }
        long rx = 0;
        while (true) {
            if (x == 0) {
                return (int)rx;
            }
            rx *= 10; 
            rx = rx + (x % 10);
            if (rx > Integer.MAX_VALUE) {
                return 0;
            }
            x /= 10;
        }
    }
}
```

## [9. 回文数](https://leetcode-cn.com/problems/palindrome-number/)

> 双指针

```java
class Solution {
    public boolean isPalindrome(int x) {
        if (x < 0)
            return false;
        String xStr = x + "";
        int left = 0, right = xStr.length() - 1;
        while (left <= right) {
            if (xStr.charAt(left) != xStr.charAt(right))
                return false;
            left++;
            right--;
        }
        return true;
    }
}
```

## [20. 有效的括号](https://leetcode-cn.com/problems/valid-parentheses/)

> 栈

```java
class Solution {
    private boolean isMatch(char a, char b) {
        return a == '(' && b == ')'
            || a == '{' && b == '}'
            || a == '[' && b == ']';
    }

    public boolean isValid(String s) {
        Stack<Character> stack = new Stack<>();
        int i = 0;
        while (i < s.length()) {
            if (!stack.isEmpty() && isMatch(stack.peek(), s.charAt(i))) {
                stack.pop();
            } else {
                stack.push(s.charAt(i));
            }
            i++;
        }
        return stack.isEmpty() ? true : false;
    }
}
```

## [136. 只出现一次的数字](https://leetcode-cn.com/problems/single-number/)

> 异或巧解

```java
class Solution {
    public int singleNumber(int[] nums) {
        int res = 0;
        for (int i = 0; i < nums.length; ++i) {
            res ^= nums[i];
        }
        return res;
    }
}
```

## [27. 移除元素](https://leetcode-cn.com/problems/remove-element/)

> 双指针（官方解法更简单）

```java
class Solution {
    public int removeElement(int[] nums, int val) {
        if (nums.length == 0) {
            return 0;
        }
        int last = nums.length - 1;
        for (int i = 0; i <= last; ++i) {
            if (nums[i] == val) {
                while (nums[last] == val && last > i)
                    last--;
                if (last == i) {
                    return last;
                }
                nums[i] = nums[last];
                last--;
            }
        }
        return last + 1;
    }
}
```

## [118. 杨辉三角](https://leetcode-cn.com/problems/pascals-triangle/)

> 二维数组

```java
class Solution {
    public List<List<Integer>> generate(int numRows) {
        List<List<Integer>> resList = new ArrayList<>();
        if (numRows == 0) { // 特殊值0的判断
            return resList;
        }
        resList.add(Arrays.asList(1)); // 添加第一行
        for (int i = 1; i < numRows; ++i) {
            List<Integer> lastRow = resList.get(i - 1);
            List<Integer> tempRow = new ArrayList<>();
            tempRow.add(1);
            for (int j = 0; j < lastRow.size() - 1; ++j) {
                tempRow.add(lastRow.get(j) + lastRow.get(j + 1));
            }
            tempRow.add(1);
            resList.add(tempRow);
        }
        return resList;
    }
}
```

## [22. 括号生成](https://leetcode-cn.com/problems/generate-parentheses/)

> dp

```java
class Solution {
    public List<String> generateParenthesis(int n) {
      	// s = (a)b, N(a) + N(b) + 1 = N(s)
      	// Set i = N(s), j = N(a) => N(a) = i - j - 1
      	// Set dp[i]为n = i所求
        // dp[i] = "(" + dp[j] + ")" + dp[i - j - 1]; j∈[0, i-1]
        // dp[0] = [[""]], dp[1] = [[""], ["()"]]
        List<List<String>> dp = new ArrayList<>(n + 1);
        List<String> dp0 = new ArrayList<>(1);
        dp0.add("");
        dp.add(dp0);
        for (int i = 1; i <= n; ++i) {
            List<String> temp = new ArrayList<>();
            for (int j = 0; j <= i - 1; ++j) {
                for (String a: dp.get(j)) {
                    for (String b: dp.get(i - j - 1)) {
                        StringBuilder sb = new StringBuilder();
                        sb.append("(").append(a).append(")").append(b);
                        temp.add(sb.toString());
                    }
                }
            }
            dp.add(temp);
        }
        return dp.get(n);
    }
}
```

## [69. x 的平方根](https://leetcode-cn.com/problems/sqrtx/)

> 牛顿迭代法（注意精度）

```java
class Solution {
    public int mySqrt(int x) {
        // 牛顿迭代法: k_i = 0.5 * (k_i-1 + C / k_i-1)
        if (x == 0)
            return 0;
        double k = (double)x; //中间值
        double y = 0.; // 结果值
        while(Math.abs(k - y) > 1e-7) { // 控制精度
            y = k; // 更新y
            k = 0.5d * (k + (double)x / k); // 更新k
        }
        System.out.println(y);
        return (int)y;
    }
}
```

## [169. 多数元素](https://leetcode-cn.com/problems/majority-element/)

> 排序（有时间复杂度更低的算法-分治法、摩尔投票法）

```java
class Solution {
    public int majorityElement(int[] nums) {
        Arrays.sort(nums);
        return nums[nums.length / 2];
    }
}
```

## [41. 缺失的第一个正数](https://leetcode-cn.com/problems/first-missing-positive/)

> 哈希表

```java
class Solution {
    public int firstMissingPositive(int[] nums) {
        Set<Integer> numSet = new HashSet<>();
        for (int i: nums) {
            numSet.add(i);
        }
        int j = 1;
        while (j <= Integer.MAX_VALUE) {
            if (!numSet.contains(j)) {
                return j;
            }
            j++;
        }
        return -1;
    }
}
```

## [328. 奇偶链表](https://leetcode-cn.com/problems/odd-even-linked-list/)

> 法一：采用插入的方式将奇数插到最后一个奇数的后面（效率不高）

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode oddEvenList(ListNode head) {
        if (head == null)
            return null;
        if (head.next == null)
            return head;
        int index = 2; // 判断奇偶坐标
        ListNode oddTail = head; // 当前最后一个奇数节点
        ListNode evenTail = head.next; // 当前最后一个偶数节点
        ListNode curNode = head.next; // 当前节点
        while (curNode != null) {
            if ((index & 1) == 1) {
                System.out.println(index);
                // 奇数节点
                ListNode temp = curNode; // 记录当前节点
                curNode = curNode.next; // 迭代到下一节点
                // 当前节点插入到最后奇数后面和最后奇数下一个之前
                temp.next = oddTail.next; 
                oddTail.next = temp;
                // 更新tail
                oddTail = temp;
                evenTail.next = curNode;
            } else {
                // 偶数节点，看下一个
                evenTail = curNode;
                curNode = curNode.next;
            }
            index++;
        }
        return head;
    }
}
```

> 法二：双指针穿针引线法（时间100%）

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode oddEvenList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode o = head; // oddTail
        ListNode e = head.next; // evenTail
        ListNode eh = head.next; // evenHead
        while(true) {
            o.next = e.next;
            if (o.next == null) {
                o.next = eh;
                e = null;
                break;
            }
            o = o.next;
            e.next = o.next;
            if (e.next == null) {
                o.next = eh;
                e = null;
                break;
            }
            e = e.next;
        }
        return head;
    }
}
```

## [530. 二叉搜索树的最小绝对差](https://leetcode-cn.com/problems/minimum-absolute-difference-in-bst/)

> DFS==中序遍历

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    private int min = Integer.MAX_VALUE;
    private int pre = Integer.MAX_VALUE;

    public int getMinimumDifference(TreeNode root) {
        dfs(root);
        return min;
    }

    private void dfs(TreeNode root) {
        if (root == null) {
            return;
        }
        // 左分支处理
        dfs(root.left);
        // 根结点处理
        if (pre != Integer.MAX_VALUE)
            min = Math.min(min, root.val - pre);
        pre = root.val;
        // 右分支处理
        dfs(root.right);
    }
}
```

## [14. 最长公共前缀](https://leetcode-cn.com/problems/longest-common-prefix/)

> LCP
>
> TODO：二分方法可以进一步优化

```java
class Solution {
    public String longestCommonPrefix(String[] strs) {
        if (strs.length == 0)
            return "";
        for (int i = 0; i < strs[0].length(); ++i) { // 第i列
            char c = strs[0].charAt(i); // 第一个字符串的第i列作为判断字符
            for (int j = 1; j < strs.length; ++j) { // 第j个字符串
                if (strs[j].length() <= i || strs[j].charAt(i) != c) { // 直接结束
                    return strs[0].substring(0, i);
                }
            }
        }
        return strs[0]; // 只有一个字符串
    }
}
```

## [66. 加一](https://leetcode-cn.com/problems/plus-one/)

> 高精度整数计算

```java
class Solution {
    public int[] plusOne(int[] digits) {
        if (digits[digits.length - 1] < 9) { // 最后一位小于9直接加1
            digits[digits.length - 1] += 1;
            return digits;
        } else { // 最后一位为9
            int i = digits.length;
            while(--i >= 0) {
                if (digits[i] == 9) { // 为9则变0
                    digits[i] = 0;
                } else { // 不为9则加1
                    digits[i] += 1;
                    return digits;
                }
            }
            int[] tmpDigits = new int[digits.length + 1];
            tmpDigits[0] = 1;
            return tmpDigits;
        }
    }
}
```

## [1480. 一维数组的动态和](https://leetcode-cn.com/problems/running-sum-of-1d-array/)

> 循环

```java
class Solution {
    public int[] runningSum(int[] nums) {
        for (int i = 1; i < nums.length; ++i) {
            nums[i] = nums[i - 1] + nums[i];
        }
        return nums;
    }
}
```

## [剑指 Offer 58 - II. 左旋转字符串](https://leetcode-cn.com/problems/zuo-xuan-zhuan-zi-fu-chuan-lcof/)

> 字符串

```java
class Solution {
    public String reverseLeftWords(String s, int n) {
        return s.substring(n, s.length()) + s.substring(0, n);
    }
}
```

## [1512. 好数对的数目](https://leetcode-cn.com/problems/number-of-good-pairs/)

> 计数数组

```java
class Solution {
    public int numIdenticalPairs(int[] nums) {
        int ans = 0;
        // Map<Integer, Integer> map = new HashMap<>();
        // for (int i : nums) {
        //     map.put(i, map.getOrDefault(i, 0) + 1);
        // }
        // for (Integer count : map.values()) {
        //     if (count > 1) {
        //         ans += count * (count - 1) / 2;
        //     }
        // }
        // 巧妙解法
        int[] tmp = new int[101];
        for (int num : nums) {
            ans += tmp[num]++;
        }
        return ans;
    }
}
```

## [1476. 子矩形查询](https://leetcode-cn.com/problems/subrectangle-queries/)

> 数组

```java
class SubrectangleQueries {

    private int[][] mRectangle;

    public SubrectangleQueries(int[][] rectangle) {
        mRectangle = rectangle;
    }
    
    public void updateSubrectangle(int row1, int col1, int row2, int col2, int newValue) {
        for (int i = row1; i <= row2; ++i) {
            for (int j = col1; j <= col2; ++j) {
                mRectangle[i][j] = newValue;
            }
        }
    }
    
    public int getValue(int row, int col) {
        return mRectangle[row][col];
    }
}

/**
 * Your SubrectangleQueries object will be instantiated and called as such:
 * SubrectangleQueries obj = new SubrectangleQueries(rectangle);
 * obj.updateSubrectangle(row1,col1,row2,col2,newValue);
 * int param_2 = obj.getValue(row,col);
 */
```

## [24. 两两交换链表中的节点](https://leetcode-cn.com/problems/swap-nodes-in-pairs/)

> 链表
>
> TODO：用递归方法会简单很多，迭代法有个哑结点的技巧可以参考

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode swapPairs(ListNode head) { // 1->2->3->4 => 2->1->4->3
        ListNode res = head != null && head.next != null ? head.next : head; // 前两个为空的特殊情况
        while(head != null && head.next != null) { // 前两个不为空的时候
            ListNode tmp = head.next; // 记录第二个值
            if (tmp.next != null && tmp.next.next != null) { // 第三个和第四个值都不为空
                head.next = tmp.next.next;
            } else if (tmp.next != null) { // 第三个不为空，第四个为空
                head.next = tmp.next;
            } else { // 第三个和第四个都为空
                head.next = null;
                tmp.next = head;
                break;
            }
            ListNode tmp2 = tmp.next; // 记录第三个值
            tmp.next = head; // 第二个指向第一个
            head = tmp2; // 把第三个值置为下一轮的头
        }
        return res;
    }
}
```

## [剑指 Offer 64. 求1+2+…+n](https://leetcode-cn.com/problems/qiu-12n-lcof/)

> 短路技巧

```java
class Solution {
    public int sumNums(int n) {
        boolean b = n > 0 && (n += sumNums(n - 1)) > 0;
        return n;
    }
}
```

## [1002. 查找常用字符](https://leetcode-cn.com/problems/find-common-characters/)

> 字母数组计数，hash表

```java
class Solution {
    public List<String> commonChars(String[] A) {
        int[] minArr = new int[26];
        for (int m = 0; m < A.length; ++m) { // 每一个字符串
            if (m != 0) { // 更新最小化数组
                int[] curArr = new int[26];
                for (char c : A[m].toCharArray()) { // 统计每个字符串的字母个数
                    curArr[c - 'a']++;
                }
                for (int i = 0; i < 26; ++i) { // 更新最小数组
                    minArr[i] = Math.min(minArr[i], curArr[i]);
                }
            } else { // 第一次初始化最小数组
                for (char c : A[m].toCharArray()) {
                    minArr[c - 'a']++;
                }
            }
        }
        List<String> ret = new ArrayList<>();
        for (int j = 0; j < 26; ++j) {
            for (int k = 0; k < minArr[j]; ++k) {
                ret.add((char)(j + 'a') + "");
            }
        }
        return ret;
    }
}
```

## [695. 岛屿的最大面积](https://leetcode-cn.com/problems/max-area-of-island/)

> 二维DFS，沉岛思想

```java
class Solution {
    public int maxAreaOfIsland(int[][] grid) {
        int ans = 0;
        for (int i = 0; i < grid.length; ++i) {
            for (int j = 0; j < grid[i].length; ++j) {
                if (grid[i][j] == 1)
                    ans = Math.max(ans, dfs(grid, i, j));
            }
        }
        return ans;
    }

    private int dfs(int[][] grid, int i, int j) {
        if (i < 0 || j < 0 || i >= grid.length || j >= grid[0].length || grid[i][j] == 0) { // 当前为0或者i,j非法
            return 0;
        } else {
            grid[i][j] = 0; // 沉岛
            int count = 1;
            count += dfs(grid, i + 1, j);
            count += dfs(grid, i, j + 1);
            count += dfs(grid, i - 1, j);
            count += dfs(grid, i, j - 1);
            return count;
        }
    }
}
```

## [1114. 按序打印](https://leetcode-cn.com/problems/print-in-order/)

> 多线程

```java
class Foo {

    private int cur = 1;

    public Foo() {
        
    }

    public synchronized void first(Runnable printFirst) throws InterruptedException {
        while (cur != 1) {
            wait();
        }
        // printFirst.run() outputs "first". Do not change or remove this line.
        printFirst.run();
        cur = 2;
        notifyAll();
    }

    public synchronized void second(Runnable printSecond) throws InterruptedException {
        while (cur != 2) {
            wait();
        }
        // printSecond.run() outputs "second". Do not change or remove this line.
        printSecond.run();
        cur = 3;
        notifyAll();
    }

    public synchronized void third(Runnable printThird) throws InterruptedException {
        while (cur != 3) {
            wait();
        }
        // printThird.run() outputs "third". Do not change or remove this line.
        printThird.run();
        cur = 1;
        notifyAll();
    }
}
```

## [1115. 交替打印FooBar](https://leetcode-cn.com/problems/print-foobar-alternately/)

> 多线程
>
> TODO：可以用信号量简单实现

```java
class FooBar {
    private int n;

    private int flag = 0;

    public FooBar(int n) {
        this.n = n;
    }

    public synchronized void foo(Runnable printFoo) throws InterruptedException {
        while (this.n != 0) {
            while (flag != 0) {
                wait();
            }
            if (this.n == 0)
                break;
            printFoo.run();
            flag = 1;
            notifyAll();
        }
    }

    public synchronized void bar(Runnable printBar) throws InterruptedException {
        while (this.n != 0) {
            while (flag != 1) {
                wait();
            }
            this.n--;
            printBar.run();
            flag = 0;
            notifyAll();
        }
    }
}
```

## [1116. 打印零与奇偶数](https://leetcode-cn.com/problems/print-zero-even-odd/)

> 多线程

```java
class ZeroEvenOdd {
    private int n;

    private int flag = 0;
    
    public ZeroEvenOdd(int n) {
        this.n = n;
    }

    // printNumber.accept(x) outputs "x", where x is an integer.
    public synchronized void zero(IntConsumer printNumber) throws InterruptedException {
        for (int i = 1; i <= n; i++) {
            while (flag != 0) {
                wait();
            }
            printNumber.accept(0);
            if ((i & 1) == 1) { // i为奇数
                flag = 1;
            } else {
                flag = 2;
            }
            notifyAll();
        }
    }

    public synchronized void even(IntConsumer printNumber) throws InterruptedException {
        for (int i = 2; i <= n; i += 2) {
            while (flag != 2) {
                wait();
            }
            printNumber.accept(i);
            flag = 0;
            notifyAll();
        }
    }

    public synchronized void odd(IntConsumer printNumber) throws InterruptedException {
        for (int i = 1; i <= n; i += 2) {
            while (flag != 1) {
                wait();
            }
            printNumber.accept(i);
            flag = 0;
            notifyAll();
        }
    }
}
```

## [116. 填充每个节点的下一个右侧节点指针](https://leetcode-cn.com/problems/populating-next-right-pointers-in-each-node/)

> BFS，层序遍历

```java
class Solution {
    public Node connect(Node root) {
        levelOrder(root);
        return root;
    }

    private void levelOrder(Node root) {
        Queue<Node> q = new LinkedList<Node>();
        if (root == null || root.left == null)
            return;
        q.offer(root);
        while (!q.isEmpty()) {
            Node tmp = q.poll(); // 从头部取出
            if (tmp.left == null)
                continue;
            tmp.left.next = tmp.right;
            if (tmp.next == null)
                tmp.right.next = null;
            else
                tmp.right.next = tmp.next.left;
            if (tmp.left != null)
                q.offer(tmp.left);
            if (tmp.right != null)
                q.offer(tmp.right);
        }
    }
}
```

## [977. 有序数组的平方](https://leetcode-cn.com/problems/squares-of-a-sorted-array/)

> 归并排序，双指针
>
> 执行：100%

```java
class Solution {
    public int[] sortedSquares(int[] A) {
        // 直接排序法
        // int idx = 0;
        // for (int i : A) {
        //     A[idx++] = i * i;
        // }
        // Arrays.sort(A);
        // return A;
        // 归并排序
        int[] ret = new int[A.length];
        int l = 0, r = A.length - 1, end = A.length - 1;
        while (l <= r) {
            int lt = A[l] * A[l];
            int rt = A[r] * A[r];
            if (l == r) {
                ret[end--] = lt;
                break;
            } else if (lt == rt) {
                ret[end--] = lt;
                ret[end--] = rt;
                l++;
                r--;
            } else if (lt > rt) {
                ret[end--] = lt;
                l++;
            } else if (rt > lt) {
                ret[end--] = rt;
                r--;
            }
        }
        return ret;
    }
}
```

## [1572. 矩阵对角线元素的和](https://leetcode-cn.com/problems/matrix-diagonal-sum/)

> 数组

```java
class Solution {
    public int diagonalSum(int[][] mat) {
        int ans = 0;
        for (int i = 0; i < mat.length; ++i) {
            ans += mat[i][i] + mat[i][mat.length - i - 1];
        }
        if ((mat.length & 1) == 1) { // 奇数:需要考虑重复
            ans -= mat[mat.length / 2][mat.length / 2];
        }
        return ans;
    }
}
```

## [1295. 统计位数为偶数的数字](https://leetcode-cn.com/problems/find-numbers-with-even-number-of-digits/)

> 数组

```java
class Solution {
    private int[] table = new int[]{9, 99, 999, 9999, 99999, 999999, 9999999, 99999999, 999999999, Integer.MAX_VALUE};

    public int findNumbers(int[] nums) {
        // int ans = 0;
        // for (int num : nums) {
        //     if ((String.valueOf(num).length() & 1) == 0) {
        // //  if ((Integer.toString(num).length() & 1) == 0) {
        //         ans += 1;
        //     }
        // }
        // return ans;

        int ans = 0;
        
        for (int num : nums) {
            int size = sizeOf(num);
            if ((size & 1) == 0) {
                ans += 1;
            }
        }
        return ans;
    }

    private int sizeOf(int num) {
        for (int i = table.length - 1; i >= 0; --i) {
            if (num >= table[i]) {
                return i + 2;
            }
        }
        return 1;
    }
}
```

## [1266. 访问所有点的最小时间](https://leetcode-cn.com/problems/minimum-time-visiting-all-points/)

> 几何，数组

```java
class Solution {
    public int minTimeToVisitAllPoints(int[][] points) {
        int ans = 0;
        for (int i = 1; i < points.length; ++i) {
            ans += minTimeBetween(points[i - 1], points[i]);
        }
        return ans;
    }

    // 最短时间为先斜着走在直着走，也就是相减较大的值（切比雪夫距离）
    private int minTimeBetween(int[] sp, int[] ep) {
        return Math.max(Math.abs(sp[0] - ep[0]), Math.abs(sp[1] - ep[1]));
    }
}
```

## [52. N皇后 II](https://leetcode-cn.com/problems/n-queens-ii/)

> 回溯算法

```java
class Solution {
    public int totalNQueens(int n) {
        return solve(n, 0, 0, 0, 0);
    }

    public int solve(int n, int row, int col, int dia1, int dia2) {
        if (row == n) { // 最后一行
            return 1;
        } else {
            int count = 0;
            int aps = ((1 << n) - 1) & (~(col | dia1 | dia2)); // 当前行可以放置的值
            while (aps != 0) {
                int pos = aps & (-aps);
                aps = aps & (aps - 1);
                count += solve(n, row + 1, col | pos, (dia1 | pos) << 1, (dia2 | pos) >> 1);
            }
            return count;
        }
    }
}
```



## [19. 删除链表的倒数第N个节点](https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list/)

> 链表，双指针，快慢指针

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode fastNode = head;
        while (n-- >= 0) {
            if (fastNode == null) { // 特殊情况，删除头元素
                head = head.next;
                return head;
            }
            fastNode = fastNode.next;
        }
        ListNode slowNode = head;
        while (fastNode != null) {
            fastNode = fastNode.next;
            slowNode = slowNode.next;
        }
        slowNode.next = slowNode.next.next;
        return head;
    }
}
```

## [844. 比较含退格的字符串](https://leetcode-cn.com/problems/backspace-string-compare/)

> 字符串，栈

```java
class Solution {
    public boolean backspaceCompare(String S, String T) {
        if (getFinalString(S).equals(getFinalString(T))) {
            return true;
        }
        return false;
    }

    private String getFinalString(String str) {
        LinkedList<Character> stack = new LinkedList<Character>();
        char[] charArr = str.toCharArray();
        for (char c : charArr) {
            if (c == '#') {
                if (!stack.isEmpty()) {
                    stack.removeLast();
                }
            } else {
                stack.addLast(c);
            }
        }
        StringBuilder sb = new StringBuilder();
        for (Character c : stack) {
            sb.append(c);
        }
        return sb.toString();
    }
}
```

## [143. 重排链表](https://leetcode-cn.com/problems/reorder-list/)

> 链表

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public void reorderList(ListNode head) {
        List<ListNode> list = new ArrayList<ListNode>();
        while (head != null) {
            list.add(head);
            head = head.next;
        }
        int len = list.size();
        for (int i = 0; i < len; ++i) {
            if (i >= len / 2) {
                list.get(i).next = null;
                break;
            }
            list.get(i).next = list.get(len - i - 1);
            list.get(len - i - 1).next = list.get(i + 1);
        }
        return;
    }
}
```

## [925. 长按键入](https://leetcode-cn.com/problems/long-pressed-name/)

> 双指针，字符串

```java
class Solution {
    public boolean isLongPressedName(String name, String typed) {
        int i = 0, j = 0;
        while (j < typed.length()) {
            if (i < name.length() && name.charAt(i) == typed.charAt(j)) {
                i++;
                j++;
            } else if (j > 0 && typed.charAt(j) == typed.charAt(j - 1)) {
                j++;
            } else {
                return false;
            }
        }
        return i == name.length();
    }
}
```

## [763. 划分字母区间](https://leetcode-cn.com/problems/partition-labels/)

> 双指针，贪心

```java
class Solution {
    public List<Integer> partitionLabels(String S) {
        int[] last = new int[26];
        for (int i = 0; i < S.length(); ++i) {
            last[S.charAt(i) - 'a'] = i;
        }
        List<Integer> ret = new ArrayList<Integer>();
        int start = 0;
        int end = 0;
        for (int i = 0; i < S.length(); ++i) {
            end = Math.max(end, last[S.charAt(i) - 'a']);
            if (i == end) {
                ret.add(end - start + 1);
                start = end + 1;
            }
        }
        return ret;
    }
}
```

## [234. 回文链表](https://leetcode-cn.com/problems/palindrome-linked-list/)

> 链表，双指针
>
> O(n)，O(1)

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    // 翻转链表
    private ListNode reverse(ListNode head) {
        ListNode cur = head;
        while (cur != null && cur.next != null) {
            ListNode p = cur.next;
            cur.next = p.next;
            p.next = head;
            head = p;
        }
        return head;
    }

    public boolean isPalindrome(ListNode head) {
        ListNode fast = head;
        ListNode slow = head;
        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;
        }
        // 翻转右边链表
        ListNode l2 = null;
        if (fast != null) { // 奇数个，slow是中间值
            l2 = reverse(slow.next);
        } else { // 偶数个，slow是右边链表第一个值
            l2 = reverse(slow);
        }
        ListNode test1 = head;
        ListNode test2 = l2;
        while (test2 != null) {
            if (test1.val != test2.val) {
                return false;
            }
            test1 = test1.next;
            test2 = test2.next;
        }
        return true;
    }
}
```

## [1024. 视频拼接](https://leetcode-cn.com/problems/video-stitching/)

> 最小覆盖，动态规划，贪心
>
> TODO：贪心

```java
class Solution {
    public int videoStitching(int[][] clips, int T) {
        // 动态规划
        // 状态定义：dp[i]表示[0,i)覆盖最小片段数（i<=T）
        // 转移方程：dp[i] = Min(dp[i], dp[aj] + 1) (aj<i<=bj)
        // 初始条件：i=1,dp[0]=0
        // 求：dp[T]
        int[] dp = new int[T + 1];
        int maxVal = Integer.MAX_VALUE - 1;
        Arrays.fill(dp, maxVal);
        dp[0] = 0;
        for (int i = 1; i <= T; ++i) {
            for (int j = 0; j < clips.length; ++j) {
                if (clips[j][0] < i && i <= clips[j][1]) {
                    dp[i] = Math.min(dp[i], dp[clips[j][0]] + 1);
                }
            }
        }
        return dp[T] == maxVal ? -1 : dp[T];
    }
}
```

## [845. 数组中的最长山脉](https://leetcode-cn.com/problems/longest-mountain-in-array/)

> 模拟法
>
> TODO：双指针，动态规划

笨办法：

```java
class Solution {
    public int longestMountain(int[] A) {
        int ret = 0;
        int cur = 0;
        boolean up = false;
        boolean down = false;
        // 在下坡，最有一个，直接更新退出
        // 不在上下坡，且后一个大于前一个，cur = 1，up = true
        // 不在上下坡，且后一个小于等于前一个，不用考虑
        // 在上坡，且后一个大于前一个，cur += 1，up = true
        // 在上坡，且后一个等于前一个，cur = 0，up = false
        // 在上坡，且后一个小于前一个，cur += 1，up = false，down = true（正好最后一个的特殊考虑）
        // 在下坡，且后一个小于前一个，cur += 1，down = true
        // 在下坡，且后一个大于前一个，更新ret，up = true，down = false，cur = 2
        // 在下坡，且后一个等于前一个，更新ret，up = false，down = false，cur = 0
        for (int i = 0; i < A.length; ++i) {
            if (i + 1 == A.length && down) {
                ret = Math.max(ret, cur + 1);
                break;
            } else if (i + 1 == A.length) {
                break;
            }
            if (!up && !down && A[i + 1] > A[i]) {
                cur = 1;
                up = true;
            } else if (up && !down && A[i + 1] > A[i]) {
                cur += 1;
                up = true;
            } else if (up && !down && A[i + 1] == A[i]) {
                cur = 0;
                up = false;
            } else if (up && !down && A[i + 1] < A[i]) {
                cur += 1;
                up = false;
                down = true;
            } else if (!up && down && A[i + 1] < A[i]) {
                cur += 1;
                down = true;
            } else if (!up && down && A[i + 1] > A[i]) {
                ret = Math.max(ret, cur + 1);
                up = true;
                down = false;
                cur = 1;
            } else if (!up && down && A[i + 1] == A[i]) {
                ret = Math.max(ret, cur + 1);
                up = false;
                down = false;
                cur = 0;
            }
        }
        return ret;
    }
}
```

整理一下：

```java
class Solution {
    public int longestMountain(int[] A) {
        int ret = 0;
        int cur = 0;
        boolean up = false;
        boolean down = false;
        for (int i = 0; i < A.length; ++i) {
            if (down && (i + 1 == A.length || A[i + 1] >= A[i]))
                ret = Math.max(ret, cur + 1);
            if (i + 1 == A.length)
                break;
            if (A[i + 1] > A[i]) {
                cur = up ? cur + 1 : 1;
                down = false;
                up = true;
            } else if (A[i + 1] < A[i]) {
                cur = (!up && !down) ? 0 : cur + 1;
                down = (!up && !down) ? false : true;
                up = false;
            } else {
                cur = 0;
                down = false;
                up = false;
            }
        }
        return ret;
    }
}
```

## [1365. 有多少小于当前数字的数字](https://leetcode-cn.com/problems/how-many-numbers-are-smaller-than-the-current-number/)

> 数组，哈希表，计数排序，快速排序

暴力解

```java
class Solution {
    public int[] smallerNumbersThanCurrent(int[] nums) {
        int[] s = new int[nums.length];
        for (int i = 0; i < nums.length; ++i) {
            int count = 0;
            for (int j = 0; j < nums.length; ++j) {
                if (i != j && nums[j] < nums[i]) {
                    count++;
                }
            }
            s[i] = count;
        }
        return s;
    }
}
```

计数排序

> 执行用时：1 ms, 在所有 Java 提交中击败了100.00%的用户
>
> 内存消耗：38.3 MB, 在所有 Java 提交中击败了97.67%的用户

```java
class Solution {
    private int sum(int[] countArray, int end) {
        int ans = 0;
        for (int i = 0; i < end; ++i) {
            ans += countArray[i];
        }
        return ans;
    }

    public int[] smallerNumbersThanCurrent(int[] nums) {
        int[] countArray = new int[101];
        int[] retArray = new int[nums.length];
        for (int i : nums) {
            countArray[i]++;
        }
        for (int i = 0; i < nums.length; ++i) {
            retArray[i] = sum(countArray, nums[i]);
        }
        return retArray;
    }
}
```

## [670. 最大交换](https://leetcode-cn.com/problems/maximum-swap/)

> 数组，数学

```java
class Solution {
    private int swap(StringBuilder intStr, int i, int j) {
        char tmp = intStr.charAt(i);
        intStr.setCharAt(i, intStr.charAt(j));
        intStr.setCharAt(j, tmp);
        return Integer.parseInt(intStr.toString());
    }

    public int maximumSwap(int num) {
        // 从高位向低位遍历数字，找到最大数字对应的**最后一个**位置，如果比第一位数组**大**，则该数字与第一位交换并返回，否则从第二位往后找最大数字对应的最后一个位置，如果第二位往后最大数字大于第二位数字，则该数与第二位数字交换，依次类推直至最低位。
        StringBuilder intSb = new StringBuilder(String.valueOf(num));
        if (intSb.length() == 1)
            return num;
        for (int i = 0; i < intSb.length(); ++i) {
            int maxIdx = i;
            int maxVal = intSb.charAt(i);
            for (int j  = i + 1; j < intSb.length(); ++j) {
                if (maxVal <= intSb.charAt(j)) {
                    maxIdx = j;
                    maxVal = intSb.charAt(j);
                }
            }
            if (maxIdx != i && maxVal != intSb.charAt(i)) {
                return swap(intSb, i, maxIdx);
            }
        }
        return num;
    }
}
```

## [478. 在圆内随机生成点](https://leetcode-cn.com/problems/generate-random-point-in-a-circle/)

> 数学，随机，拒绝采样

拒绝采样

```java
class Solution {
    private double xc, yc, r;

    public Solution(double radius, double x_center, double y_center) {
        xc = x_center;
        yc = y_center;
        r = radius;
    }
    
    public double[] randPoint() {
        Random ran = new Random();
        double xl = xc - r;
        double yb = yc - r;
        while (true) {
            double rx = ran.nextDouble() * 2 * r + xl;
            double ry = ran.nextDouble() * 2 * r + yb;
            double x = rx - xc;
            double y = ry - yc;
            if (x * x + y * y <= r * r) {
                return new double[]{rx, ry};
            }
        }
    }
}

/**
 * Your Solution object will be instantiated and called as such:
 * Solution obj = new Solution(radius, x_center, y_center);
 * double[] param_1 = obj.randPoint();
 */
```

极坐标
$$
(r*\sqrt{Random(0,1)}*cos2\pi Random(0,1)+x_{center},r*\sqrt{Random(0,1)}*sin2\pi Random(0,1)+y_{center})
$$

```java
class Solution {
    private double xc, yc, r;

    public Solution(double radius, double x_center, double y_center) {
        xc = x_center;
        yc = y_center;
        r = radius;
    }
    
    public double[] randPoint() {
        double d = Math.sqrt(Math.random()) * r;
        double theta = Math.random() * 2 * Math.PI;
        return new double[]{d * Math.cos(theta) + xc, d * Math.sin(theta) + yc};
    }
}

/**
 * Your Solution object will be instantiated and called as such:
 * Solution obj = new Solution(radius, x_center, y_center);
 * double[] param_1 = obj.randPoint();
 */
```

## [LCP 06. 拿硬币](https://leetcode-cn.com/problems/na-ying-bi/)

> 判断奇偶

```java
class Solution {
    public int minCount(int[] coins) {
        int ans = 0;
        for (int i : coins) {
            if ((i & 1) == 1) {
                ans += i / 2 + 1;
            } else {
                ans += i / 2;
            }
        }
        return ans;
    }
}
```

## [LCP 01. 猜数字](https://leetcode-cn.com/problems/guess-numbers/)

> 数组

```java
class Solution {
    public int game(int[] guess, int[] answer) {
        int ans = 0;
        for (int i = 0; i < 3; ++i) {
            if (guess[i] == answer[i])
                ans += 1;
        }
        return ans;
    }
}
```

## [LCP 17. 速算机器人](https://leetcode-cn.com/problems/nGK0Fy/)

> 模拟计算

```java
class Solution {
    private int x = 1;
    private int y = 0;

    private void A() {
        x = 2 * x + y;
    }

    private void B() {
        y = 2 * y + x;
    }

    public int calculate(String s) {
        int ans = 0;
        for (char c : s.toCharArray()) {
            if (c == 'A') {
                A();
            } else {
                B();
            }
        }
        return x + y;
    }
}
```

## [LCP 02. 分式化简](https://leetcode-cn.com/problems/deep-dark-fraction/)

> 数学
>
> 执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户
>
> 内存消耗：36 MB, 在所有 Java 提交中击败了99.14%的用户

```java
class Solution {
    public int[] fraction(int[] cont) {
        if (cont.length == 1) {
            return new int[]{cont[0], 1};
        }
        int n = cont[cont.length - 1];
        int m = 1;
        for (int i = cont.length - 2; i >= 0; --i) {
            int tmp = n;
            n = cont[i] * n + m;
            m = tmp;
        }
        return new int[]{n, m};
    }
}
```

## [144. 二叉树的前序遍历](https://leetcode-cn.com/problems/binary-tree-preorder-traversal/)

> 二叉树，栈
>
> 执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户
>
> 内存消耗：36.6 MB, 在所有 Java 提交中击败了95.01%的用户
>
> TODO：Morris 遍历

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    public List<Integer> preorderTraversal(TreeNode root) {
        if (root == null)
            return new ArrayList<Integer>();
        List<Integer> retList = new ArrayList<Integer>();
        LinkedList<TreeNode> nodeList = new LinkedList<TreeNode>();
        nodeList.add(root);
        while (!nodeList.isEmpty()) {
            TreeNode node = nodeList.removeLast();
            retList.add(node.val);
            if (node.right != null) {
                nodeList.add(node.right);
            }
            if (node.left != null) {
                nodeList.add(node.left);
            }
        }
        return retList;
    }
}
```

## [1207. 独一无二的出现次数](https://leetcode-cn.com/problems/unique-number-of-occurrences/)

> 哈希表
>
> 执行用时：1 ms, 在所有 Java 提交中击败了100.00%的用户
>
> 内存消耗：37.6 MB, 在所有 Java 提交中击败了13.80%的用户

```java
class Solution {
    public boolean uniqueOccurrences(int[] arr) {
        int[] countArr = new int[2001];
        for (int i : arr) {
            countArr[1000 + i] += 1;
        }
        int[] uniqueArr = new int[arr.length + 1];
        for (int j : countArr) {
            if (j == 0)
                continue;
            if (uniqueArr[j] >= 1) {
                return false;
            } else {
                uniqueArr[j] = 1;
            }
        }
        return true;
    }
}
```

## [1588. 所有奇数长度子数组的和](https://leetcode-cn.com/problems/sum-of-all-odd-length-subarrays/)

> 数组

暴力法（O(n^3)）

```java
class Solution {
    public int sumOddLengthSubarrays(int[] arr) {
        int ans = 0;
        for (int i = 1; i <= arr.length; i += 2) {
            for (int start = 0; start + i <= arr.length; ++start) {
                if (i == 1) {
                    ans += arr[start];
                    continue;
                }
                for (int j = start; j <= start + i - 1; ++j)
                    ans += arr[j];
            }
        }
        return ans;
    }
}
```

计算出现的次数（O(n)）

> 执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户

```java
class Solution {
    // i+1=1,len-i=5：左侧0个右侧4个：偶数1*3+奇数0*2=3
    // i+1=2,len-i=4：左侧1个右侧3个：偶数1*2+奇数1*2=4
    // i+1=3,len-i=3：左侧2个右侧2个：偶数2*2+奇数1*1=5
    // i+1=4,len-i=2：左侧3个右侧1个：偶数2*1+奇数2*1=4
    // i+1=5,len-i=1：左侧4个右侧0个：偶数3*1+奇数2*0=3
    public int sumOddLengthSubarrays(int[] arr) {
        int ans = 0;
        for (int i = 0; i < arr.length; ++i) {
            int l = i + 1;
            int r = arr.length - i;
            ans += ((l / 2) * (r / 2) + ((l + 1) / 2) * ((r + 1) / 2)) * arr[i];
        }
        return ans;
    }
}
```

## [129. 求根到叶子节点数字之和](https://leetcode-cn.com/problems/sum-root-to-leaf-numbers/)

> 树，DFS
>
> 执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户
>
> 内存消耗：36.2 MB, 在所有 Java 提交中击败了89.82%的用户

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    private int ans = 0;

    private void dfs(TreeNode curNode, int lastNum) {
        if (curNode == null) {
            return;
        }
        if (curNode.left == null && curNode.right == null) {
            ans += lastNum * 10 + curNode.val;
            return;
        }
        if (curNode.left != null) {
            dfs(curNode.left, lastNum * 10 + curNode.val);
        }
        if (curNode.right != null) {
            dfs(curNode.right, lastNum * 10 + curNode.val);
        }
    }

    public int sumNumbers(TreeNode root) {
        dfs(root, 0);
        return ans;
    }
}
```

## [463. 岛屿的周长](https://leetcode-cn.com/problems/island-perimeter/)

> dfs

遍历法（空间小）

```java
class Solution {
    public int islandPerimeter(int[][] grid) {
        int ans = 0;
        for (int i = 0; i < grid.length; ++i) {
            for (int j = 0; j < grid[i].length; ++j) {
                if (grid[i][j] == 1) {
                    if (i - 1 < 0 || grid[i - 1][j] == 0)
                    ans++;
                if (i + 1 >= grid.length || grid[i + 1][j] == 0)
                    ans++;
                if (j - 1 < 0 || grid[i][j - 1] == 0)
                    ans++;
                if (j + 1 >= grid[i].length || grid[i][j + 1] == 0)
                    ans++;
                }
            }
        }
        return ans;
    }
}
```

DFS

```java
class Solution {
    public int islandPerimeter(int[][] grid) {
        for (int i = 0; i < grid.length; ++i) {
            for (int j = 0; j < grid[0].length; ++j) {
                if (grid[i][j] == 1) {
                    return dfs(grid, i, j);
                }
            }
        }
        return 0;
    }

    private int dfs(int[][] grid, int m, int n) {
        if (!(0 <= m && m < grid.length && 0 <= n && n < grid[0].length) || grid[m][n] == 0) { // 从岛出网格或者接触水域，周长+1
            return 1;
        }
        if (grid[m][n] != 1) { // 边缘的水域或者已经访问过的，周长不变
            return 0;
        }
        grid[m][n] = 2; // 记录已经访问过的
        return dfs(grid, m + 1, n) + dfs(grid, m - 1, n) + dfs(grid, m, n - 1) + dfs(grid, m, n + 1); // 周长求和
    }
}
```

## [381. O(1) 时间插入、删除和获取随机元素 - 允许重复](https://leetcode-cn.com/problems/insert-delete-getrandom-o1-duplicates-allowed/)

> 设计，数组，哈希表
>
> ArrayList(值)+Map<值,Set(索引)>

```java
class RandomizedCollection {
    int n;
    HashMap<Integer, Set<Integer>> map;
    List<Integer> table;

    private void swap(int a, int b) {
        if (a == b)
            return;
        int temp = table.get(a);
        table.set(a, table.get(b));
        table.set(b, temp);
    }

    /** Initialize your data structure here. */
    public RandomizedCollection() {
        n = 0;
        map = new HashMap<Integer, Set<Integer>>();
        table = new ArrayList<Integer>(n);
    }
    
    /** Inserts a value to the collection. Returns true if the collection did not already contain the specified element. */
    public boolean insert(int val) {
        boolean ret = false;
        table.add(val);
        Set<Integer> s;
        if (map.containsKey(val)) {
            s = map.get(val);
        } else {
            s = new HashSet<Integer>();
            ret = true;
        }
        s.add(n);
        map.put(val, s);
        n++;
        return ret;
    }
    
    /** Removes a value from the collection. Returns true if the collection contained the specified element. */
    public boolean remove(int val) {
        if (!map.containsKey(val)) {
            return false;
        }
        int last = table.get(n - 1);
        Set<Integer> valSet = map.get(val);
        int valIndex = (int)valSet.iterator().next();
        swap(n - 1, valIndex);
        // 删除val
        table.remove(n - 1);
        valSet.remove(valIndex);
        if (valSet.size() == 0) {
            map.remove(val);
        }
        // 更新原来最后一个值
        if (n - 1 != valIndex) {
            map.get(last).remove(n - 1);
            map.get(last).add(valIndex);
        }
        n--;
        return true;
    }
    
    /** Get a random element from the collection. */
    public int getRandom() {
        Random r = new Random();
        int ri = r.nextInt(n);
        return table.get(ri);
    }
}

/**
 * Your RandomizedCollection object will be instantiated and called as such:
 * RandomizedCollection obj = new RandomizedCollection();
 * boolean param_1 = obj.insert(val);
 * boolean param_2 = obj.remove(val);
 * int param_3 = obj.getRandom();
 */
```

## [1614. 括号的最大嵌套深度](https://leetcode-cn.com/problems/maximum-nesting-depth-of-the-parentheses/)

> 字符串

```java
class Solution {
    public int maxDepth(String s) {
        int depth = 0;
        int maxDepth = 0;
        for (char c : s.toCharArray()) {
            if (c == '(') {
                depth++;
                maxDepth = Math.max(maxDepth, depth);
            } else if (c == ')') {
                depth--;
            }
        }
        return depth == 0 ? maxDepth : 0;
    }
}
```

## [1389. 按既定顺序创建目标数组](https://leetcode-cn.com/problems/create-target-array-in-the-given-order/)

> 数组

```java
class Solution {
    public int[] createTargetArray(int[] nums, int[] index) {
        List<Integer> ret = new ArrayList<Integer>(nums.length);
        for (int i = 0; i < nums.length; ++i) {
            ret.add(index[i], nums[i]);
        }
        return ret.stream().mapToInt(e->e).toArray();
    }
}
```

## [140. 单词拆分 II](https://leetcode-cn.com/problems/word-break-ii/)

> DP，回溯

先判断是否可以拆分（防止超时），再用自底向上方法拆分

```java
class Solution {
    private boolean check(String s, List<String> wordDict) {
        Set<String> wordDictSet = new HashSet(wordDict);
        boolean[] dp = new boolean[s.length() + 1];
        dp[0] = true;
        for (int i = 1; i <= s.length(); i++) {
            for (int j = 0; j < i; j++) {
                if (dp[j] && wordDictSet.contains(s.substring(j, i))) {
                    dp[i] = true;
                    break;
                }
            }
        }
        return dp[s.length()];
    }

    public List<String> wordBreak(String s, List<String> wordDict) {
        // dp[k] = if(word[m,k) in s) {dp[k].add(所有的dp[m] + ' ' + word[m, k))} (m in [0,k))
        // dp[0] = [""]
        // 求dp[n]
        if (!check(s, wordDict))
            return new ArrayList<String>();
        List<List<String>> dp = new ArrayList<List<String>>();
        for (int i = 0; i <= s.length(); ++i) {
            List<String> initVal = new ArrayList<String>();
            dp.add(initVal);
        }
        for (int k = 1; k <= s.length(); ++k) {
            for (int m = 0; m < k; ++m) {
                if (wordDict.contains(s.substring(m, k))) { // 尾部找到单词
                    if (m == 0) {
                        dp.get(k).add(s.substring(0, k));
                    } else {
                        for (String str : dp.get(m)) {
                            dp.get(k).add(str + " " + s.substring(m, k));
                        }
                    }
                 }
            }
        }
        return dp.get(s.length());
    }
}
```

直接自顶向下搜索回溯+剪枝

```java
class Solution {
    public List<String> wordBreak(String s, List<String> wordDict) {
        Map<Integer, List<List<String>>> map = new HashMap<Integer, List<List<String>>>();
        List<List<String>> wordBreaks = backtrack(s, s.length(), new HashSet<String>(wordDict), 0, map);
        List<String> breakList = new LinkedList<String>();
        for (List<String> wordBreak : wordBreaks) {
            breakList.add(String.join(" ", wordBreak));
        }
        return breakList;
    }

    public List<List<String>> backtrack(String s, int length, Set<String> wordSet, int index, Map<Integer, List<List<String>>> map) {
        if (!map.containsKey(index)) {
            List<List<String>> wordBreaks = new LinkedList<List<String>>();
            if (index == length) {
                wordBreaks.add(new LinkedList<String>());
            }
            for (int i = index + 1; i <= length; i++) {
                String word = s.substring(index, i);
                if (wordSet.contains(word)) {
                    List<List<String>> nextWordBreaks = backtrack(s, length, wordSet, i, map);
                    for (List<String> nextWordBreak : nextWordBreaks) {
                        LinkedList<String> wordBreak = new LinkedList<String>(nextWordBreak);
                        wordBreak.offerFirst(word);
                        wordBreaks.add(wordBreak);
                    }
                }
            }
            map.put(index, wordBreaks);
        }
        return map.get(index);
    }
}
```

## [349. 两个数组的交集](https://leetcode-cn.com/problems/intersection-of-two-arrays/)

> 排序、哈希表、双指针、二分查找

```java
class Solution {
    public int[] intersection(int[] nums1, int[] nums2) {
        Set<Integer> set1 = new HashSet<>();
        for (int n1: nums1) {
            set1.add(n1);
        }
        Set<Integer> set2 = new HashSet<>();
        for (int n2: nums2) {
            set2.add(n2);
        }
        return getIntersection(set1, set2);
    }

    private int[] getIntersection(Set<Integer> set1, Set<Integer> set2) {
        if (set1.size() > set2.size()) {
            getIntersection(set2, set1);
        }
        List<Integer> retList = new ArrayList<Integer>();
        for (Integer i : set1) {
            if (set2.contains(i)) {
                retList.add(i);
            }
        }
        int[] ret = new int[retList.size()];
        int index = 0;
        for (Integer j : retList) {
            ret[index++] = j;
        }
        return ret;
    }
}
```

## [941. 有效的山脉数组](https://leetcode-cn.com/problems/valid-mountain-array/)

> 数组

```java
class Solution {
    public boolean validMountainArray(int[] A) {
        int n = A.length;
        if (n == 1 || n == 0)
            return false;
        int left = 0, right = n - 1;
        while (left < n - 1) {
            if (A[left] < A[left + 1]) {
                left++;
            } else {
                break;
            }
        }
        while (right > 1) {
            if (A[right] < A[right - 1]) {
                right--;
            } else {
                break;
            }
        }
        if (left == right && left != 0 && right != n - 1)
            return true;
        return false;
    }
}
```

## [57. 插入区间](https://leetcode-cn.com/problems/insert-interval/)

> 排序，数组
>
> 执行用时：1 ms, 在所有 Java 提交中击败了99.65%的用户
>
> 内存消耗：40.9 MB, 在所有 Java 提交中击败了71.74%的用户

```java
class Solution {
    public int[][] insert(int[][] intervals, int[] newInterval) {
        // 插入I[l,r]，原数组O[ol, or]
        // ① I[1] < O[0]
        // ② I[0] > O[1]
        // ③ 合并为[min(l,ol),max(r,or)]
        List<int[]> retList = new ArrayList<>();
        for (int[] item : intervals) {
            if (newInterval[1] < item[0]) { // 插入偏左，插入进组，原数组成为新插入
                int[] tmp = new int[]{newInterval[0], newInterval[1]};
                retList.add(tmp);
                newInterval[0] = item[0];
                newInterval[1] = item[1];
            } else if (newInterval[0] > item[1]) { // 插入靠右，原数组进组，插入继续
                retList.add(item);
            } else { // 两者重叠或者相接，合并后更新插入
                newInterval[0] = Math.min(item[0], newInterval[0]);
                newInterval[1] = Math.max(item[1], newInterval[1]);
            }
        }
        retList.add(newInterval);
        int[][] retArr = new int[retList.size()][2];
        for (int i = 0; i < retList.size(); ++i) {
            retArr[i] = retList.get(i);
        }
        return retArr;
    }
}
```

## [127. 单词接龙](https://leetcode-cn.com/problems/word-ladder/)

> 双端BFS，图

```java
class Solution {
    public int ladderLength(String beginWord, String endWord, List<String> wordList) {
        int end = wordList.indexOf(endWord);
        if (end == -1)
            return 0;
        wordList.add(beginWord);
        // 双端BFS
        Queue<String> q1 = new LinkedList<>();
        Queue<String> q2 = new LinkedList<>();
        q1.offer(beginWord);
        q2.offer(endWord);
        // 访问记录
        Set<String> v1 = new HashSet<>();
        Set<String> v2 = new HashSet<>();
        v1.add(beginWord);
        v2.add(endWord);
        // wordList存到HashSet中提高检索效率
        Set<String> wordSet = new HashSet<>(wordList);
        // 结果
        int count = 0;
        while(!q1.isEmpty() && !q2.isEmpty()) {
            count++;
            // 从小的开始遍历
            if (q1.size() > q2.size()) {
                Queue<String> tmpQueue = q1;
                q1 = q2;
                q2 = tmpQueue;
                Set<String> tmpSet = v1;
                v1 = v2;
                v2 = tmpSet;
            }
            int sizeSmall = q1.size(); 
            while (sizeSmall-- > 0) {
                String s = q1.poll();
                char[] sChars = s.toCharArray();
                for (int i = 0; i < s.length(); ++i) {
                    char cTmp = sChars[i]; // 记录改变前的值
                    for (char j = 'a'; j <= 'z'; ++j) {
                        sChars[i] = j;
                        String newS = new String(sChars);
                        if (v1.contains(newS)) { // 新单词已经访问过
                            continue;
                        }
                        if (v2.contains(newS)) { // 相遇
                            return count + 1;
                        }
                        if (wordSet.contains(newS)) { // 新单词未访问且未相遇且包含在字典中，加入访问
                            q1.offer(newS);
                            v1.add(newS);
                        }
                    }
                    sChars[i] = cTmp; // 恢复
                }
            }
        }
        return 0;
    }
}
```

## [1356. 根据数字二进制下 1 的数目排序](https://leetcode-cn.com/problems/sort-integers-by-the-number-of-1-bits/)

> 排序，位运算

```java
class Solution {
    private Integer countOne(Integer i) {
        Integer num = 0;
        while (i != 0) {
            if ((i & 1) == 1)
                num++;
            i >>= 1;
        }
        return num;
    }

    public int[] sortByBits(int[] arr) {
        int len = arr.length;
        Integer[] arrInteger = new Integer[len];
        for (int i = 0; i < len; ++i) {
            arrInteger[i] = Integer.valueOf(arr[i]);
        }
        Arrays.sort(arrInteger, new Comparator<Integer>() {
            @Override
            public int compare(Integer a, Integer b) {
                int ca = countOne(a);
                int cb = countOne(b);
                if (ca > cb)
                    return 1;
                else if (ca < cb)
                    return -1;
                else {
                    return a.compareTo(b);
                }
            }
        });
        int[] ret = new int[len];
        for (int j = 0; j < len; ++j) {
            ret[j] = arrInteger[j].intValue();
        }
        return ret;
    }
}
```

巧用高位存bit数

```java
class Solution {
    public int[] sortByBits(int[] arr) {
        for (int i = 0; i < arr.length; i++) {
            arr[i] += Integer.bitCount(arr[i]) * 100000;
        }
        Arrays.sort(arr);
        for (int i = 0; i < arr.length; i++) {
            arr[i] %= 100000;
        }
        return arr;
    }
}
```

## [327. 区间和的个数](https://leetcode-cn.com/problems/count-of-range-sum/)

> 排序，树状数组，线段树，二分查找，分治算法，前缀和
>
> TODO：高级数据结构

暴力法

```java
class Solution {
    public int countRangeSum(int[] nums, int lower, int upper) {
        int n = nums.length;
        // 计算前缀和
        long[] preSum = new long[n + 1];
        long sum = 0;
        for (int i = 0; i < n; ++i) {
            sum += nums[i];
            preSum[i + 1] = sum;
        }
        int count = 0;
        for (int i = 1; i <= n; ++i) {
            for (int j = 0; j < i; ++j) {
                long tmp = preSum[i] - preSum[j];
                if (lower <= tmp && tmp <= upper) {
                    count++;
                }
            }
        }
        return count;
    }
}
```

## [717. 1比特与2比特字符](https://leetcode-cn.com/problems/1-bit-and-2-bit-characters/)

> 数组

复杂方法

```java
class Solution {
    private boolean canSplit(int[] bits, int right) {
        if (right == 0) // [0]
            return bits[0] == 0;
        if (right == 1) // [10]/[11]/[00]
            return bits[0] == 1 || (bits[0] == 0 && bits[1] == 0);
        if (bits[right] == 0) // [...00] or [...10]
            return bits[right - 1] == 0 ? canSplit(bits, right - 1) : (canSplit(bits, right - 1) || canSplit(bits, right - 2));
        else // [...11] or [...01]
            return bits[right - 1] == 1 ? canSplit(bits, right - 2) : false;
    }

    public boolean isOneBitCharacter(int[] bits) {
        int len = bits.length;
        if (len == 1 || bits[len - 2] == 0) // [0]、[00]
            return true;
        if (len == 2 && bits[len - 2] == 1) // [10]
            return false;
        return !canSplit(bits, len - 3); // [...10]
    }
}
```

简单方法（直接扫描）扫描到1就+2，扫描到0就+1。上面的方法好蠢

执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：37.6 MB, 在所有 Java 提交中击败了96.13%的用户

```java
class Solution {
    public boolean isOneBitCharacter(int[] bits) {
        int i = 0;
        while (i < bits.length - 1) {
            i += bits[i] + 1;
        }
        return i == bits.length - 1;
    }
}
```

## [258. 各位相加](https://leetcode-cn.com/problems/add-digits/)

> 数学
>
> 12,345 = 1 × (9,999 + 1) + 2 × (999 + 1) + 3 × (99 + 1) + 4 × (9 + 1) + 5.
> 12,345 = (1 × 9,999 + 2 × 999 + 3 × 99 + 4 × 9) + (1 + 2 + 3 + 4 + 5).

```java
class Solution {
    public int addDigits(int num) {
        return (num - 1) % 9 + 1;
    }
}
```

## [973. 最接近原点的 K 个点](https://leetcode-cn.com/problems/k-closest-points-to-origin/)

> 堆，排序，分治算法，快排

函数法

```java
class Solution {
    public int[][] kClosest(int[][] points, int K) {
        Arrays.sort(points, new Comparator<int[]>() {
            public int compare(int[] point1, int[] point2) {
                return (point1[0] * point1[0] + point1[1] * point1[1]) - (point2[0] * point2[0] + point2[1] * point2[1]);
            }
        });
        return Arrays.copyOfRange(points, 0, K);
    }
}
```

```java
class Solution {
    public int[][] kClosest(int[][] points, int K) {
        int[][] ans = new int[K][2];
        Arrays.sort(points, (int[] o1, int[] o2) -> (o1[0] * o1[0] + o1[1] * o1[1] - o2[0] * o2[0] - o2[1] * o2[1]));
        System.arraycopy(points, 0, ans, 0, K);
        return ans;
    }
}
```

快排 TODO

```java
class Solution {
    Random rand = new Random();

    public int[][] kClosest(int[][] points, int K) {
        int n = points.length;
        random_select(points, 0, n - 1, K);
        return Arrays.copyOfRange(points, 0, K);
    }

    public void random_select(int[][] points, int left, int right, int K) {
        int pivotId = left + rand.nextInt(right - left + 1);
        int pivot = points[pivotId][0] * points[pivotId][0] + points[pivotId][1] * points[pivotId][1];
        swap(points, right, pivotId);
        int i = left - 1;
        for (int j = left; j < right; ++j) {
            int dist = points[j][0] * points[j][0] + points[j][1] * points[j][1];
            if (dist <= pivot) {
                ++i;
                swap(points, i, j);
            }
        }
        ++i;
        swap(points, i, right);
        // [left, i-1] 都小于等于 pivot, [i+1, right] 都大于 pivot
        if (K < i - left + 1) {
            random_select(points, left, i - 1, K);
        } else if (K > i - left + 1) {
            random_select(points, i + 1, right, K - (i - left + 1));
        }
    }

    public void swap(int[][] points, int index1, int index2) {
        int[] temp = points[index1];
        points[index1] = points[index2];
        points[index2] = temp;
    }
}
```

优先队列 TODO

```java
class Solution {
    public int[][] kClosest(int[][] points, int K) {
        PriorityQueue<int[]> pq = new PriorityQueue<int[]>(new Comparator<int[]>() {
            public int compare(int[] array1, int[] array2) {
                return array2[0] - array1[0];
            }
        });
        for (int i = 0; i < K; ++i) {
            pq.offer(new int[]{points[i][0] * points[i][0] + points[i][1] * points[i][1], i});
        }
        int n = points.length;
        for (int i = K; i < n; ++i) {
            int dist = points[i][0] * points[i][0] + points[i][1] * points[i][1];
            if (dist < pq.peek()[0]) {
                pq.poll();
                pq.offer(new int[]{dist, i});
            }
        }
        int[][] ans = new int[K][2];
        for (int i = 0; i < K; ++i) {
            ans[i] = points[pq.poll()[1]];
        }
        return ans;
    }
}
```

## [514. 自由之路](https://leetcode-cn.com/problems/freedom-trail/)

> 二维动态规划，字母数组

```java
class Solution {
    public int findRotateSteps(String ring, String key) {
        // 二维dp
        // 定义：dp[i][j]表示 i [0-m) key的位置，j [0-n) ring的位置，且ring[j] == key[i]
        // 初始化：dp[0][j]=ring中首字母到0位置的最短距离；其他值默认为最大值
        // 递推方程：dp[i][j]=min(dp[i][j],i-1到i的最小值)
        // 结果：dp[m-1]中最小值+m
        int m = key.length();
        int n = ring.length();
        // 技巧，字母数组稀疏化成位置列表数组
        List<Integer>[] characters = new ArrayList[26];
        for (int i = 0; i < 26; ++i) {
            characters[i] = new ArrayList<Integer>();
        }
        for (int i = 0; i < n; ++i) {
            characters[ring.charAt(i) - 'a'].add(i);
        }
        int[][] dp = new int[m][n];
        for (int i = 0; i < m; ++i) {
            Arrays.fill(dp[i], Integer.MAX_VALUE);
        }
        // 初始化
        for (int i : characters[key.charAt(0) - 'a']) {
            dp[0][i] = Math.min(i - 0, n - i);
        }
        for (int i = 1; i < m; ++i) { // 遍历key
            for (int j : characters[key.charAt(i) - 'a']) {
                for (int k : characters[key.charAt(i - 1) - 'a']) {
                    dp[i][j] = Math.min(dp[i][j], dp[i - 1][k] + Math.min(Math.abs(j - k), n - Math.abs(j - k)));
                }
            }
        }
        return Arrays.stream(dp[m - 1]).min().getAsInt() + m;
    }
}
```

## [922. 按奇偶排序数组 II](https://leetcode-cn.com/problems/sort-array-by-parity-ii/)

> 排序，数组

执行用时：2 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：41.4 MB, 在所有 Java 提交中击败了33.37%的用户

```java
class Solution {
    public int[] sortArrayByParityII(int[] A) {
        int len = A.length;
        int[] ret = new int[len];
        int p1 = 0, p2 = 1;
        for (int i = 0; i < len; ++i) {
            if ((A[i] & 1) == 1) {
                ret[p2] = A[i];
                p2 += 2;
            } else {
                ret[p1] = A[i];
                p1 += 2;
            }
        }
        return ret;
    }
}
```

## [328. 奇偶链表](https://leetcode-cn.com/problems/odd-even-linked-list/)

> 链表

双指针分别前移后偶的头接奇的尾，注意判断是否为null

执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：38.2 MB, 在所有 Java 提交中击败了84.52%的用户

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode oddEvenList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode oddHead = head, evenHead = head.next, p1 = head, p2 = head.next;
        while (true) {
            if (p1.next != null) {
                p1.next = p1.next.next;
                if (p1.next != null)
                    p1 = p1.next;
            } else {
                p1.next = evenHead;
                break;
            }
            if (p2.next != null) {
                p2.next = p2.next.next;
                if (p2.next != null)
                    p2 = p2.next;
            } else {
                p1.next = evenHead;
                break;
            }
        }
        return oddHead;
    }
}
```

## [1122. 数组的相对排序](https://leetcode-cn.com/problems/relative-sort-array/)

> 排序，数组

计数数组

执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：38.5 MB, 在所有 Java 提交中击败了50.31%的用户

```java
class Solution {
    public int[] relativeSortArray(int[] arr1, int[] arr2) {
        int[] countArr = new int[1001];
        for (int i : arr1) {
            countArr[i]++;
        }
        int[] ret = new int[arr1.length];
        int index = 0;
        for (int j : arr2) {
            int jCount = countArr[j];
            while (jCount-- > 0) {
                ret[index++] = j;
            }
            countArr[j] = 0;
        }
        for (int i = 0; i < countArr.length; ++i) {
            if (countArr[i] != 0) {
                int count = countArr[i];
                while (count-- > 0)
                    ret[index++] = i;
            }
        }
        return ret;
    }
}
```

## [402. 移掉K位数字](https://leetcode-cn.com/problems/remove-k-digits/)

> 单调栈，双向队列，贪心

```java
class Solution {
    public String removeKdigits(String num, int k) {
        // 维护单调递增栈
        Deque<Character> deque = new LinkedList<Character>();
        int len = num.length();
        for (int i = 0; i < len; ++i) {
            Character c = num.charAt(i);
            while (!deque.isEmpty() && k > 0 && c < deque.peekLast()) {
                deque.pollLast();
                k--;
            }
            deque.offerLast(c);
        }
        for (int i = 0; i < k; ++i) {
            deque.pollLast();
        }
        StringBuilder sb = new StringBuilder();
        Boolean leadingZero = true;
        while(!deque.isEmpty()) {
            while (leadingZero && deque.peekFirst() == '0' && deque.size() > 1)
                deque.pollFirst();
            leadingZero = false;
            sb.append(deque.pollFirst());
        }
        return sb.length() == 0 ? "0" : sb.toString();
    }
}
```

注：TODO学习下这个https://leetcode-cn.com/problems/remove-k-digits/solution/yi-zhao-chi-bian-li-kou-si-dao-ti-ma-ma-zai-ye-b-5/

## [406. 根据身高重建队列](https://leetcode-cn.com/problems/queue-reconstruction-by-height/)

> 排序
>
> TODO值得再做一遍

```java
class Solution {
    public int[][] reconstructQueue(int[][] people) {
        Arrays.sort(people, new Comparator<int[]>() {
            @Override
            public int compare(int[] a, int[] b) {
                if (a[0] != b[0]) {
                    return a[0] - b[0];
                } else {
                    return b[1] - a[1];
                }
            }
        });
        int len = people.length;
        int[][] ret = new int[len][];
        for (int[] person : people) {
            int spaces = person[1] + 1;
            for (int i = 0; i < len; ++i) {
                if (ret[i] == null) {
                    --spaces;
                    if (spaces == 0) {
                        ret[i] = person;
                        break;
                    }
                }
            }
        }
        return ret;
    }
}
```

## [283. 移动零](https://leetcode-cn.com/problems/move-zeroes/)

> 数组，双指针

执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：38.7 MB, 在所有 Java 提交中击败了82.23%的用户

```java
class Solution {
    public void moveZeroes(int[] nums) {
        int left = 0, right = 0, len = nums.length;
        boolean findZero = false;
        while (right < len) {
            if (!findZero && nums[right] == 0) {
                findZero = true;
                left = right;
            } else if (left != right && findZero && nums[right] != 0) {
                nums[left] = nums[right];
                nums[right] = 0;
                left++;
            }
            right++;
        }
    }
}
```

## [148. 排序链表](https://leetcode-cn.com/problems/sort-list/)

> 排序，链表，快慢指针，归并排序

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode sortList(ListNode head) {
        // nlogn -> 归并算法
        if (head == null || head.next == null) {
            return head;
        }
        ListNode fast = head.next, slow = head;
        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;
        }
        ListNode middle = slow.next;
        slow.next = null;
        ListNode left = sortList(head);
        ListNode right = sortList(middle);
        ListNode newHead = new ListNode(0);
        ListNode res = newHead;
        while (left != null && right != null) {
            if (left.val < right.val) {
                newHead.next = left;
                left = left.next;
            } else {
                newHead.next = right;
                right = right.next;
            }
            newHead = newHead.next;
        }
        newHead.next = left != null ? left : right;
        return res.next;
    }
}
```

## [242. 有效的字母异位词](https://leetcode-cn.com/problems/valid-anagram/)

> 排序，哈希表

执行用时：2 ms, 在所有 Java 提交中击败了99.91%的用户

内存消耗：38.6 MB, 在所有 Java 提交中击败了81.39%的用户

```java
class Solution {
    public boolean isAnagram(String s, String t) {
        int[] count = new int[26];
        for (char c : s.toCharArray()) {
            count[c - 'a']++;
        }
        for (char c : t.toCharArray()) {
            count[c - 'a']--;
        }
        for (int i = 0; i < 26; ++i) {
            if (count[i] != 0)
                return false;
        }
        return true;
    }
}
```

## [452. 用最少数量的箭引爆气球](https://leetcode-cn.com/problems/minimum-number-of-arrows-to-burst-balloons/)

> 贪心，排序

```java
class Solution {
    public int findMinArrowShots(int[][] points) {
        // 按照右边界排序，从右边界往下射，删除能射中的，结果+1，然后从射不中的最小的右边界继续往下射，一次类推。
        if (points.length == 0)
            return 0;
        Arrays.sort(points, new Comparator<int[]> () {
            @Override
            public int compare(int[] a, int[] b) {
                if (a[1] > b[1])
                    return 1;
                else if (a[1] < b[1])
                    return -1;
                else
                    return 0;
            }
        });
        int right = points[0][1];
        int count = 1;
        for (int i = 1; i < points.length; ++i) {
            if (points[i][0] > right) { // 没有交集
                right = points[i][1];
                count++;
            }
        }
        return count;
    }
}
```

## [222. 完全二叉树的节点个数](https://leetcode-cn.com/problems/count-complete-tree-nodes/)

> 树，二分查找

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    private int treeHeight(TreeNode root) { // 树的高度
        return root == null ? 0 : 1 + treeHeight(root.left);
    }

    public int countNodes(TreeNode root) {
        int count = 0, h = treeHeight(root);
        while (root != null) {
            if (treeHeight(root.right) == h - 1) {
                count += 1 << (h - 1);
                root = root.right;
            } else {
                count += 1 << (h - 2);
                root = root.left;
            }
            h--;
        }
        return count;
    }
}
```

TODO位运算

## [1370. 上升下降字符串](https://leetcode-cn.com/problems/increasing-decreasing-string/)

> 排序，字符串

```java
class Solution {
    public String sortString(String s) {
        int[] count = new int[26];
        for (char c : s.toCharArray()) {
            count[c - 'a']++;
        }
        int len = s.length();
        StringBuilder sb = new StringBuilder();
        while (true) {
            if (len == 0)
                break;
            for (int i = 0; i < 26; ++i) {
                if (count[i] != 0) {
                    sb.append((char)(i + 'a'));
                    count[i]--;
                    len--;
                    if (len == 0) {
                        return sb.toString();
                    }
                }
            }
            for (int i = 25; i >= 0; --i) {
                if (count[i] != 0) {
                    sb.append((char)(i + 'a'));
                    count[i]--;
                    len--;
                    if (len == 0) {
                        return sb.toString();
                    }
                }
            }
        }
        return sb.toString();
    }
}
```

## [164. 最大间距](https://leetcode-cn.com/problems/maximum-gap/)

> 排序

直接排序O(nlogn)

```
class Solution {
    public int maximumGap(int[] nums) {
        if (nums.length < 2)
            return 0;
        Arrays.sort(nums);
        int max = 0;
        for (int i = 0; i < nums.length - 1; ++i) {
            max = Math.max(max, Math.abs(nums[i + 1] - nums[i]));
        }
        return max;
    }
}
```

桶排序

```
class Solution {
    // 线性时间复杂度和空间复杂度 不能用Arrays.sort
    public int maximumGap(int[] nums) {
        if (nums.length < 2) return 0;
        int len = nums.length;

        // 找出最大值和最小值 为了方便后面确定桶的数量
        int max = -1, min = Integer.MAX_VALUE;
        for (int i  = 0; i < len; i++) {
            max = Math.max(nums[i], max);
            min = Math.min(nums[i], min);
        }

        // 排除nums全部为一样的数字，nums = [1,1,1,1,1,1];
        if (max - min == 0) return 0;
        // 用于存放每个桶的最大值
        int[] bucketMin = new int[len - 1];
        // 用于存放每个桶的最小值
        int[] bucketMax = new int[len - 1];
        Arrays.fill(bucketMax, -1);
        Arrays.fill(bucketMin, Integer.MAX_VALUE);

        // 确定桶的间距
        int interval = (int)Math.ceil((double)(max - min) / (len - 1));
        for (int i = 0; i < len; i++) {
            // 找到每一个值所对应桶的索引
            int index = (nums[i] - min) / interval;
            if (nums[i] == min || nums[i] == max) continue;
            // 更新每个桶的数据
            bucketMax[index] = Math.max(bucketMax[index], nums[i]);
            bucketMin[index] = Math.min(bucketMin[index], nums[i]);
        }

        // maxGap 表示桶之间最大的差距
        int maxGap = 0;
        // preMax 表示前一个桶的最大值
        int preMax = min;
        for (int i = 0; i < len - 1; i++) {
            // 表示某一个桶为空
            // 但凡某一个桶不为空，都会在前面的数据中更新掉bucketMax的值
            if (bucketMax[i] == -1) continue;
            maxGap = Math.max(bucketMin[i] - preMax, maxGap);
            preMax = bucketMax[i];
        }
        // [1,10000000]
        maxGap = Math.max(maxGap, max - preMax);
        return maxGap;
    }
}
```

## [454. 四数相加 II](https://leetcode-cn.com/problems/4sum-ii/)

> 哈希表

```java
class Solution {
    public int fourSumCount(int[] A, int[] B, int[] C, int[] D) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int a: A) {
            for (int b : B) {
                Integer sum = a + b;
                map.put(sum, map.getOrDefault(sum, 0) + 1);
            }
        }
        int count = 0;
        for (int c: C) {
            for (int d: D) {
                Integer s = map.get(-(c + d));
                if (s != null && s != 0)
                    count += s;
            }
        }
        return count;
    }
}
```

## [767. 重构字符串](https://leetcode-cn.com/problems/reorganize-string/)

> TODO：堆，贪心，排序，字符串

间隔法

```java
class Solution {
    public String reorganizeString(String S) {
        int len = S.length();
        int[] charCount = new int[26];
        int threshold = (len + 1) >> 1; // 有任何数超过这个值就是“”
        int maxIndex = 0, maxCount = 0;
        for (char c: S.toCharArray()) { // 计数+阀值判断+最多值记录
            int count = ++charCount[c - 'a'];
            if (count > threshold) {
                return "";
            }
            if (count > maxCount) {
                maxIndex = c - 'a';
                maxCount = count;
            }
        }
        char[] ret = new char[len];
        int i = 0;
        while (charCount[maxIndex]-- > 0) { // 最大值放在偶数位0，2，4...
            ret[i] = (char)(maxIndex + 'a');
            i += 2;
        }
        int j = 0;
        for (j = 0; j < 26; ++j) { // 其他接着放偶数位直到放完，然后从奇数位开始放
            if (i >= len)
                i = 1;
            while (charCount[j]-- > 0) {
                ret[i] = (char)(j + 'a');
                i += 2;
                if (i >= len)
                    i = 1;
            }
        }
        return new String(ret);
    }
}
```

## [321. 拼接最大数](https://leetcode-cn.com/problems/create-maximum-number/)

> 单调栈，贪心

执行用时：8 ms, 在所有 Java 提交中击败了97.00%的用户

内存消耗：38.9 MB, 在所有 Java 提交中击败了84.56%的用户

```java
class Solution {
    // 通过单调栈找到最大序列
    private int[] getMaxArr(int[] arr, int count) {
        if (count == 0)
            return new int[0];
        int[] stack = new int[count];
        Arrays.fill(stack, 10);
        stack[0] = arr[0];
        int top = 0;
        int len = arr.length;
        for (int i = 1; i < len; ++i) {
            if (stack[top] >= arr[i]) {
                // 栈顶元素大于等于待放入的元素，有空位，则元素入栈
                if (top < count - 1) {
                    stack[++top] = arr[i];
                }
            } else {
                // 栈顶元素小于待放入的元素
                int maxDeleteNum = len - i + top - count + 1;
                if (maxDeleteNum <= 0) {
                    stack[++top] = arr[i];
                } else {
                    while (maxDeleteNum-- > 0 && top >= 0) {
                        if (stack[top] < arr[i]) {
                            top--;
                        } else {
                            break;
                        }
                    }
                    stack[++top] = arr[i];
                }
            }
        }
        return stack;
    }

    // 合并两个较大的数组
    private int[] mergeArr(int[] arr1, int[] arr2) {
        int i = 0, j = 0;
        int len1 = arr1.length, len2 = arr2.length;
        if (len1 == 0) {
            return arr2;
        }
        if (len2 == 0) {
            return arr1;
        }
        int[] ret = new int[len1 + len2];
        int count = 0;
        while (i < len1 && j < len2) {
            if (arr1[i] < arr2[j]) {
                ret[count++] = arr2[j++];
            } else if (arr1[i] > arr2[j]) {
                ret[count++] = arr1[i++];
            } else {
                int offset = 0;
                boolean flag = false;
                while ((i + offset < len1) && (j + offset < len2)) {
                    if (arr1[i + offset] > arr2[j + offset]) {
                        ret[count++] = arr1[i++];
                        flag = true;
                        break;
                    } else if (arr1[i + offset] < arr2[j + offset]) {
                        ret[count++] = arr2[j++];
                        flag = true;
                        break;
                    }
                    offset++;
                }
                if (!flag) {
                    if (len1 == i + offset) {
                        ret[count++] = arr2[j++];
                    } else {
                        ret[count++] = arr1[i++];
                    }
                }
            }
        }
        if (i < len1) {
            while(i < len1)
                ret[count++] = arr1[i++];
        }
        if (j < len2) {
            while(j < len2)
                ret[count++] = arr2[j++];
        }
        return ret;
    }

    // 取得更大的数组
    private int[] compareArr(int[] arr1, int[] arr2, int len) {
        for (int i = 0; i < len; ++i) {
            if (arr1[i] > arr2[i]) {
                return arr1;
            } else if (arr1[i] < arr2[i]) {
                return arr2;
            }
        }
        return arr1;
    }

    public int[] maxNumber(int[] nums1, int[] nums2, int k) {
        int m = nums1.length;
        int n = nums2.length;
        if (m == 0)
            return nums2;
        if (n == 0)
            return nums1;
        int[] ret = new int[k];
        // 获取可以枚举的l1,l2，满足l1+l2=k && l1<=m && l2 <= n
        for (int i = 0; i < k; ++i) {
            if (k - i > n || i > m)
                continue;
            else {
                int[] maxArr1 = getMaxArr(nums1, i);
                int[] maxArr2 = getMaxArr(nums2, k - i);
                // test(maxArr1);
                // test(maxArr2);
                int[] mergeRes = mergeArr(maxArr1, maxArr2);
                // test(mergeRes);
                ret = compareArr(ret, mergeRes, k);
                // test(ret);
            }
        }
        return ret;
    }

    private void test(int[] arr) {
        for (int i : arr) {
            System.out.print(i);
        }
        System.out.println();
    }
}
```

## [204. 计数质数](https://leetcode-cn.com/problems/count-primes/)

> 数学，埃氏筛

```java
class Solution {
    public int countPrimes(int n) {
        int[] isPrime = new int[n]; // 0~n-1的埃氏筛
        Arrays.fill(isPrime, 1);
        int count = 0;
        for (int i = 2; i < n; ++i) {
            if (isPrime[i] == 1) {
                count++;
                if ((long) i * i < n) {
                    for (int j = i * i; j < n; j += i) {
                        isPrime[j] = 0;
                    }
                }
            }
        }
        return count;
    }
}
```

```java
class Solution {
    public int countPrimes(int n) {
        List<Integer> primes = new ArrayList<Integer>(); // 线性筛，存储已有质数
        int[] isPrime = new int[n];
        Arrays.fill(isPrime, 1);
        for (int i = 2; i < n; ++i) {
            if (isPrime[i] == 1) {
                primes.add(i);
            }
            for (int j = 0; j < primes.size() && i * primes.get(j) < n; ++j) {
                isPrime[i * primes.get(j)] = 0;
                if (i % primes.get(j) == 0) {
                    break;
                }
            }
        }
        return primes.size();
    }
}
```

## [621. 任务调度器](https://leetcode-cn.com/problems/task-scheduler/)

> 贪心，数组

```java
class Solution {
    public int leastInterval(char[] tasks, int n) {
      	// 情况一：n > 所有任务数 
        // [AB ][AB ][AB]，一共最多的数组，n+1列，最后一列只有最多计数的字母个数。
        // 情况二：n <= 所有任务数 tasks.length
        int[] count = new int[26];
        for (char c: tasks) { // 字母计数数组
            count[c - 'A']++;
        }
        Arrays.sort(count);
        int maxTimes = count[25]; // 最大值
        int maxCount = 1;
        for (int i = 25; i >= 1; --i) {
            if (count[i] == count[i - 1])
                maxCount++;
            else
                break;
        }
        return Math.max((maxTimes - 1) * (n + 1) + maxCount, tasks.length);
    }
}
```

## [861. 翻转矩阵后的得分](https://leetcode-cn.com/problems/score-after-flipping-matrix/)

> 贪心算法

执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：36.3 MB, 在所有 Java 提交中击败了65.25%的用户

```java
class Solution {
    public int matrixScore(int[][] A) {
        if (A.length == 0)
            return 0;
        int m = A.length, n = A[0].length, ans = 0;
        if (n == 0)
            return 0;
        for (int i = 0; i < m; ++i) { // 首列均为1
            if (A[i][0] == 0) { // 移动该行
                A[i][0] = 1;
                for (int j = 1; j < n; ++j) {
                    A[i][j] = A[i][j] == 0 ? 1 : 0;
                }
            }
        }
        ans += m * Math.pow(2, (n - 1));
        for (int j = 1; j < n; ++j) {
            int count = 0;
            for (int i = 0; i < m; ++i) {
                count += A[i][j];
            }
            count = count >= m - count ? count : m - count;
            ans += count * Math.pow(2, (n - j - 1));
        }
        return ans;
    }
}
```

## [842. 将数组拆分成斐波那契序列](https://leetcode-cn.com/problems/split-array-into-fibonacci-sequence/)

> 贪心，字符串，回溯，dfs，剪枝

执行用时：1 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：36.7 MB, 在所有 Java 提交中击败了96.10%的用户

```java
class Solution {
    private String s = "";
    private int slen = 0;

    public List<Integer> splitIntoFibonacci(String S) {
        // DFS+回溯+剪枝
        // dfs表示从某个位置往后是否可以找到一个数加入结果列表
        // dfs(结果列表，位置)->是否可分
        // 结束条件：位置到末尾了，通过是否满足F.length>=3来判断true/false
        // 剪枝1：计算出的数字大于前两数之和，可以提前false
        // 剪枝2：计算出的数字首位为0，可以提前false
        // 剪枝3：计算出的数字超出整数范围，可以提前false
        // 回溯：如果后续无法分割，则结果列表最后一个删掉，再加个数字试试
        this.slen = S.length();
        this.s = S;
        ArrayList<Integer> ret = new ArrayList<>(16);
        return dfs(ret, 0) ? ret : new ArrayList<>();
    }

    private boolean dfs(ArrayList<Integer> ret, int index) {
        if (index == slen) {
            return ret.size() > 2;
        }
        long num = 0L;
        for (int i = index; i < this.slen; ++i) {
            num = 10 * num + (this.s.charAt(i) - '0');
            if (num > Integer.MAX_VALUE) { // 剪枝3
                return false;
            }
            if (i > index && this.s.charAt(index) == '0') { // 剪枝2
                return false;
            }
            int size = ret.size();
            if (size >= 2) {
                int lastSum = ret.get(size - 1) + ret.get(size - 2);
                if (lastSum == (int)num) {
                    ret.add((int)num);
                    if (dfs(ret, i + 1)) {
                        return true;
                    } else {
                        ret.remove(ret.size() - 1);
                    }
                } else if (lastSum > (int)num) {
                    continue;
                } else { // 剪枝1
                    return false;
                }
            }
            if (size < 2) {
                ret.add((int)num);
                if (dfs(ret, i + 1)) {
                    return true;
                } else {
                    ret.remove(ret.size() - 1);
                }
            }
        }
        return false;
    }
}
```

## [62. 不同路径](https://leetcode-cn.com/problems/unique-paths/)

> 数组，动态规划，数学

执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：35.4 MB, 在所有 Java 提交中击败了49.21%的用户

1. DP

```java
class Solution {
    public int uniquePaths(int m, int n) {
        int[][] nums = new int[m][n];
        for (int i = 0; i < m; ++i) {
            nums[i][0] = 1;
        }
        for (int j = 0; j < n; ++j) {
            nums[0][j] = 1;
        }
        for (int i = 1; i < m; ++i) {
            for (int j = 1; j < n; ++j) {
                nums[i][j] = nums[i - 1][j] + nums[i][j - 1];
            }
        }
        return nums[m - 1][n - 1];
    }
}
```

2. 数学

从m+n-2条路径中选出m-1条向下的
$$
C^{m-1}_{m+n-2}=\frac{(m+n-2)!}{(m-1)!(n-1)!}=\frac{(m+n-2)(m+n-3)\cdot\cdot\cdot n}{(m-1)(m-2)\cdot\cdot\cdot 1}
$$

```java
class Solution {
    public int uniquePaths(int m, int n) {
        long ans = 1;
        for (int x = n, y = 1; y < m; ++x, ++y) {
            ans = ans * x / y; // 必须先算乘，所以不能写成ans *= x / y
        }
        return (int)ans;
    }
}
```

## [860. 柠檬水找零](https://leetcode-cn.com/problems/lemonade-change/)

> 贪心

执行用时：2 ms, 在所有 Java 提交中击败了99.72%的用户

内存消耗：39.6 MB, 在所有 Java 提交中击败了55.38%的用户

```java
class Solution {
    public boolean lemonadeChange(int[] bills) {
        int[] store = new int[2];
        if (bills.length == 0)
            return true;
        for (int i = 0; i < bills.length; ++i) {
            if (bills[i] == 5) {
                store[0]++;
            } else if (bills[i] == 10) {
                if (store[0] > 0) {
                    store[0]--;
                } else {
                    return false;
                }
                store[1]++;
            } else if (bills[i] == 20) { // 20=5+5+5/10+5
                if (store[1] > 0 && store[0] > 0) {
                    store[1]--;
                    store[0]--;
                } else if (store[0] > 2) {
                    store[0] -= 3;
                } else {
                    return false;
                }
            }
        }
        return true;
    }
}
```

## [649. Dota2 参议院](https://leetcode-cn.com/problems/dota2-senate/)

> 贪心

```java
class Solution {
    public String predictPartyVictory(String senate) {
        int len = senate.length();
        Queue<Integer> qr = new LinkedList<>();
        Queue<Integer> qd = new LinkedList<>();
        for (int i = 0; i < len; ++i) {
            if (senate.charAt(i) == 'R')
                qr.offer(i);
            else
                qd.offer(i);
        }
        while (!qr.isEmpty() && !qd.isEmpty()) {
            Integer qri = qr.poll(), qdi = qd.poll();
            if (qri < qdi) { // R来投票
                qr.offer(qri + len);
            } else {
                qd.offer(qdi + len);
            }
        }
        return qr.isEmpty() ? "Dire" : "Radiant";
    }
}
```

## [376. 摆动序列](https://leetcode-cn.com/problems/wiggle-subsequence/)

> dp，贪心
>
> TODO：还可以用DP或者贪心来做

```java
class Solution {
    public int wiggleMaxLength(int[] nums) {
        int len = nums.length;
        if (len < 2)
            return len;
        if (len == 2) {
            return nums[0] == nums[1] ? 1 : 2;
        }
        // 数组去重
        ArrayList<Integer> al = new ArrayList<>();
        for (int i: nums) {
            if (al.isEmpty() || !al.isEmpty() && al.get(al.size() - 1) != i) {
                al.add(i);
            }
        }
        int count = al.size();
        for (int i = 0; i < al.size() - 2; ++i) {
            int a = al.get(i), b = al.get(i + 1), c = al.get(i + 2);
            if (a < b && b < c || a > b && b > c) {
                count--;
            }
        }
        return count;
    }
}
```

## [172. 阶乘后的零](https://leetcode-cn.com/problems/factorial-trailing-zeroes/)

> 数学

求n/(5的各个次幂)即可

```java
class Solution {
    public int trailingZeroes(int n) {
        int five = 0;
        for (int j = 5; j <= n; j *= 5) {
            five += n / j;
        }
        return five;
    }
}
```

## [217. 存在重复元素](https://leetcode-cn.com/problems/contains-duplicate/)

> 哈希表，数组

```java
class Solution {
    public boolean containsDuplicate(int[] nums) {
        Set<Integer> set = new HashSet<>();
        for (int i: nums) {
            if (set.contains(i)) {
                return true;
            } else {
                set.add(i);
            }
        }
        return false;
    }
}
```

## [49. 字母异位词分组](https://leetcode-cn.com/problems/group-anagrams/)

> 哈希表，字符串

```java
class Solution {
    public List<List<String>> groupAnagrams(String[] strs) {
        Map<String, List<String>> map = new HashMap<>();
        for (String str: strs) {
            char[] s = str.toCharArray();
            Arrays.sort(s);
            List<String> sameList = map.getOrDefault(new String(s), new ArrayList<>());
            sameList.add(str);
            map.put(new String(s), sameList);
        }
        return new ArrayList<List<String>>(map.values());
    }
}
```

## [738. 单调递增的数字](https://leetcode-cn.com/problems/monotone-increasing-digits/)

> 贪心

```java
class Solution {
    public int monotoneIncreasingDigits(int N) {
        char[] nChar = Integer.toString(N).toCharArray();
        int len = nChar.length;
        for (int i = 0; i < len - 1; ++i) {
            if (nChar[i + 1] < nChar[i]) {
                int start = i;
                if(i - 1 >= 0 && nChar[i] == nChar[i - 1]) {
                    start = i - 1;
                    while (start > 0 && nChar[start] == nChar[start - 1]) {
                        start--;
                    }
                }
                nChar[start] -= 1;
                for (int k = start + 1; k < len; ++k) {
                    nChar[k] = '9';
                }
                break;
            }
        }
        return Integer.parseInt(new String(nChar));
    }
}
```

## [290. 单词规律](https://leetcode-cn.com/problems/word-pattern/)

> 哈希表

执行用时：1 ms, 在所有 Java 提交中击败了98.94%的用户

内存消耗：36.4 MB, 在所有 Java 提交中击败了76.31%的用户

```java
class Solution {
    public boolean wordPattern(String pattern, String s) {
        Map<Character, String> map = new HashMap<>();
        List<String> hasList = new ArrayList<>();
        String[] sarr = s.split(" ");
        int i = 0;
        int len = pattern.length();
        if (len != sarr.length) {
            return false;
        }
        for (char c: pattern.toCharArray()) {
            String tmp = map.get(c);
            if (tmp != null) {
                if (!tmp.equals(sarr[i])) {
                    return false;
                }
            } else {
                if (hasList.contains(sarr[i])) {
                    return false;
                }
                map.put(c, sarr[i]);
                hasList.add(sarr[i]);
            }
            i++;
        }
        return true;
    }
}
```

## [389. 找不同](https://leetcode-cn.com/problems/find-the-difference/)

> 哈希表，位运算

```java
class Solution {
    public char findTheDifference(String s, String t) {
        int[] charCount = new int[26];
        for (char c: s.toCharArray()) {
            charCount[c - 'a']++;
        }
        for (char c: t.toCharArray()) {
            charCount[c - 'a']--;
            if (charCount[c - 'a'] < 0) {
                return c;
            }
        }
        return '0';
    }
}
```

TODO：位运算、求和两种方法可以优化空间到O(1)

## [48. 旋转图像](https://leetcode-cn.com/problems/rotate-image/)

> 二维数组

执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：38.3 MB, 在所有 Java 提交中击败了95.26%的用户

STEP1：找到旋转前后坐标变化：
顺时针变换为(x, y) -> (y, N-x-1)，
逆时针变换为(x, y) -> (N-y-1, x)。

STEP2：找到最少的旋转组（左上，右上，右下，左下4个需要交换的数为一组），以左上坐标为代表分别为：
(0,0)(0,1)...(0,N-2),
(1,1)(1,1)...(1,N-3),
...,
((N-1)/2,(N-1)/2)...((N-1)/2,(N-1)/2)
遍历所有旋转组分别进行四数交换处理即可。

```java
class Solution {
    public void rotate(int[][] matrix) {
        int N = matrix.length;
        if (N == 0)
            return;
        int end = (N - 1) / 2;
        int span = N - 2;
        for (int i = 0; i <= end; ++i, --span) {
            if (i == span && (N & 1) == 1)
                return;
            for (int j = i; j <= span; ++j) {
                int tmp = matrix[i][j];
                matrix[i][j] = matrix[N - j - 1][i];
                matrix[N - j - 1][i] = matrix[N - i - 1][N - j - 1];
                matrix[N - i - 1][N - j - 1] = matrix[j][N - i - 1];
                matrix[j][N - i - 1] = tmp;
            }
        }
    }
}
```

## [316. 去除重复字母](https://leetcode-cn.com/problems/remove-duplicate-letters/)

> 字符串，贪心，栈

```java
class Solution {
    public String removeDuplicateLetters(String s) {
        int[] num = new int[26];
        boolean[] visit = new boolean[26];
        int len = s.length();
        for (int i = 0; i < len; ++i) {
            num[s.charAt(i) - 'a']++;
        }
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < len; ++i) {
            char c = s.charAt(i);
            if (!visit[c - 'a']) { // 栈中没出现过
                while (sb.length() > 0 && sb.charAt(sb.length() - 1) > c) { // 当前字母比当前栈顶小
                    if(num[sb.charAt(sb.length() - 1) - 'a'] > 0) { // 栈顶后续还有，访问记录设为false，不断删除栈顶元素
                        visit[sb.charAt(sb.length() - 1) - 'a'] = false;
                        sb.deleteCharAt(sb.length() - 1);
                    } else {
                        break;
                    }
                }
                visit[c - 'a'] = true;
                sb.append(c);
            }
            num[c - 'a']--;
        }
        return sb.toString();
    }
}
```

## [746. 使用最小花费爬楼梯](https://leetcode-cn.com/problems/min-cost-climbing-stairs/)

> 数组，DP

```java
class Solution {
    public int minCostClimbingStairs(int[] cost) {
        // 转移方程：dp[i] = min(dp[i-1]+cost[i], dp[i-2]+cost[i])
        // 初始化：dp[0]=0/dp[1]=0
        int dp_0 = cost[0], dp_1 = cost[1];
        for (int i = 2; i < cost.length; ++i) {
            int tmp = dp_1;
            dp_1 = Math.min(dp_0 + cost[i], dp_1 + cost[i]);
            dp_0 = tmp;
        }
        return Math.min(dp_1, dp_0);
    }
}
```

## [103. 二叉树的锯齿形层序遍历](https://leetcode-cn.com/problems/binary-tree-zigzag-level-order-traversal/)

> 二叉树，BFS，双端队列

TODO：可以用双端队列优化

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        boolean direction = true;
        List<List<TreeNode>> retTree = new ArrayList<>();
        List<TreeNode> rootList = new ArrayList<>();
        rootList.add(root);
        retTree.add(rootList);
        int deep = 0;
        while (retTree.get(deep).size() > 0) {
            List<TreeNode> newList = new ArrayList<>();
            for (int i = retTree.get(deep).size() - 1; i >= 0; --i) {
                if (retTree.get(deep).get(i) != null) {
                    if (direction) {
                        newList.add(retTree.get(deep).get(i).right);
                        newList.add(retTree.get(deep).get(i).left);
                    } else {
                        newList.add(retTree.get(deep).get(i).left);
                        newList.add(retTree.get(deep).get(i).right);
                    }
                }
            }
            retTree.add(newList);
            deep++;
            direction = direction ? false : true;
        }
        List<List<Integer>> ret = new ArrayList<>();
        for (List<TreeNode> lt: retTree) {
            if (lt.size() == 0)
                break;
            List<Integer> l = new ArrayList<>();
            for (TreeNode t: lt) {
                if (t != null) {
                    l.add(t.val);
                }
            }
            if (l.size() > 0)
                ret.add(l);
            else
                break;
        }
        return ret;
    }
}
```

## [387. 字符串中的第一个唯一字符](https://leetcode-cn.com/problems/first-unique-character-in-a-string/)

> 哈希表，字符串

```java
class Solution {
    public int firstUniqChar(String s) {
        int[] count = new int[26];
        char[] charArr = s.toCharArray();
        for (char c: charArr) {
            count[c - 'a']++;
        }
        for (int i = 0; i < charArr.length; ++i) {
            if (count[charArr[i] - 'a'] == 1)
                return i;
        }
        return -1;
    }
}
```

## [455. 分发饼干](https://leetcode-cn.com/problems/assign-cookies/)

> 贪心

```java
class Solution {
    public int findContentChildren(int[] g, int[] s) {
        Arrays.sort(g);
        Arrays.sort(s);
        int i = 0, j = 0, count = 0;
        while (i < g.length && j < s.length) {
            if (g[i] > s[j]) {
                ++j;
            } else if (g[i] <= s[j]) {
                count++;
                ++i;
                ++j;
            }
        }
        return count;
    }
}
```

## [85. 最大矩形](https://leetcode-cn.com/problems/maximal-rectangle/)

> 单调栈，数组，哈希表，动态规划

暴力法

```java
class Solution {
    private void p(Object o) {
        System.out.println(o.toString());
    }

    public int maximalRectangle(char[][] matrix) {
        int rows = matrix.length; // 行数4
        if (rows == 0)
            return 0;
        int cols = matrix[0].length; // 列数5
        int max = 0;
        for (int j = 0; j < rows; ++j) {
            for (int i = 0; i < cols; ++i) {
                if (matrix[j][i] == '0')
                    continue; 
                int w = 1;
                int h = 1;
                while (j + h - 1 < rows) {
                    if (matrix[j + h - 1][i] == '0') {
                        break;
                    }
                    if (w == 1 && h == 1) {
                        while (i + w - 1 < cols && matrix[j + h - 1][i + w - 1] == '1') {
                            w++;
                        }
                        w--;
                    }
                    int x = 1;
                    while (x <= w && matrix[j + h - 1][i + x - 1] == '1') {
                        x++;
                    }
                    w = Math.min(w, x - 1);
                    max = Math.max(max, w * h);
                    h++;
                }
                max = Math.max(max, w * (h - 1));
            }
        }
        return max;
    }
}
```

单调栈（参考windliang）

```java
public int maximalRectangle(char[][] matrix) {
    if (matrix.length == 0) {
        return 0;
    }
    int[] heights = new int[matrix[0].length];
    int maxArea = 0;
    for (int row = 0; row < matrix.length; row++) {
        //遍历每一列，更新高度
        for (int col = 0; col < matrix[0].length; col++) {
            if (matrix[row][col] == '1') {
                heights[col] += 1;
            } else {
                heights[col] = 0;
            }
        }
        maxArea = Math.max(maxArea, largestRectangleArea(heights));
    }
    return maxArea;
}

public int largestRectangleArea(int[] heights) {
    int maxArea = 0;
    Stack<Integer> stack = new Stack<>();
    int p = 0;
    while (p < heights.length) {
        //栈空入栈
        if (stack.isEmpty()) {
            stack.push(p);
            p++;
        } else {
            int top = stack.peek();
            //当前高度大于栈顶，入栈
            if (heights[p] >= heights[top]) {
                stack.push(p);
                p++;
            } else {
                //保存栈顶高度
                int height = heights[stack.pop()];
                //左边第一个小于当前柱子的下标
                int leftLessMin = stack.isEmpty() ? -1 : stack.peek();
                //右边第一个小于当前柱子的下标
                int RightLessMin = p;
                //计算面积
                int area = (RightLessMin - leftLessMin - 1) * height;
                maxArea = Math.max(area, maxArea);
            }
        }
    }
    while (!stack.isEmpty()) {
        //保存栈顶高度
        int height = heights[stack.pop()];
        //左边第一个小于当前柱子的下标
        int leftLessMin = stack.isEmpty() ? -1 : stack.peek();
        //右边没有小于当前高度的柱子，所以赋值为数组的长度便于计算
        int RightLessMin = heights.length;
        int area = (RightLessMin - leftLessMin - 1) * height;
        maxArea = Math.max(area, maxArea);
    }
    return maxArea;
}
```

## [205. 同构字符串](https://leetcode-cn.com/problems/isomorphic-strings/)

> 哈希

```java
class Solution {
    public boolean isIsomorphic(String s, String t) {
        Map<Character, Character> map = new HashMap<>();
        Set<Character> set = new HashSet<>();
        if (s.length() != t.length())
            return false;
        for (int i = 0; i < s.length(); ++i) {
            if (!map.containsKey(s.charAt(i))) {
                if (set.contains(t.charAt(i)))
                    return false;
                map.put(s.charAt(i), t.charAt(i));
                set.add(t.charAt(i));
            } else {
                if (t.charAt(i) != map.get(s.charAt(i)))
                    return false;
            }
        }
        return true;
    }
}
```

用数组（TODO）

```java
class Solution {
    public boolean isIsomorphic(String s, String t) {
        char[] cs = s.toCharArray();
        char[] ct = t.toCharArray();
        int[] sa = new int[256];
        int[] ta = new int[256];
        for (int i = 0; i < cs.length; i++) {
            if (sa[cs[i]] != ta[ct[i]]) {
                return false;
            }
            sa[cs[i]] = i + 1;
            ta[ct[i]] = i + 1;
        }
        return true;
    }
}
```

## [188. 买卖股票的最佳时机 IV](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iv/)

> DP

TODO 重点看一下

```java
class Solution {
    public int maxProfit(int k, int[] prices) {
        // 设dp_1[i][j]表示第i天买入/持有正好j次交易的最大收益， dp_0[i][j]表示第i天卖出/未持有正好j次交易的最大收益
        // j \in [0, k]
        // dp_1[i][j] = max(dp_0[i-1][j]-prices[i], dp_1[i-1][j])
        // dp_0[i][j] = max(dp_0[i-1][j], dp_1[i-1][j-1]+prices[i])
        // 求：max(dp_0[n-1][0...k])
        // 初始化：dp_1[0][0]=-prices[0],dp_1[0][1...k]=-无穷；
        //        dp_0[0][0]=0,dp_0[0][1...k]=-无穷；
        if (prices.length == 0)
            return 0;
        int len = prices.length;
        k = Math.min(k, len / 2);
        int[][] dp_0 = new int[len][k + 1];
        int[][] dp_1 = new int[len][k + 1];
        dp_0[0][0] = 0;
        dp_1[0][0] = -prices[0];
        for (int i = 1; i <= k; ++i) {
            dp_1[0][i] = dp_0[0][i] = Integer.MIN_VALUE / 2; // 防止越界
        }
        for (int i = 1; i < len; ++i) {
            dp_1[i][0] = Math.max(dp_0[i - 1][0] - prices[i], dp_1[i - 1][0]); // j == 0讨论
            for (int j = 1; j <= k; ++j) {
                dp_1[i][j] = Math.max(dp_0[i - 1][j] - prices[i], dp_1[i - 1][j]);
                dp_0[i][j] = Math.max(dp_0[i - 1][j], dp_1[i - 1][j - 1] + prices[i]);
            }
        }
        int maxValue = 0;
        for (int i = 0; i <= k; ++i) {
            maxValue = Math.max(maxValue, dp_0[len - 1][i]);
        }
        return maxValue;
    }
}
```

## [330. 按要求补齐数组](https://leetcode-cn.com/problems/patching-array/)

> 贪心

TODO 主要是想法

```java
class Solution {
    public int minPatches(int[] nums, int n) {
        int patches = 0;
        long x = 1;
        int length = nums.length, index = 0;
        while (x <= n) { // [1..x]都可以遍历到
            if (index < length && nums[index] <= x) {
                x += nums[index];
                index++;
            } else {
                x *= 2;
                patches++;
            }
        }
        return patches;
    }
}
```

## [1046. 最后一块石头的重量](https://leetcode-cn.com/problems/last-stone-weight/)

> 最大堆

```java
class Solution {
    public int lastStoneWeight(int[] stones) {
        PriorityQueue<Integer> pl = new PriorityQueue<>((x, y) -> y - x);
        for (int i: stones) {
            pl.offer(i);
        }
        while (pl.size() > 1) {
            int y = pl.poll();
            int x = pl.poll();
            if (y != x) {
                pl.offer(y - x);
            }
        }
        return pl.isEmpty() ? 0 : pl.poll();
    }
}
```

## [435. 无重叠区间](https://leetcode-cn.com/problems/non-overlapping-intervals/)

> 贪心

执行用时：1 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：38.4 MB, 在所有 Java 提交中击败了69.35%的用户

```java
class Solution {
    public int eraseOverlapIntervals(int[][] intervals) {
        Arrays.sort(intervals, new Comparator<int[]>() {
            public int compare(int[] a, int[] b) {
                return a[0] - b[0];
            }
        });
        int count = 0, i = 0, j = 1, n = intervals.length;
        while (j < n) {
            int[] a = intervals[i];
            int[] b = intervals[j];
            if (b[0] >= a[1]) { // a和b不覆盖，i=j，j++
                i = j;
            } else { // a和b覆盖，删除[1]大的，count++，i置为未删除的那个，j++
                count++;
                if (b[1] < a[1]) // 删除a
                    i = j;
            }
            j++;
        }
        return count;
    }
}
```

## [605. 种花问题](https://leetcode-cn.com/problems/can-place-flowers/)

> 数组

执行用时：1 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：39.8 MB, 在所有 Java 提交中击败了78.54%的用户

```java
class Solution {
    public boolean canPlaceFlowers(int[] flowerbed, int n) {
        int fn = flowerbed.length;
        if (n == 0)
            return true;
        if (fn == 1)
            return flowerbed[0] == 0;
        for (int i = 0; i < fn; ++i) {
            if ((i == 0 && flowerbed[0] == 0 && flowerbed[1] == 0) ||
                (i == fn - 1 && flowerbed[i] == 0 && flowerbed[i - 1] == 0) ||
                (i != 0 && i != fn - 1 && flowerbed[i] == 0 && flowerbed[i - 1] == 0 && flowerbed[i + 1] == 0)) {
                flowerbed[i] = 1;
                if (--n == 0) // 提前退出
                    return true;
            }
        }
        return false;
    }
}
```



## [239. 滑动窗口最大值](https://leetcode-cn.com/problems/sliding-window-maximum/)

> 滑动窗口，单调栈

先判断n==0和数组为1的情况。
然后从第一个位置开始模拟种植，如有：
1.刚开始种植且第一个和第二个位置都为0，即可种植在第一个位置
2.最后位置种植且最后一个和倒数第二个位置都为0，即可种植在最后一个位置
3.在中间种植且自己和前后都为0，即可种植在中间
种植操作：
1.修改当前位置为1
2.n--，判断n已经扣减为0，是则返回true
如果循环结束还无法返回true，则返回false

```java
class Solution {
    public int[] maxSlidingWindow(int[] nums, int k) {
        int n = nums.length;
        if (n < 2)
            return nums;
        int[] ans = new int[n - k + 1];
        Deque<Integer> deque = new LinkedList<Integer>(); // k单调递减栈（存位置）
        for (int i = 0; i < n; ++i) { // i表示右界
            while(!deque.isEmpty() && nums[deque.peekLast()] < nums[i]) { // 加入一个新值，维护栈
                deque.pollLast();
            }
            deque.offerLast(i); // 加入新值
            if (deque.peekFirst() <= i - k) { // 判断队首有效性
                deque.pollFirst();
            }
            if (i >= k - 1) { // k-1后持续加入最大值
                ans[i + 1 - k] = nums[deque.peekFirst()];
            }
        }
        return ans;
    }
}
```

## [86. 分隔链表](https://leetcode-cn.com/problems/partition-list/)

> 链表，双指针

用ps记录小数链，pb记录大数链，将原始链条分开，然后大数链接到小数链尾部

执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：37.4 MB, 在所有 Java 提交中击败了97.45%的用户

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public ListNode partition(ListNode head, int x) {
        ListNode ps = new ListNode(0), pb = new ListNode(0);
        ListNode psh = ps, pbh = pb;
        while (head != null) {
            if (head.val < x) {
                ps.next = head;
                ps = ps.next;
            } else {
                pb.next = head;
                pb = pb.next;
            }
            ListNode tmp = head;
            head = head.next;
            tmp.next = null; // 打断原始连接，不然会有环
        }
        ps.next = pbh.next;
        return psh.next;
    }
}
```

## [509. 斐波那契数](https://leetcode-cn.com/problems/fibonacci-number/)

> 数组

```java
class Solution {
    public int fib(int n) {
        if (n < 2)
            return n;
        int a = 0, b = 1;
        for (int i = 0; i < n - 2; ++i) {
            int ans = a + b;
            a = b;
            b = ans;
        }
        return a + b;
    }
}
```

## [830. 较大分组的位置](https://leetcode-cn.com/problems/positions-of-large-groups/)

> 数组

执行用时：1 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：38.3 MB, 在所有 Java 提交中击败了95.91%的用户

```java
class Solution {
    public List<List<Integer>> largeGroupPositions(String s) {
        List<List<Integer>> ans = new ArrayList<>();
        int n = s.length();
        if (n < 3)
            return ans;
        for (int i = 2; i < n; ++i) {
            char c = s.charAt(i);
            if (c == s.charAt(i - 1) && c == s.charAt(i - 2)) {
                int j = i - 2;
                while (i < n && s.charAt(i) == c) {
                    ++i;
                }
                ans.add(Arrays.asList(j, i - 1));
                i += 1;
            }
        }
        return ans;
    }
}
```

## [399. 除法求值](https://leetcode-cn.com/problems/evaluate-division/)

> 并查集，图

执行用时：1 ms, 在所有 Java 提交中击败了97.24%的用户

内存消耗：37.3 MB, 在所有 Java 提交中击败了46.02%的用户

```java
class Solution {
    public double[] calcEquation(List<List<String>> equations, double[] values, List<List<String>> queries) {
        // 第一步：构造并查集
        int len = equations.size() * 2;
        UnionFind unionFind = new UnionFind(len);
        // 第二步：对equations中元素统一编码为int
        Map<String, Integer> map = new HashMap<>(len);
        int i = 0, j = 0;
        for (List<String> equation: equations) {
            String str1 = equation.get(0);
            String str2 = equation.get(1);
            if (!map.containsKey(str1)) {
                map.put(str1, i);
                ++i;
            }
            if (!map.containsKey(str2)) {
                map.put(str2, i);
                ++i;
            }
            unionFind.union(map.get(str1), map.get(str2), values[j++]);
        }
        // 第三步：根据提问查找结果
        double[] res = new double[queries.size()];
        int k = 0;
        for (List<String> query: queries) {
            Integer v1 = map.get(query.get(0));
            Integer v2 = map.get(query.get(1));
            if (v1 == null || v2 == null) {
                res[k++] = -1.0d;
            } else {
                res[k++] = unionFind.isConnected(v1, v2);
            }
        }
        return res;
    }

    private class UnionFind { // 并查集内部类
        int[] parent; // 每个元素的直接父类集合
        double[] weight; // 每个元素到直接父类的权重
        
        // 构造方法：根据容量构造出均指向自己的并查集
        UnionFind(int n) {
            parent = new int[n];
            weight = new double[n];
            for (int i = 0; i < n; ++i) {
                parent[i] = i;
                weight[i] = 1.0d;
            }
        }
        
        // 合并方法
        public void union(int x, int y, double val) {
            int px = find(x); // 找到x的直接父类
            int py = find(y); // 找到y的直接父类
            if (px == py)
                return;
            parent[px] = py;
            weight[px] = val * weight[y] / weight[x];
        }

        // 查找直接父类并路径压缩
        private int find(int x) {
            if (x != parent[x]) { // 上面还有人
                int tmpPx = parent[x]; // 记录临时的父元素
                parent[x] = find(parent[x]); // 递归发现父元素并更新parent[]
                weight[x] *= weight[tmpPx]; // 当前元素weight是乘上其原始父元素（可能已经变成子元素了）的weight值
            }
            return parent[x];
        }

        // 检查是否在一个集中（所有值合并后），在的话返回相除结果
        public double isConnected(int x, int y) {
            int px = find(x);
            int py = find(y);
            if (px == py) {
                return weight[x] / weight[y];
            }
            return -1.0d;
        }
    }
}
```

## [547. 省份数量](https://leetcode-cn.com/problems/number-of-provinces/)

> 并查集，图，dfs

执行用时：1 ms, 在所有 Java 提交中击败了99.49%的用户

内存消耗：39.3 MB, 在所有 Java 提交中击败了69.07%的用户

```java
class Solution {
    public int findCircleNum(int[][] isConnected) {
        int n = isConnected.length;
        int[] parents = new int[n];
        // 并查集初始化
        for (int i = 0; i < n; ++i) {
            parents[i] = i;
        }
        // 并查集合并
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                if (isConnected[i][j] == 1)
                    union(parents, i, j);
            }
        }
        int count = 0;
        for (int i = 0; i < n; ++i) {
            if (i == parents[i])
                count++;
        }
        return count;
    }

    // 并查集联合
    private void union(int[] parents, int x, int y) {
        parents[find(parents, x)] = find(parents, y);
    }

    // 并查集找父且压缩路径
    private int find(int[] parents, int x) {
        if (x != parents[x]) {
            parents[x] = find(parents, parents[x]);
        }
        return parents[x];
    }
}
```

## [189. 旋转数组](https://leetcode-cn.com/problems/rotate-array/)

> 数组

整体移动法（效率不算高）

执行用时：1 ms, 在所有 Java 提交中击败了54.78%的用户

内存消耗：38.8 MB, 在所有 Java 提交中击败了81.24%的用户

```java
class Solution {
    private void swap(int[] nums, int i, int j) {
        int tmp = nums[i];
        nums[i] = nums[j];
        nums[j] = tmp;
    }

    // 整体浮动
    private void blockFloat(int[] nums, int start, int end, int k, boolean direction) {
        if (k == 0)
            return;
        while (end - start + 1 - k >= k) {
            if (direction) { // 向左浮动
                for (int j = 0; j < k; ++j) {
                    swap(nums, end - j, end - j - k);
                }
                end -= k;
            } else { // 向右浮动
                for (int j = 0; j < k; ++j) {
                    swap(nums, start + j, start + j + k);
                }
                start += k;
            }
        }
        blockFloat(nums, start, end, end - start - k + 1, !direction);
    }

    public void rotate(int[] nums, int k) {
        int n = nums.length;
        k %= n;
        blockFloat(nums, 0, n - 1, k, true);
    }
}
```

使用额外数组

执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：39.1 MB, 在所有 Java 提交中击败了32.37%的用户

```java
class Solution {
    public void rotate(int[] nums, int k) {
        int n = nums.length;
        int[] newArr = new int[n];
        for (int i = 0; i < n; ++i) {
            newArr[(i + k) % n] = nums[i];
        }
        System.arraycopy(newArr, 0, nums, 0, n);
    }
}
```

环装替代

执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：38.7 MB, 在所有 Java 提交中击败了93.74%的用户

```java
class Solution {
    public void rotate(int[] nums, int k) {
        int n = nums.length;
        k = k % n;
        int count = gcd(k, n);
        for (int start = 0; start < count; ++start) {
            int current = start;
            int prev = nums[start];
            do {
                int next = (current + k) % n;
                int temp = nums[next];
                nums[next] = prev;
                prev = temp;
                current = next;
            } while (start != current);
        }
    }

    public int gcd(int x, int y) {
        return y > 0 ? gcd(y, x % y) : x;
    }
}
```

数组翻转

执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：38.6 MB, 在所有 Java 提交中击败了96.24%的用户

```java
class Solution {
    public void rotate(int[] nums, int k) {
        k %= nums.length;
        reverse(nums, 0, nums.length - 1);
        reverse(nums, 0, k - 1);
        reverse(nums, k, nums.length - 1);
    }

    public void reverse(int[] nums, int start, int end) {
        while (start < end) {
            int temp = nums[start];
            nums[start] = nums[end];
            nums[end] = temp;
            start += 1;
            end -= 1;
        }
    }
}
```

## [123. 买卖股票的最佳时机 III](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iii/)

> dp

```java
class Solution {
    public int maxProfit(int[] prices) {
        // 递推方程:
        // 状态分5种——
        // 从未交易过:dp[i][0] = 0（不需要记录，最后比较0就行）
        // 一次买入状态:dp[i][1] = max(dp[i-1][1],-prices[i]) 之前买入过不操作/今日买入
        // 一次买卖状态:dp[i][2] = max(dp[i-1][2],dp[i-1][1]+prices[i]) 昨日一次买卖未操作/之前没卖，今天卖出
        // 一次买卖后再买状态:dp[i][3] = max(dp[i-1][3],dp[i-1][2]-prices[i])
        // 二次买卖状态:dp[i][4] = max(dp[i-1][4],dp[i-1][3]+prices[i])
        
        // 初始化dp[0][1]=-prices[0],dp[0][2]=-无穷,dp[0][3]=-无穷,dp[0][4]=-无穷
        // 求max(0, dp[n-1][2], dp[n-1][4])
        int n = prices.length;
        if (n == 0)
            return 0;
        int dp_i_1 = -prices[0], dp_i_2 = -Integer.MIN_VALUE;
        int dp_i_3 = -Integer.MIN_VALUE, dp_i_4 = -Integer.MIN_VALUE;
        for (int i = 1; i < n; ++i) {
            dp_i_1 = Math.max(dp_i_1, -prices[i]);
            dp_i_2 = Math.max(dp_i_2, dp_i_1 + prices[i]);
            dp_i_3 = Math.max(dp_i_3, dp_i_2 - prices[i]);
            dp_i_4 = Math.max(dp_i_4, dp_i_3 + prices[i]);
        }
        int tmp = Math.max(dp_i_2, dp_i_4);
        return tmp < 0 ? 0 : tmp;
    }
}
```

## [228. 汇总区间](https://leetcode-cn.com/problems/summary-ranges/)

> 数组

TODO：使用StringBuilder可能会提高效率

```java
class Solution {
    public List<String> summaryRanges(int[] nums) {
        int n = nums.length;
        List<String> ret = new ArrayList<>();
        if (n == 0)
            return ret;
        if (n == 1) {
            ret.add(nums[0] + "");
            return ret;
        }
        int l = 0, r = 0;
        for (int i = 1; i < n; ++i) {
            if (nums[i] == nums[i - 1] + 1) {
                r++;
            } else {
                if (l != i - 1) {
                    ret.add(nums[l] + "->" + nums[i - 1]);
                } else {
                    ret.add(nums[l] + "");
                }
                l = i;
                r = i;
            }
        }
        if (l != n - 1) {
            ret.add(nums[l] + "->" + nums[n - 1]);
        } else {
            ret.add(nums[l] + "");
        }
        return ret;
    }
}
```

## [1202. 交换字符串中的元素](https://leetcode-cn.com/problems/smallest-string-with-swaps/)

> 并查集，数组

```java
class Solution {
    private int[] parents; // 并查集(索引)
    private int[] ranks; // 按秩合并

    public String smallestStringWithSwaps(String s, List<List<Integer>> pairs) {
        int n = s.length();
        parents = new int[n];
        ranks = new int[n];
        // 并查集初始化
        for (int i = 0; i < n; ++i) {
            parents[i] = i;
            ranks[i] = 1;
        }
        // 聚类合并
        for (List<Integer> l: pairs) {
            Integer v1 = l.get(0);
            Integer v2 = l.get(1);
            union(v1, v2);
        }
        // 每个类记录优先队列
        Map<Integer, PriorityQueue<Character>> map = new HashMap<>(n);
        for (int i = 0; i < n; ++i) {
            // int pi = find(i);
            // if (map.containsKey(pi)) {
            //     Queue q = map.get(pi);
            //     q.offer(s.charAt(i));
            // } else {
            //     PriorityQueue q = new PriorityQueue<>();
            //     q.offer(s.charAt(i));
            //     map.put(pi, q);
            // }
            map.computeIfAbsent(find(i), key -> new PriorityQueue<>()).offer(s.charAt(i)); // 一句话代替
        }
        // 组合字符串
        StringBuilder sb = new StringBuilder(n);
        for (int i = 0; i < n; ++i) {
            int pi = find(i);
            PriorityQueue pq = map.get(pi);
            sb.append(pq.poll());
        }
        return sb.toString();
    }

    private int find(int x) {
        if (parents[x] != x) {
            parents[x] = find(parents[x]);
        }
        return parents[x];
    }

    private void union(int a, int b) {
        int p1 = find(a);
        int p2 = find(b);
        if (p1 == p2)
            return;
        if (ranks[p1] == ranks[p2]) {
            parents[p1] = p2;
            ranks[p1]++;
        } else if (ranks[p1] > ranks[p2]) {
            parents[p2] = p1;
        } else {
            parents[p1] = p2;
        }
    }
}
```

## [1672. 最富有客户的资产总量](https://leetcode-cn.com/problems/richest-customer-wealth/)

> 数组

执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：37.9 MB, 在所有 Java 提交中击败了89.55%的用户

```java
class Solution {
    public int maximumWealth(int[][] accounts) {
        int max = 0;
        for (int i = 0; i < accounts.length; ++i) {
            int tmpMax = 0;
            for (int j = 0; j < accounts[0].length; ++j) {
                tmpMax += accounts[i][j];
            }
            max = Math.max(max, tmpMax);
        }
        return max;
    }
}
```

## [1678. 设计 Goal 解析器](https://leetcode-cn.com/problems/goal-parser-interpretation/)

> 字符串

```java
class Solution {
    public String interpret(String command) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < command.length(); ++i) {
            if (command.charAt(i) == 'G') {
                sb.append('G');
            } else {
                if (command.charAt(i + 1) == ')') {
                    sb.append('o');
                    i++;
                } else {
                    sb.append("al");
                    i += 3;
                }
            }
        }
        return sb.toString();
    }
}
```

## [1203. 项目管理](https://leetcode-cn.com/problems/sort-items-by-groups-respecting-dependencies/)

> 拓扑排序，BFS，图

```java
class Solution {
    private ArrayList<Integer> topoLogicalSort(List<Integer>[] adj, int[] inDegree, int n) {
        ArrayList<Integer> res = new ArrayList<>();
        Queue<Integer> q = new LinkedList<>();
        for (int i = 0; i < n; ++i) {
            if (inDegree[i] == 0) {
                q.offer(i);
            }
        }
        while (!q.isEmpty()) {
            Integer item = q.poll();
            res.add(item);
            if (adj[item] != null) {
                for (Integer i: adj[item]) {
                    inDegree[i]--;
                    if (inDegree[i] == 0) {
                        q.offer(i);
                    }
                }
            }
        }
        if (res.size() == n)
            return res;
        else
            return new ArrayList<>();
    }

    public int[] sortItems(int n, int m, int[] group, List<List<Integer>> beforeItems) {
        // 满足拓扑排序条件：有向无环图（DAG）先后排序
        // 第一步：将所有无组的项目标注为某个独立组（小技巧）
        for (int i = 0; i < n; ++i) {
            if (group[i] == -1) {
                group[i] = m;
                m++;
            }
        }
        // 第二步：构建组邻接表并拓扑排序
        List<Integer>[] groupAdj = new ArrayList[m];
        int[] groupInDegree = new int[m];
        for (int i = 0; i < group.length; ++i) {
            int currentGroup = group[i];
            for (Integer j: beforeItems.get(i)) {
                int jGroup = group[j];
                if (currentGroup != jGroup) {
                    if (groupAdj[jGroup] == null)
                        groupAdj[jGroup] = new ArrayList<>();
                    groupAdj[jGroup].add(currentGroup);
                    groupInDegree[currentGroup]++;
                }
            }
        }
        ArrayList<Integer> groupTopo = topoLogicalSort(groupAdj, groupInDegree, m);
        if (groupTopo.size() == 0)
            return new int[0];
        // 第三步：构建项目邻接表并拓扑排序
        List<Integer>[] itemAdj = new ArrayList[n]; // 邻接表（后继）
        int[] itemInDegree = new int[n]; // 入度
        for (int i = 0; i < beforeItems.size(); ++i) {
            for (Integer it: beforeItems.get(i)) {
                if (itemAdj[it] == null)
                    itemAdj[it] = new ArrayList<>();
                itemAdj[it].add(i);
                itemInDegree[i]++;
            }
        }
        ArrayList<Integer> itemTopo = topoLogicalSort(itemAdj, itemInDegree, n);
        if (itemTopo.size() == 0)
            return new int[0];
        // 第四步：组和项目关系
        Map<Integer, List<Integer>> map = new HashMap<>();
        for (Integer item: itemTopo) {
            map.computeIfAbsent(group[item], key -> new ArrayList<>()).add(item);
        }
        // 第五步：构造结果
        int[] res = new int[n];
        int count = 0;
        for (int groupIndex: groupTopo) {
            List<Integer> items = map.getOrDefault(groupIndex, new ArrayList<>());
            for (Integer item: items)
                res[count++] = item;
        }
        return res;
    }
}
```

## [684. 冗余连接](https://leetcode-cn.com/problems/redundant-connection/)

> 并查集，树，图

```java
class Solution {
    public int[] findRedundantConnection(int[][] edges) {
        // 并查集
        // 初始时候，每个点单独是自己的连通分量
        // 合并，如果两个点不属于一个连同分量则合并，否则有环
        int n = edges.length; // 点和边数量一致，都是n
        int[] parents = new int[n + 1];
        for (int i = 1; i <= n; ++i) {
            parents[i] = i;
        }
        for (int i = 0; i < n; ++i) {
            int pu = find(parents, edges[i][0]);
            int pv = find(parents, edges[i][1]);
            if (pu != pv) {
                parents[pu] = pv;
            } else {
                return edges[i]; // 因为只可能有一个环，成环必然是最后一个满足条件的边加入
            }
        }
        return new int[0];
    }

    private int find(int[] parents, int x) {
        if (parents[x] != x) {
            parents[x] = find(parents, parents[x]);
        }
        return parents[x];
    }
}
```

## [1018. 可被 5 整除的二进制前缀](https://leetcode-cn.com/problems/binary-prefix-divisible-by-5/)

> 数组

```java
class Solution {
    public List<Boolean> prefixesDivBy5(int[] A) {
        List<Boolean> res = new ArrayList<>();
        int n = A.length;
        int tmp = 0;
        for (int i = 0; i < n; ++i) {
            tmp = ((tmp * 2) + A[i]) % 5; // 只保留余数
            res.add(tmp == 0);
        }
        return res;
    }
}
```

## [947. 移除最多的同行或同列石头](https://leetcode-cn.com/problems/most-stones-removed-with-same-row-or-column/)

> 并查集

```java
class Solution {
    public int removeStones(int[][] stones) {
        // 并查集
        UnionFind uf = new UnionFind();
        for (int[] stone: stones) {
            uf.union(stone[0] + 10001, stone[1]);
        }
        return stones.length - uf.getCount();
    }

    class UnionFind {
        private Map<Integer, Integer> parents;
        private int count;

        UnionFind() {
            parents = new HashMap<>();
            count = 0;
        }

        public int getCount() {
            return this.count;
        }
        
        public void union(int a, int b) {
            int pa = find(a);
            int pb = find(b);
            if (pa != pb) {
                parents.put(pa, pb);
                count--;
            }
        }

        public int find(int x) {
            if (!parents.containsKey(x)) {
                parents.put(x, x);
                count++;
            }
            if (x != parents.get(x)) {
                parents.put(x, find(parents.get(x)));
            }
            return parents.get(x);
        }
    }
}
```

## [1232. 缀点成线](https://leetcode-cn.com/problems/check-if-it-is-a-straight-line/)

> 数学

向量法

```java
class Solution {
    public boolean checkStraightLine(int[][] coordinates) {
        int len = coordinates.length;
        if (len == 2)
            return true;
        for (int i = 2; i < len; ++i) {
            int[] x1 = coordinates[i - 2];
            int[] x2 = coordinates[i - 1];
            int[] x3 = coordinates[i];
            int[] a1 = {x2[0] - x1[0], x2[1] - x1[1]};
            int[] a2 = {x3[0] - x2[0], x3[1] - x2[1]};
            if (a2[0] * a1[1] != a2[1] * a1[0])
                return false;
        }
        return true;
    }
}
```

坐标方程法

```java

class Solution {
    public boolean checkStraightLine(int[][] coordinates) {
        int deltaX = coordinates[0][0], deltaY = coordinates[0][1];
        int n = coordinates.length;
        for (int i = 0; i < n; i++) { // 变换到过O点的直线
            coordinates[i][0] -= deltaX;
            coordinates[i][1] -= deltaY;
        }
        int A = coordinates[1][1], B = -coordinates[1][0];
        for (int i = 2; i < n; i++) {
            int x = coordinates[i][0], y = coordinates[i][1];
            if (A * x + B * y != 0) {
                return false;
            }
        }
        return true;
    }
}
```

## [721. 账户合并](https://leetcode-cn.com/problems/accounts-merge/)

> 并查集

```java
class Solution {
    public List<List<String>> accountsMerge(List<List<String>> accounts) {
        // 预处理：给emails编号，记录emails的账户名
        Map<String, Integer> mapNumber = new HashMap<>();
        Map<String, String> mapName = new HashMap<>();
        int count = 0;
        for (List<String> ls: accounts) {
            String accName = ls.get(0);
            for (int i = 1; i < ls.size(); ++i) {
                String email = ls.get(i);
                if (!mapNumber.containsKey(email)) {
                    mapNumber.put(email, count++);
                    mapName.put(email, accName);
                }
            }
        }
        // 对编完号的emails进行Union
        int emailLen = mapNumber.keySet().size();
        UnionFind uf = new UnionFind(emailLen);
        for (List<String> ls: accounts) {
            Integer first = mapNumber.get(ls.get(1));
            for (int i = 2; i < ls.size(); ++i) {
                uf.union(first, mapNumber.get(ls.get(i)));
            }
        }
        // 记录所有的连通分量（爸爸和孩子们）
        Map<Integer, List<String>> pcs = new HashMap<>();
        for (String email: mapNumber.keySet()) {
            Integer i = mapNumber.get(email);
            int p = uf.find(i);
            List<String> ls = pcs.getOrDefault(p, new ArrayList<>());
            ls.add(email);
            pcs.put(p, ls);
        }
        // 按照要求输出
        List<List<String>> res = new ArrayList<>();
        for (List<String> ls: pcs.values()) {
            List<String> tmp = new ArrayList<>();
            Collections.sort(ls);
            tmp.add(mapName.get(ls.get(0)));
            tmp.addAll(ls);
            res.add(tmp);
        }
        return res;
    }

    private class UnionFind {
        private int[] parents;

        UnionFind(int n) {
            parents = new int[n];
            for (int i = 0; i < n; ++i) {
                parents[i] = i;
            }
        }
        
        public int find(int x) {
            if (parents[x] != x) {
                parents[x] = find(parents[x]);
            }
            return parents[x];
        }

        public void union(int a, int b) {
            parents[find(a)] = find(b);
        }
    }
}
```

## [1584. 连接所有点的最小费用](https://leetcode-cn.com/problems/min-cost-to-connect-all-points/)

> 最小生成树，Prim，Kruskal，并查集

```java
class Solution {
    private int getDis(int i, int j, int[][]  points) {
        int xi = points[i][0];
        int xj = points[j][0];
        int yi = points[i][1];
        int yj = points[j][1];
        return Math.abs(xi - xj) + Math.abs(yi - yj);
    }

    public int minCostConnectPoints(int[][] points) {
        // 初始化所有边
        int n = points.length;
        UnionFind uf = new UnionFind(n);
        List<Edge> le = new ArrayList<>();
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                int d = getDis(i, j, points);
                le.add(new Edge(d, i, j));
            }
        }
        // 对列表排序
        Collections.sort(le, new Comparator<Edge>(){
            @Override
            public int compare(Edge e1, Edge e2) {
                return e1.dis - e2.dis;
            }
        });
        // Kruskal：加小边，如果连通则加入
        int res = 0, count = 1;
        for (Edge e: le) {
            if (uf.union(e.x, e.y)) {
                res += e.dis;
                count++;
                if (count == n) {
                    break;
                }
            }
        }
        return res;
    }

    private class UnionFind {
        int[] parents;

        UnionFind(int n) {
            parents = new int[n];
            for (int i = 0; i < n; ++i) {
                parents[i] = i;
            }
        }

        public boolean union(int a, int b) {
            int pa = find(a);
            int pb = find(b);
            if (pa == pb)
                return false;
            parents[pa] = pb;
            return true;
        }

        public int find(int x) {
            if (parents[x] != x) {
                parents[x] = find(parents[x]);
            }
            return parents[x];
        }
    }

    private class Edge {
        int dis, x, y;
        Edge(int dis, int x, int y) {
            this.dis = dis;
            this.x = x;
            this.y = y;
        }
    }
}
```

TODO：Prim算法，图优化

## [628. 三个数的最大乘积](https://leetcode-cn.com/problems/maximum-product-of-three-numbers/)

> 数组

排序法

```java
class Solution {
    public int maximumProduct(int[] nums) {
        Arrays.sort(nums);
        // 最大的三个 vs 最小两个*最大一个
        int len = nums.length;
        int tmp1 = nums[len - 1] * nums[len - 2] * nums[len - 3];
        int tmp2 = nums[0] * nums[1] * nums[len - 1];
        return Math.max(tmp1, tmp2);
    }
}
```

直接找到5个数

```java
class Solution {
    public int maximumProduct(int[] nums) {
        int min1 = 1000, min2 = 1000, max3 = -1000, max2 = -1000, max1 = -1000;
        for (int num: nums) {
            if (num < min1) {
                min2 = min1;
                min1 = num;
            } else if (num < min2) {
                min2 = num;
            }
            if (num > max1) {
                max3 = max2;
                max2 = max1;
                max1 = num;
            } else if (num > max2) {
                max3 = max2;
                max2 = num;
            } else if (num > max3) {
                max3 = num;
            }
        }
        return Math.max(min1 * min2 * max1, max3 * max2 * max1);
    }
}
```

## [989. 数组形式的整数加法](https://leetcode-cn.com/problems/add-to-array-form-of-integer/)

> 数组

模拟相加

```java
class Solution {
    public List<Integer> addToArrayForm(int[] A, int K) {
        List<Integer> res = new ArrayList<>();
        int len = A.length;
        for (int i = len - 1; i >= 0; --i) {
            int tmp = K % 10 + A[i];
            K /= 10;
            if (tmp >= 10) {
                K++;
                tmp -= 10;
            }
            res.add(tmp);
        }
        for (; K > 0; K /= 10) {
            res.add(K % 10);
        }
        Collections.reverse(res);
        return res;
    }
}
```

把K加到数组最后一个

```java
class Solution {
    public List<Integer> addToArrayForm(int[] A, int K) {
        List<Integer> res = new ArrayList<Integer>();
        int n = A.length;
        for (int i = n - 1; i >= 0 || K > 0; --i, K /= 10) {
            if (i >= 0) {
                K += A[i];
            }
            res.add(K % 10);
        }
        Collections.reverse(res);
        return res;
    }
}
```

## [1319. 连通网络的操作次数](https://leetcode-cn.com/problems/number-of-operations-to-make-network-connected/)

> 并查集

```java
class Solution {
    public int makeConnected(int n, int[][] connections) {
      	if (connections.length < n - 1)
            return -1;
        UnionFind uf = new UnionFind(n);
        int res = 0;
        int freeLine = 0;
        for (int[] connect: connections) {
            if (uf.isConnected(connect[0], connect[1])) { // 已经连接
                freeLine++;
            }
            uf.union(connect[0], connect[1]);
        }
        for (int i = 0; i < n; ++i) {
            if (uf.find(i) != 0) {
                if (freeLine > 0) {
                    freeLine--;
                    uf.union(i, 0);
                    res++;
                } else
                    return -1;
            }
        }
        return res;
    }

    private class UnionFind {
        private int[] parents;

        UnionFind(int n) {
            parents = new int[n];
            for (int i = 0; i < n; ++i) {
                parents[i] = i;
            }
        }

        public int find(int x) {
            if (parents[x] != x) {
                parents[x] = find(parents[x]);
            }
            return parents[x];
        }

        public boolean isConnected(int a, int b) {
            return find(a) == find(b);
        }

        public void union(int a, int b) {
            int pa = find(a);
            int pb = find(b);
            if (pb != pa) {
                if (pa > pb)
                    parents[pa] = pb;
                else
                    parents[pb] = pa;
            }
        }
    }
}
```

## [1689. 十-二进制数的最少数目](https://leetcode-cn.com/problems/partitioning-into-minimum-number-of-deci-binary-numbers/)

> 数组

直接求

```java
class Solution {
    public int minPartitions(String n) {
        int res = 0;
        for (char c: n.toCharArray()) {
            int tmp = c - '0';
            if (tmp > res)
                res = tmp;
        }
        return res;
    }
}
```

排序后取最大

```java
class Solution {
    public int minPartitions(String n) {
        char[] nc = n.toCharArray();
        Arrays.sort(nc);
        return nc[nc.length - 1] - '0';
    }
}
```

## [674. 最长连续递增序列](https://leetcode-cn.com/problems/longest-continuous-increasing-subsequence/)

> 数组

```java
class Solution {
    public int findLengthOfLCIS(int[] nums) {
        int res = 0, n = nums.length;
        if (n <= 1)
            return n;
        int tmp = 1;
        for (int i = 1; i < n; ++i) {
            if (nums[i] > nums[i - 1]) {
                tmp++;
            } else {
                res = Math.max(res, tmp);
                tmp = 1;
            }
        }
        res = Math.max(res, tmp);
        return res;
    }
}
```

## [959. 由斜杠划分区域](https://leetcode-cn.com/problems/regions-cut-by-slashes/)

> 并查集

```java
class Solution {
    // 顺时针：上面0-右边1-下面2-左边3
    public int regionsBySlashes(String[] grid) {
        int n = grid.length;
        int N = n * n * 4;
        UnionFind uf = new UnionFind(N);
        for (int i = 0; i < n; ++i) { // 纵坐标
            char[] row = grid[i].toCharArray();
            for (int j = 0; j < n; ++j) { // 横坐标
                int index = (i * n + j) * 4; // 0号坐标
                char c = row[j];
                if (c == '/') { // 合并0、3和1、2
                    uf.union(index, index + 3);
                    uf.union(index + 1, index + 2);
                } else if (c == '\\') { // 合并0、1和2、3
                    uf.union(index, index + 1);
                    uf.union(index + 2, index + 3);
                } else { // 合并0、1、2、3
                    uf.union(index, index + 1);
                    uf.union(index + 1, index + 2);
                    uf.union(index + 2, index + 3);
                }
                // 向下试探
                if (i < n - 1) {
                    int downIndex = ((i + 1) * n + j) * 4;
                    uf.union(index + 2, downIndex);
                }
                // 向右试探
                if (j < n - 1) {
                    int rightIndex = (i * n + j + 1) * 4 + 3;
                    uf.union(index + 1, rightIndex);
                }
            }
        }
        return uf.getCount();
    }

    private class UnionFind {
        private int[] parents;
        private int count;

        public int getCount() {
            return this.count;
        }

        UnionFind(int n) {
            parents = new int[n];
            for (int i = 0; i < n; ++i) {
                parents[i] = i;
            }
            count = n;
        }

        public void union(int a, int b) {
            int pa = find(a);
            int pb = find(b);
            if (pa > pb) {
                parents[pa] = pb;
                count--;
            } else if (pa < pb) {
                parents[pb] = pa;
                count--;
            }
        }

        public int find(int x) {
            if (parents[x] != x) {
                parents[x] = find(parents[x]);
            }
            return parents[x];
        }
    }
}
```

## [1128. 等价多米诺骨牌对的数量](https://leetcode-cn.com/problems/number-of-equivalent-domino-pairs/)

> 数组

```java
class Solution {
    public int numEquivDominoPairs(int[][] dominoes) {
        int count = 0;
        int[] num = new int[100];
        for (int[] d: dominoes) {
            int index = d[0] < d[1] ? d[0] * 10 + d[1] : d[1] * 10 + d[0];
            count += num[index];
            num[index]++;
        }
        return count;
    }
}
```

## [1579. 保证图可完全遍历](https://leetcode-cn.com/problems/remove-max-number-of-edges-to-keep-graph-fully-traversable/)

> 并查集

```java
class Solution {
    public int maxNumEdgesToRemove(int n, int[][] edges) {
        UnionFind uf = new UnionFind(n + 1);
        List<int[]> aList = new ArrayList<>();
        List<int[]> bList = new ArrayList<>();
        List<int[]> cList = new ArrayList<>();
        for (int[] edge: edges) {
            switch(edge[0]) {
                case 1:
                    aList.add(edge);
                    break;
                case 2:
                    bList.add(edge);
                    break;
                default:
                    cList.add(edge);
            }
        }
        // 优先加入公共边
        for (int[] edge: cList) {
            int curCount = uf.union(edge[1], edge[2]);
            if (curCount == 1) { // 通过公共边已经可以完全遍历
                return edges.length - uf.getUnionCount() + uf.getDelCount();
            }
        }
        int cDelCount = cList.size() - uf.getUnionCount() + uf.getDelCount();
        int aDelCount = getDelInfo(uf, aList);
        if (aDelCount == -1)
            return -1;
        int bDelCount = getDelInfo(uf, bList);
        if (bDelCount == -1)
            return -1;
        return cDelCount + aDelCount + bDelCount;
    }

    private int getDelInfo(UnionFind uf, List<int[]> list) {
        UnionFind ufx = uf.copyuf();
        for (int[] edge: list) {
            int curCount = ufx.union(edge[1], edge[2]);
            if (curCount == 1) { // 通过x边已经可以完全遍历x点
                return list.size() - ufx.getUnionCount() + ufx.getDelCount();
            }
        }
        return -1;
    }

    private class UnionFind {
        private int[] parents;
        private int count; // 连通分量
        private int unionCount; // 合并次数
        private int delCount; // 删除的边数

        public int getUnionCount() {
            return this.unionCount;
        }

        public void setCount(int count) {
            this.count = count;
        }

        public void setUnionCount(int unionCount) {
            this.unionCount = unionCount;
        }

        public int getDelCount() {
            return this.delCount;
        }

        public void setDelCount(int delCount) {
            this.delCount = delCount;
        }

        public void setParents(int[] parents) {
            this.parents = Arrays.copyOf(parents, parents.length);
        }

        public UnionFind copyuf() {
            UnionFind uf = new UnionFind(this.parents.length);
            uf.setCount(this.count);
            uf.setParents(this.parents);
            uf.setDelCount(0);
            uf.setUnionCount(0);
            return uf;
        }

        UnionFind(int n) {
            parents = new int[n];
            for (int i = 1; i < n; ++i) {
                parents[i] = i;
            }
            count = n - 1;
        }

        public int union(int a, int b) {
            this.unionCount++;
            int pa = find(a);
            int pb = find(b);
            if (pa > pb) {
                parents[pa] = pb;
                count--;
            } else if (pa < pb) {
                parents[pb] = pa;
                count--;
            } else {
                delCount++;
            }
            return this.count;
        }

        public int find(int x) {
            if (parents[x] != x) {
                parents[x] = find(parents[x]);
            }
            return parents[x];
        }
    }
}
```

## [724. 寻找数组的中心索引](https://leetcode-cn.com/problems/find-pivot-index/)

> 数组，前缀和

左右前缀和比较

```java
class Solution {
    public int pivotIndex(int[] nums) {
        int len = nums.length;
        int[] lSums = new int[len + 2];
        lSums[0] = 0;
        lSums[len + 1] = 0;
        int[] rSums = new int[len + 2];
        rSums[0] = 0;
        rSums[len + 1] = 0;
        int tmp = 0;
        for (int i = 0; i < len; ++i) {
            tmp += nums[i];
            lSums[i + 1] = tmp;
        }
        tmp = 0;
        for (int j = len - 1; j > 0; --j) {
            tmp += nums[j];
            rSums[j + 1] = tmp;
        }
        if (len == 0) {
            return -1;
        }
        for (int i = 2; i < rSums.length; ++i) {
            if (rSums[i] == lSums[i - 2]) {
                return i - 2;
            }
        }
        return -1;
    }
}
```

巧用和

```java
class Solution {
    public int pivotIndex(int[] nums) {
        // int total = Arrays.stream(nums).sum(); // 可以但没必要，效率低
        int total = 0;
        for (int i: nums) {
            total += i;
        }
        int sum = 0;
        for (int i = 0; i < nums.length; ++i) {
            if (sum * 2 + nums[i] == total)
                return i;
            sum += nums[i];
        }
        return -1;
    }
}
```

## [1631. 最小体力消耗路径](https://leetcode-cn.com/problems/path-with-minimum-effort/)

> 并查集，二分，DFS，BFS

直接DFS搜索，会超时，不保证正确

```java
class Solution {
    private int tmp_max = Integer.MAX_VALUE;

    public int minimumEffortPath(int[][] heights) {
        int rows = heights.length;
        int columns = heights[0].length;
        int[][] visited = new int[rows][columns];
        return dfs(0, 0, rows, columns, visited, heights, 0, heights[0][0]);
    }

    private int dfs(int i, int j, int rows, int columns, int[][] visited, int[][] heights, int max, int last) {
        if (i == rows - 1 && j == columns - 1) { // 走到终点，在公共变量对比记录下最好成绩，并返回自身成绩
            int m = Math.max(max, Math.abs(last - heights[i][j]));
            tmp_max = Math.min(tmp_max, m);
            return m;
        }
        if (i >= rows || j >= columns || i < 0 || j < 0 || visited[i][j] == 1) { // 走到不符合条件的点
            return -1; // 表示绝路
        }
        int cur = Math.abs(last - heights[i][j]); // 当前差值
        if (cur >= tmp_max) // 已经不比当前最好成绩好了就不用往下比了
            return -1;
        max = Math.max(max, cur);
        int[] res = new int[4];
        visited[i][j] = 1; // 标记已访问
        res[0] = dfs(i + 1, j, rows, columns, visited, heights, max, heights[i][j]);
        res[1] = dfs(i, j + 1, rows, columns, visited, heights, max, heights[i][j]);
        res[2] = dfs(i - 1, j, rows, columns, visited, heights, max, heights[i][j]);
        res[3] = dfs(i, j - 1, rows, columns, visited, heights, max, heights[i][j]);
        int min = -1; // 默认走不通
        for (int r: res) {
            if (r != -1) { // 有能走通的更新走通的最小值
                if (min == -1)
                    min = r;
                else
                    min = Math.min(min, r);
            }
        }
        visited[i][j] = 0; // 处理完了，标记未访问
        return min;
    }
}
```

并查集

```java
class Solution {
    public int minimumEffortPath(int[][] heights) {
        // 并查集：思路是权值从小到大加入并查集，如果0和len能联通，则返回最小的值
        int rows = heights.length;
        int columns = heights[0].length;
        int len = rows * columns;
        List<int[]> list = new ArrayList<>();
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < columns; ++j) {
                int curIndex = i * columns + j;
                if (j + 1 < columns) {
                    int rightIndex = i * columns + j + 1;
                    list.add(new int[]{curIndex, rightIndex, Math.abs(heights[i][j] - heights[i][j + 1])});
                }
                if (i + 1 < rows) {
                    int bottomIndex = (i + 1) * columns + j;
                    list.add(new int[]{curIndex, bottomIndex, Math.abs(heights[i][j] - heights[i + 1][j])});
                }
            }
        }
        Collections.sort(list, new Comparator<int[]>() {
            @Override
            public int compare(int[] a, int[] b) {
                return a[2] - b[2];
            }
        });
        UnionFind uf = new UnionFind(len);
        for (int[] item: list) {
            uf.union(item[0], item[1]);
            if (uf.isConnected(0, len - 1))
                return item[2];
        }
        return 0;
    }

    private class UnionFind {
        private int[] parents;

        UnionFind(int n) {
            parents = new int[n];
            for (int i = 0; i < n; ++i) {
                parents[i] = i;
            }
        }

        public void union(int a, int b) {
            int pa = find(a);
            int pb = find(b);
            if (pa > pb) {
                parents[pb] = pa;
            } else if (pb > pa) {
                parents[pa] = pb;
            }
        }

        public int find(int x) {
            return parents[x] == x ? x : (parents[x] = find(parents[x]));
        }

        public boolean isConnected(int a, int b) {
            return find(a) == find(b);
        }
    }
}
```

二分+BFS

```java
class Solution {
    public int minimumEffortPath(int[][] heights) {
        // 二分搜索，0~999999
        int l = 0, r = 999999;
        int rows = heights.length;
        int columns = heights[0].length;
        int res = 0;
        while (l <= r) {
            int mid = (l + r) / 2;
            boolean[][] visited = new boolean[rows][columns];
            visited[0][0] = true;
            Queue<int[]> q = new LinkedList<>();
            q.offer(new int[]{0, 0}); // 加入左上第一个坐标和值
            while (!q.isEmpty()) {
                int[] item = q.poll();
                int x = item[0], y = item[1];
                int[][] dirs = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}}; // 上下左右
                for (int i = 0; i < 4; ++i) {
                    int nx = dirs[i][0] + x;
                    int ny = dirs[i][1] + y;
                    if (nx < rows && ny < columns && nx >= 0 && ny >= 0 && !visited[nx][ny]) {
                        int delta = Math.abs(heights[x][y] - heights[nx][ny]);
                        if (delta <= mid) {
                            q.offer(new int[]{nx, ny});
                            visited[nx][ny] = true;
                        }
                    }
                }
            }
            if (visited[rows - 1][columns - 1]) {
                res = mid;
                r = mid - 1;
            } else {
                l = mid + 1;
            }
        }
        return res;
    }
}
```

Dijkstra（效率最高）

```java
class Solution {
    public int minimumEffortPath(int[][] heights) {
        // Dijkstra算法
        int rows = heights.length;
        int columns = heights[0].length;
        int n = rows * columns;
        int[][] dirs = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        // 初始化访问数组
        boolean[] visited = new boolean[n];
        // 初始化距离数组
        int[] dist = new int[n];
        Arrays.fill(dist, Integer.MAX_VALUE);
        dist[0] = 0;
        // 初始化OPEN优先队列
        PriorityQueue<int[]> pq = new PriorityQueue<int[]>(new Comparator<int[]>() {
            public int compare(int[] e1, int[] e2) {
                return e1[2] - e2[2];
            }
        });
        pq.offer(new int[]{0, 0, 0}); // 坐标(0,0)距离为0
        // Dijkstra
        while (!pq.isEmpty()) {
            int[] item = pq.poll(); // 距离小的优先
            int x = item[0], y = item[1], odist = item[2];
            int id = x * columns + y;
            if (visited[id])
                continue;
            if (x == rows - 1 && y == columns - 1)
                break;
            visited[id] = true;
            for (int i = 0; i < 4; ++i) {
                int nx = dirs[i][0] + x;
                int ny = dirs[i][1] + y;
                int nid = nx * columns + ny;
                if (nx >=0 && ny >=0 && nx < rows && ny < columns) {
                    int cost = Math.abs(heights[nx][ny] - heights[x][y]); // 差值(cost)计算
                    int ndist = Math.max(odist, cost); // g(new)=g(old)[+]cost
                    if (ndist < dist[nid]) { // g(new)<new的原本dist
                        dist[nid] = ndist; // 更新dist
                        pq.offer(new int[]{nx, ny, ndist});
                    }  
                }
            }
        }
        return dist[n - 1];
    }
}
```

## [778. 水位上升的泳池中游泳](https://leetcode-cn.com/problems/swim-in-rising-water/)

> 并查集，二分，BFS，DFS

并查集

```java
class Solution {
    public int swimInWater(int[][] grid) {
        int res = 0;
        int N = grid.length;
        int n = N * N;
        UnionFind uf = new UnionFind(n);
        int min = 0;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                if (grid[i][j] < min)
                    min = grid[i][j];
            }
        }
        while (true) {
            int curHeight = min + res;
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    if (i - 1 >= 0 && curHeight >= grid[i - 1][j] &&  curHeight >= grid[i][j]) {
                        uf.union((i - 1) * N + j, i * N + j);
                    }
                    if (j - 1 >= 0 && curHeight >= grid[i][j - 1] &&  curHeight >= grid[i][j]) {
                        uf.union(i * N + j - 1, i * N + j);
                    }
                    if (uf.isConnectedZero(n - 1)) {
                        return res;
                    }
                }
            }
            res++;
        }
    }

    private class UnionFind {
        private int[] parents;

        UnionFind(int n) {
            parents = new int[n];
            for (int i = 0; i < n; ++i) {
                parents[i] = i;
            }
        }

        private int find(int x) {
            return parents[x] == x ? x : (parents[x] = find(parents[x]));
        }

        private void union(int a, int b) {
            int pa = find(a);
            int pb = find(b);
            if (pa > pb) {
                parents[pa] = pb;
            } else if (pa < pb) {
                parents[pb] = pa;
            }
        }

        private boolean isConnectedZero(int a) {
            return find(a) == 0;
        }
    }
}
```

TODO：更简单的并查集思路、二分+DFS/BFS、Dijkstra

## [839. 相似字符串组](https://leetcode-cn.com/problems/similar-string-groups/)

> 并查集

```java
class Solution {
    public int numSimilarGroups(String[] strs) {
        int n = strs.length;
        UnionFind uf = new UnionFind(n);
        for (int i = 0; i < n - 1; ++i) {
            for (int j = i + 1; j < n; ++j) {
                if (!uf.isConnected(i, j) && isSimilar(strs[i], strs[j])) {
                    uf.union(i, j);
                }
            }
        }
        return uf.getRegion();
    }

    private boolean isSimilar(String s1, String s2) {
        int count = 0;
        for (int i = 0; i < s1.length(); ++i) {
            if (s1.charAt(i) != s2.charAt(i)) {
                count++;
                if (count > 2)
                    return false;
            }
        }
        return true;
    }

    private class UnionFind {
        private int[] parents;
        private int region;

        public int getRegion() {
            return this.region;
        }

        UnionFind(int n) {
            parents = new int[n];
            for (int i = 0; i < n; ++i) {
                parents[i] = i;
            }
            region = n;
        }

        private void union(int a, int b) {
            int pa = find(a);
            int pb = find(b);
            if (pa < pb) {
                parents[pb] = pa;
                region--;
            } else if (pa > pb) {
                parents[pa] = pb;
                region--;
            }
        }

        private int find(int x) {
            return parents[x] == x ? x : (parents[x] = find(parents[x]));
        }

        private boolean isConnected(int a, int b) {
            return find(a) == find(b);
        }
    }
}
```

## [888. 公平的糖果棒交换](https://leetcode-cn.com/problems/fair-candy-swap/)

> 数组

```java
class Solution {
    public int[] fairCandySwap(int[] A, int[] B) {
        boolean[] a = new boolean[100001];
        boolean[] b = new boolean[100001];
        int countA = 0, countB = 0;
        for (int i = 0; i < A.length; ++i) {
            countA += A[i];
            a[A[i]] = true;
        }
        for (int i = 0; i < B.length; ++i) {
            countB += B[i];
            b[B[i]] = true;
        }
        int delta = (countA - countB) / 2;
        for (int i = 0; i < 100001; ++i) {
            if (i + delta < 100001 && i + delta > 0 && a[i + delta] && b[i]) {
                return new int[]{i + delta, i};
            }
        }
        return new int[0];
    }
}
```



# Java算法模板

## BFS

### 如果不需要确定当前遍历到了哪一层

```java
queue.push(root)
while queue 不空：
    cur = queue.pop()
    for 节点 in cur的所有相邻节点：
        if 该节点有效且未访问过：
            queue.push(该节点)
```

### 如果要确定当前遍历到了哪一层

> level表示二叉树遍历到哪一层或者图走了几步、size表示在当前层有多少个元素

```java
queue.push(root)
level = 0
while queue 不空：
    size = queue.size()
    while (size --) {
        cur = queue.pop()
        for 节点 in cur的所有相邻节点：
            if 该节点有效且未被访问过：
                queue.push(该节点)
    }
    level ++;
```

## DFS

> 判出口（终点、越界）->剪枝->扩展->标记->递归->还原

```java
/*
 * Return true if there is a path from cur to target.
 */
boolean DFS(Node cur, Node target, Set<Node> visited) {
    return true if cur is target;
    for (next : each neighbor of cur) {
        if (next is not in visited) {
            add next to visted;
            return true if DFS(next, target, visited) == true;
        }
    }
    return false;
}
```

- 二叉树递归

```java
private static List<Integer> result = new ArrayList<>();
public static List<Integer> DFS(TreeNode root) {
		if (root == null) {
				return null;
		}
		result.add(root.val);
		if (root.left != null) {
				DFS(root.left);
		}
		if (root.right != null) {
 				DFS(root.right);
		}
		return result;
}
```

- 二叉树非递归 LinkedList

```java
public static List<Integer> DFS(TreeNode root) {
		if (root == null) {
				return null;
		}
		Stack<TreeNode> stack = new Stack<>();
		stack.push(root);
		List<Integer> result = new ArrayList<>();
		while (!stack.isEmpty()) {
				TreeNode treeNode = stack.pop();
				result.add(treeNode.val);
				if (treeNode.right != null) {
						stack.push(treeNode.right);
				}
				if (treeNode.left != null) {
						stack.push(treeNode.left);
				}
		}
    return result;
}
```

## 二分查找

```java
int findBoundary(int[] nums, int target, int direction) {
        int left = 0, right = nums.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] < target) {
                // 更新left
                left = mid + 1;
            } else if (nums[mid] > target) {
                // 更新right
                right = mid - 1;
            } else if (nums[mid] == target) {
                if (direction == 0) {
                    // 找左边界，收缩右边界
                    right = mid - 1;
                } else {
                    // 找右边界，收缩左边界
                    left = mid + 1;
                }
            }
        }
        if (direction == 0) {
            if (left >= nums.length || nums[left] != target)
                return -1;
            return left;
        } else {
            if (right < 0 || nums[right] != target)
                return -1;
            return right;
        }
}
```

## 快排

```java
void QuickSort(int arr, int left, int right) {
    if(left >= right)
        return;
    int i = left, j = right, pivot = arr[left];
    while(i < j) {
				while(i<j && arr[j] >= pivot) j--;
            arr[i] = arr[j];
        while(i<j && arr[i] <= pivot) i++;
            arr[j] = arr[i];
		}
    arr[i] = pivot;
    QuickSort(arr, left, i - 1);
    QuickSort(arr, i + 1, right);
}
```

## 回溯法

```java
List<List<Integer>> result = new ArrayList<>();
private void backtrack(路径, 选择列表) {
    if(满足结束条件) {
    		result.add(路径)
        return
    }
    for (选择: 选择列表) {
    		做选择
        backtrack(路径, 选择列表)
        撤销选择
    }
}
```

## 树

### 递归



### 迭代

### 前序遍历

```java
	public static void preOrder(TreeNode root){
		if(root != null){
			System.out.print(root.val + " ");
			preOrder(root.left);
			preOrder(root.right);
		}
	}
```

```java
public static ArrayList preOrder1(TreeNode root){
	Stack<TreeNode> stack = new Stack<TreeNode>();
	ArrayList alist = new ArrayList();
	TreeNode p = root;
	while(p != null || !stack.empty()){
		while(p != null){
			alist.add(p.val);
			stack.push(p);
			p = p.left;
		}
		if(!stack.empty()){
			TreeNode temp = stack.pop();
			p = temp.right;
		}
	}
	return alist;
}
```
### 中序遍历

```java
	public static void inOrder(TreeNode root){
		if(root != null){
			inOrder(root.left);
			System.out.print(root.val + " ");
			inOrder(root.right);
		}
	}
```

```java
public static ArrayList inOrder1(TreeNode root){
	ArrayList alist = new ArrayList();
	Stack<TreeNode> stack = new Stack<TreeNode>();
	TreeNode p = root;
	while(p != null || !stack.empty()){
		while(p != null){
			stack.push(p);
			p = p.left;
		}
		if(!stack.empty()){
			TreeNode temp = stack.pop();
			alist.add(temp.val);
			p = temp.right;
		}
	}
	return alist;
}
```
### 后序遍历

```java
	public static void postOrder(TreeNode root){
		if(root != null){
			postOrder(root.left);
			postOrder(root.right);
			System.out.print(root.val + " ");
		}
	}
```

```java
	public static ArrayList postOrder1(TreeNode root){
		ArrayList alist = new ArrayList();
		Stack<TreeNode> stack = new Stack<TreeNode>();
		if(root == null)
			return alist;
		TreeNode cur,pre = null;
		stack.push(root);
		while(!stack.empty()){
			cur = stack.peek();
			if((cur.left == null && cur.right == null) || (pre != null && (cur.left == pre || cur.right == pre))){
				TreeNode temp = stack.pop();
				alist.add(temp.val);
				pre = temp;
			}
			else{
				if(cur.right != null)
					stack.push(cur.right);
				if(cur.left != null)
					stack.push(cur.left);
			}
		}
		return alist;
	}
```

### 层序遍历

```java
	private static void levelOrder(TreeNode root) {
		Queue<TreeNode> queue = new LinkedList<TreeNode>();
		if(root == null)
			return;
		queue.offer(root);
		while(!queue.isEmpty()){
			TreeNode temp  = queue.poll();
			System.out.print(temp.val + " ");
			if(temp.left != null)
				queue.offer(temp.left);
			if(temp.right != null)
				queue.offer(temp.right);
		}
	}
```

### 构造完全二叉树



### 并查集

### 前缀树

## 图

### Dijkstra算法

> T-O(n^2+e)/S-O(n+e)

### Floyd-Warshall算法

> T-O(n^3)/S-O(n^2)

### Bellman-Ford算法

> T-O(ne)/S-O(n)

### 最小生成树

### Kruskal算法

### Prim算法

### 拓扑排序

## 双指针

### 查找子字符串

## 动态规划

### 状态搜索

### 贪心

## 滑动窗口

```java
public String minWindow(String s, String t) {
        int left = 0, right = 0; // 滑动窗口前后指针
        int min = Integer.MAX_VALUE; // 最小子串的长度
        int start = 0, end = 0; // 最小子串的左右位置
        int count = 0; // 相同字符的个数

        Map<Character, Integer> tMap = new HashMap<>(); // target串的字符计数（目标）
        Map<Character, Integer> sMap = new HashMap<>(); // source串的字符计数（窗口）

        // 初始化target串的字符计数
        for (int i = 0; i < t.length(); ++i) {
            tMap.put(t.charAt(i), tMap.getOrDefault(t.charAt(i), 0) + 1);
        }

        while (right < s.length()) {
            char c = s.charAt(right);
            // 更新窗口状态
            if (tMap.containsKey(c)) { // 是所求字符
                sMap.put(c, sMap.getOrDefault(c, 0) + 1); // 存字符进窗口
                if (tMap.get(c).compareTo(sMap.get(c)) == 0) { // 看是不是该字符达标
                    count++;
                }
            }
            right++; // 右滑动扩大
            while (count == tMap.size()) {
                // 满足条件，更新最值
                if (min > right - left) {
                    end = right;
                    start = left;
                    min = right - left;
                }
                char d = s.charAt(left);
                // 更新窗口状态
                if (tMap.containsKey(d)) {
                    sMap.put(d, sMap.get(d) - 1);
                    if (tMap.get(d) > sMap.get(d)) {
                        count--;
                    }
                }
                left++; //左滑动缩小
            }
        }
        return min == Integer.MIN_VALUE ? "" : s.substring(start, end);
    }
```

## 单调栈

## 最大公约数gcd

```java
public int gcd(int x, int y) {
    return y > 0 ? gcd(y, x % y) : x;
}
```

## 最小公倍数lcm

## A*算法

参考https://zhuanlan.zhihu.com/p/108344917

```java
* 初始化open_set和close_set；
* 将起点加入open_set中，并设置优先级为0（优先级最高）；
* 如果open_set不为空，则从open_set中选取优先级最高的节点n：
    * 如果节点n为终点，则：
        * 从终点开始逐步追踪parent节点，一直达到起点；
        * 返回找到的结果路径，算法结束；
    * 如果节点n不是终点，则：
        * 将节点n从open_set中删除，并加入close_set中；
        * 遍历节点n所有的邻近节点：
            * 如果邻近节点m在close_set中，则：
                * 跳过，选取下一个邻近节点
            * 如果邻近节点m在open_set中，则：
                * 判断节点n到节点m的 F(n) + cost[n,m] 值是否 < 节点m的 F(m) 。来尝试更新该点，重新设置f值和父节点等数据
            * 如果邻近节点m也不在open_set中，则：
                * 设置节点m的parent为节点n
                * 计算节点m的优先级
                * 将节点m加入open_set中
```



# Java常用数据结构

## Queue

- offer/add 添加
- poll/remove 删除
- peek/element 查询头部
- isEmpty 是否为空
- size 长度

- 最大堆 PriorityQueue<T>((a, b) -> b - a)
- 最小堆 PriorityQueue<T>()

## ArrayList

- 

## HashMap

## HashSet

# Java常见技巧

+ 数组初始化：Arrays.fill(arr, Integer.MAX_VALUE)

+ 数组复制：Arrays.copyOfRange(nums, 0, k)

+ ArrayList简单构造：Arrays.asList(a,b)

+ ArrayList2int[]：list.stream().mapToInt(e->e).toArray()或list.stream().mapToInt(Integer::valueOf).toArray();

+ 创建哑节点 `dummyHead`，令 `dummyHead.next = head`。引入哑节点是为了便于在 `head` 节点之前插入节点。

+ 计数哈希表构建：map.put(val, map.getOrDefault(val, 0) + 1);

+ 数组倒序：必须是Integer[]，Arrays.sort(A, Collections.reverseOrder()); 或者从后往前处理

+ 二维数组排序

  ```
  Arrays.sort(arr, new Comparator<?>() {
  		@Override
  		private int Compare(int[] a, int[] b) {
  				return a[0] > b[0] ? -1: 1;
  		}
  })
  ```

- 获取数组最大值：Arrays.stream(arr).max().getAsInt();

- 数组复制

  ```java
  System.arraycopy(Object src【原数组】, int srcPos【原数组开始位置】, Object dest【目标数组】, int destPos【目标数组开始位置】, int length【拷贝长度】);
  ```

- 数组求和：int total = Arrays.stream(nums).sum();  // 效率偏低

# 未完成

## [493. 翻转对](https://leetcode-cn.com/problems/reverse-pairs/)

## [147. 对链表进行插入排序](https://leetcode-cn.com/problems/insertion-sort-list/)

## [135. 分发糖果](https://leetcode-cn.com/problems/candy/)

## 803. 打砖块

## [1489. 找到最小生成树里的关键边和伪关键边](https://leetcode-cn.com/problems/bricks-falling-when-hit/)


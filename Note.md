

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

## [424. 替换后的最长重复字符](https://leetcode-cn.com/problems/longest-repeating-character-replacement/)

> 双指针，滑动数组

TODO：非传统意义滑动数组，有技巧

```java
class Solution {
    public int characterReplacement(String s, int k) {
        int[] nums = new int[26]; // 频数数组
        int maxn = 0; // 历史最大值
        char[] arr = s.toCharArray();
        int n = s.length();
        int left = 0, right = 0;
        while (right < n) {
            int tmp = ++nums[arr[right] - 'A']; // 更新纳入数组元素的频数
            maxn = Math.max(maxn, tmp); // 更新历史最大值
            if (right - left + 1 - maxn > k) { // 非重复元素大于k时同步右移左右指针
                --nums[arr[left] - 'A'];
                left++;
            }
            right++;
        }
        return right - left;
    }
}
```

## [295. 数据流的中位数](https://leetcode-cn.com/problems/find-median-from-data-stream/)

> 堆，设计

```java
class MedianFinder {
    private Queue<Integer> small;
    private Queue<Integer> big;

    /** initialize your data structure here. */
    public MedianFinder() {
        small = new PriorityQueue<>(Collections.reverseOrder()); // 大顶堆
        big = new PriorityQueue<>(); // 小顶堆
    }
    
    public void addNum(int num) {
        // 左右相等 >2，移左填右；其他情况，填左
        // 左大右小 <1，移右填左；其他情况，填右
        if (small.isEmpty()) { // 初始化
            small.offer(num);
            return;
        }
        if (small.size() == big.size()) {
            if (num > big.peek()) {
                small.offer(big.poll());
                big.offer(num);
            } else {
                small.offer(num);
            }
        } else if (small.size() > big.size()) {
            if (num < small.peek()) {
                big.offer(small.poll());
                small.offer(num);
            } else {
                big.offer(num);
            }
        }
    }
    
    public double findMedian() {
        if (small.size() == big.size())
            return (small.peek() + big.peek()) / 2.0;
        else
            return small.peek();
    }
}

/**
 * Your MedianFinder object will be instantiated and called as such:
 * MedianFinder obj = new MedianFinder();
 * obj.addNum(num);
 * double param_2 = obj.findMedian();
 */
```

简单写法，效率偏低

```java
class MedianFinder {
    private Queue<Integer> small;
    private Queue<Integer> big;
    private int count;

    /** initialize your data structure here. */
    public MedianFinder() {
        small = new PriorityQueue<>(Collections.reverseOrder()); // 大顶堆
        big = new PriorityQueue<>(); // 小顶堆
        count = 0;
    }
    
    public void addNum(int num) {
        count++;
        small.offer(num);
        big.offer(small.poll());
        if ((count & 1) == 1) {
            small.offer(big.poll());
        }
    }
    
    public double findMedian() {
        if ((count & 1) == 0)
            return (small.peek() + big.peek()) / 2.0;
        else
            return small.peek();
    }
}

/**
 * Your MedianFinder object will be instantiated and called as such:
 * MedianFinder obj = new MedianFinder();
 * obj.addNum(num);
 * double param_2 = obj.findMedian();
 */
```

## [643. 子数组最大平均数 I](https://leetcode-cn.com/problems/maximum-average-subarray-i/)

> 数组，滑动窗口

执行用时：2 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：42.6 MB, 在所有 Java 提交中击败了74.54%的用户

```java
class Solution {
    public double findMaxAverage(int[] nums, int k) {
        int sum = 0, n = nums.length;
        for (int i = 0; i < k; ++i) {
            sum += nums[i];
        }
        int tmp = sum;
        for (int i = k; i < n; ++i) {
            tmp -= nums[i - k] - nums[i];
            if (tmp > sum)
                sum = tmp;
        }
        return (double)sum / k;
    }
}
```

## [1208. 尽可能使字符串相等](https://leetcode-cn.com/problems/get-equal-substrings-within-budget/)

> 数组，滑动窗口

前缀和+二分查找（单调增序列）

```java
class Solution {
    public int equalSubstring(String s, String t, int maxCost) {
        int n = s.length(), maxLen = 0;
        int[] diff = new int[n + 1];
        for (int i = 0; i < n; ++i) {
            diff[i + 1] = diff[i] + Math.abs(s.charAt(i) - t.charAt(i));
        }
        for (int i = 1; i <= n; ++i) {
            int start = bs(diff, 0, i, diff[i] - maxCost);
            maxLen = Math.max(maxLen, i - start);
        }
        return maxLen;
    }

    private int bs(int[] arr, int l, int r, int target) {
        while (l < r) {
            int mid = (r - l) / 2 + l;
            if (arr[mid] < target) {
                l = mid + 1;
            } else  {
                r = mid;
            }
        }
        return l;
    }
}
```

双指针

```java
class Solution {
    public int equalSubstring(String s, String t, int maxCost) {
        int l = 0, r = 0, curMaxCost = 0, maxLen = 0, n = s.length();
        while (r < n) {
            curMaxCost += Math.abs(s.charAt(r) - t.charAt(r));
            while(curMaxCost > maxCost) {
                curMaxCost -= Math.abs(s.charAt(l) - t.charAt(l));
                l++;
            }
            maxLen = Math.max(maxLen, r - l + 1);
            r++;
        }
        return maxLen;
    }
}
```

## [1423. 可获得的最大点数](https://leetcode-cn.com/problems/maximum-points-you-can-obtain-from-cards/)

> 数组，滑动数组

逆向思维，取中间一段滑动

```java
class Solution {
    public int maxScore(int[] cardPoints, int k) {
        // int total = Arrays.stream(cardPoints).sum();
        int sum = 0, n = cardPoints.length, w = n - k, total = 0;
        for (int i = 0; i < n; ++i) {
            total += cardPoints[i];
            if (i == w - 1)
                sum = total;
        }
        int tmp = sum;
        for (int i = 0; i < k; ++i) {
            tmp += cardPoints[i + w] - cardPoints[i];
            sum = Math.min(sum, tmp);
        }
        return total - sum;
    }
}
```

## [771. 宝石与石头](https://leetcode-cn.com/problems/jewels-and-stones/)

> 哈希表

执行用时：1 ms, 在所有 Java 提交中击败了98.11%的用户

内存消耗：36.5 MB, 在所有 Java 提交中击败了97.97%的用户

```java
class Solution {
    public int numJewelsInStones(String jewels, String stones) {
        boolean[] jc = new boolean[58]; // 'A'~'z'有58个元素
        for (char c: jewels.toCharArray()) {
            jc[c - 'A'] = true;
        }
        int ans = 0;
        for (char c: stones.toCharArray()) {
            if (jc[c - 'A'])
                ans++;
        }
        return ans;
    }
}
```

## [1486. 数组异或操作](https://leetcode-cn.com/problems/xor-operation-in-an-array/)

> 位运算，数组

```java
class Solution {
    public int xorOperation(int n, int start) {
        int res = 0;
        for (int i = 0; i < n; ++i) {
            res ^= start + 2 * i;
        }
        return res;
    }
}
```

## [1470. 重新排列数组](https://leetcode-cn.com/problems/shuffle-the-array/)

> 数组

```java
class Solution {
    public int[] shuffle(int[] nums, int n) {
        int[] res = new int[2 * n];
        for (int i = 0; i < n; ++i) {
            res[i * 2] = nums[i];
            res[i * 2 + 1] = nums[i + n];
        }
        return res;
    }
}
```

## [1720. 解码异或后的数组](https://leetcode-cn.com/problems/decode-xored-array/)

> 位运算

```java
class Solution {
    public int[] decode(int[] encoded, int first) {
        int n = encoded.length;
        int[] res = new int[n + 1];
        res[0] = first;
        for (int i = 0; i < n; ++i) {
            res[i + 1] = res[i] ^ encoded[i];
        }
        return res;
    }
}
```

## [1603. 设计停车系统](https://leetcode-cn.com/problems/design-parking-system/)

> 设计

```java
class ParkingSystem {

    private int[] capacity;

    public ParkingSystem(int big, int medium, int small) {
        capacity = new int[]{big, medium, small};
    }
    
    public boolean addCar(int carType) {
        if (capacity[carType - 1] == 0)
            return false;
        capacity[carType - 1]--;
        return true;
    }
}
```

## [237. 删除链表中的节点](https://leetcode-cn.com/problems/delete-node-in-a-linked-list/)

> 链表

执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：38 MB, 在所有 Java 提交中击败了18.55%的用户

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
    public void deleteNode(ListNode node) {
        node.val = node.next.val;
        node.next = node.next.next;
    }
}	
```

## [1313. 解压缩编码列表](https://leetcode-cn.com/problems/decompress-run-length-encoded-list/)

> 数组

```java
class Solution {
    public int[] decompressRLElist(int[] nums) {
        int n = nums.length, total = 0;
        for (int i = 0; i < n; i += 2) {
            total += nums[i];
        }
        int[] ret = new int[total];
        int index = 0;
        List<Integer> list = new ArrayList<>();
        for (int i = 0; i < n; i += 2) {
            int freq = nums[i];
            int val = nums[i + 1];
            for (int j = 0; j < freq; ++j)
                ret[index++] = val;
        }
        return ret;
    }
}
```

## [1662. 检查两个字符串数组是否相等](https://leetcode-cn.com/problems/check-if-two-string-arrays-are-equivalent/)

> 字符串

StringBuilder拼接

```java
class Solution {
    public boolean arrayStringsAreEqual(String[] word1, String[] word2) {
        StringBuilder sb1 = new StringBuilder(), sb2 = new StringBuilder();
        for (String str: word1) {
            sb1.append(str);
        }
        for (String str: word2) {
            sb2.append(str);
        }
        return sb1.toString().equals(sb2.toString());
    }
}
```

join

```java
class Solution {
    public boolean arrayStringsAreEqual(String[] word1, String[] word2) {
        return String.join("", word1).equals(String.join("", word2));
    }
}
```

## [627. 变更性别](https://leetcode-cn.com/problems/swap-salary/)

> SQL

巧用ASCII和【普通版】，使用CHAR()和ASCII()

执行用时：194 ms, 在所有 MySQL 提交中击败了41.85%的用户

内存消耗：0 B, 在所有 MySQL 提交中击败了100.00%的用户

```sql
UPDATE salary SET sex = CHAR(ASCII('m') + ASCII('f') - ASCII(sex));
```

巧用ASCII和【优化版】，使用CHAR()和ASCII()

执行用时：156 ms, 在所有 MySQL 提交中击败了99.84%的用户

内存消耗：0 B, 在所有 MySQL 提交中击败了100.00%的用户

```sql
UPDATE salary SET sex = CHAR(11 ^ ASCII(sex));
```

巧用REPLACE()【普通版】

执行用时：206 ms, 在所有 MySQL 提交中击败了29.92%的用户

内存消耗：0 B, 在所有 MySQL 提交中击败了100.00%的用户

```sql
UPDATE salary SET sex = REPLACE("fm", sex, "");
```


巧用REPLACE()【优化版】

执行用时：175 ms, 在所有 MySQL 提交中击败了81.98%的用户

内存消耗：0 B, 在所有 MySQL 提交中击败了100.00%的用户

```sql
UPDATE salary SET sex = REPLACE("fm", sex, "") WHERE sex != "";
```


使用IF()

执行用时：156 ms, 在所有 MySQL 提交中击败了99.84%的用户

内存消耗：0 B, 在所有 MySQL 提交中击败了100.00%的用户

```sql
UPDATE salary SET sex = IF(sex = 'm', 'f', 'm');
```


使用CASE...WHEN

执行用时：192 ms, 在所有 MySQL 提交中击败了44.25%的用户

内存消耗：0 B, 在所有 MySQL 提交中击败了100.00%的用户

```sql
UPDATE salary
SET
    sex = CASE sex
        WHEN 'm' THEN 'f'
        ELSE 'm'
    END;
```

## [665. 非递减数列](https://leetcode-cn.com/problems/non-decreasing-array/)

> 数组

执行用时：1 ms, 在所有 Java 提交中击败了99.54%的用户

内存消耗：39.5 MB, 在所有 Java 提交中击败了94.30%的用户

```java
class Solution {
    public boolean checkPossibility(int[] nums) {
        int n = nums.length;
        boolean flag = false;
        if (n < 3)
            return true;
        for (int i = 2; i < n; ++i) {
            int a = nums[i - 2], b = nums[i - 1], c = nums[i];
            if (a <= b && b <= c || a == b && b <=c || a <= b && b == c) // 123 继续
                continue;
            if (flag)
                return false;
            flag = true;
            if (a < b && b > c && a <= c) // 132 改第二个
                nums[i - 1] = nums[i - 2];
            else if (a > b && b < c && a <= c) // 213 改第二个
                nums[i - 1] = nums[i - 2];
            else if (a <= b && b > c && a >= c) // 231 改第三个
                nums[i] = nums[i - 1];
            else if (a > b && b <= c && a >= c) // 312/311 改第一个
                nums[i - 2] = nums[i - 1];
            else if (a > b && b > c) // 321 false
                return false;
        }
        return true;
    }
}
```

## [978. 最长湍流子数组](https://leetcode-cn.com/problems/longest-turbulent-subarray/)

> 数组，滑动数组，动态规划

循环两遍

```java
class Solution {
    public int maxTurbulenceSize(int[] arr) {
        int n = arr.length;
        if (n == 1)
            return 1;
        int maxLen = 1;
        for (int l = 0, r = 1; r < n; ++r) { // 奇大
            if ((r & 1) == 0 && arr[r] >= arr[r - 1] || (r & 1) == 1 && arr[r] <= arr[r - 1]) { // 不满足
                maxLen = Math.max(maxLen, r - l);
                l = r;
                continue;
            }
            if (r == n - 1) // 最后一个特殊处理
                maxLen = Math.max(maxLen, r - l + 1);
        }
        for (int l = 0, r = 1; r < n; ++r) { // 偶大
            if ((r & 1) == 0 && arr[r] <= arr[r - 1] || (r & 1) == 1 && arr[r] >= arr[r - 1]) {
                maxLen = Math.max(maxLen, r - l);
                l = r;
                continue;
            }
            if (r == n - 1)
                maxLen = Math.max(maxLen, r - l + 1);
        }
        return maxLen;
    }
}
```

循环一遍，比较三者

```java
class Solution {
    public int maxTurbulenceSize(int[] arr) {
        int n = arr.length, ans = 1, l = 0, r = 0;
        while (r < n - 1) {
            if (l == r) {
                if (arr[l] == arr[l + 1]) { // 相等同时右移
                    l++;
                }
                r++;
            } else {
                if (arr[r - 1] < arr[r] && arr[r] > arr[r + 1] || arr[r - 1] > arr[r] && arr[r] < arr[r + 1])
                    r++;
                else
                    l = r;
            }
            ans = Math.max(ans, r - l + 1);
        }
        return ans;
    }
}
```

动态规划

执行用时：4 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：41.4 MB, 在所有 Java 提交中击败了86.69%的用户

```java
class Solution {
    public int maxTurbulenceSize(int[] arr) {
        // 1.DP含义
        // dp[i][0]表示以arr[i]结尾在增长的最大长度
        // dp[i][1]表示以arr[i]结尾在降低的最大长度
        // 2.转移方程
        // arr[i-1]<arr[i]满足增长:dp[i][0]=dp[i-1][1]+1; dp[i][1]=1
        // arr[i-1]>arr[i]满足降低:dp[i][0]=1; dp[i][1]=dp[i-1][0]+1
        // arr[i-1]==arr[i]:dp[i][0]=dp[i][1]=1
        // 3.初始化
        // dp[0][0]=dp[0][1]=1
        // 4.结果
        // dp[i]最大值
        int max = 1, dp_i_0 = 1, dp_i_1 = 1, n = arr.length;
        for (int i = 1; i < n; ++i) {
            if (arr[i - 1] < arr[i]) {
                dp_i_0 = dp_i_1 + 1;
                dp_i_1 = 1;
                max = Math.max(max, dp_i_0);
            } else if (arr[i - 1] > arr[i]) {
                dp_i_1 = dp_i_0 + 1;
                dp_i_0 = 1;
                max = Math.max(max, dp_i_1);
            } else {
                dp_i_0 = 1;
                dp_i_1 = 1;
            }
        }
        return max;
    }
}
```

## [992. K 个不同整数的子数组](https://leetcode-cn.com/problems/subarrays-with-k-different-integers/)

> 双指针，滑动数组，频数数组

```java
class Solution {
    public int subarraysWithKDistinct(int[] A, int K) {
        // 恰好为K的个数 = [包含1~K的总数] - [包含1~K-1的总数]
        return disTotal(A, K) - disTotal(A, K - 1); 
    }

    private int disTotal(int[] A, int K) {
        int l = 0, r = 0, n = A.length, ans = 0, count = 0; // count:[l, r)里不同整数的个数
        int[] freq = new int[n + 1]; // 频数数组
        while (r < n) {
            if (freq[A[r]] == 0) // 新数
                count++;
            freq[A[r]]++;
            while (count > K) {
                freq[A[l]]--;
                if (freq[A[l]] == 0) // 只剩最后一个了
                    count--;
                l++; // 左指针右移
            }
            ans += r - l + 1; // [l, r]区间的长度就是对计数的贡献
            r++; // 右指针右移
        }
        return ans;
    }
}
```

## [1431. 拥有最多糖果的孩子](https://leetcode-cn.com/problems/kids-with-the-greatest-number-of-candies/)

> 数组

```java
class Solution {
    public List<Boolean> kidsWithCandies(int[] candies, int extraCandies) {
        int max = 0;
        for (int i = 0; i < candies.length; ++i)
            max = Math.max(max, candies[i]);
        List<Boolean> ret = new ArrayList<>();
        for (int i = 0; i < candies.length; ++i)
            ret.add(candies[i] + extraCandies >= max);
        return ret;
    }
}
```

## [567. 字符串的排列](https://leetcode-cn.com/problems/permutation-in-string/)

> 双指针，滑动窗口

执行用时：5 ms, 在所有 Java 提交中击败了92.74%的用户

内存消耗：38.7 MB, 在所有 Java 提交中击败了35.09%的用户

```java
class Solution {
    public boolean checkInclusion(String s1, String s2) {
        int n1 = s1.length(), n2 = s2.length();
        if (n1 > n2)
            return false;
        int[] freq = new int[26];
        for (char c: s1.toCharArray()) {
            freq[c - 'a']++;
        }
        int l = 0, r = n1 - 1;
        int[] tmp = new int[26];
        for (int i = 0; i < n1; ++i) {
            tmp[s2.charAt(i) - 'a']++;
        }
        while (r < n2) {
            if (!camp(tmp, freq)) {
                tmp[s2.charAt(l) - 'a']--;
                l++;
                r++;
                if (r == n2)
                    return false;
                tmp[s2.charAt(r) - 'a']++;
            } else {
                return true;
            }
        }
        return false;
    }

    private boolean camp(int[] a, int[] b) {
        for (int i = 0; i < 26; ++i) {
            if (a[i] != b[i])
                return false;
        }
        return true;
    }
}
```

TODO：可以记录窗口内不相等字符个数来减少比较（camp）次数 or 官方题解方法

## [703. 数据流中的第 K 大元素](https://leetcode-cn.com/problems/kth-largest-element-in-a-stream/)

> 堆，设计

```java
class KthLargest {
    private PriorityQueue<Integer> pq;
    private int k;

    public KthLargest(int k, int[] nums) {
        this.k = k;
        pq = new PriorityQueue<>(k); // 构造k长的小顶堆
        for (int num: nums) {
            this.add(num);
        }
    }
    
    public int add(int val) {
        pq.offer(val);
        if (pq.size() > this.k)
            pq.poll();
        return pq.peek();
    }
}

/**
 * Your KthLargest object will be instantiated and called as such:
 * KthLargest obj = new KthLargest(k, nums);
 * int param_1 = obj.add(val);
 */
```

## [119. 杨辉三角 II](https://leetcode-cn.com/problems/pascals-triangle-ii/)

> 数组，递归，递推

递归

```java
class Solution {
    public List<Integer> getRow(int rowIndex) {
        List<Integer> ret = new ArrayList<>();
        ret.add(1);
        if (rowIndex == 0)
            return ret;
        else if (rowIndex == 1) {
            ret.add(1);
            return ret;
        }
        List<Integer> lastRow = getRow(rowIndex - 1);
        for (int i = 0; i < lastRow.size() - 1; ++i) {
            ret.add(lastRow.get(i) + lastRow.get(i + 1));
        }
        ret.add(1);
        return ret;
    }
}
```

递推（从后往前加）（TODO）

```java
class Solution {
    public List<Integer> getRow(int rowIndex) {
        List<Integer> row = new ArrayList<Integer>();
        row.add(1);
        for (int i = 1; i <= rowIndex; ++i) {
            row.add(0);
            for (int j = i; j > 0; --j) {
                row.set(j, row.get(j) + row.get(j - 1));
            }
        }
        return row;
    }
}
```

数学（TODO）

```java
class Solution {
    public List<Integer> getRow(int rowIndex) {
        List<Integer> row = new ArrayList<Integer>();
        row.add(1);
        for (int i = 1; i <= rowIndex; ++i) {
            row.add((int) ((long) row.get(i - 1) * (rowIndex - i + 1) / i));
        }
        return row;
    }
}
```

## [448. 找到所有数组中消失的数字](https://leetcode-cn.com/problems/find-all-numbers-disappeared-in-an-array/)

> 数组

巧妙利用原数组

```java
class Solution {
    public List<Integer> findDisappearedNumbers(int[] nums) {
        int n = nums.length;
        for (int num: nums) {
            int index = (num - 1) % n;
            nums[index] += n;
        }
        List<Integer> ret = new ArrayList<>();
        for (int i = 0; i < n; ++i) {
            if (nums[i] <= n)
                ret.add(i + 1);
        }
        return ret;
    }
}
```

## [765. 情侣牵手](https://leetcode-cn.com/problems/couples-holding-hands/)

> 并查集，图

```java
class Solution {
    public int minSwapsCouples(int[] row) {
        // 情侣对数（最终连通分量）- 初始连通分量 = 最小交换次数
        int len = row.length; // 座位数
        int N = len / 2; // 情侣对数
        UnionFind uf = new UnionFind(N);
        for (int i = 0; i < len; i += 2) {
            uf.union(row[i] / 2, row[i + 1] / 2);
        }
        return N - uf.getCount();
    }

    private class UnionFind {
        private int[] parents;
        private int count; // 连通分量

        private int getCount() {
            return this.count;
        }

        UnionFind(int n) {
            parents = new int[n];
            for (int i = 0; i < n; ++i) {
                parents[i] = i;
            }
            this.count = n;
        }

        private void union(int a, int b) {
            int pa = find(a);
            int pb = find(b);
            if (pa == pb)
                return;
            parents[pb] = pa;
            this.count--;
        }

        private int find(int x) {
            return parents[x] == x ? x : (parents[x] = find(parents[x]));
        }
    }
}
```

## [485. 最大连续1的个数](https://leetcode-cn.com/problems/max-consecutive-ones/)

> 数组

```java
class Solution {
    public int findMaxConsecutiveOnes(int[] nums) {
        int count = 0, max = 0;
        for (int num: nums) {
            if (num == 1) {
                count++;
            } else {
                max = Math.max(max, count);
                count = 0;
            }
        }
        return Math.max(max, count);
    }
}
```

## [561. 数组拆分 I](https://leetcode-cn.com/problems/array-partition-i/)

> 数组

```java
class Solution {
    public int arrayPairSum(int[] nums) {
        Arrays.sort(nums);
        int ans = 0;
        for (int i = 0; i < nums.length; i += 2) {
            ans += nums[i];
        }
        return ans;
    }
}
```

## [566. 重塑矩阵](https://leetcode-cn.com/problems/reshape-the-matrix/)

> 数组

执行用时：1 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：39.5 MB, 在所有 Java 提交中击败了66.92%的用户

```java
class Solution {
    public int[][] matrixReshape(int[][] nums, int r, int c) {
        int ro = nums.length, co = nums[0].length;
        if (r * c != ro * co)
            return nums;
        int[][] ans = new int[r][c];
        for (int i = 0; i < ro; ++i) {
            for (int j = 0; j < co; ++j) {
                int flat = co * i + j;
                ans[flat / c][flat % c] = nums[i][j];
            }
        }
        return ans;
    }
}
```

## [995. K 连续位的最小翻转次数](https://leetcode-cn.com/problems/minimum-number-of-k-consecutive-bit-flips/)

> 差分数组，滑动窗口

差分数组（模）

```java
class Solution {
    public int minKBitFlips(int[] A, int K) {
        int n = A.length;
        int[] diff = new int[n + 1]; // 相邻两个元素翻转次数的差
        int ans = 0, revCnt = 0;
        for (int i = 0; i < n; ++i) {
            revCnt += diff[i]; // 累加和即当前元素的翻转次数
            if ((A[i] + revCnt) % 2 == 0) { // （当前元素值+翻转次数）是偶数=>当前为0
                if (i + K > n) { // 可翻转子数组超出限制
                    return -1;
                }
                ++ans; // 实际翻转次数+1
                ++revCnt; // 当前元素翻转次数+1
                --diff[i + K]; // 最后一个元素翻转次数-1
            }
        }
        return ans;
    }
}
```

差分数组（异或优化）

```java
class Solution {
    public int minKBitFlips(int[] A, int K) {
        int n = A.length;
        int[] diff = new int[n + 1];
        int ans = 0, revCnt = 0;
        for (int i = 0; i < n; ++i) {
            revCnt ^= diff[i];
            if (A[i] == revCnt) { // A[i] ^ revCnt == 0
                if (i + K > n) {
                    return -1;
                }
                ++ans;
                revCnt ^= 1;
                diff[i + K] ^= 1;
            }
        }
        return ans;
    }
}
```

滑动窗口

```java
class Solution {
    public int minKBitFlips(int[] A, int K) {
        int n = A.length;
        int ans = 0, revCnt = 0;
        for (int i = 0; i < n; ++i) {
            if (i >= K && A[i - K] > 1) {
                revCnt ^= 1;
                A[i - K] -= 2; // 复原数组元素，若允许修改数组 A，则可以省略
            }
            if (A[i] == revCnt) {
                if (i + K > n) {
                    return -1;
                }
                ++ans;
                revCnt ^= 1;
                A[i] += 2;
            }
        }
        return ans;
    }
}
```

## [1004. 最大连续1的个数 III](https://leetcode-cn.com/problems/max-consecutive-ones-iii/)

> 滑动窗口，前缀和，BS

不会变小的滑动窗口

```java
class Solution {
    public int longestOnes(int[] A, int K) {
        int l = 0, r = 0, n = A.length, count = 0, ans = 0;
        boolean flag = false;
        while (r < n) {
            if (A[r] == 0) // 更新右状态
                count++;
            if (count <= K) { // 满足
                ans = r - l + 1; // 更新结果
            } else { // 不满足，步进左右
                if (A[l++] == 0) // 移动左指针，更新左状态
                    count--;
            }
            r++; // 移动右指针
        }
        return ans;
    }
}
```

构造前缀和数组 TODO

前缀和+二分（经典）

```java
class Solution {
    public int longestOnes(int[] A, int K) {
        int n = A.length;
        int[] P = new int[n + 1];
        for (int i = 1; i <= n; ++i) {
            P[i] = P[i - 1] + (1 - A[i - 1]);
        }
        int ans = 0;
        for (int right = 0; right < n; ++right) {
            int left = binarySearch(P, P[right + 1] - K);
            ans = Math.max(ans, right - left + 1);
        }
        return ans;
    }

    public int binarySearch(int[] P, int target) {
        int low = 0, high = P.length - 1;
        while (low < high) {
            int mid = (high - low) / 2 + low;
            if (P[mid] < target) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        return low;
    }
}
```

前缀和+滑动窗口

```java
class Solution {
    public int longestOnes(int[] A, int K) {
        int n = A.length;
        int left = 0, lsum = 0, rsum = 0;
        int ans = 0;
        for (int right = 0; right < n; ++right) {
            rsum += 1 - A[right];
            while (lsum < rsum - K) {
                lsum += 1 - A[left];
                ++left;
            }
            ans = Math.max(ans, right - left + 1);
        }
        return ans;
    }
}
```

## [697. 数组的度](https://leetcode-cn.com/problems/degree-of-an-array/)

> 滑动窗口

用两个map

执行用时：25 ms, 在所有 Java 提交中击败了73.47%的用户

内存消耗：42.1 MB, 在所有 Java 提交中击败了61.59%的用户

```java
class Solution {
    public int findShortestSubArray(int[] nums) {
        int max = 0, n = nums.length;
        Map<Integer, Integer> fMap = new HashMap<>();
        for (int num: nums) {
            int fraq = fMap.getOrDefault(num, 0) + 1;
            fMap.put(num, fraq);
            max = Math.max(max, fraq);
        }

        int l = 0, r = 0, minLen = Integer.MAX_VALUE;
        Map<Integer, Integer> fCurMap = new HashMap<>();
        while (r < n) {
            // 当前r对应值频数+1
            fCurMap.put(nums[r], fCurMap.getOrDefault(nums[r], 0) + 1);
            if (fCurMap.get(nums[r]) != max) // 右移右指针
                r++;
            else {
                while (fCurMap.get(nums[r]) == max) { // 满足条件，收缩左指针
                    fCurMap.put(nums[l], fCurMap.get(nums[l]) - 1);
                    l++;
                }
                r++;
                minLen = Math.min(minLen, r - l + 1);
            }
        }
        return minLen;
    }
}
```

每个数都记录频数和最早出现位置和最晚出现位置，代码更清晰 TODO

```java
class Solution {
    public int findShortestSubArray(int[] nums) {
        Map<Integer, int[]> map = new HashMap<Integer, int[]>();
        int n = nums.length;
        for (int i = 0; i < n; i++) {
            if (map.containsKey(nums[i])) {
                map.get(nums[i])[0]++;
                map.get(nums[i])[2] = i;
            } else {
                map.put(nums[i], new int[]{1, i, i});
            }
        }
        int maxNum = 0, minLen = 0;
        for (Map.Entry<Integer, int[]> entry : map.entrySet()) {
            int[] arr = entry.getValue();
            if (maxNum < arr[0]) { // 更新最大值和最小长度
                maxNum = arr[0];
                minLen = arr[2] - arr[1] + 1;
            } else if (maxNum == arr[0]) { // 更新最小长度
                if (minLen > arr[2] - arr[1] + 1) {
                    minLen = arr[2] - arr[1] + 1;
                }
            }
        }
        return minLen;
    }
}
```

## [682. 棒球比赛](https://leetcode-cn.com/problems/baseball-game/)

> 栈，双端队列

执行用时：3 ms, 在所有 Java 提交中击败了81.80%的用户

内存消耗：37.5 MB, 在所有 Java 提交中击败了88.35%的用户

```java
class Solution {
    public int calPoints(String[] ops) {
        Deque<Integer> nums = new LinkedList<>();
        for (String op: ops) {
            if (op.equals("C")) {
                nums.pollLast();
            } else if (op.equals("D")) {
                nums.offer(nums.peekLast() * 2);
            } else if (op.equals("+")) {
                int a = nums.pollLast();
                int b = nums.peekLast();
                int sum = a + b;
                nums.offer(a);
                nums.offer(sum);
            } else {
                nums.offer(Integer.valueOf(op));
            }
        }
        int sum = 0;
        for (Integer i: nums) {
            sum += i;
        }
        return sum;
    }
}
```

## [1438. 绝对差不超过限制的最长连续子数组](https://leetcode-cn.com/problems/longest-continuous-subarray-with-absolute-diff-less-than-or-equal-to-limit/)

> 滑动窗口，双端队列，单调队列

两个优先队列

执行用时：191 ms, 在所有 Java 提交中击败了20.44%的用户

内存消耗：57 MB, 在所有 Java 提交中击败了40.04%的用户

```java
class Solution {
    public int longestSubarray(int[] nums, int limit) {
        PriorityQueue<Integer> spq = new PriorityQueue<>();
        PriorityQueue<Integer> bpq = new PriorityQueue<>(Collections.reverseOrder());
        int l = 0, r = 0, n = nums.length, ans = 0;
        while (r < n) {
            // 更新r状态
            spq.offer(nums[r]);
            bpq.offer(nums[r]);
            while (bpq.peek() - spq.peek() > limit) { // 不满足，收缩l
                spq.remove(nums[l]);
                bpq.remove(nums[l]);
                l++;
            }
            // 满足，更新
            ans = Math.max(ans, r - l + 1);
            r++;
        }
        return ans;
    }
}
```

两个双端队列（因为只需要最大值和最小值，所以维护大小两个单调队列即可）

执行用时：33 ms, 在所有 Java 提交中击败了93.43%的用户

内存消耗：47.8 MB, 在所有 Java 提交中击败了78.77%的用户

```java
class Solution {
    public int longestSubarray(int[] nums, int limit) {
        Deque<Integer> minDeque = new ArrayDeque<>(); // 维护最小单调队列，排除出现在中间的较大值，头部最小值
        Deque<Integer> maxDeque = new ArrayDeque<>(); // 维护最大单调队列，排除出现在中间的较小值，头部最大值
        int n = nums.length, l = 0, r = 0, ans = 0;
        while (r < n) {
            // 用r维护状态
            while (!minDeque.isEmpty() && nums[r] < minDeque.peekLast())
                minDeque.pollLast();
            while (!maxDeque.isEmpty() && nums[r] > maxDeque.peekLast())
                maxDeque.pollLast();
            minDeque.offerLast(nums[r]);
            maxDeque.offerLast(nums[r]);
            while (maxDeque.peekFirst() - minDeque.peekFirst() > limit) {
                // 收缩l
                // minDeque.remove(nums[l]);
                // maxDeque.remove(nums[l]);
                // TODO 很妙
                if (nums[l] == minDeque.peekFirst())
                    minDeque.pollFirst();
                if (nums[l] == maxDeque.peekFirst())
                    maxDeque.pollFirst();
                l++;
            }
            ans = Math.max(ans, r - l + 1);
            r++;
        }
        return ans;
    }
}
```

用平衡树，如TreeMap

执行用时：85 ms, 在所有 Java 提交中击败了32.85%的用户

内存消耗：47.5 MB, 在所有 Java 提交中击败了82.68%的用户

```java
class Solution {
    public int longestSubarray(int[] nums, int limit) {
        TreeMap<Integer, Integer> map = new TreeMap<>();
        int l = 0, r = 0, n = nums.length, ans = 0;
        while (r < n) {
            // 更新r状态
            map.put(nums[r], map.getOrDefault(nums[r], 0) + 1);
            while (map.lastKey() - map.firstKey() > limit) { // 不满足，收缩l
                map.put(nums[l], map.getOrDefault(nums[l], 0) - 1);
                if (map.get(nums[l]) == 0)
                    map.remove(nums[l]);
                l++;
            }
            // 满足，更新
            ans = Math.max(ans, r - l + 1);
            r++;
        }
        return ans;
    }
}
```

## [766. 托普利茨矩阵](https://leetcode-cn.com/problems/toeplitz-matrix/)

> 数组

执行用时：1 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：38.6 MB, 在所有 Java 提交中击败了59.23%的用户

```java
class Solution {
    public boolean isToeplitzMatrix(int[][] matrix) {
        int m = matrix.length, n = matrix[0].length;
        for (int i = 1; i < m; ++i) {
            for (int j = 1; j < n; ++j) {
                if (matrix[i][j] != matrix[i - 1][j - 1]) {
                    return false;
                }
            }
        }
        return true;
    }
}
```

## [5685. 交替合并字符串](https://leetcode-cn.com/problems/merge-strings-alternately/)

> 字符串

执行用时：1 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：37 MB, 在所有 Java 提交中击败了100.00%的用户

```java
class Solution {
    public String mergeAlternately(String word1, String word2) {
        StringBuilder sb = new StringBuilder();
        int len1 = word1.length(), len2 = word2.length();
        int len = Math.min(len1, len2);
        int maxLen = Math.max(len1, len2);
        for (int i = 0; i < len; ++i) {
            sb.append(word1.charAt(i));
            sb.append(word2.charAt(i));
        }
        for (int i = len; i < maxLen; ++i) {
            if (len1 > len2) {
                sb.append(word1.charAt(i));
            } else {
                sb.append(word2.charAt(i));
            }
        }
        return sb.toString();
    }
}
```

## [5686. 移动所有球到每个盒子所需的最小操作数](https://leetcode-cn.com/problems/minimum-number-of-operations-to-move-all-balls-to-each-box/)

> 贪心算法，数组

执行用时：3 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：39.2 MB, 在所有 Java 提交中击败了100.00%的用户

```java
class Solution {
    public int[] minOperations(String boxes) {
        int n = boxes.length();
        int[] ans = new int[n];
        int leftOne = 0, rightOne = 0;
        for (int i = 0; i < n; ++i) {
            if (boxes.charAt(i) == '1') {
                rightOne++;
                ans[0] += i;
            }
        }
        if (boxes.charAt(0) == '1') {
            leftOne++;
            rightOne--;
        }
        for (int i = 1; i < n; ++i) {
            if (boxes.charAt(i) == '1') {
                rightOne--;
                leftOne++;
                ans[i] = ans[i - 1] + (leftOne - 1) - (rightOne + 1);
            } else {
                ans[i] = ans[i - 1] + leftOne - rightOne;
            }
        }
        return ans;
    }
}
```

## [5687. 执行乘法运算的最大分数](https://leetcode-cn.com/problems/maximum-score-from-performing-multiplication-operations/)

> 动态规划

执行用时：60 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：47.6 MB, 在所有 Java 提交中击败了100.00%的用户

```java
class Solution {
    public int maximumScore(int[] nums, int[] multipliers) {
        // 状态定义：dp[i][l][r]表示第i步后的最大分数，l、r表示执行后的左右坐标,r=n-i+l => dp[i][l]
        // 转移方程：dp[i][l] = max(dp[i-1][l-1]+选左,dp[i-1][l]+选右)
        // 初始化：dp[0][0~m]=0
        //        dp[1~m][0~m]=-无穷
        // 结果：max(dp[m])
        int n = nums.length, m = multipliers.length;
        int[][] dp = new int[m + 1][m + 1];
        for (int i = 0; i < m; ++i)
            for (int j = 0; j <= m; ++j)
                dp[i + 1][j] = Integer.MIN_VALUE;
        for (int i = 1; i <= m; ++i) { // [1,m]表示第1轮到第m轮
            for (int j = 0; j <= i; ++j) { // 表示处理完之后左边删除了几个
                int mul = multipliers[i - 1];
                if (j == i) { // 全选左
                    dp[i][j] = dp[i - 1][j - 1] + mul * nums[j - 1];
                } else if (j == 0) { // 全选右
                    dp[i][j] = dp[i - 1][j] + mul * nums[n - i + j];
                } else {
                    int left = dp[i - 1][j - 1] + mul * nums[j - 1];
                    int right = dp[i - 1][j] + mul * nums[n - i + j];
                    dp[i][j] = Math.max(left, right);
                }
            }
        }
        int ans = Integer.MIN_VALUE;
        for (int i = 1; i <= m; ++i) {
            ans = Math.max(ans, dp[m][i]);
        }
        return ans;
    }
}
```

## [516. 最长回文子序列](https://leetcode-cn.com/problems/longest-palindromic-subsequence/)

> 动态规划，回文，最长子序列

```java
class Solution {
    public int longestPalindromeSubseq(String s) {
        // 定义：dp[i][j]表示从i字符到j字符的回文子序列长度
        // 转移方程：char(i)==char(j) => dp[i][j]=dp[i+1][j-1]+2;
        //         char(i)!=char(j) => dp[i][j]=max(dp[i+1][j],dp[i][j-1])
        // 初始化：dp[i][i]=1
        // 求：dp[0][n-1]
        // 注意遍历顺序
        int n = s.length();
        int[][] dp = new int[n][n];
        for (int i = 0; i < n; ++i) {
            dp[i][i] = 1;
        }
        for (int i = n - 1; i >= 0; --i) {
            for (int j = i + 1; j < n; ++j) {
                if (s.charAt(i) == s.charAt(j)) {
                    dp[i][j] = dp[i + 1][j - 1] + 2;
                } else {
                    dp[i][j] = Math.max(dp[i + 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp[0][n - 1];
    }
}
```

## [5688. 由子序列构造的最长回文串的长度](https://leetcode-cn.com/problems/maximize-palindrome-length-from-subsequences/)

> 动态规划，回文，最长子序列

```java
class Solution {
    public int longestPalindrome(String word1, String word2) {
        // 定义：dp[i][j]表示从i字符到j字符的回文子序列长度
        // 转移方程：char(i)==char(j) => dp[i][j]=dp[i+1][j-1]+2;
        //         char(i)!=char(j) => dp[i][j]=max(dp[i+1][j],dp[i][j-1])
        // 初始化：dp[i][i]=1
        // 求：dp[0][n-1]
        String s = word1 + word2;
        int n = s.length(), x = word1.length(), ans = 0;
        int[][] dp = new int[n][n];
        for (int i = n - 1; i >= 0; --i) {
            dp[i][i] = 1;
            for (int j = i + 1; j < n; ++j) {
                if (s.charAt(i) == s.charAt(j)) {
                    dp[i][j] = dp[i + 1][j - 1] + 2;
                    // 当i，j满足非空条件时才更新结果
                    if (i < x && x <= j) {
                        ans = Math.max(ans, dp[i][j]);
                    }
                } else {
                    dp[i][j] = Math.max(dp[i + 1][j], dp[i][j - 1]);
                }
            }
        }
        return ans;
    }
}
```

## [1052. 爱生气的书店老板](https://leetcode-cn.com/problems/grumpy-bookstore-owner/)

> 滑动窗口

```java
class Solution {
    public int maxSatisfied(int[] customers, int[] grumpy, int X) {
        int n = customers.length;
        int sum = 0;
        for (int i = 0; i < n; ++i) { // 原始结果
            if (grumpy[i] == 0)
                sum += customers[i];
        }
        int change = 0;
        for (int i = 0; i < X; ++i) {
            change += grumpy[i] * customers[i];
        }
        int maxChange = change;
        for (int i = X; i < n; ++i) { // 求最大增长量
            change += grumpy[i] * customers[i] - grumpy[i - X] * customers[i - X];
            maxChange = Math.max(maxChange, change);
        }
        return sum + maxChange;
    }
}
```

## [面试题 02.03. 删除中间节点](https://leetcode-cn.com/problems/delete-middle-node-lcci/)

> 链表

执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：37.6 MB, 在所有 Java 提交中击败了93.45%的用户

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
    public void deleteNode(ListNode node) {
        node.val = node.next.val;
        node.next = node.next.next;
    }
}
```

## [1684. 统计一致字符串的数目](https://leetcode-cn.com/problems/count-the-number-of-consistent-strings/)

> 字符串，字符数组

```java
class Solution {
    public int countConsistentStrings(String allowed, String[] words) {
        int[] chars = new int[26];
        for (char c: allowed.toCharArray()) {
            chars[c - 'a']++;
        }
        int count = words.length;
        for (String word: words) {
            for (char c: word.toCharArray()) {
                if (chars[c - 'a'] == 0) {
                    count--;
                    break;
                }
            }
        }
        return count;
    }
}
```

## [1108. IP 地址无效化](https://leetcode-cn.com/problems/defanging-an-ip-address/)

> 字符串

```java
class Solution {
    public String defangIPaddr(String address) {
        StringBuilder sb = new StringBuilder();
        for (char c: address.toCharArray()) {
            if (c == '.')
                sb.append('[').append('.').append(']');
            else
                sb.append(c);
        }
        return sb.toString();
    }
}
```

## [1732. 找到最高海拔](https://leetcode-cn.com/problems/find-the-highest-altitude/)

> 数组

```java
class Solution {
    public int largestAltitude(int[] gain) {
        int maxHeight = 0;
        int height = 0;
        for (int i: gain) {
            height += i;
            maxHeight = Math.max(maxHeight, height);
        }
        return maxHeight;
    }
}
```

## [1290. 二进制链表转整数](https://leetcode-cn.com/problems/convert-binary-number-in-a-linked-list-to-integer/)

> 位运算，链表

执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：35.7 MB, 在所有 Java 提交中击败了89.80%的用户

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
    public int getDecimalValue(ListNode head) {
        int ans = 0;
        while (head != null) {
            ans = ans * 2 + head.val;
            head = head.next;
        }
        return ans;
    }
}
```

## [1688. 比赛中的配对次数](https://leetcode-cn.com/problems/count-of-matches-in-tournament/)

> 递归

执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：35.5 MB, 在所有 Java 提交中击败了9.78%的用户

```java
class Solution {
    public int numberOfMatches(int n) {
        int ans = 0;
        while (n > 1) {
            ans += n / 2;
            n = ((n & 1) == 1) ? ((n - 1) / 2 + 1) : n / 2;
        }
        return ans;
    }
}
```

直接求

```java
class Solution {
    public int numberOfMatches(int n) {
        return n - 1;
    }
}
```

## [832. 翻转图像](https://leetcode-cn.com/problems/flipping-an-image/)

> 数组

执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：38.5 MB, 在所有 Java 提交中击败了79.36%的用户

```java
class Solution {
    public int[][] flipAndInvertImage(int[][] A) {
        int m = A.length;
        int n = A[0].length;
        for (int i = 0; i < m; ++i) {
            int len = (n & 1) == 1 ? (n / 2 + 1) : n / 2;
            for (int j = 0; j < len; ++j) {
                if (A[i][j] == A[i][n - j - 1]) {
                    A[i][j] ^= 1;
                    A[i][n - j - 1] = A[i][j];
                }
            }
        }
        return A;
    }
}
```

## [867. 转置矩阵](https://leetcode-cn.com/problems/transpose-matrix/)

> 数组

```java
class Solution {
    public int[][] transpose(int[][] matrix) {
        int m = matrix.length;
        int n = matrix[0].length;
        int[][] ans = new int[n][m];
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                ans[j][i] = matrix[i][j];
            }
        }
        return ans;
    }
}
```

## [1178. 猜字谜](https://leetcode-cn.com/problems/number-of-valid-words-for-each-puzzle/)

> 位运算，哈希表

```java
class Solution {
    public List<Integer> findNumOfValidWords(String[] words, String[] puzzles) {
        int wn = words.length, pn = puzzles.length;
        Map<Integer, Integer> fraq = new HashMap<>(); // key:字符int编码，val:数量
        for (int i = 0; i < wn; ++i) {
            int key = 0;
            for (char c: words[i].toCharArray())
                key |= (1 << (c - 'a'));
            if (Integer.bitCount(key) <= 7)
                fraq.put(key, fraq.getOrDefault(key, 0) + 1);
        }
        List<Integer> res = new ArrayList<>();
        for (String puzzle: puzzles) {
            int count = 0;
            // 枚举法1
            // 枚举1-6的这6个位置的01情况，一共1<<6种选择
            for (int choice = 0; choice < (1 << 6); ++choice) {
                int mask = 0;
                // 检查除了首位外每种情况下哪几位是1
                for (int i = 0; i < 6; ++i) {
                    if ((choice & (1 << i)) != 0) {
                        mask |= (1 << (puzzle.charAt(i + 1) - 'a'));
                    }
                }
                // 首位处理
                mask |= (1 << (puzzle.charAt(0) - 'a'));
                if (fraq.containsKey(mask)) {
                    count+= fraq.get(mask);
                }
            }
            // 枚举法2：通用“枚举二进制子集”
            // int mask = 0;
            // for (int i = 1; i < 7; ++i) {
            //     mask |= (1 << (puzzle.charAt(i) - 'a'));
            // }
            // int subset = mask;
            // do {
            //     int s = subset | (1 << (puzzle.charAt(0) - 'a'));
            //     if (fraq.containsKey(s)) {
            //         count += fraq.get(s);
            //     }
            //     subset = (subset - 1) & mask;
            // } while (subset != mask);
            
            res.add(count);
        }
        return res;
    }
}
```

TODO 字典树

## [395. 至少有K个重复字符的最长子串](https://leetcode-cn.com/problems/longest-substring-with-at-least-k-repeating-characters/)

> 递归，分治，滑动窗口

执行用时：1 ms, 在所有 Java 提交中击败了78.51%的用户

内存消耗：36.1 MB, 在所有 Java 提交中击败了95.23%的用户

```java
class Solution {
    public int longestSubstring(String s, int k) {
        // 分治法
        int n = s.length();
        return dfs(s, 0, n - 1, k);
    }

    private int dfs(String s, int l, int r, int k) {
        int[] cnt = new int[26]; // 频数数组
        for (int i = l; i <= r; ++i) {
            cnt[s.charAt(i) - 'a']++;
        }
        char split = 0; // 分割字符
        for (int i = 0; i < 26; ++i) {
            if (cnt[i] != 0 && cnt[i] < k) {
                split = (char)(i + 'a');
            }
        }
        if (split == 0) { // 没有小于k的，满足要求
            return r - l + 1;
        }
        // 有需要分割的
        int start = l, end = l, status = 0, max = 0;
        for (int i = l; i <= r; ++i) {
            if (status == 0 && s.charAt(i) != split) {
                status = 1;
                start = i;
            } else if (status == 1 && (s.charAt(i) == split || i == r)) {
                end = s.charAt(i) != split ? i : i - 1;
                if (end - start + 1 >= k)
                    max = Math.max(max, dfs(s, start, end, k));
                start = i;
                status = 0;
            }
        }
        return max;
    }
}
```

TODO 滑动窗口（不好理解）

## [896. 单调数列](https://leetcode-cn.com/problems/monotonic-array/)

> 数组

执行用时：1 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：46.7 MB, 在所有 Java 提交中击败了56.41%的用户

```java
class Solution {
    public boolean isMonotonic(int[] A) {
        int state = 0; // 0表示之前为=或者还未设置，1表示之前为>=，2表示之前为<=
        int n = A.length;
        for (int i = 1; i < n; ++i) {
            if (A[i] > A[i - 1]) {
                if (state == 0)
                    state = 1;
                else if (state == 2)
                    return false;
            } else if (A[i] < A[i - 1]) {
                if (state == 0)
                    state = 2;
                else if (state == 1)
                    return false;
            }
        }
        return true;
    }
}
```

执行用时：2 ms, 在所有 Java 提交中击败了40.70%的用户

内存消耗：47 MB, 在所有 Java 提交中击败了8.92%的用户

```java
class Solution {
    public boolean isMonotonic(int[] A) {
        // 既遇到>又遇到<则不是
        int n = A.length;
        boolean asc = true, desc = true; // 1 1 1;1 0 1; 0 1 1;0 0 0
        for (int i = 1; i < n; ++i) {
            if (A[i] > A[i - 1]) {
                asc = false;
            } else if (A[i] < A[i - 1]) {
                desc = false;
            }
        }
        return asc || desc;
    }
}
```

## [303. 区域和检索 - 数组不可变](https://leetcode-cn.com/problems/range-sum-query-immutable/)

> 前缀和

执行用时：9 ms, 在所有 Java 提交中击败了99.83%的用户

内存消耗：41 MB, 在所有 Java 提交中击败了97.71%的用户

```java
class NumArray {

    private int[] prefix;

    public NumArray(int[] nums) {
        int tmp = 0;
        int n = nums.length;
        this.prefix = new int[n + 1];
        for (int i = 1; i <= n; ++i) {
            tmp += nums[i - 1];
            this.prefix[i] = tmp;
        }
    }
    
    public int sumRange(int i, int j) {
        return this.prefix[j + 1] - this.prefix[i];
    }
}

/**
 * Your NumArray object will be instantiated and called as such:
 * NumArray obj = new NumArray(nums);
 * int param_1 = obj.sumRange(i,j);
 */
```

## [304. 二维区域和检索 - 矩阵不可变](https://leetcode-cn.com/problems/range-sum-query-2d-immutable/)

> 二维矩阵，前缀和

```java
class NumMatrix {

    private int[][] preSum;

    public NumMatrix(int[][] matrix) {
        int m = matrix.length;
        if (m > 0) {
            int n = matrix[0].length;
            preSum = new int[m + 1][n + 1];
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) {
                    preSum[i + 1][j + 1] = preSum[i + 1][j] + preSum[i][j + 1] - preSum[i][j] + matrix[i][j];
                }
            }
        }
    }
    
    public int sumRegion(int row1, int col1, int row2, int col2) {
        return preSum[row2 + 1][col2 + 1] - preSum[row2 + 1][col1] - preSum[row1][col2 + 1] + preSum[row1][col1];
    }
}

/**
 * Your NumMatrix object will be instantiated and called as such:
 * NumMatrix obj = new NumMatrix(matrix);
 * int param_1 = obj.sumRegion(row1,col1,row2,col2);
 */
```

## [338. 比特位计数](https://leetcode-cn.com/problems/counting-bits/)

> 位运算，动态规划

找规律

```java
class Solution {
    public int[] countBits(int num) {
        // 0(0)
        // 1	1(1)
        // 2	1(2) 2(3)
        // 4	1(4) 2(5) 2(6) 3(7)
        // 8	1(8) 2(9) 2(10) 3(11) 2(12) 3(13) 3(14) 4(15)
        // 16	1 2 2 3 2 3 3 4
        // 32	1 2 2 3 2 3 3 4 2 3 3 4 3 4 4 5
        int[] ans = new int[num + 1];
        if (num == 0)
            return ans;
        ans[1] = 1;
        int tmp = 2;
        for (int i = 2; i <= num; ++i) {
            if (i >= tmp && i < (int)tmp * 1.5) {
                ans[i] = ans[i - tmp / 2];
            } else {
                if (i == tmp * 2) {
                    tmp = i;
                }
                ans[i] = ans[i - tmp] + 1;
            }
        }
        return ans;
    }
}
```

暴力求解

```java
class Solution {
    public int[] countBits(int num) {
        int[] bits = new int[num + 1];
        for (int i = 0; i <= num; i++) {
            bits[i] = countOnes(i);
        }
        return bits;
    }

    public int countOnes(int x) {
        // return Integer.bitCount(x);
        int ones = 0;
        while (x > 0) {
            x &= (x - 1);
            ones++;
        }
        return ones;
    }
}
```

TODO 以下各方法的分析和技巧

动态规划——最高有效位

```java
class Solution {
    public int[] countBits(int num) {
        int[] bits = new int[num + 1];
        int highBit = 0;
        for (int i = 1; i <= num; i++) {
            if ((i & (i - 1)) == 0) {
                highBit = i;
            }
            bits[i] = bits[i - highBit] + 1;
        }
        return bits;
    }
}
```

动态规划——最低有效位

```java
class Solution {
    public int[] countBits(int num) {
        int[] bits = new int[num + 1];
        for (int i = 1; i <= num; i++) {
            bits[i] = bits[i >> 1] + (i & 1);
        }
        return bits;
    }
}
```

动态规划——最低设置位

```java
class Solution {
    public int[] countBits(int num) {
        int[] bits = new int[num + 1];
        for (int i = 1; i <= num; i++) {
            bits[i] = bits[i & (i - 1)] + 1;
        }
        return bits;
    }
}
```

## [232. 用栈实现队列](https://leetcode-cn.com/problems/implement-queue-using-stacks/)

> 栈，设计

```java
class MyQueue {
    private Deque<Integer> dqHelper;
    private Deque<Integer> dqMain;

    /** Initialize your data structure here. */
    public MyQueue() {
        this.dqHelper = new LinkedList<>();
        this.dqMain = new LinkedList<>();
    }
    
    /** Push element x to the back of queue. */
    public void push(int x) {
        while (!this.dqMain.isEmpty()) {
            this.dqHelper.offer(this.dqMain.pollLast());
        }
        dqMain.offer(x);
        while (!this.dqHelper.isEmpty()) {
            this.dqMain.offer(this.dqHelper.pollLast());
        }
    }
    
    /** Removes the element from in front of queue and returns that element. */
    public int pop() {
        return this.dqMain.pollLast();
    }
    
    /** Get the front element. */
    public int peek() {
        return this.dqMain.peekLast();
    }
    
    /** Returns whether the queue is empty. */
    public boolean empty() {
        return this.dqMain.isEmpty();
    }
}

/**
 * Your MyQueue object will be instantiated and called as such:
 * MyQueue obj = new MyQueue();
 * obj.push(x);
 * int param_2 = obj.pop();
 * int param_3 = obj.peek();
 * boolean param_4 = obj.empty();
 */
```

不用每次都倒完，顺序栈取完了再倒

```java
class MyQueue {
    Deque<Integer> inStack;
    Deque<Integer> outStack;

    public MyQueue() {
        inStack = new LinkedList<Integer>();
        outStack = new LinkedList<Integer>();
    }
    
    public void push(int x) {
        inStack.push(x);
    }
    
    public int pop() {
        if (outStack.isEmpty()) {
            in2out();
        }
        return outStack.pop();
    }
    
    public int peek() {
        if (outStack.isEmpty()) {
            in2out();
        }
        return outStack.peek();
    }
    
    public boolean empty() {
        return inStack.isEmpty() && outStack.isEmpty();
    }

    private void in2out() {
        while (!inStack.isEmpty()) {
            outStack.push(inStack.pop());
        }
    }
}
```

## [503. 下一个更大元素 II](https://leetcode-cn.com/problems/next-greater-element-ii/)

> 栈，循环数组，单调栈

暴力法

```java
class Solution {
    public int[] nextGreaterElements(int[] nums) {
        int n = nums.length;
        int[] ans = new int[n];
        for (int i = 0; i < n; ++i) {
            int count = 0;
            int tmp = nums[i];
            while (count < n) {
                int cur = nums[(i + count) % n];
                if (cur > tmp) {
                    ans[i] = cur;
                    break;
                }
                count++;
            }
            if (count == n)
                ans[i] = -1;
        }
        return ans;
    }
}
```

单调栈 + 拉直数组

```java
class Solution {
    public int[] nextGreaterElements(int[] nums) {
        int n = nums.length;
        int[] ans = new int[n];
        Arrays.fill(ans, -1);
        Deque<Integer> stack = new LinkedList<>();
        // 把循环数组拉直，用单调栈（单调不增）
        for (int i = 0; i < n * 2 - 1; ++i) {
            while (!stack.isEmpty() && nums[stack.peek()] < nums[i % n]) { // 栈顶小于当前值，弹栈，并更新它对应位置结果
                ans[stack.pop()] = nums[i % n];
            }
            stack.push(i % n);
        }
        return ans;
    }
}
```

## [131. 分割回文串](https://leetcode-cn.com/problems/palindrome-partitioning/)

> 深度优先搜索，回溯，动态规划，记忆化搜索

DP+DFS+回溯

```java
class Solution {
    private List<List<String>> res = new ArrayList<>();
    private List<String> tmp = new ArrayList<>();
    private int n = 0;
    private boolean[][] dp;

    public List<List<String>> partition(String s) {
        n = s.length();
        // dp[i][j]表示[i,j]是否为回文串
        dp = new boolean[n][n];
        for (int i = 0; i < n; ++i) {
            Arrays.fill(dp[i], true);
        }
        for (int i = n - 1; i >= 0; --i) {
            for (int j = i + 1; j < n; ++j) {
                dp[i][j] = dp[i + 1][j - 1] && (s.charAt(i) == s.charAt(j));
            }
        }
        // 搜索+回溯
        dfs(s, 0);
        return res;
    }

    private void dfs(String s, int i) {
        if (i == n) { // 该分支检查完毕，保存路径
            res.add(new ArrayList<String>(tmp));
            return;
        }
        for (int j = i; j < n; ++j) { // 固定i，遍历每个j
            if (dp[i][j]) {
                tmp.add(s.substring(i, j + 1));
                dfs(s, j + 1);
                tmp.remove(tmp.size() - 1); // 回溯
            }
        }
    }
}
```

记忆化搜索

```java
class Solution {
    private List<List<String>> res = new ArrayList<>();
    private List<String> tmp = new ArrayList<>();
    private int n = 0;
    private int[][] mem; // 0表示未搜索，1表示是回文串，-1表示不是回文串

    public List<List<String>> partition(String s) {
        n = s.length();
        mem = new int[n][n];
        // 记忆化搜索+回溯
        dfs(s, 0);
        return res;
    }

    private void dfs(String s, int i) {
        if (i == n) { // 该分支检查完毕，保存路径
            res.add(new ArrayList<String>(tmp));
            return;
        }
        for (int j = i; j < n; ++j) { // 固定i，遍历每个j
            if (isPalindrome(s, i, j) == 1) {
                tmp.add(s.substring(i, j + 1));
                dfs(s, j + 1);
                tmp.remove(tmp.size() - 1); // 回溯
            }
        }
    }

    private int isPalindrome(String s, int i, int j) {
        if (mem[i][j] != 0) {
            return mem[i][j];
        }
        if (i >= j) { // 只有一个数或者不存在
            mem[i][j] = 1;
        } else if (s.charAt(i) == s.charAt(j)) {
            mem[i][j] = isPalindrome(s, i + 1, j - 1);
        } else {
            mem[i][j] = -1;
        }
        return mem[i][j];
    }
}
```

## [132. 分割回文串 II](https://leetcode-cn.com/problems/palindrome-partitioning-ii/)

> 动态规划

两重DP

```java
class Solution {
    public int minCut(String s) {
        int n = s.length();
        // 先记录所有子串是否是回文串
        boolean[][] dpIsPal = new boolean[n][n];
        for (int i = 0; i < n; ++i) {
            Arrays.fill(dpIsPal[i], true);
        }
        for (int i = n - 1; i >= 0; --i) {
            for (int j = i + 1; j < n; ++j) {
                dpIsPal[i][j] = (s.charAt(i) == s.charAt(j)) && dpIsPal[i + 1][j - 1];
            }
        }
        // 计算最小dp
        // 定义：dp[0...i]表示s[0,i]最小
        // 递推：dp[0...i]=dp[0...i]是回文串=>0；dp[0...i]不是回文串=>min(dp[0...j]+1)(j<i && s[j+1,i]是回文串)
        // 初始化：dp[0...i] = MAX
        // 结果：dp[n - 1]
        int[] dp = new int[n];
        Arrays.fill(dp, Integer.MAX_VALUE);
        for (int i = 0; i < n; ++i) {
            if (dpIsPal[0][i]) {
                dp[i] = 0;
            } else {
                for (int j = 0; j < i; ++j) {
                    if (dpIsPal[j + 1][i]) {
                        dp[i] = Math.min(dp[i], dp[j] + 1);
                    }
                }
            }
        }
        return dp[n - 1];
    }
}
```

## [1047. 删除字符串中的所有相邻重复项](https://leetcode-cn.com/problems/remove-all-adjacent-duplicates-in-string/)

> 栈

开一个栈

```java
class Solution {
    public String removeDuplicates(String S) {
        Deque<Character> stack = new LinkedList<>();
        for (char c: S.toCharArray()) {
            if (!stack.isEmpty() && stack.peekLast() == c) {
                stack.pollLast();
            } else {
                stack.offerLast(c);
            }
        }
        StringBuilder sb = new StringBuilder();
        for (Character c: stack) {
            sb.append(c);
        }
        return sb.toString();
    }
}
```

直接用StringBuilder

```java
class Solution {
    public String removeDuplicates(String S) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < S.length(); ++i) {
            char c = S.charAt(i);
            int last = sb.length() - 1;
            if (last >= 0 && c == sb.charAt(last)) {
                sb.deleteCharAt(last);
            } else {
                sb.append(c);
            }
        }
        return sb.toString();
    }
}
```

直接用字符数组

```java
class Solution {
    public String removeDuplicates(String S) {
        if(S.length() == 1) return S;
        char[] ss = S.toCharArray();
        int index = -1;
        for (char c : ss) {
            if (index != -1 && c == ss[index]) {
                index--;
            } else {
                ++index;
                ss[index] = c;
            }
        }
        return new String(ss, 0, index + 1);
    }
}
```

## [224. 基本计算器](https://leetcode-cn.com/problems/basic-calculator/)

> 栈，数学

```java
class Solution {
    public int calculate(String s) {
        StringBuilder symbols = new StringBuilder(); // 符号栈
        Deque<Integer> nums = new LinkedList<>(); // 数字栈
        int ans = 0;
        char[] chs = s.toCharArray();
        int len = chs.length;
        boolean lastIsNum = false;
        char lastChar = '^';
        for (int i = 0; i < len; ++i) {
            char ch = chs[i];
            if (ch == '+' || ch == '-') { // 查看前面有+-不，有的话就求一下
                if (lastChar == '^' || lastChar == '(') { // 前面为空或者左括号要做负数处理
                    nums.offerLast(0);
                }
                if (symbols.length() != 0) {
                    int lastIndex = symbols.length() - 1;
                    char lastCh = symbols.charAt(lastIndex);
                    if (lastCh == '+') {
                        nums.offerLast(nums.pollLast() + nums.pollLast());
                        symbols.deleteCharAt(lastIndex);
                    } else if (lastCh == '-') {
                        int tmp = nums.pollLast();
                        nums.offerLast(nums.pollLast() - tmp);
                        symbols.deleteCharAt(lastIndex);
                    }
                }
                symbols.append(ch);
                lastIsNum = false;
                lastChar = ch;
            } else if (ch == '(') {
                symbols.append(ch);
                lastIsNum = false;
                lastChar = ch;
            } else if (ch == ')') {
                int lastIndex = symbols.length() - 1;
                char lastCh = symbols.charAt(lastIndex);
                if (lastCh == '+') {
                    nums.offerLast(nums.pollLast() + nums.pollLast());
                    symbols.deleteCharAt(lastIndex);
                    symbols.deleteCharAt(lastIndex - 1); // 删除左括号
                } else if (lastCh == '-') {
                    int tmp = nums.pollLast();
                    nums.offerLast(nums.pollLast() - tmp);
                    symbols.deleteCharAt(lastIndex);
                    symbols.deleteCharAt(lastIndex - 1); // 删除左括号
                } else if (lastCh == '(')
                    symbols.deleteCharAt(lastIndex); // 删除左括号
                lastIsNum = false;
                lastChar = ch;
            } else if (ch - '0' >= 0 && ch - '0' <= 9) { // 数字
                if (lastIsNum) {
                    nums.offerLast(nums.pollLast() * 10 + (ch - '0'));
                } else {
                    nums.offerLast(ch - '0');
                }
                lastIsNum = true;
                lastChar = ch;
            } else { // 空格
                lastIsNum = false;
            }
        }
        if (symbols.length() != 0) {
            int lastIndex = symbols.length() - 1;
            char lastCh = symbols.charAt(lastIndex);
            if (lastCh == '+') {
                nums.offerLast(nums.pollLast() + nums.pollLast());
            } else if (lastCh == '-') {
                int tmp = nums.pollLast();
                nums.offerLast(nums.pollLast() - tmp);
            }
        }
        return nums.pop();
    }
}
```

括号展开，只用符号入栈

```java
class Solution {
    public int calculate(String s) {
        // 将源表达式展开，即(1-(-2+3)) => [+1] + ( [+2] + [-3] )
        Deque<Integer> signs = new LinkedList<>(); // 正负栈 {-1, 1}
        signs.push(1);
        char[] chs = s.toCharArray();
        int n = s.length();
        int sign = 1;
        int ans = 0;
        for (int i = 0; i < n; ++i) {
            if (chs[i] == ' ') {
                continue;
            } else if (chs[i] == '(') {
                signs.push(sign);
            } else if (chs[i] == ')') {
                signs.pop();
            } else if (chs[i] == '+') {
                sign = signs.peek();
            } else if (chs[i] == '-') {
                sign = -signs.peek();
            } else { // 数字
                long num = 0;
                while (i < n && Character.isDigit(s.charAt(i))) {
                    num = num * 10 + (s.charAt(i) - '0');
                    ++i;
                }
                ans += sign * num;
                i--;
            }
        }
        return ans;
    }
}
```

## [227. 基本计算器 II](https://leetcode-cn.com/problems/basic-calculator-ii/)

> 栈，字符串

*/直接算，+-入栈

```java
class Solution {
    public int calculate(String s) {
        char[] chs = s.toCharArray();
        Deque<Integer> nums = new LinkedList<>();
        int n = s.length();
        char lastOp = ' ';
        int i = 0;
        while (i < n) {
            if (chs[i] == ' ') {
                ++i;
            } else {
                char ch = chs[i];
                i = Character.isDigit(ch) ? i : i + 1;
                // 找到下一个整数
                int nextNum = 0;
                while (i < n) {
                    if (Character.isDigit(chs[i]))
                        nextNum = nextNum * 10 + (s.charAt(i) - '0');
                    else if (chs[i] != ' ')
                        break;
                    i++;
                }
                if (Character.isDigit(ch)) {
                    nums.push(nextNum);
                } else { // 运算符
                    if (ch == '*') {
                        nums.push(nums.pop() * nextNum);
                    } else if (ch == '/') {
                        nums.push(nums.pop() / nextNum);
                    } else {
                        if (lastOp != ' ') {
                            int tmp = nums.pop();
                            if (lastOp == '+')
                                nums.push(nums.pop() + tmp);
                            else
                                nums.push(nums.pop() - tmp);
                        }
                        nums.push(nextNum);
                        lastOp = ch;
                    }
                }
            }
        }
        if (nums.size() > 1) {
            int tmp = nums.pop();
            return lastOp == '+' ? (tmp + nums.pop()) : (nums.pop() - tmp);
        }
        return nums.pop();
    }
}
```

更清晰的写法

```java
class Solution {
    public int calculate(String s) {
        Deque<Integer> nums = new LinkedList<>();
        s = s + '+'; // 便于结束
        char[] chs = s.toCharArray();
        char preSign = '+'; // 前面一个符号
        int num = 0; // 计算整数
        for (int i = 0; i < s.length(); ++i) {
            char ch = s.charAt(i);
            if (Character.isDigit(ch)) { // 数字
                num = num * 10 + (ch - '0');
            } else if (ch != ' '){ // 符号
                switch(preSign) {
                    case '+':
                        nums.push(num);
                        break;
                    case '-':
                        nums.push(-num);
                        break;
                    case '*':
                        nums.push(nums.pop() * num);
                        break;
                    default:
                        nums.push(nums.pop() / num);
                }
                preSign = ch;
                num = 0;
            }
        }
        int ans = 0;
        while (!nums.isEmpty()) {
            ans += nums.pop();
        }
        return ans;
    }
}
```

## [331. 验证二叉树的前序序列化](https://leetcode-cn.com/problems/verify-preorder-serialization-of-a-binary-tree/)

> 栈

用栈的核心原因是前序遍历的本质是递归，可以用栈模拟递归

栈：split+槽位

```java
class Solution {
    public boolean isValidSerialization(String preorder) {
        String[] nodes = preorder.split(",");
        Deque<Integer> stack = new LinkedList<>();
        stack.push(1);
        for (int i = 0; i < nodes.length; ++i) {
            if (stack.isEmpty())
                return false;
            int top = stack.pop() - 1; // 占一个槽位
            if (top > 0) {
                stack.push(top);
            }
            if (!nodes[i].equals("#")) { // 栈顶-1，push(2)
                stack.push(2);
            }
        }
        return stack.isEmpty();
    }
}
```

栈：直接char+槽位（快一些）

```java
class Solution {
    public boolean isValidSerialization(String preorder) {
        Deque<Integer> stack = new LinkedList<>();
        int n = preorder.length();
        stack.push(1);
        for (int i = 0; i < n; ++i) {
            if (stack.isEmpty())
                return false;
            int top = stack.pop() - 1; // 栈顶-1
            if (top > 0) {
                stack.push(top);
            }
            if (preorder.charAt(i) != '#') { // push(2)
                stack.push(2);
                while (i < n && preorder.charAt(i) != ',') {
                    i++;
                }
            } else { // 跳过逗号
                ++i;
            }
        }
        return stack.isEmpty();
    }
}
```

计数替代栈

```java
class Solution {
    public boolean isValidSerialization(String preorder) {
        int n = preorder.length();
        int i = 0;
        int slots = 1;
        while (i < n) {
            if (slots == 0) {
                return false;
            }
            if (preorder.charAt(i) == ',') {
                i++;
            } else if (preorder.charAt(i) == '#') {
                slots--;
                i++;
            } else {
                while (i < n && preorder.charAt(i) != ',') // 读一个数字
                    i++;
                slots++; // slots = slots - 1 + 2
            }
        }
        return slots == 0;
    }
}
```

TODO 入度出度计算、反向替代、正向替代（x##->#）

## [705. 设计哈希集合](https://leetcode-cn.com/problems/design-hashset/)

> 设计，哈希表

```java
class MyHashSet {
    private static final int CAP = 769; // 默认容量
    private LinkedList[] data;

    /** Initialize your data structure here. */
    public MyHashSet() {
        data = new LinkedList[CAP];
        for (int i = 0; i < CAP; ++i) {
            data[i] = new LinkedList<Integer>();           
        }
    }
    
    public void add(int key) {
        LinkedList<Integer> node = data[hash(key)];
        if (node.size() != 0) {
            if (node.indexOf(key) != -1)
                return;
        }
        node.push(key);
    }
    
    public void remove(int key) {
        LinkedList<Integer> node = data[hash(key)];
        if (node.size() != 0) {
            int index = node.indexOf(key);
            if (index != -1)
                node.remove(index);
        }
    }
    
    /** Returns true if this set contains the specified element */
    public boolean contains(int key) {
        LinkedList<Integer> node = data[hash(key)];
        if (node.size() != 0) {
            if (node.indexOf(key) != -1)
                return true;
        }
        return false;
    }

    private int hash(int key) {
        return key % CAP;
    }
}

/**
 * Your MyHashSet object will be instantiated and called as such:
 * MyHashSet obj = new MyHashSet();
 * obj.add(key);
 * obj.remove(key);
 * boolean param_3 = obj.contains(key);
 */
```

## [706. 设计哈希映射](https://leetcode-cn.com/problems/design-hashmap/)

> 设计，哈希表

```java
class MyHashMap {
    private static final int BASE = 769;
    private LinkedList[] data;

    /** Initialize your data structure here. */
    public MyHashMap() {
        data = new LinkedList[BASE];
        for (int i = 0; i < BASE; ++i) {
            data[i] = new LinkedList<Node>();
        }
    }
    
    /** value will always be non-negative. */
    public void put(int key, int value) {
        LinkedList<Node> nodes = data[hash(key)];
        for (Node node: nodes) {
            if (node.getKey() == key) {
                node.setValue(value);
                return;
            }
        }
        nodes.push(new Node(key, value));
    }
    
    /** Returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key */
    public int get(int key) {
        LinkedList<Node> nodes = data[hash(key)];
        for (Node node: nodes) {
            if (node.getKey() == key) {
                return node.getValue();
            }
        }
        return -1;
    }
    
    /** Removes the mapping of the specified value key if this map contains a mapping for the key */
    public void remove(int key) {
        LinkedList<Node> nodes = data[hash(key)];
        for (Node node: nodes) {
            if (node.getKey() == key) {
                nodes.remove(node);
                return;
            }
        }
    }

    private int hash(int key) {
        return key % BASE;
    }

    private class Node {
        private int key;
        private int value;

        Node(int key, int value) {
            this.key = key;
            this.value = value;
        }

        private int getKey() {
            return this.key;
        }

        private int getValue() {
            return this.value;
        }

        private void setKey(int key) {
            this.key = key;
        }

        private void setValue(int value) {
            this.value = value;
        }
    }
}

/**
 * Your MyHashMap object will be instantiated and called as such:
 * MyHashMap obj = new MyHashMap();
 * obj.put(key,value);
 * int param_2 = obj.get(key);
 * obj.remove(key);
 */
```

## [54. 螺旋矩阵](https://leetcode-cn.com/problems/spiral-matrix/)

> 数组

执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：36.8 MB, 在所有 Java 提交中击败了16.53%的用户

```java
class Solution {
    public List<Integer> spiralOrder(int[][] matrix) {
        int m = matrix.length;
        int n = matrix[0].length;
        int[][] direction = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
        int cur = 0;
        List<Integer> ans = new ArrayList<>();
        int i = 0, j = 0;
        boolean[][] visited = new boolean[m][n];
        while (true) {
            if (i < 0 || j < 0 || i == n || j == m)
                break;
            if (!visited[j][i]) {
                ans.add(matrix[j][i]);
                visited[j][i] = true;
                if ((cur == 2 && i == 0) ||
                    (cur == 3 && j == 0) || 
                    (cur == 0 && i == n - 1) ||
                    (cur == 1 && j == m - 1))
                    cur = (cur + 1) & 3;
                j += direction[cur][0];
                i += direction[cur][1];
            } else {
                j -= direction[cur][0];
                i -= direction[cur][1];
                cur = (cur + 1) & 3;
                j += direction[cur][0];
                i += direction[cur][1];
                if (visited[j][i]) {
                    break;
                }
            }
        }
        return ans;
    }
}
```

## [59. 螺旋矩阵 II](https://leetcode-cn.com/problems/spiral-matrix-ii/)

> 数组

执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：36.4 MB, 在所有 Java 提交中击败了87.57%的用户

```java
class Solution {
    public int[][] generateMatrix(int n) {
        int count = 1;
        int[][] directions = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
        int dir = 0;
        boolean[][] visited = new boolean[n][n];
        int[][] ans = new int[n][n];
        int i = 0, j = 0;
        while(true) {
            if (i < 0 || j < 0 || i >= n || j >= n)
                break;
            if (!visited[j][i]) {
                visited[j][i] = true;
                ans[j][i] = count;
                count++;
                if ((dir == 0 && i == n - 1) ||
                    (dir == 1 && j == n - 1) ||
                    (dir == 2 && i == 0) ||
                    (dir == 3 && j == 0)) {
                        dir = (dir + 1) & 3;
                    }
                j += directions[dir][0];
                i += directions[dir][1];
            } else {
                j -= directions[dir][0];
                i -= directions[dir][1];
                dir = (dir + 1) & 3;
                j += directions[dir][0];
                i += directions[dir][1];
                if (visited[j][i])
                    break;
            }
        }
        return ans;
    }
}
```

简洁写法

```java
class Solution {
    public int[][] generateMatrix(int n) {
        int max = n * n;
        int cur = 1;
        int[][] matrix = new int[n][n];
        int row = 0, col = 0;
        int[][] dirs = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
        int dirIdx = 0;
        while (cur <= max) {
            matrix[row][col] = cur;
            cur++;
            int nextRow = row + dirs[dirIdx][0], nextCol = col + dirs[dirIdx][1];
            if (nextRow < 0 || nextRow >= n ||
                nextCol < 0 || nextCol >= n || 
                matrix[nextRow][nextCol] != 0) {
                dirIdx = (dirIdx + 1) & 3;
            }
            row += dirs[dirIdx][0];
            col += dirs[dirIdx][1];
        }
        return matrix;
    }
}
```

## [115. 不同的子序列](https://leetcode-cn.com/problems/distinct-subsequences/)

> 字符串，动态规划

```java
class Solution {
    public int numDistinct(String s, String t) {
        // 定义：dp[i][j]表示[i~m][j~n]匹配的次数
        // 递推：dp[i][j] = dp[i+1][j]
        // 初始化：dp[i][n] = 
        // 求：dp[0][0]
        int m = s.length(), n = t.length();
        if (m < n) {
            return 0;
        }
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 0; i <= m; i++) {
            dp[i][n] = 1;
        }
        for (int i = m - 1; i >= 0; i--) {
            char sChar = s.charAt(i);
            for (int j = n - 1; j >= 0; j--) {
                char tChar = t.charAt(j);
                if (sChar == tChar) {
                    dp[i][j] = dp[i + 1][j + 1] + dp[i + 1][j];
                } else {
                    dp[i][j] = dp[i + 1][j];
                }
            }
        }
        return dp[0][0];
    }
}
```

## [92. 反转链表 II](https://leetcode-cn.com/problems/reverse-linked-list-ii/)

> 链表

执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：36 MB, 在所有 Java 提交中击败了69.72%的用户

头插法

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
    public ListNode reverseBetween(ListNode head, int left, int right) {
        if (left == right)
            return head;
        int count = 1;
        ListNode curNode = head;
        ListNode leftNode = new ListNode(0, head);
        while (count < right && curNode != null) {
            if (count >= left) { // 当前下一个节点提到leftNode的右边
                ListNode nextNode = curNode.next;
                if (nextNode != null) {
                    curNode.next = nextNode.next;
                    nextNode.next = leftNode.next;
                }
                leftNode.next = nextNode;
            } else {
                if (count == left - 1) { // 记录下左节点
                    leftNode = curNode;
                }
                curNode = curNode.next;
            }
            count++;
        }
        return left == 1 ? leftNode.next : head; // 注意left为1的特殊处理（dummyNode）
    }
}
```

简洁写法

```java
class Solution {
    public ListNode reverseBetween(ListNode head, int left, int right) {
        // 设置 dummyNode 是这一类问题的一般做法
        ListNode dummyNode = new ListNode(-1);
        dummyNode.next = head;
        ListNode pre = dummyNode;
        for (int i = 0; i < left - 1; i++) {
            pre = pre.next;
        }
        ListNode cur = pre.next;
        ListNode next;
        for (int i = 0; i < right - left; i++) {
            next = cur.next;
            cur.next = next.next;
            next.next = pre.next;
            pre.next = next;
        }
        return dummyNode.next;
    }
```

## [150. 逆波兰表达式求值](https://leetcode-cn.com/problems/evaluate-reverse-polish-notation/)

> 栈

```java
class Solution {
    public int evalRPN(String[] tokens) {
        Deque<Integer> nums = new LinkedList<>();
        for (String token: tokens) {
            if (token.equals("+")) {
                nums.push(nums.pop() + nums.pop());
            } else if (token.equals("-")) {
                Integer tmp = nums.pop();
                nums.push(nums.pop() - tmp);
            } else if (token.equals("*")) {
                nums.push(nums.pop() * nums.pop());
            } else if (token.equals("/")) {
                Integer tmp = nums.pop();
                nums.push(nums.pop() / tmp);
            } else {
                nums.push(Integer.valueOf(token));
            }
        }
        return Integer.valueOf(nums.pop());
    }
}
```

## [73. 矩阵置零](https://leetcode-cn.com/problems/set-matrix-zeroes/)

> 数组

用hashset存储O(m+n)

```java
class Solution {
    public void setZeroes(int[][] matrix) {
        Set<Integer> row = new HashSet<>();
        Set<Integer> col = new HashSet<>();
        int m = matrix.length;
        int n = matrix[0].length;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (matrix[i][j] == 0) {
                    if (!row.contains(i)) {
                        row.add(i);
                        for (int k = 0; k < n; ++k) {
                            matrix[i][k] = 0;
                        }
                    }
                    if (!col.contains(j)) {
                        col.add(j);
                        for (int k = 0; k < m; ++k) {
                            matrix[k][j] = 0;
                        }
                    }
                }
            }
        }
    }
}
```

用一个变量O(1)

```java
class Solution {
    public void setZeroes(int[][] matrix) {
        // 用1个变量存储第一行原本是否有0
        boolean isZero = false;
        int m = matrix.length;
        int n = matrix[0].length;
        for (int j = 0; j < n; ++j) {
            if (matrix[0][j] == 0)
                isZero = true;
        }
        for (int i = 1; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (j == 0 && matrix[i][0] == 0) {
                    matrix[0][0] = 0;
                }
                if (matrix[i][j] == 0) {
                    matrix[0][j] = 0;
                    matrix[i][0] = 0;
                }
            }
        }
        for (int i = 1; i < m; ++i) {
            for (int j = 1; j < n; ++j) {
                if (matrix[i][0] == 0 || matrix[0][j] == 0)
                    matrix[i][j] = 0;
            }
        }
        if (matrix[0][0] == 0) {
            for (int i = 1; i < m; ++i)
                matrix[i][0] = 0;
        }
        if (isZero) {
            for (int j = 0; j < n; ++j)
                matrix[0][j] = 0;
        }
    }
}
```

优化版

```java
class Solution {
    public void setZeroes(int[][] matrix) {
        int m = matrix.length, n = matrix[0].length;
        boolean flagCol0 = false;
        for (int i = 0; i < m; i++) {
            if (matrix[i][0] == 0) {
                flagCol0 = true;
            }
            for (int j = 1; j < n; j++) {
                if (matrix[i][j] == 0) {
                    matrix[i][0] = matrix[0][j] = 0;
                }
            }
        }
        for (int i = m - 1; i >= 0; i--) { // 保证第一行不先更新覆盖了，最后再处理第一行
            for (int j = 1; j < n; j++) {
                if (matrix[i][0] == 0 || matrix[0][j] == 0) {
                    matrix[i][j] = 0;
                }
            }
            if (flagCol0) {
                matrix[i][0] = 0;
            }
        }
    }
}
```

## [191. 位1的个数](https://leetcode-cn.com/problems/number-of-1-bits/)

> 位运算

直接用函数

执行用时：1 ms, 在所有 Java 提交中击败了95.76%的用户

内存消耗：35.3 MB, 在所有 Java 提交中击败了61.15%的用户

```java
public class Solution {
    // you need to treat n as an unsigned value
    public int hammingWeight(int n) {
        return Integer.bitCount(n);
    }
}
```

循环检查

```java
public class Solution {
    public int hammingWeight(int n) {
        int count = 0;
        for (int i = 0; i < 32; i++) {
            if ((n & (1 << i)) != 0) {
                count++;
            }
        }
        return count;
    }
}
```

位运算求1个数的技巧

```java

public class Solution {
    public int hammingWeight(int n) {
        int count = 0;
        while (n != 0) {
            n &= n - 1;
            count++;
        }
        return count;
    }
}
```

## [341. 扁平化嵌套列表迭代器](https://leetcode-cn.com/problems/flatten-nested-list-iterator/)

> 栈，dfs，设计

DFS

```java
/**
 * // This is the interface that allows for creating nested lists.
 * // You should not implement it, or speculate about its implementation
 * public interface NestedInteger {
 *
 *     // @return true if this NestedInteger holds a single integer, rather than a nested list.
 *     public boolean isInteger();
 *
 *     // @return the single integer that this NestedInteger holds, if it holds a single integer
 *     // Return null if this NestedInteger holds a nested list
 *     public Integer getInteger();
 *
 *     // @return the nested list that this NestedInteger holds, if it holds a nested list
 *     // Return null if this NestedInteger holds a single integer
 *     public List<NestedInteger> getList();
 * }
 */
public class NestedIterator implements Iterator<Integer> {

    private List<Integer> data = new LinkedList<>();
    private Iterator<Integer> dataIt;

    public NestedIterator(List<NestedInteger> nestedList) {
        dfs(nestedList);
        dataIt = this.data.iterator();
    }

    private void dfs(List<NestedInteger> list) {
        for (NestedInteger ni: list) {
            if (ni.isInteger()) {
                data.add(ni.getInteger());
            } else {
                dfs(ni.getList());
            }
        }
    }

    @Override
    public Integer next() {
        return dataIt.next();
    }

    @Override
    public boolean hasNext() {
        return dataIt.hasNext();
    }
}

/**
 * Your NestedIterator object will be instantiated and called as such:
 * NestedIterator i = new NestedIterator(nestedList);
 * while (i.hasNext()) v[f()] = i.next();
 */
```

栈 TODO

```java
/**
 * // This is the interface that allows for creating nested lists.
 * // You should not implement it, or speculate about its implementation
 * public interface NestedInteger {
 *
 *     // @return true if this NestedInteger holds a single integer, rather than a nested list.
 *     public boolean isInteger();
 *
 *     // @return the single integer that this NestedInteger holds, if it holds a single integer
 *     // Return null if this NestedInteger holds a nested list
 *     public Integer getInteger();
 *
 *     // @return the nested list that this NestedInteger holds, if it holds a nested list
 *     // Return null if this NestedInteger holds a single integer
 *     public List<NestedInteger> getList();
 * }
 */
public class NestedIterator implements Iterator<Integer> {
    // 存储列表的当前遍历位置
    private Deque<Iterator<NestedInteger>> stack;

    public NestedIterator(List<NestedInteger> nestedList) {
        stack = new LinkedList<Iterator<NestedInteger>>();
        stack.push(nestedList.iterator());
    }

    @Override
    public Integer next() {
        // 由于保证调用 next 之前会调用 hasNext，直接返回栈顶列表的当前元素
        return stack.peek().next().getInteger();
    }

    @Override
    public boolean hasNext() {
        while (!stack.isEmpty()) {
            Iterator<NestedInteger> it = stack.peek();
            if (!it.hasNext()) { // 遍历到当前列表末尾，出栈
                stack.pop();
                continue;
            }
            // 若取出的元素是整数，则通过创建一个额外的列表将其重新放入栈中
            NestedInteger nest = it.next();
            if (nest.isInteger()) {
                List<NestedInteger> list = new ArrayList<NestedInteger>();
                list.add(nest);
                stack.push(list.iterator());
                return true;
            }
            stack.push(nest.getList().iterator());
        }
        return false;
    }
}

/**
 * Your NestedIterator object will be instantiated and called as such:
 * NestedIterator i = new NestedIterator(nestedList);
 * while (i.hasNext()) v[f()] = i.next();
 */
```

## [456. 132 模式](https://leetcode-cn.com/problems/132-pattern/)

> 栈，单调栈

遍历1

```java
class Solution {
    public boolean find132pattern(int[] nums) {
        int n = nums.length;
        if (n < 3)
            return false;
        int candidateC = -1; // 候选次大下标
        Deque<Integer> stack = new LinkedList<>();
        for (int i = n - 1; i >= 0; --i) { // 从后往前维护单调递增栈，栈顶为候选最大值的下标
            while (!stack.isEmpty() && nums[i] > nums[stack.peek()]) {
                candidateC = stack.pop(); // 最后一个弹出的就是相对栈顶小的最大值（候选次大）下标
            }
            stack.push(i);
            if (candidateC != -1 && nums[candidateC] > nums[i]) // 如果当前遍历到的值比候选次大值下，则存在
                return true;
        }
        return false;
    }
}
```

遍历2、3 TODO

## [84. 柱状图中最大的矩形](https://leetcode-cn.com/problems/largest-rectangle-in-histogram/)

> 栈，单调栈，数组

```java
class Solution {
    public int largestRectangleArea(int[] heights) {
        int n = heights.length;
        int nn = n + 2;
        int[] newHeights = new int[nn];
        System.arraycopy(heights, 0, newHeights, 1, n);
        Deque<Integer> stack = new ArrayDeque<>();
        int area = 0;
        for (int i = 0; i < nn; ++i) {
            while (!stack.isEmpty() && newHeights[stack.peek()] > newHeights[i]) {
                int h = newHeights[stack.pop()];
                area = Math.max(area, (i - stack.peek() - 1) * h);
            }
            stack.push(i);
        }
        return area;
    }
}
```

## [82. 删除排序链表中的重复元素 II](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list-ii/)

> 链表

非递归

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
    public ListNode deleteDuplicates(ListNode head) {
        if (head == null)
            return null;
        ListNode dummyNode = new ListNode(0, head);
        ListNode curNode = dummyNode;
        while (curNode.next != null && curNode.next.next != null) {
            if (curNode.next.val == curNode.next.next.val) {
                // 删除后续所有重复的值
                int x = curNode.next.val;
                while (curNode.next != null && x == curNode.next.val) {
                    curNode.next = curNode.next.next; // 删除下一个
                }
            } else {
                curNode = curNode.next;
            }
        }
        return dummyNode.next;
    }
}
```

递归

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
    public ListNode deleteDuplicates(ListNode head) {
        if (head == null || head.next == null)
            return head;
        if (head.val != head.next.val) {
            head.next = deleteDuplicates(head.next);
            return head;
        } else {
            int x = head.val;
            while (head != null && head.val == x) {
                head = head.next;
            }
            return deleteDuplicates(head);
        }
    }
}
```

## [1773. 统计匹配检索规则的物品数量](https://leetcode-cn.com/problems/count-items-matching-a-rule/)

> 数组，字符串

```java
class Solution {
    public int countMatches(List<List<String>> items, String ruleKey, String ruleValue) {
        int ans = 0;
        int index = 0;
        switch(ruleKey) {
            case "color":
                index = 1;
                break;
            case "name":
                index = 2;
                break;
            default:
                index = 0;
        }
        for (List<String> item: items) {
            if (item.get(index).equals(ruleValue)) {
                ans++;
            }
        }
        return ans;
    }
}
```

## [61. 旋转链表](https://leetcode-cn.com/problems/rotate-list/)

> 链表，双指针

执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：37.9 MB, 在所有 Java 提交中击败了42.07%的用户

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
    public ListNode rotateRight(ListNode head, int k) {
        if (head == null)
            return null;
        ListNode p = head, tail = head;
        int n = 0;
        while (p != null) {
            if (n > 0)
                tail = tail.next;
            p = p.next;
            n++;
        }
        p = head;
        tail.next = head;
        n = n - k % n - 1;
        while (n > 0) {
            p = p.next;
            n--;
        }
        ListNode res = p.next;
        p.next = null;
        return res;
    }
}
```

## [173. 二叉搜索树迭代器](https://leetcode-cn.com/problems/binary-search-tree-iterator/)

> 栈，树，设计

用链表

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
class BSTIterator {

    List<Integer> nodes = new LinkedList<>();
    Iterator<Integer> it;

    public BSTIterator(TreeNode root) {
        nodes = init(root);
        it = nodes.iterator();
    }

    private List<Integer> init(TreeNode root) {
        if (root == null)
            return new LinkedList<>();
        else {
            List<Integer> tmp = new LinkedList<>();
            tmp.addAll(init(root.left));
            tmp.add(root.val);
            tmp.addAll(init(root.right));
            return tmp;
        }
    }
    
    public int next() {
        return it.next();
    }
    
    public boolean hasNext() {
        return it.hasNext();
    }
}

/**
 * Your BSTIterator object will be instantiated and called as such:
 * BSTIterator obj = new BSTIterator(root);
 * int param_1 = obj.next();
 * boolean param_2 = obj.hasNext();
 */
```

用数组

```java
class BSTIterator {
    private int idx;
    private List<Integer> arr;

    public BSTIterator(TreeNode root) {
        idx = 0;
        arr = new ArrayList<Integer>();
        inorderTraversal(root, arr);
    }

    public int next() {
        return arr.get(idx++);
    }

    public boolean hasNext() {
        return idx < arr.size();
    }

    private void inorderTraversal(TreeNode root, List<Integer> arr) {
        if (root == null) {
            return;
        }
        inorderTraversal(root.left, arr);
        arr.add(root.val);
        inorderTraversal(root.right, arr);
    }
}
```

栈（节省空间）

```java
class BSTIterator {
    Deque<TreeNode> stack;
    TreeNode cur;

    public BSTIterator(TreeNode root) {
        cur = root;
        stack = new LinkedList<>();
    }
    
    public int next() {
        while (cur != null) { // 找到最左
            stack.push(cur);
            cur = cur.left;
        }
        cur = stack.pop(); // 取出最左的值
        int ret = cur.val;
        cur = cur.right; // 更新到最左的右节点作为下一个
        return ret;
    }
    
    public boolean hasNext() {
        return !stack.isEmpty() || cur != null;
    }
}
```

## [190. 颠倒二进制位](https://leetcode-cn.com/problems/reverse-bits/)

> 位运算

执行用时：1 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：38.3 MB, 在所有 Java 提交中击败了21.25%的用户

用2的n次方和long

```java
public class Solution {
    // you need treat n as an unsigned value
    public int reverseBits(int n) {
        int i = 0;
        long res = 0;
        while (i < 32) {
            if ((n & 1) == 1) {
                res += (long)Math.pow(2, 31 - i);
            }
            n >>= 1;
            i++;
        }
        return (int)res;
    }
}
```

用逻辑右移

```java
public class Solution {
    public int reverseBits(int n) {
        int res = 0;
        for (int i = 0; i < 32 && n != 0; ++i) {
            res |= (n & 1) << (31 - i);
            n >>>= 1;
        }
        return res;
    }
}
```

空出一位

```java
public class Solution {
    public int reverseBits(int n) {
        int res = 0;
        for (int i = 0; i < 32; i++) {
            res <<= 1; //空出一位
            res += n & 1; //加上n最后一位
            n >>= 1;
        }
        return res;
    }
}
```

分治

```java
public class Solution {
    private static final int M1 = 0x55555555; // 01010101010101010101010101010101
    private static final int M2 = 0x33333333; // 00110011001100110011001100110011
    private static final int M4 = 0x0f0f0f0f; // 00001111000011110000111100001111
    private static final int M8 = 0x00ff00ff; // 00000000111111110000000011111111

    public int reverseBits(int n) {
        n = n >>> 1 & M1 | (n & M1) << 1; // 偶数位和奇数位交换位置
        n = n >>> 2 & M2 | (n & M2) << 2;
        n = n >>> 4 & M4 | (n & M4) << 4;
        n = n >>> 8 & M8 | (n & M8) << 8;
        return n >>> 16 | n << 16;
    }
}
```

## [74. 搜索二维矩阵](https://leetcode-cn.com/problems/search-a-2d-matrix/)

> 二维数组，二分查找，BS

一遍二分

```java
class Solution {
    public boolean searchMatrix(int[][] matrix, int target) {
        int m = matrix.length, n = matrix[0].length;
        int l = 0, r = m * n - 1;
        while (l <= r) {
            int mid = l + (r - l) / 2;
            int candidate = matrix[mid / n][mid % n];
            if (target < candidate) {
                r = mid - 1;
            } else if (target > candidate){
                l = mid + 1;
            } else {
                return true;
            }
        }
        return false;
    }
}
```

两遍二分

```java
class Solution {
    public boolean searchMatrix(int[][] matrix, int target) {
        int rowIndex = binarySearchFirstColumn(matrix, target);
        if (rowIndex < 0) {
            return false;
        }
        return binarySearchRow(matrix[rowIndex], target);
    }

    public int binarySearchFirstColumn(int[][] matrix, int target) {
        int low = -1, high = matrix.length - 1;
        while (low < high) {
            int mid = (high - low + 1) / 2 + low;
            if (matrix[mid][0] <= target) {
                low = mid;
            } else {
                high = mid - 1;
            }
        }
        return low;
    }

    public boolean binarySearchRow(int[] row, int target) {
        int low = 0, high = row.length - 1;
        while (low <= high) {
            int mid = (high - low) / 2 + low;
            if (row[mid] == target) {
                return true;
            } else if (row[mid] > target) {
                high = mid - 1;
            } else {
                low = mid + 1;
            }
        }
        return false;
    }
}
```

## [90. 子集 II](https://leetcode-cn.com/problems/subsets-ii/)

> 数组，回溯，DFS，二进制枚举

二进制枚举

```java
class Solution {
    public List<List<Integer>> subsetsWithDup(int[] nums) {
        // 先排序，如果前一个相同的数没有选择，则可以忽略掉
        List<List<Integer>> res = new ArrayList<>();
        int n = nums.length;
        Arrays.sort(nums);
        int len = 1 << n;
        for (int mask = 0; mask < len; ++mask) { // 用二进制遍历所有情况，mask有n位
            boolean flag = true;
            List<Integer> tmp = new ArrayList<>();
            for (int i = 0; i < n; ++i) {
                if ((mask & (1 << i)) != 0) { // 第i位为1
                    if (i > 0 && nums[i - 1] == nums[i] && (mask & (1 << (i - 1))) == 0) { // 第i-1位
                        flag = false;
                        break;
                    }
                    tmp.add(nums[i]);
                }
            }
            if (flag) {
                res.add(tmp);
            }
        }
        return res;
    }
}
```

DFS+回溯

```java
class Solution {
    List<List<Integer>> res = new ArrayList<>();
    List<Integer> tmp = new ArrayList<>();

    public List<List<Integer>> subsetsWithDup(int[] nums) {
        Arrays.sort(nums);
        dfs(false, 0, nums);
        return res;
    }

    private void dfs(boolean lastChoose, int cur, int[] nums) { // 前一个是否选择，当前选择，所有选择
        // 走到结束
        if (cur == nums.length) {
            res.add(new ArrayList<>(tmp)); // 注意新建对象
            return;
        }
        // 不选择
        dfs(false, cur + 1, nums);
        // 选择
        if (cur > 0 && !lastChoose && nums[cur - 1] == nums[cur]) { // 无论之后怎么选都是重复的
            return;
        }
        tmp.add(nums[cur]);
        dfs(true, cur + 1, nums);
        // 回溯
        tmp.remove(tmp.size() - 1);
    }
}
```

## [面试题 17.21. 直方图的水量](https://leetcode-cn.com/problems/volume-of-histogram-lcci/)

> 数组，栈，动态规划，双指针

双指针（可以背诵）

```java
class Solution {
    public int trap(int[] height) {
        int left = 0, right = height.length - 1;
        int leftMax = 0, rightMax = 0;
        int ans = 0;
        while (left < right) {
            leftMax = Math.max(leftMax, height[left]);
            rightMax = Math.max(rightMax, height[right]);
            if (height[left] < height[right]) {
                ans += leftMax - height[left];
                left++;
            } else {
                ans += rightMax - height[right];
                right--;
            }
        }
        return ans;
    }
}
```

动态规划（其实就是用两个数组记忆最大值）

```java
class Solution {
    public int trap(int[] height) {
        int n = height.length;
        if (n == 0)
            return 0;
        int[] leftMax = new int[n];
        leftMax[0] = height[0];
        for (int i = 1; i < n; ++i) {
            leftMax[i] = Math.max(height[i], leftMax[i - 1]);
        }
        int[] rightMax = new int[n];
        rightMax[n - 1] = height[n - 1];
        for (int i = n - 2; i >= 0; --i) {
            rightMax[i] = Math.max(height[i], rightMax[i + 1]);
        }
        int ans = 0;
        for (int i = 0; i < n; ++i) {
            ans += Math.min(leftMax[i], rightMax[i]) - height[i];
        }
        return ans;
    }
}
```

单调栈 TODO

```java
class Solution {
    public int trap(int[] height) {
        int ans = 0;
        Deque<Integer> stack = new LinkedList<Integer>();
        int n = height.length;
        for (int i = 0; i < n; ++i) {
            while (!stack.isEmpty() && height[i] > height[stack.peek()]) {
                int top = stack.pop();
                if (stack.isEmpty()) {
                    break;
                }
                int left = stack.peek();
                int currWidth = i - left - 1;
                int currHeight = Math.min(height[left], height[i]) - height[top];
                ans += currWidth * currHeight;
            }
            stack.push(i);
        }
        return ans;
    }
}
```

面向行求解

```java
class Solution {
    public int trap(int[] height) {
        int ans = 0;
        int n = height.length;
        int l = 0, r = height.length - 1;
        int blackSum = 0;
        for (int i = 0; i < n; ++i) {
            blackSum += height[i];
        }
        int level = 0;
        int allSum = 0;
        while (l <= r) {
            level++;
            while (l <= r && height[l] < level) l++;
            while (l <= r && height[r] < level) r--;
            allSum += r - l + 1;
        }
        return allSum - blackSum;
    }
}
```

## [1143. 最长公共子序列](https://leetcode-cn.com/problems/longest-common-subsequence/)

> 动态规划，LCS

建议背诵

```java
class Solution {
    public int longestCommonSubsequence(String text1, String text2) {
        // dp[i+1][j+1]: text1(0~i)和text2(0~j)的LCS
        // dp[i+1][j+1] = max(dp[i][j]+1(v[i]=v[j])), dp[i+1][j], dp[i][j+1])
        // 初始化：dp[i][0] = 0; dp[0][j] = 0;
        // 结果：dp[m][n]
        int m = text1.length(), n = text2.length();
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 0; i < m; i++) {
            char c1 = text1.charAt(i);
            for (int j = 0; j < n; j++) {
                char c2 = text2.charAt(j);
                if (c1 == c2) {
                    dp[i + 1][j + 1] = dp[i][j] + 1;
                } else {
                    dp[i + 1][j + 1] = Math.max(dp[i + 1][j], dp[i][j + 1]);
                }
            }
        }
        return dp[m][n];
    }
}
```

## [781. 森林中的兔子](https://leetcode-cn.com/problems/rabbits-in-forest/)

> 哈希表，数学

执行用时：1 ms, 在所有 Java 提交中击败了96.97%的用户

内存消耗：37.8 MB, 在所有 Java 提交中击败了60.98%的用户

```java
class Solution {
    // 开1000的数组统计不同报数者的数量，因为不同报数者必然不是一个颜色。
    // 然后遍历所有报数者大于0的报数：
    // 情况一：如果报数者数量多于报数：
    // 1）恰好可以分成几组，那所有兔子都可以在组内，那结果就是报数者数量；
    // 2）分组后还有剩余，排除分组后的余数满足情况二
    // 情况二：如果报数者数量少于等于报数：
    // 结果为(报数+1)只
    public int numRabbits(int[] answers) {
        int[] count = new int[1000];
        for (int i = 0; i < answers.length; ++i) {
            count[answers[i]]++;
        }
        int ans = 0;
        for (int i = 0; i < 1000; ++i) {
            if (count[i] > 0) {
                if (count[i] > i) { // 报数者数量多于报的数字
                    if (count[i] % (i + 1) == 0) // 正好可以分成count[i]/(i+1)组
                        ans += count[i];
                    else { // 分成count[i]/(i+1)组后还有剩余兔子
                        int tmp = i + 1;
                        ans += count[i] / tmp * tmp + tmp;
                    }
                } else { // 报数者数量少于等于报的数字
                    ans += i + 1;
                }
            }
        }
        return ans;
    }
}
```

一个公式 TODO

```java
class Solution {
    public int numRabbits(int[] answers) {
        int[] count = new int[1000];
        for (int i = 0; i < answers.length; ++i) {
            count[answers[i]]++;
        }
        int ans = 0;
        for (int i = 0; i < 1000; ++i) {
            if (count[i] > 0) {
                int tmp = i + 1;
                ans += (count[i] + i) / tmp * tmp;
            }
        }
        return ans;
    }
}
```

## [1450. 在既定时间做作业的学生人数](https://leetcode-cn.com/problems/number-of-students-doing-homework-at-a-given-time/)

> 数组

```java
class Solution {
    public int busyStudent(int[] startTime, int[] endTime, int queryTime) {
        int ans = 0;
        for (int i = 0; i < startTime.length; ++i) {
            if (startTime[i] <= queryTime && queryTime <= endTime[i]) {
                ans++;
            }
        }
        return ans;
    }
}
```

## [80. 删除有序数组中的重复项 II](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array-ii/)

> 数组，双指针

```java
class Solution {
    public int removeDuplicates(int[] nums) {
        int n = nums.length;
        int state = 1; // 1表示一个，2表示两个，3表示多个
        int last = nums[0];
        int len = 1;
        for (int i = 1; i < n; ++i) {
            if (last == nums[i]) {
                state++;
            } else {
                state = 1;
            }
            if (state <= 2) {
                nums[len++] = nums[i];
            }
            last = nums[i];
        }
        return len;
    }
}
```

快慢指针简单写法

```java
class Solution {
    public int removeDuplicates(int[] nums) {
        int n = nums.length;
        if (n <= 2) {
            return n;
        }
        int slow = 2, fast = 2;
        while (fast < n) {
            if (nums[slow - 2] != nums[fast]) {
                nums[slow] = nums[fast];
                ++slow;
            }
            ++fast;
        }
        return slow;
    }
}
```

## [81. 搜索旋转排序数组 II](https://leetcode-cn.com/problems/search-in-rotated-sorted-array-ii/)

> 数组，二分查找

执行用时：1 ms, 在所有 Java 提交中击败了88.78%的用户

内存消耗：38 MB, 在所有 Java 提交中击败了90.96%的用户

```java
class Solution {
    public boolean search(int[] nums, int target) {
        int n = nums.length;
        int binIndex = 0;
        for (int i = 1; i < n; ++i) {
            if (nums[i] < nums[i - 1]) {
                binIndex = i;
            }
        }
        if (binIndex == 0)
            return bs(nums, 0, n, target);
        else if (target > nums[0])
            return bs(nums, 0, binIndex, target);
        else if (target < nums[0])
            return bs(nums, binIndex, n, target);
        else
            return true;
    }

    private boolean bs(int[] nums, int l, int r, int target) { // [l,r)
        if (l == r) {
            if (l < nums.length)
                return nums[l] == target;
            else
                return false;
        }
        int mid = l + (r - l) / 2;
        if (nums[mid] < target) {
            return bs(nums, mid + 1, r, target);
        } else if (nums[mid] > target) {
            return bs(nums, l, mid, target);
        } else {
            return true;
        }
    }
}
```

直接暴力

执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：38.3 MB, 在所有 Java 提交中击败了28.49%的用户

```java
class Solution {
    public boolean search(int[] nums, int target) {
        for (int i = 0; i < nums.length; ++i) {
            if (nums[i] == target)
                return true;
        }
        return false;
    }
}
```

## [153. 寻找旋转排序数组中的最小值](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array/)

> 数组，二分查找

暴力方法

执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：38 MB, 在所有 Java 提交中击败了33.82%的用户

```java
class Solution {
    public int findMin(int[] nums) {
        for (int i = 1; i < nums.length; ++i) {
            if (nums[i] < nums[0])
                return nums[i];
        }
        return nums[0];
    }
}
```

二分 TODO

```java
class Solution {
    public int findMin(int[] nums) {
        int low = 0;
        int high = nums.length - 1;
        while (low < high) {
            int pivot = low + (high - low) / 2;
            if (nums[pivot] < nums[high]) {
                high = pivot;
            } else {
                low = pivot + 1;
            }
        }
        return nums[low];
    }
}
```

## [154. 寻找旋转排序数组中的最小值 II](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array-ii/)

> 数组，二分查找

直接暴力

```java
class Solution {
    public int findMin(int[] nums) {
        int min = nums[0];
        for (int i: nums) {
            if (i <= min)
                min = i;
        }
        return min;
    }
}
```

二分

```java
class Solution {
    public int findMin(int[] nums) {
        int l = 0, r = nums.length - 1;
        while (l < r) {
            int mid = l + (r - l) / 2;
            if (nums[mid] < nums[r]) {
                r = mid;
            } else if (nums[mid] > nums[r]) {
                l = mid + 1;
            } else {
                r--; // 忽略右边（关键）
            }
        }
        return nums[l];
    }
}
```

## [263. 丑数](https://leetcode-cn.com/problems/ugly-number/)

> 数学

执行用时：1 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：35.8 MB, 在所有 Java 提交中击败了5.44%的用户

```java
class Solution {
    public boolean isUgly(int n) {
        if (n <= 0)
            return false;
        while (n != 1) {
            if ((n & 1) == 0) {
                n >>= 1;
            } else if (n % 3 == 0) {
                n /= 3;
            } else if (n % 5 == 0){
                n /= 5;
            } else {
                return false;
            }
        }
        return true;
    }
}
```

官方写法（更简单）

```java
class Solution {
    public boolean isUgly(int n) {
        if (n <= 0)
            return false;
        int[] factors = {2, 3, 5};
        for (int factor: factors) {
            while (n % factor == 0)
                n /= factor;
        }
        return n == 1;
    }
}
```

## [264. 丑数 II](https://leetcode-cn.com/problems/ugly-number-ii/)

> 堆，数学，动态规划

最小堆 TODO

```java
class Solution {
    public int nthUglyNumber(int n) {
        // 每次把最小值的2x/3x/5x放入最小堆
        int[] factors = {2, 3, 5};
        Set<Long> unique = new HashSet<>();
        PriorityQueue<Long> pq = new PriorityQueue<>();
        unique.add(1L);
        pq.offer(1L);
        int ugly = 0;
        for (int i = 0; i < n; ++i) {
            long top = pq.poll();
            ugly = (int)top;
            for (int factor: factors) {
                long num = top * factor;
                if (unique.add(num)) // 小技巧
                    pq.offer(num);
            }
        }
        return ugly;
    }
}
```

动态规划

```java
class Solution {
    public int nthUglyNumber(int n) {
        // 谁乘了最小就把谁的指针前移一个数
        int[] dp = new int[n + 1];
        dp[1] = 1;
        int p2 = 1, p3 = 1, p5 = 1;
        for (int i = 2; i <= n; ++i) {
            int n2 = dp[p2] * 2, n3 = dp[p3] * 3, n5 = dp[p5] * 5;
            dp[i] = Math.min(Math.min(n2, n3), n5);
            if (dp[i] == n2) {
                p2++; 
            }
            if (dp[i] == n3) {
                p3++;
            }
            if (dp[i] == n5) {
                p5++;
            }
        }
        return dp[n];
    }
}
```

## [179. 最大数](https://leetcode-cn.com/problems/largest-number/)

> 排序

正反拼接字符串排序（注意删除前导0）

用PriorityQueue

```java
class Solution {
    public String largestNumber(int[] nums) {
        int n = nums.length;
        PriorityQueue<String> pq = new PriorityQueue<>((x, y) -> {
            if (x.compareTo(y) == 0)
                return 0;
            String str1 = x + y;
            String str2 = y + x;
            for (int i = 0; i < str1.length(); ++i) {
                if (str1.charAt(i) > str2.charAt(i)) {
                    return -1;
                } else if (str1.charAt(i) < str2.charAt(i)) {
                    return 1;
                }
            }
            return 0;
        });
        for (int num: nums) {
            pq.offer(num + "");
        }
        StringBuilder ans = new StringBuilder();
        while (!pq.isEmpty()) {
            ans.append(pq.poll());
        }
        while(ans.length() > 1 && ans.charAt(0) == '0') {
            ans.deleteCharAt(0);
        }
        return ans.toString();
    }
}
```

直接用排序

执行用时：6 ms, 在所有 Java 提交中击败了91.49%的用户

内存消耗：37.9 MB, 在所有 Java 提交中击败了69.86%的用户

```java
class Solution {
    public String largestNumber(int[] nums) {
        int n = nums.length;
        String[] numStrs = new String[n];
        for (int i = 0; i < n; ++i) {
            numStrs[i] = String.valueOf(nums[i]);
        }
        Arrays.sort(numStrs, (x, y) -> (y + x).compareTo(x + y));
        int len = numStrs.length;
        StringBuilder ans = new StringBuilder();
        int k = 0;
        while (k < len - 1 && numStrs[k].equals("0"))
            k++;
        for (int i = k; i < len; ++i) {
            ans.append(numStrs[i]);
        }
        return ans.toString();
    }
}
```

## [783. 二叉搜索树节点最小距离](https://leetcode-cn.com/problems/minimum-distance-between-bst-nodes/)

> 树，DFS，递归

执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：36 MB, 在所有 Java 提交中击败了57.79%的用户

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
    private static final int MAX = 100001;

    public int minDiffInBST(TreeNode root) {
        if (root == null) {
            return MAX;
        }
        int left = MAX, right = MAX, minLeft = MAX, minRight = MAX;
        if (root.left != null) {
            left = root.val - max(root.left);
            minLeft = minDiffInBST(root.left);
        }
        if (root.right != null) {
            right = min(root.right) - root.val;
            minRight = minDiffInBST(root.right);
        }
        return Math.min(Math.min(left, right), Math.min(minLeft, minRight));
    }

    private int max(TreeNode root) { // 左子树的最大值
        int ans = root.val;
        while (root.right != null) {
            ans = root.right.val;
            root = root.right;
        }
        return ans;
    }

    private int min(TreeNode root) { // 右子树的最小值
        int ans = root.val;
        while (root.left != null) {
            ans = root.left.val;
            root = root.left;
        }
        return ans;
    }
}
```

中序遍历 TODO

**二叉搜索树中序遍历得到的值序列是递增有序的**

执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：36.1 MB, 在所有 Java 提交中击败了47.38%的用户

```java
class Solution {
    int pre;
    int ans;

    public int minDiffInBST(TreeNode root) {
        ans = Integer.MAX_VALUE;
        pre = -1;
        dfs(root);
        return ans;
    }

    public void dfs(TreeNode root) {
        if (root == null) {
            return;
        }
        dfs(root.left);
        if (pre == -1) {
            pre = root.val;
        } else {
            ans = Math.min(ans, root.val - pre);
            pre = root.val;
        }
        dfs(root.right);
    }
}
```

## [208. 实现 Trie (前缀树)](https://leetcode-cn.com/problems/implement-trie-prefix-tree/)

> 设计，字典树，前缀树，Trie

执行用时：38 ms, 在所有 Java 提交中击败了96.74%的用户

内存消耗：49.6 MB, 在所有 Java 提交中击败了12.38%的用户

```java
class Trie {

    private TrieNode head;

    /** Initialize your data structure here. */
    public Trie() {
        head = new TrieNode();
    }
    
    /** Inserts a word into the trie. */
    public void insert(String word) {
        TrieNode p = head;
        for (char c: word.toCharArray()) {
            TrieNode[] ts = p.children;
            if (ts[c - 'a'] == null) {
                ts[c - 'a'] = new TrieNode();
            }
            p = ts[c - 'a'];
        }
        p.isTail = true;
    }
    
    /** Returns if the word is in the trie. */
    public boolean search(String word) {
        TrieNode p = head;
        for (char c: word.toCharArray()) {
            TrieNode[] ts = p.children;
            if (ts[c - 'a'] == null) {
                return false;
            }
            p = ts[c - 'a'];
        }
        return p.isTail;
    }
    
    /** Returns if there is any word in the trie that starts with the given prefix. */
    public boolean startsWith(String prefix) {
        TrieNode p = head;
        for (char c: prefix.toCharArray()) {
            TrieNode[] ts = p.children;
            if (ts[c - 'a'] == null) {
                return false;
            }
            p = ts[c - 'a'];
        }
        return true;
    }

    private class TrieNode {
        public TrieNode[] children;
        public boolean isTail = false; // 是否有该字符串结尾的单词

        TrieNode() {
            this.children = new TrieNode[27];
        }
    }
}

/**
 * Your Trie object will be instantiated and called as such:
 * Trie obj = new Trie();
 * obj.insert(word);
 * boolean param_2 = obj.search(word);
 * boolean param_3 = obj.startsWith(prefix);
 */
```

官方写法更简单

```java
class Trie {
    private Trie[] children;
    private boolean isEnd;

    public Trie() {
        children = new Trie[26];
        isEnd = false;
    }
    
    public void insert(String word) {
        Trie node = this;
        for (int i = 0; i < word.length(); i++) {
            char ch = word.charAt(i);
            int index = ch - 'a';
            if (node.children[index] == null) {
                node.children[index] = new Trie();
            }
            node = node.children[index];
        }
        node.isEnd = true;
    }
    
    public boolean search(String word) {
        Trie node = searchPrefix(word);
        return node != null && node.isEnd;
    }
    
    public boolean startsWith(String prefix) {
        return searchPrefix(prefix) != null;
    }

    private Trie searchPrefix(String prefix) {
        Trie node = this;
        for (int i = 0; i < prefix.length(); i++) {
            char ch = prefix.charAt(i);
            int index = ch - 'a';
            if (node.children[index] == null) {
                return null;
            }
            node = node.children[index];
        }
        return node;
    }
}
```

## [213. 打家劫舍 II](https://leetcode-cn.com/problems/house-robber-ii/)

> 动态规划

执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：36 MB, 在所有 Java 提交中击败了22.84%的用户

```java
class Solution {
    public int rob(int[] nums) {
        // dp[i]表示当前i位置盗窃的最高金额
        // dp[i] = max(dp[i-2]+nums[i], dp[i-1])
        // dp[i] = 0
        // dp[n-1]
        int n = nums.length;
        if (n == 1)
            return nums[0];
        if (n == 2)
            return Math.max(nums[0], nums[1]);
        if (n == 3)
            return Math.max(nums[0], Math.max(nums[1], nums[2]));
        return Math.max(getDpI(nums[0], 1, n - 1, nums), getDpI(0, 1, n, nums)); // 第一个取不取分开讨论
    }

    private int getDpI(int dp_i, int start, int end, int[] nums) {
        int dp_i_1 = 0;
        for (int i = start; i < end; ++i) {
            int dp_old = dp_i;
            dp_i = Math.max(dp_i_1 + nums[i], dp_i);
            dp_i_1 = dp_old;
        }
        return dp_i;
    }
}
```

## [220. 存在重复元素 III](https://leetcode-cn.com/problems/contains-duplicate-iii/)

> 排序，TreeSet，滑动窗口，桶

TreeSet

```java
class Solution {
    public boolean containsNearbyAlmostDuplicate(int[] nums, int k, int t) {
        TreeSet<Long> set = new TreeSet<>();
        for (int i = 0; i < nums.length; ++i) {
            long num = (long)nums[i];
            // 用ceiling函数查找大于或等于给定的元素的最小元素或null
            Long min = set.ceiling(num - (long)t);
            if (min != null && min <= (num + (long)t))
                return true;
            set.add(num);
            if (i >= k) // 维护窗口
                set.remove((long)nums[i - k]);
        }
        return false;
    }
}
```

桶

```java
class Solution {
    public boolean containsNearbyAlmostDuplicate(int[] nums, int k, int t) {
        // bucket .../-1[-t-1,0)/0[0, t+1)/1[t+1, 2t+2)/....
        // 滑动数组内，属于一个桶，必然满足条件，在不相邻桶，必然不满足条件，在相邻桶，需要比较后判断
        int n = nums.length;
        Map<Long, Long> bucket = new HashMap<>(); // Map<getId(v), v>
        long w = (long)(t + 1);
        for (int i = 0; i < n; ++i) {
            long num = (long)nums[i];
            long id = getId(num, w);
            if (bucket.containsKey(id))
                return true;
            if (bucket.containsKey(id - 1) && num - bucket.get(id - 1) < w)
                return true;
            if (bucket.containsKey(id + 1) && bucket.get(id + 1) - num < w)
                return true;
            bucket.put(id, num);
            if (i >= k)
                bucket.remove(getId((long)nums[i - k], w));
        }
        return false;
    }

    private long getId(long x, long w) {
        if (x >= 0)
            return x / w;
        return (x + 1) / w - 1;
    }
}
```

## [26. 删除有序数组中的重复项](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array/)

> 数组，双指针

执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：40.1 MB, 在所有 Java 提交中击败了81.61%的用户

```java
class Solution {
    public int removeDuplicates(int[] nums) {
        int n = nums.length;
        if (n <= 1)
            return n;
        int fast = 1, slow = 0;
        while (fast != n) {
            if (nums[fast] != nums[slow]) {
                slow++;
                nums[slow] = nums[fast];
            }
            fast++;
        }
        return slow + 1;
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

简单版

```java
int bs(int[] arr, int l, int r, int target) { // 找到最靠近target且>=target的index
    while (l < r) {
        int mid = (r - l) / 2 + l;
        if (arr[mid] < target) {
            l = mid + 1;
        } else  {
            r = mid;
        }
    }
    return l;
}
```

完整版

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

## 排序

<img src="https://i.loli.net/2021/02/28/iAOlWyNMd9tuwn3.png" alt="image-20210228093649556" style="zoom:50%;" />

### 快排

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

- 不用计算连通量

```java
private class UnionFind {
    private int[] parents;

    UnionFind(int n) {
        parents = new int[n];
        for (int i = 0; i < n; ++i) {
            parents[i] = i;
        }
    }

    public int find(int x) {
        return parents[x] == x ? x : parents[x] = find(parents[x]);
    }

    public void union(int a, int b) {
        parents[find(a)] = find(b);
    }
}
```

- 计算连通量

```java
private class UnionFind {
    private int[] parents;
    private int count;

    UnionFind(int n) {
        parents = new int[n];
        for (int i = 0; i < n; ++i) {
            parents[i] = i;
        }
        count = n;
    }

    public int find(int x) {
        return parents[x] == x ? x : parents[x] = find(parents[x]);
    }

    public void union(int a, int b) {
        int pa = find(a);
        int pb = find(b);
        if (pa != pb) {
            count--;
        }
        parents[pa] = pb;
    }
}
```

### Trie树

> 又称字典树、前缀树、单词查找树、键树，多叉哈希树
>
> Trie树典型应用是用于快速检索（最长前缀匹配），统计，排序和保存大量的字符串，所以经常被搜索引擎系统用于文本词频统计，搜索提示等场景。它的优点是最大限度地减少无谓的字符串比较，查询效率比较高。

```java
public void add(TrieNode root, String word) {
    TrieNode cur = root;
    for (int i = 0; i < word.length(); ++i) {
        char ch = word.charAt(i);
        if (cur.child[ch - 'a'] == null) {
            cur.child[ch - 'a'] = new TrieNode();
        }
        cur = cur.child[ch - 'a'];
    }
    ++cur.frequency;
}

public void find(TrieNode cur) {
    // 常用树的回溯或者DFS进行搜索
}

class TrieNode {
    int frequency; // 频数统计（可选）
    TrieNode[] child;

    public TrieNode() {
        frequency = 0;
        child = new TrieNode[26];
    }
}
```

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

## 滑动窗口

- 求最大窗口大小（for版本）

```java
int  maxLen = 0; // 窗口最大值
int sum = 0; // 当前状态
for(int l = 0, r = 0; r < arr.length; r++) {
    sum += arr[r]; // 根据加入右指针更新状态
    if (sum <= maxCost) { // 满足约束
        maxLen = Math.max(maxLen, r - l + 1); // 更新窗口最大值
    } else { // 不满足约束
        sum -= arr[l]; // 根据去除左指针更新状态
        l++; // 左指针右移
    }
}
return maxLen;
```

- 求最大窗口大小（双while版本）

模板

```java
while(r < n){
    UPDATE STATE(r)
    while(WRONG){
        UPDATE STATE(l)
        l++
    }
    MAXORMIN(ans)
    r++
}
```

```java
int l = 0, r = 0, sum = 0, maxLen = 0; // 初始化左右指针，状态值，窗口大小
while (r < n) { // 右指针不越界
    sum += arr[r]; // 根据加入右指针更新状态
    while(sum > maxCost) { // 不符合约束条件
        sum -= arr[l]; // 根据去除左指针更新状态
        l++; // 左指针右移
    }
    maxLen = Math.max(maxLen, r - l + 1); // 更新窗口最大值
    r++; // 右指针右移
}
return maxLen;
```

```java
int l = 0, r = 0, sum = 0, maxLen = 0; // 初始化左右指针，状态值，窗口大小
while (r < n) { // 右指针不越界
    sum += arr[r]; // 根据加入右指针更新状态
    r++; // 右指针右移
    while(sum > maxCost) { // 不符合约束条件
        sum -= arr[l]; // 根据去除左指针更新状态
        l++; // 左指针右移
    }
    maxLen = Math.max(maxLen, r - l); // 更新窗口最大值（注意区别）
}
return maxLen;
```

- 窗口不需要减小

```java
int l = 0, r = 0, sum = 0;
for(; r < n; r++) {
    if(sum + arr[r] > maxCost) { // 不满足约束，左右指针同时移动
        sum += arr[r] - arr[l]; // 更新状态
        l++;
    } else { // 满足约束，右指针移动
        sum += arr[r]; // 更新状态
    }
}
return r - l;
```

- 复杂样板

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

## 动态规划

思考模板

- 状态定义：一维还是二维？
- 转移方程：有几种状态？状态之间依赖关系？
- 初始条件
- 结果表示
- 是否可以采用优化：滚动数组/滚动变量

### 状态搜索

### 贪心

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

**Queue VS Deque**

| `Queue` Method                                               | Equivalent `Deque` Method                                    |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`add(e)`](https://docs.oracle.com/javase/10/docs/api/java/util/Deque.html#add(E)) | [`addLast(e)`](https://docs.oracle.com/javase/10/docs/api/java/util/Deque.html#addLast(E)) |
| [`offer(e)`](https://docs.oracle.com/javase/10/docs/api/java/util/Deque.html#offer(E)) | [`offerLast(e)`](https://docs.oracle.com/javase/10/docs/api/java/util/Deque.html#offerLast(E)) |
| [`remove()`](https://docs.oracle.com/javase/10/docs/api/java/util/Deque.html#remove()) | [`removeFirst()`](https://docs.oracle.com/javase/10/docs/api/java/util/Deque.html#removeFirst()) |
| [`poll()`](https://docs.oracle.com/javase/10/docs/api/java/util/Deque.html#poll()) | [`pollFirst()`](https://docs.oracle.com/javase/10/docs/api/java/util/Deque.html#pollFirst()) |
| [`element()`](https://docs.oracle.com/javase/10/docs/api/java/util/Deque.html#element()) | [`getFirst()`](https://docs.oracle.com/javase/10/docs/api/java/util/Deque.html#getFirst()) |
| [`peek()`](https://docs.oracle.com/javase/10/docs/api/java/util/Deque.html#peek()) | [`peekFirst()`](https://docs.oracle.com/javase/10/docs/api/java/util/Deque.html#peekFirst()) |

**Stack VS Deque**

| Stack Method                                                 | Equivalent `Deque` Method                                    |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`push(e)`](https://docs.oracle.com/javase/10/docs/api/java/util/Deque.html#push(E)) | [`addFirst(e)`](https://docs.oracle.com/javase/10/docs/api/java/util/Deque.html#addFirst(E)) |
| [`pop()`](https://docs.oracle.com/javase/10/docs/api/java/util/Deque.html#pop()) | [`removeFirst()`](https://docs.oracle.com/javase/10/docs/api/java/util/Deque.html#removeFirst()) |
| [`peek()`](https://docs.oracle.com/javase/10/docs/api/java/util/Deque.html#peek()) | [`peekFirst()`](https://docs.oracle.com/javase/10/docs/api/java/util/Deque.html#peekFirst()) |

## Queue 队列

- offer/add 添加
- poll/remove 删除
- peek/element 查询头部
- size/isEmpty 容量

## Deque 双端队列

![image-20210221013330334](https://i.loli.net/2021/02/21/kEis72yF9G3Q8HM.png)

- add/addAll/addFirst/addLast/offer/offerFirst/offerLast/push 添加
- poll/pollFirst/pollLast/pop/remove/removeFirst/removeLast 删除
- element/getFirst/getLast/peek/peekFirst/peekLast 获取查询
- contains 存在查询
- iterator/descendingIterator 迭代查询
- size 容量

ArrayDeque 双端队列数组实现（几乎没有容量限制，线程不安全，禁止null）

- 作为队列FIFO：add=addLast, offer=offerLast, remove=removeFirst, poll=pollFirst, element=getFirst, peek=peekFirst
- 作为堆栈FILO（替代Stack）：push=addFirst, pop=removeFirst, peek=peekFirst
- clear 删除
- toArray 转化
- clone 浅拷贝

LinkedList 双端队列链表实现

## ArrayList 动态数组

- add/addAll 添加
- remove/removeAll/removeRange/removeIf/clear 删除
- set/replaceAll 修改
- get 获取查询
- contains/containsAll/indexOf/lastIndexOf 存在查询
- forEach/iterator/listIterator/spliterator 迭代查询
- size/isEmpty/ensureCapacity/trimToSize 容量
- clone 浅拷贝
- subList/retainAll 截取
- sort 排序
- toArray/toString 转化

## HashMap 散列表

- put/putAll/putIfAbsent/merge/compute/computeIfAbsent/computeIfPresent 添加
- remove/clear 删除
- replace/replaceAll 修改
- get/getOrDefault 获取查询
- containsKey/containsValue 存在查询
- keySet/values/entrySet/forEach 迭代查询
- size/isEmpty 容量
- clone 浅拷贝

## HashSet 散列集合

- add/addAll 添加
- remove/removeAll/retainAll/clear 删除
- contains 存在查询
- iterator/spliterator 迭代查询
- size/isEmpty 容量
- clone 浅拷贝

## TreeMap 平衡树（红黑树）

- put/putAll 添加
- remove/clear 删除
- replace/replaceAll 修改
- entrySet/firstEntry/lastEntry/floorEntry/higherEntry/lowerEntry/ceilingEntry/descendingMap/pollFirstEntry/pollLastEntry 获取KV查询
- keySet/firstKey/lastKey/floorKey/higherKey/lowerKey/ceilingKey/descendingKeySet/navigableKeySet 获取K查询
- get/values 获取V查询
- containsKey/containsValue 存在查询
- forEach 迭代查询
- headMap/tailMap/subMap 区间查询
- comparator 获取排序
- size 容量
- clone 浅拷贝

## TreeSet 有序集合

- add/addAll 添加
- pollFirst/pollLast/remove/clear 删除
- first/last/floor/ceiling/higher/lower 获取查询
- contains 存在查询
- iterator/descendingIterator 获取迭代器
- descendingSet/headSet/subSet/tailSet 获取视图
- size/isEmpty 容量
- clone 浅拷贝

# Java数据转化

- `int[]` 转 `List<Integer>`

```java
List<Integer> list = Arrays.stream(intArr).boxed().collect(Collectors.toList());
List<Integer> list = IntStream.of(intArr).boxed().collect(Collectors.toList());
```

- `int[]` 转 `Integer[]`

```java
Integer[] integerArr = Arrays.stream(intArr).boxed().toArray(Integer[]::new);
```

- `List<Integer>` 转 `Integer[]`

```java
Integer[] integerArr = integerList.toArray(new Integer[0]);
```

- `List<Integer>` 转 `int[]`

```java
int[] intArr = integerList.stream().mapToInt(Integer::valueOf).toArray();
int[] intArr = integerList.stream().mapToInt(e->e).toArray();
```

- `Integer[]` 转 `int[]`

```java
int[] intArr = Arrays.stream(integerArr).mapToInt(Integer::valueOf).toArray();
```

- `Integer[]` 转 `List<Integer>`

```java
// 不支持add()/remove()
List<Integer> integerList = Arrays.asList(integerArr);
// 支持add()/remove()
List<Integer> integerList = new ArrayList<>();
boolean b = Collections.addAll(integerList, integersArr)
```

- `String[]` 转 `List<String>`

```java
List<String> stringList = Arrays.asList(stringArr);
```

- `List<String>`转`String[]`

```java
String[] stringArr = stringList.toArray(new String[0]);
```

- `String[]` 转 `String`

```java
String.join("", stringArr)
```

- `String`转`Integer`

```java
Integer i = Integer.valueOf(str);
Integer i = Integer.parseInt(str);
```

- `char[]`转`String`

```java
String str = new String(charArr)
```

- `int`转`String`

```java
String str = String.valueOf(i)
```

# Java常见技巧

+ 数组初始化：Arrays.fill(arr, Integer.MAX_VALUE)

+ 数组复制：Arrays.copyOfRange(nums, 0, k) 或者 System.arraycopy(srcArr, srcPos, destArr, destPos, length)

+ ArrayList简单构造：Arrays.asList(a,b)

+ 创建哑节点 `dummyHead`，令 `dummyHead.next = head`。引入哑节点是为了便于在 `head` 节点之前插入节点。

+ 计数哈希表构建：map.put(val, map.getOrDefault(val, 0) + 1);

+ 哈希表（复杂value）获取key并操作，eg：

	```java
	Map<Integer, PriorityQueue<Character>> map = new HashMap<>();
	map.computeIfAbsent(key, key -> new PriorityQueue<>()).offer(ch);
	```

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

- 大顶堆：new PriorityQueue<>(Collections.reverseOrder());或者new PriorityQueue<>((x,y)->y-x);

- 打印数组

	```java
	Object[] arr;
	for (int i = 0; i < arr.length; i++) { 
	    System.out.print(arr[i] + ", "); 
	}
	
	for(Object item: arr) 
	    System.out.println(item + ", ");
	
	System.out.println(Arrays.toString(arr));
	
	System.out.println(Arrays.asList(arr));
	
	Arrays.asList(arr).stream().forEach(item -> System.out.println(item));
	```

- 复杂Map遍历

	```java
	// 例如Map<Integer, int[]>
	for (Map.Entry<Integer, int[]> entry : map.entrySet()) {
	    int[] arr = entry.getValue();
	    //...
	}
	```

- 0和1翻转：n ^= 1

- int整数表示字符串字母集合（不考虑char出现的频数）：对每个ch计算mask |= (1 << (ch - 'a'))

- int的bit数

	```java
	public int countOnes(int x) {
	    // return Integer.bitCount(x);
	    int ones = 0;
	    while (x > 0) {
	        x &= (x - 1); // 最低位的1变成0
	        ones++;
	    }
	    return ones;
	}
	```

- [Integer常用函数和位运算技巧](https://blog.csdn.net/youyou1543724847/article/details/52385775)

- 检查回文串（经典DP子串遍历）

	```java
	// dp[i][j]表示[i,j]是否为回文串
	boolean[][] dp = new boolean[n][n];
	for (int i = 0; i < n; ++i) {
	    Arrays.fill(dp[i], true);
	}
	for (int i = n - 1; i >= 0; --i) { // 经典子串遍历
	    for (int j = i + 1; j < n; ++j) {
	        dp[i][j] = dp[i + 1][j - 1] && (s.charAt(i) == s.charAt(j));
	    }
	}
	```

- 判断字符是否是数字：Character.isDigit(ch)

# 未完成

## [493. 翻转对](https://leetcode-cn.com/problems/reverse-pairs/)

## [147. 对链表进行插入排序](https://leetcode-cn.com/problems/insertion-sort-list/)

## [135. 分发糖果](https://leetcode-cn.com/problems/candy/)

## 803. 打砖块

## [1489. 找到最小生成树里的关键边和伪关键边](https://leetcode-cn.com/problems/bricks-falling-when-hit/)

## [480. 滑动窗口中位数](https://leetcode-cn.com/problems/sliding-window-median/)【重点看看】

## [354. 俄罗斯套娃信封问题](https://leetcode-cn.com/problems/russian-doll-envelopes/)【重点看看】

## [1006. 笨阶乘](https://leetcode-cn.com/problems/clumsy-factorial/)

## [87. 扰乱字符串](https://leetcode-cn.com/problems/scramble-string/)


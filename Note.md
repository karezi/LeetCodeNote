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

暴力法

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

数学法

```java
class Solution {
    public int xorOperation(int n, int start) {
        // ans = (n&1)+2*(f(start/2-1)^f(start/2+n-1))
        // f(x)={x[4k],1[4k+1],x+1[4k+2],0[4k+3]}
        int s = start >> 1;
        return (n & start & 1) | (func(s - 1) ^ func(s + n - 1)) << 1;
    }

    private int func(int x) {
        int remainder = x & 3;
        switch(remainder) {
            case 0:
                return x;
            case 1:
                return 1;
            case 2:
                return x + 1;
            default:
                return 0;
        }
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
list+steam
```java
class Solution {
    public int calPoints(String[] ops) {
        int res = 0;
        List<Integer> scores = new ArrayList<>();
        for (String op: ops) {
            int size = scores.size();
            if (op.equals("+")) {
                scores.add(scores.get(size - 1) + scores.get(size - 2));
            } else if (op.equals("D")) {
                scores.add(2 * scores.get(size - 1));
            } else if (op.equals("C")) {
                scores.remove(scores.get(size - 1));
            } else {
                scores.add(Integer.parseInt(op));
            }
        }
        return scores.stream().reduce(Integer::sum).orElse(0);
    }
}
```
数组+switch
执行用时：1 ms, 在所有 Java 提交中击败了100.00%的用户
内存消耗：39.5 MB, 在所有 Java 提交中击败了34.12%的用户
```java
class Solution {
    public int calPoints(String[] ops) {
        int res = 0, idx = 0;
        int[] scores = new int[1001];
        for (String op: ops) {
            idx++;
            switch (op.charAt(0)) {
                case '+':
                    scores[idx] = scores[idx - 1] + scores[idx - 2];
                    res += scores[idx - 1] + scores[idx - 2];
                    break;
                case 'D':
                    scores[idx] = 2 * scores[idx - 1];
                    res += 2 * scores[idx - 1];
                    break;
                case 'C':
                    res -= scores[--idx];
                    idx--;
                    break;
                default:
                    scores[idx] = Integer.parseInt(op);
                    res += scores[idx];
            }
        }
        return res;
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

> 动态规划，回文，最长子序列，序列DP

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
        // 1    1(1)
        // 2    1(2) 2(3)
        // 4    1(4) 2(5) 2(6) 3(7)
        // 8    1(8) 2(9) 2(10) 3(11) 2(12) 3(13) 3(14) 4(15)
        // 16    1 2 2 3 2 3 3 4
        // 32    1 2 2 3 2 3 3 4 2 3 3 4 3 4 4 5
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
二分
```java
class Solution {
    public int findMin(int[] nums) {
        int n = nums.length;
        int l = 0, r = n - 1;
        while (l < r) {
            int mid = l + (r - l) / 2;
            if (nums[mid] < nums[r]) {
                r = mid;
            } else {
                l = mid + 1;
            }
        }
        return nums[l];
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

## [13. 罗马数字转整数](https://leetcode-cn.com/problems/roman-to-integer/)

> 数学，字符串

```java
class Solution {
    public int romanToInt(String s) {
        int ans = 0;
        int n = s.length();
        for (int i = 0; i < n; ++i) {
            char c = s.charAt(i);
            if (c == 'I') {
                if (i + 1 < n && s.charAt(i + 1) == 'V') {
                    ans += 4;
                    i++;
                } else if (i + 1 < n && s.charAt(i + 1) == 'X') {
                    ans += 9;
                    i++;
                } else {
                    ans += 1;
                }
            } else if (c == 'V') {
                ans += 5;
            } else if (c == 'X') {
                if (i + 1 < n && s.charAt(i + 1) == 'L') {
                    ans += 40;
                    i++;
                } else if (i + 1 < n && s.charAt(i + 1) == 'C') {
                    ans += 90;
                    i++;
                } else {
                    ans += 10;
                }
            } else if (c == 'L') {
                ans += 50;
            } else if (c == 'C') {
                if (i + 1 < n && s.charAt(i + 1) == 'D') {
                    ans += 400;
                    i++;
                } else if (i + 1 < n && s.charAt(i + 1) == 'M') {
                    ans += 900;
                    i++;
                } else {
                    ans += 100;
                }
            } else if (c == 'D') {
                ans += 500;
            } else if (c == 'M') {
                ans += 1000;
            }
        }
        return ans;
    }
}
```

## [91. 解码方法](https://leetcode-cn.com/problems/decode-ways/)

> 字符串，动态规划

执行用时：1 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：36.5 MB, 在所有 Java 提交中击败了92.22%的用户

```java
class Solution {
    public int numDecodings(String s) {
        // 求DP步骤数的经典题目，用递归也可以但是会超时，TODO 细节多建议默写
        // 表示：dp[i]表示s[0,i-1]的解码个数
        // 方程：dp[i]=dp[i-1] <s[i-1]是>或dp[i]=dp[i-2] <s[i-2]s[i-1]是> 这些都加一起
        // 初始化：dp[0]=1 表示第一个为空字符串，有一种解码方式
        // 结果：dp[n]
        int n = s.length();
        int[] dp = new int[n + 1];
        dp[0] = 1;
        for (int i = 1; i <= n; ++i) {
            if (s.charAt(i - 1) != '0') {
                dp[i] += dp[i - 1];
            }
            if (i > 1 && check(s.charAt(i - 2), s.charAt(i - 1))) {
                dp[i] += dp[i - 2];
            }
        }
        return dp[n];
    }

    private boolean check(char a, char b) {
        if (a == '0')
            return false;
        int num = (a - '0') * 10 + (b - '0');
        return num > 0 && num <= 26;
    }
}
```

## [363. 矩形区域不超过 K 的最大数值和](https://leetcode-cn.com/problems/max-sum-of-rectangle-no-larger-than-k/)

> 二分查找，动态规划，前缀和

前缀和

```java
class Solution {
    public int maxSumSubmatrix(int[][] matrix, int k) {
        int m = matrix.length, n = matrix[0].length;
        int[][] pref = new int[m + 1][n + 1];
        for (int i = 1; i <= m; ++i) {
            for (int j = 1; j <= n; ++j) {
                pref[i][j] = pref[i - 1][j] + pref[i][j - 1] - pref[i - 1][j - 1] + matrix[i - 1][j - 1];
            }
        }
        int max = Integer.MIN_VALUE;
        for (int x1 = 0; x1 < m; ++x1) {
            for (int y1 = 0; y1 < n; ++y1) {
                for (int x2 = x1 + 1; x2 <= m; ++x2) {
                    for (int y2 = y1 + 1; y2 <= n; ++y2) {
                        int area = getArea(pref, x1, y1, x2, y2);
                        if (area == k)
                            return k;
                        else if (area < k)
                            max = Math.max(max, area);
                    }
                }
            }
        }
        return max;
    }

    private int getArea(int[][] pref, int x1, int y1, int x2, int y2) {
        return pref[x2][y2] - pref[x1][y2] - pref[x2][y1] + pref[x1][y1];
    }
}
```

又用到set.ceiling->二分查找 TODO

```java
class Solution {
    public int maxSumSubmatrix(int[][] matrix, int k) {
        int ans = Integer.MIN_VALUE;
        int m = matrix.length, n = matrix[0].length;
        for (int i = 0; i < m; ++i) { // 枚举上边界
            int[] sum = new int[n];
            for (int j = i; j < m; ++j) { // 枚举下边界
                for (int c = 0; c < n; ++c) {
                    sum[c] += matrix[j][c]; // 更新每列的元素和(a0,a1,...,an-1)
                }
                TreeSet<Integer> sumSet = new TreeSet<Integer>();
                sumSet.add(0);
                int s = 0; // Sr
                for (int v : sum) {
                    s += v;
                    Integer ceil = sumSet.ceiling(s - k); // >=s-k的最小值
                    if (ceil != null) {
                        ans = Math.max(ans, s - ceil);
                    }
                    sumSet.add(s);
                }
            }
        }
        return ans;
    }
}
```

## [368. 最大整除子集](https://leetcode-cn.com/problems/largest-divisible-subset/)

```java
class Solution {
    public List<Integer> largestDivisibleSubset(int[] nums) {
        // 含义：dp[i]表示包含nums[i]且以nums[i]为最大值的集合长度
        // 转移方程：dp[i]=max(满足(j<i&&dp[i]%dp[j]==0)]的值+1)
        // 初始化：dp[i]=1
        // 结果：先找到max(dp[i])，然后逆序倒推出结果
        Arrays.sort(nums);
        int n = nums.length;
        int[] dp = new int[n];
        Arrays.fill(dp, 1);
        int maxIndex = 0;
        int maxDp = 1;
        for (int i = 1; i < nums.length; ++i) {
            for (int j = i - 1; j >= 0; --j) {
                if (nums[i] % nums[j] == 0)
                    dp[i] = Math.max(dp[i], dp[j] + 1);
            }
            if (dp[i] > maxDp) {
                maxDp = dp[i];
                maxIndex = i;
            }
        }
        List<Integer> ans = new ArrayList<>();
        int maxVal = nums[maxIndex];
        ans.add(maxVal);
        int cur = maxDp - 1;
        for (int i = maxIndex - 1; i >= 0; --i) {
            if (dp[i] == cur && (maxVal % nums[i] == 0)) {
                ans.add(nums[i]);
                maxVal = nums[i];
                cur--;
            }
            if (cur == 0)
                break;
        }
        return ans;
    }
}
```

## [377. 组合总和 Ⅳ](https://leetcode-cn.com/problems/combination-sum-iv/)

> 动态规划

```java
class Solution {
    public int combinationSum4(int[] nums, int target) {
        // dp[i]表示和为i的组合个数
        // dp[i]=sum(dp[i-num])以num作为结尾（TODO 很妙）
        // dp[0]=1
        // dp[target]
        int n = nums.length;
        int[] dp = new int[target + 1];
        dp[0] = 1;
        for (int i = 1; i <= target; ++i) {
            for (int num: nums) {
                if (i - num >= 0) {
                    dp[i] += dp[i - num];
                }
            }
        }
        return dp[target];
    }
}
```

## [897. 递增顺序搜索树](https://leetcode-cn.com/problems/increasing-order-search-tree/)

> 树，深度优先搜索，递归

中序遍历之后生成新的树

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
    private TreeNode cur;

    public TreeNode increasingBST(TreeNode root) {
        cur = new TreeNode(0);
        TreeNode res = cur;
        dfs(root);
        return res.right;
    }

    private void dfs(TreeNode root) {
        if (root == null)
            return;
        dfs(root.left);
        cur.right = new TreeNode(root.val);
        cur = cur.right;
        dfs(root.right);
    }
}
```

在中序遍历的过程中改变节点指向

```java
class Solution {
    private TreeNode cur;

    public TreeNode increasingBST(TreeNode root) {
        TreeNode dummyNode = new TreeNode(-1);
        cur = dummyNode;
        inorder(root);
        return dummyNode.right;
    }

    public void inorder(TreeNode node) {
        if (node == null) {
            return;
        }
        inorder(node.left);
        // 在中序遍历的过程中修改节点指向
        cur.right = node;
        node.left = null;
        cur = node;
        inorder(node.right);
    }
}
```

## [1011. 在 D 天内送达包裹的能力](https://leetcode-cn.com/problems/capacity-to-ship-packages-within-d-days/)

> 数组，二分查找

```java
class Solution {
    public int shipWithinDays(int[] weights, int D) {
        // 二分
        // 查找内容：最低运载能力min(Smax)
        // 左边界：数组最大值；右边界：数组和
        int total = weights[0];
        int maxVal = weights[0];
        int n = weights.length;
        for (int i = 1; i < n; ++i) {
            if (weights[i] > maxVal)
                maxVal = weights[i];
            total += weights[i];
        }
        int l = maxVal, r = total;
        while (l < r) {
            int mid = l + (r - l) / 2; // 每天能运送的能力
            if(getMinD(mid, weights, n) > D) { // 能力太小
                l = mid + 1;
            } else if (getMinD(mid, weights, n) <= D) { // 能力太大或者正好
                r = mid;
            }
        }
        return l;
    }
    private int getMinD(int ab, int[] weights, int n) {
        int count = 0;
        int index = 0;
        int curSum = 0;
        while (index < n) {
            curSum += weights[index];
            if (curSum > ab) {
                count++;
                curSum = 0;
            } else if (curSum == ab) {
                count++;
                curSum = 0;
                index++;
            } else {
                index++;
            }
        }
        return curSum == 0 ? count : (count + 1);
    }
}
```

官方简单写法

```java
class Solution {
    public int shipWithinDays(int[] weights, int D) {
        int l = Arrays.stream(weights).max().getAsInt();
        int r = Arrays.stream(weights).sum();
        while (l < r) {
            int mid = (l + r) >>> 1;
            int need = 1; // 需运送天数（初始就一定有一次）
            int cur = 0; // 当前这一天已经运送的包裹重量之和
            for (int weight : weights) {
                if (cur + weight > mid) {
                    ++need;
                    cur = 0;
                }
                cur += weight;
            }
            if (need <= D) {
                r = mid;
            } else {
                l = mid + 1;
            }
        }
        return l;
    }
}
```

## [938. 二叉搜索树的范围和](https://leetcode-cn.com/problems/range-sum-of-bst/)

> 树，深度优先搜索，广度优先搜索，递归，二叉搜索树

深度优先搜索（时间效率高）

```java
class Solution {
    public int rangeSumBST(TreeNode root, int low, int high) {
        if (root == null)
            return 0;
        if (root.val > high) {
            return rangeSumBST(root.left, low, high);
        } else if (root.val < low) {
            return rangeSumBST(root.right, low, high); 
        } else {
            return rangeSumBST(root.left, low, high) + root.val + rangeSumBST(root.right, low, high);
        }
    }
}
```

广度优先搜索（空间效率高）

```java
class Solution {
    public int rangeSumBST(TreeNode root, int low, int high) {
        int sum = 0;
        Queue<TreeNode> q = new LinkedList<TreeNode>();
        q.offer(root);
        while (!q.isEmpty()) {
            TreeNode node = q.poll();
            if (node == null) {
                continue;
            }
            if (node.val > high) {
                q.offer(node.left);
            } else if (node.val < low) {
                q.offer(node.right);
            } else {
                sum += node.val;
                q.offer(node.left);
                q.offer(node.right);
            }
        }
        return sum;
    }
}
```

## [633. 平方数之和](https://leetcode-cn.com/problems/sum-of-square-numbers/)

> 数学

sqrt

```java
class Solution {
    public boolean judgeSquareSum(int c) {
        for (long a = 0; a * a <= c; ++a) {
            double b = Math.sqrt(c - a * a);
            if (b == (int)b)
                return true;
        }
        return false;
    }
}
```

双指针

```java
class Solution {
    public boolean judgeSquareSum(int c) {
        long l = 0, r = (long)Math.sqrt(c);
        while (l <= r) {
            long sum = l * l + r * r;
            if (sum == c) {
                return true;
            } else if (sum < c) {
                l++;
            } else {
                r--;
            }
        }
        return false;
    }
}
```

费马平方和定理 TODO

> 一个非负整数c如果能够表示为两个整数的平方和，当且仅当c的所有形如4k+3的**质因子**的幂均为偶数。

```java
class Solution {
    public boolean judgeSquareSum(int c) {
        for (int i = 2, cnt = 0; i * i <= c; i++, cnt = 0) {
            while (c % i == 0 && ++cnt > 0)
                c /= i;
            if (i % 4 == 3 && cnt % 2 != 0)
                return false;
        }
        return c % 4 != 3;
    }
}
```

## [403. 青蛙过河](https://leetcode-cn.com/problems/frog-jump/)

> DFS，记忆化，递归，动态规划

记忆化+DFS

```java
class Solution {
    private Map<Integer, Integer> stoneMap;
    private Boolean[][] mem;

    public boolean canCross(int[] stones) {
        if (stones[0] != 0 || stones[1] != 1)
            return false;
        int n = stones.length;
        mem = new Boolean[n][n]; // 注意一定要用Boolean可以通过null判断是否记忆，数组大小也注意下
        stoneMap = new HashMap<>();
        for (int i = 0; i < n; ++i) {
            stoneMap.put(stones[i], i);
        }
        return dfs(stones, 1, 1);
    }

    private boolean dfs(int[] stones, int index, int lastK) {
        if (stones.length - 1 == index)
            return true;
        if (mem[index][lastK] != null)
            return mem[index][lastK];
        for (int i = -1; i <= 1; ++i) {
            int k = lastK + i;
            if (k > 0 && stoneMap.containsKey(stones[index] + k)) { // 用map或者二分搜索都可以
                if (dfs(stones, stoneMap.get(stones[index] + k), k))
                    return mem[index][lastK] = true;
            }
        }
        return mem[index][lastK] = false;
    }
}
```

动态规划 TODO

```java
class Solution {
    public boolean canCross(int[] stones) {
        int n = stones.length;
        boolean[][] dp = new boolean[n][n];
        dp[0][0] = true;
        for (int i = 1; i < n; ++i) {
            if (stones[i] - stones[i - 1] > i) {
                return false;
            }
        }
        for (int i = 1; i < n; ++i) {
            for (int j = i - 1; j >= 0; --j) {
                int k = stones[i] - stones[j];
                if (k > j + 1) {
                    break;
                }
                dp[i][k] = dp[j][k - 1] || dp[j][k] || dp[j][k + 1];
                if (i == n - 1 && dp[i][k]) {
                    return true;
                }
            }
        }
        return false;
    }
}
```

## [137. 只出现一次的数字 II](https://leetcode-cn.com/problems/single-number-ii/)

> 哈希表，位运算

哈希表

执行用时：6 ms, 在所有 Java 提交中击败了24.03%的用户

内存消耗：38 MB, 在所有 Java 提交中击败了89.67%的用户

```java
class Solution {
    public int singleNumber(int[] nums) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int num: nums) {
            map.put(num, map.getOrDefault(num, 0) + 1);
        }
        for (Integer num: map.keySet()) {
            if (map.get(num) == 1)
                return num;
        }
        return 0;
    }
}
```

位运算

```java
class Solution {
    public int singleNumber(int[] nums) {
        int ans = 0;
        // 32位逐个相加模3
        for (int i = 0; i <= 32; ++i) {
            int total = 0;
            for (int num: nums) {
                total += (num >> i) & 1;
            }
            if (total % 3 != 0) { // 非0即1
                ans |= (1 << i);
            }
        }
        return ans;
    }
}
```

数字电路设计（优化上一个方法）TODO

```java
class Solution {
    public int singleNumber(int[] nums) {
        int a = 0, b = 0;
        for (int num : nums) {
            b = ~a & (b ^ num);
            a = ~b & (a ^ num);
        }
        return b;
    }
}
```

## [690. 员工的重要性](https://leetcode-cn.com/problems/employee-importance/)

> 深度优先搜索，广度优先搜索，哈希表

执行用时：5 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：39.4 MB, 在所有 Java 提交中击败了98.83%的用户

```java
/*
// Definition for Employee.
class Employee {
    public int id;
    public int importance;
    public List<Integer> subordinates;
};
*/

class Solution {
    public int getImportance(List<Employee> employees, int id) {
        Map<Integer, Employee> map = new HashMap<>();
        for (Employee em: employees) {
            map.put(em.id, em);
        }
        return dfs(map, id);
    }

    private int dfs(Map<Integer, Employee> map, int id) {
        Employee e = map.get(id);
        int subImportance = 0;
        for (Integer emId: e.subordinates) {
            subImportance += dfs(map, emId);
        }
        return e.importance + subImportance;
    }
}
```

## [554. 砖墙](https://leetcode-cn.com/problems/brick-wall/)

> 哈希表

执行用时：14 ms, 在所有 Java 提交中击败了83.74%的用户

内存消耗：41.4 MB, 在所有 Java 提交中击败了79.55%的用户

```java
class Solution {
    public int leastBricks(List<List<Integer>> wall) {
        int n = wall.size();
        Map<Integer, Integer> freq = new HashMap<>();
        for (List<Integer> line: wall) {
            int sum = 0;
            for (int i = 0; i < line.size() - 1; ++i) {
                sum += line.get(i);
                freq.put(sum, freq.getOrDefault(sum, 0) + 1);
            }
        }
        int maxCount = 0;
        for (Integer i: freq.keySet()) {
            if (freq.get(i) > maxCount)
                maxCount = freq.get(i);
        }
        return n - maxCount;
    }
}
```

## [1844. 将所有数字用字符替换](https://leetcode-cn.com/problems/replace-all-digits-with-characters/)

> 字符串

```java
class Solution {
    public String replaceDigits(String s) {
        char[] ca = s.toCharArray();
        for (int i = 1; i < ca.length; i += 2) {
            ca[i] = (char)(ca[i - 1] + (ca[i] - '0'));
        }
        return String.valueOf(ca); // 比new String好
    }
}
```

## [740. 删除并获得点数](https://leetcode-cn.com/problems/delete-and-earn/)

> 动态规划

```java
class Solution {
    // 转化成和数组就是《打家劫舍（经典）》那道题
    public int deleteAndEarn(int[] nums) {
        // 分组求和
        int[] sum = new int[10001];
        for (int num: nums) {
            sum[num] += num;
        }
        return rob(sum);
    }

    private int rob(int[] arr) {
        // dp[i]表示点数和
        // dp[i]=max(dp[i-1], dp[i-2]+arr[i])  0old 1old(0) 1
        // dp[0]=arr[0],dp[1]=max(arr[0],arr[1])
        // dp[1]
        int dp_0 = arr[0], dp_1 = Math.max(arr[0], arr[1]);
        for (int i = 2; i < arr.length; ++i) {
            int tmp = dp_1;
            dp_1 = Math.max(dp_1, dp_0 + arr[i]);
            dp_0 = tmp;
        }
        return dp_1;
    }
}
```

可以拆分成子数组优化 TODO

## [1822. 数组元素积的符号](https://leetcode-cn.com/problems/sign-of-the-product-of-an-array/)

> 数学

```java
class Solution {
    public int arraySign(int[] nums) {
        boolean isNeg = false;
        for (int num: nums) {
            if (num == 0) {
                return 0;
            } else if (num < 0) {
                isNeg = !isNeg;
            }
        }
        return isNeg ? -1 : 1;
    }
}
```

## [1832. 判断句子是否为全字母句](https://leetcode-cn.com/problems/check-if-the-sentence-is-pangram/)

> 字符串

HashSet

```java
class Solution {
    public boolean checkIfPangram(String sentence) {
        Set<Character> s = new HashSet<Character>();
        for (char c = 'a'; c <= 'z'; ++c) {
            s.add(c);
        }
        for (char c: sentence.toCharArray()) {
            if (s.contains(c)) {
                s.remove(c);
                if (s.size() == 0)
                    return true;
            }
        }
        return false;
    }
}
```

字符数组

```java
class Solution {
    public boolean checkIfPangram(String sentence) {
        int[] freq = new int[26];
        for (char c: sentence.toCharArray()) {
            if (freq[c - 'a'] == 0)
                freq[c - 'a'] = 1;
        }
        for (int i = 0; i < 26; ++i) {
            if (freq[i] == 0)
                return false;
        }
        return true;
    }
}
```

位运算

```java
class Solution {
    public boolean checkIfPangram(String sentence) {
        int res = 0;
        for ( char c : sentence.toCharArray()) {
            res |= 1 << (c - 'a');
            if ((res ^ 0x3ffffff) == 0) { // (1 << 26) - 1 => 26个1
                return true;
            }
        }
        return false;
    }
}
```

## [1723. 完成所有工作的最短时间](https://leetcode-cn.com/problems/find-minimum-time-to-finish-all-jobs/)

> 递归，回溯，动态规划，状态压缩

dp+状压

```java
class Solution {
    public int minimumTimeRequired(int[] jobs, int k) {
        // dp[i][j]表示0~i-1个工人分配方案位j时的最大工作时间的最小值
        // dp[i][j]=min(max(dp[i-1][j*]<前i-1个人用j的某个子集>,jobSum(~j*)<第i个人用j的某个子集的补集>))
        // dp[0][j]=jobSum[j]
        // dp[k-1][(1<<n)-1]
        // 预先生成jobSum
        int n = jobs.length;
        int len = 1 << n;
        int[] jobSum = new int[len];
        for (int i = 1; i < len; ++i) { // 遍历子集并计算每一位
            int lastOne = Integer.numberOfTrailingZeros(i);
            int lastI = i - (1 << lastOne);
            jobSum[i] = jobSum[lastI] + jobs[lastOne];
        }
        // 定义
        int[][] dp = new int[k][len];
        // 初始化
        for (int j = 0; j < len; ++j) {
            dp[0][j] = jobSum[j];
        }
        // 递推
        for (int i = 1; i < k; ++i) {
            for (int j = 0; j < len; ++j) {
                int min = Integer.MAX_VALUE;
                for (int x = j; x != 0; x = (x - 1) & j) { // 状态子集遍历技巧
                    min = Math.min(min, Math.max(dp[i - 1][x], jobSum[j - x]));
                }
                dp[i][j] = min;
            }
        }
        return dp[k - 1][len - 1];
    }
}
```

二分查找 + 回溯 + 剪枝 TODO

## [1482. 制作 m 束花所需的最少天数](https://leetcode-cn.com/problems/minimum-number-of-days-to-make-m-bouquets/)

> 数组，二分查找，滑动数组

满足条件的最小或者满足条件的最大，这种题大概率二分

执行用时：17 ms, 在所有 Java 提交中击败了97.94%的用户

内存消耗：47.2 MB, 在所有 Java 提交中击败了64.12%的用户

```java
class Solution {
    public int minDays(int[] bloomDay, int m, int k) {
        // 二分
        int n = bloomDay.length;
        int l = bloomDay[0], r = bloomDay[0];
        for (int i = 0; i < n; ++i) {
            l = Math.min(l, bloomDay[i]);
            r = Math.max(r, bloomDay[i]);
        }
        while (l < r) {
            int mid = l + (r - l) / 2;
            if (check(mid, bloomDay, n, m, k)) {
                r = mid;
            } else {
                l = mid + 1;
            }
        }
        return check(l, bloomDay, n, m, k) ? l : -1;
    }

    private boolean check(int day, int[] bloomDay, int n, int m, int k) {
        int count = 0;
        int l = 0, r = k;
        while (r <= n) {
            while (l < r && bloomDay[l] <= day)
                l++;
            if (l == r) {
                count++;
                if (count >= m)
                    return true;
            } else {
                l++;
            }
            r = l + k;
        }
        return false;
    }
}
```

## [872. 叶子相似的树](https://leetcode-cn.com/problems/leaf-similar-trees/)

> 树，深度优先搜索

执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：35.8 MB, 在所有 Java 提交中击败了98.61%的用户

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
    public boolean leafSimilar(TreeNode root1, TreeNode root2) {
        List<Integer> list1 = new ArrayList<>();
        dfs(root1, list1);
        List<Integer> list2 = new ArrayList<>();
        dfs(root2, list2);
        return list1.equals(list2); // 直接比较
    }

    private void dfs(TreeNode root, List<Integer> list) {
        if (root == null) {
            return;
        }
        if (root.left == null && root.right == null) {
            list.add(root.val);
            return;
        }
        dfs(root.left, list);
        dfs(root.right, list);
    }
}
```

## [1734. 解码异或后的排列](https://leetcode-cn.com/problems/decode-xored-permutation/)

> 位运算

妙！

```java
class Solution {
    public int[] decode(int[] encoded) {
        // e[0]=p[0]^p[1]
        // e[1]=p[1]^p[2]
        // e[2]=p[2]^p[3]
        // e[3]=p[3]^p[4]
        // p[0]^p[1]^...^p[n-1]=1^2^3...^n
        // p[1]^p[2]^...^p[n-1]=e[1]^e[3]^...^e[n-2]
        int n = encoded.length + 1;
        int total = 0;
        for (int i = 1; i <= n; ++i) {
            total ^= i;
        }
        int odd = 0;
        for (int i = 1; i < n - 1; i += 2) {
            odd ^= encoded[i];
        }
        int first = total ^ odd;
        int[] ans = new int[n];
        ans[0] = first;
        for (int i = 1; i < n; ++i) {
            ans[i] = encoded[i - 1] ^ ans[i - 1];
        }
        return ans;
    }
}
```

## [1310. 子数组异或查询](https://leetcode-cn.com/problems/xor-queries-of-a-subarray/)

> 位运算，前缀

```java
class Solution {
    public int[] xorQueries(int[] arr, int[][] queries) {
        int n = arr.length;
        int[] preXor = new int[n + 1]; // 前缀异或
        preXor[0] = 0;
        for (int i = 0; i < n; ++i) {
            preXor[i + 1] = preXor[i] ^ arr[i];
        }
        int m = queries.length;
        int[] ans = new int[m];
        for (int i = 0; i < m; ++i) {
            ans[i] = preXor[queries[i][0]] ^ preXor[queries[i][1] + 1];
        }
        return ans;
    }
}
```

## [1269. 停在原地的方案数](https://leetcode-cn.com/problems/number-of-ways-to-stay-in-the-same-place-after-some-steps/)

> 动态规划，二维动态规划

```java
class Solution {
    public int numWays(int steps, int arrLen) {
        // dp[i][j]表示第i步走到j位置的时候的方案数，0<=i<=steps，0=<j<=min(arrLen-1, steps/2+1)
        // dp[i][j]=dp[i-1][j-1]+dp[i-1][j]+dp[i-1][j+1]
        // dp[0][0]=1,dp[0][j]=0
        // dp[steps][0]
        // i可以优化掉
        int mo = 1000000007;
        int len = Math.min(arrLen - 1, steps / 2 + 1) + 1; // 最远距离
        int[] dp = new int[len];
        dp[0] = 1;
        for (int i = 1; i <= steps; ++i) { // 执行steps步骤
            int[] dpn = new int[len]; // 下一状态
            for (int j = 0; j < len; ++j) {
                dpn[j] = dp[j];
                if (j - 1 >= 0) { // 考虑-1
                    dpn[j] = (dp[j - 1] + dpn[j]) % mo;
                }
                if (j + 1 < len) { // 考虑+1
                    dpn[j] = (dp[j + 1] + dpn[j]) % mo;
                }
            }
            dp = dpn; // 更新
        }
        return dp[0];
    }
}
```

## [12. 整数转罗马数字](https://leetcode-cn.com/problems/integer-to-roman/)

> 数学，字符串

```java
class Solution {
    public String intToRoman(int num) {
        StringBuilder sb = new StringBuilder();
        int[] n = {1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1};
        String[] strs = {"M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"};
        for (int i = 0; i < 13; ++i) {
            while (num >= n[i]) {
                sb.append(strs[i]);
                num -= n[i];
            }
            if (num == 0)
                break;
        }
        return sb.toString();
    }
}
```

## [1854. 人口最多的年份](https://leetcode-cn.com/problems/maximum-population-year/)

> 数组，差分数组

暴力

```java
class Solution {
    public int maximumPopulation(int[][] logs) {
        int maxCount = 0;
        int ans = -1;
        for (int i = 1950; i <= 2050; ++i) {
            int count = 0;
            for (int[] log: logs) {
                if (log[0] <= i && i < log[1])
                    count++;
            }
            if (count > maxCount) {
                maxCount = count;
                ans = i;
            }
        }
        return ans;
    }
}
```

差分数组

「差分」也是「前缀和」的逆运算

```java
class Solution {
    public int maximumPopulation(int[][] logs) {
        int offset = 1950;
        int[] delta = new int[101];
        for (int[] log: logs) {
            ++delta[log[0] - offset];
            --delta[log[1] - offset];
        }
        int preSum = 0;
        int maxIndex = 0;
        int maxDelta = 0;
        for (int i = 0; i < 101; ++i) {
            preSum += delta[i];
            if (preSum > maxDelta) {
                maxIndex = i;
                maxDelta = preSum;
            }
        }
        return maxIndex + offset;
    }
}
```

## [421. 数组中两个数的最大异或值](https://leetcode-cn.com/problems/maximum-xor-of-two-numbers-in-an-array/)

> 位运算，字典树

HashSet

```java
class Solution {
    public int findMaximumXOR(int[] nums) {
        // 从高位到低位求最大ans，即高位尽可能是1
        int ans = 0;
        for (int i = 30; i >= 0; --i) {
            Set<Integer> s = new HashSet<>();
            for (int num: nums) {
                s.add(num >> i);
            }
            ans = ans * 2 + 1;
            boolean success = false;
            for (int num: nums) {
                if (s.contains(ans ^ (num >> i))) {
                    success = true;
                    break;
                }
            }
            if (!success)
                ans -= 1;
        }
        return ans;
    }
}
```

Trie

```java
class Solution {
    private Trie t = new Trie();

    public int findMaximumXOR(int[] nums) {
        int ans = 0;
        for (int i = 1; i < nums.length; ++i) { // 两两比较的技巧
            add(nums[i - 1]);
            ans = Math.max(ans, getMax(nums[i]));
        }
        return ans;
    }

    private void add(int num) {
        Trie cur = t;
        for (int i = 30; i >= 0; --i) {
            int p = (num >> i) & 1; // 取出第i位
            if (p == 0) {
                if (cur.left == null) {
                    cur.left = new Trie();
                }
                cur = cur.left;
            } else {
                if (cur.right == null) {
                    cur.right = new Trie();
                }
                cur = cur.right;
            }
        }
    }

    private int getMax(int num) {
        // 通过num求可以得到的最大x
        int x = 0;
        Trie cur = t;
        for (int i = 30; i >= 0; --i) {
            int p = (num >> i) & 1;
            if (p == 0) {
                if (cur.right != null) {
                    x = x * 2 + 1;
                    cur = cur.right;
                } else {
                    x *= 2;
                    cur = cur.left;
                }
            } else {
                if (cur.left != null) {
                    x = x * 2 + 1;
                    cur = cur.left;
                } else {
                    x *= 2;
                    cur = cur.right;
                }
            }
        }
        return x;
    }

    class Trie {
        Trie left = null; // left!=null表示0
        Trie right = null; // right!=null表示1
    }
}
```

## [993. 二叉树的堂兄弟节点](https://leetcode-cn.com/problems/cousins-in-binary-tree/)

> 树，深度优先搜索，广度优先搜索

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
    private int[] f = new int[2];
    private int[] d = new int[2];

    public boolean isCousins(TreeNode root, int x, int y) {
        dfs(root, x, -1, 0, 0);
        dfs(root, y, -1, 0, 1);
        return f[0] != f[1] && d[0] == d[1];
    }

    private void dfs(TreeNode root, int a, int father, int depth, int offset) {
        if (root == null) {
            return;
        }
        if (root.val == a) {
            d[offset] = depth;
            f[offset] = father;
            return;
        }
        if (root.left != null) {
            dfs(root.left, a, root.val, depth + 1, offset);
        }
        if (root.right != null) {
            dfs(root.right, a, root.val, depth + 1, offset);
        }
    }
}
```

可以提前结束

## [1442. 形成两个异或相等数组的三元组数目](https://leetcode-cn.com/problems/count-triplets-that-can-form-two-arrays-of-equal-xor/)

执行用时：1 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：35.6 MB, 在所有 Java 提交中击败了100.00%的用户

```java
class Solution {
    public int countTriplets(int[] arr) {
        int ans = 0;
        int n = arr.length;
        for (int i = n - 2; i >= 0; i--) {
            int cur = arr[i];
            for (int j = i + 1; j < n; ++j) {
                cur ^= arr[j];
                if (cur == 0)
                    ans += j - i;
            }
        }
        return ans;
    }
}
```

Hash号称可以降低复杂度TODO

## [1738. 找出第 K 大的异或坐标值](https://leetcode-cn.com/problems/find-kth-largest-xor-coordinate-value/)

> 数组，优先队列，前缀和，排序，快速选择算法

优先队列

```java
class Solution {
    public int kthLargestValue(int[][] matrix, int k) {
        int m = matrix.length, n = matrix[0].length;
        int[][] cache = new int[m][n];
        PriorityQueue<Integer> pq = new PriorityQueue<>(k);
        cache[0][0] = matrix[0][0];
        putIn(pq, k, cache[0][0]);
        for (int i = 1; i < m; ++i) {
            cache[i][0] = cache[i - 1][0] ^ matrix[i][0];
            putIn(pq, k, cache[i][0]);
        }
        for (int i = 1; i < n; ++i) {
            cache[0][i] = cache[0][i - 1] ^ matrix[0][i];
            putIn(pq, k, cache[0][i]);
        }
        for (int a = 1; a < m; ++a) {
            for (int b = 1; b < n; ++b) {
                cache[a][b] = cache[a - 1][b] ^ cache[a][b - 1] ^ cache[a - 1][b - 1] ^ matrix[a][b];
                putIn(pq, k, cache[a][b]);
            }
        }
        return pq.poll();
    }

    private void putIn(PriorityQueue<Integer> pq, int k, int x) {
        if (pq.size() < k) {
            pq.offer(x);
        } else if (pq.peek() < x){
            pq.poll();
            pq.offer(x);
        }
    }
}
```

排序、快速选择算法TODO

## [692. 前K个高频单词](https://leetcode-cn.com/problems/top-k-frequent-words/)

> 堆，字典树，哈希表，优先队列

执行用时：7 ms, 在所有 Java 提交中击败了91.95%的用户

内存消耗：38.4 MB, 在所有 Java 提交中击败了91.91%的用户

```java
class Solution {
    public List<String> topKFrequent(String[] words, int k) {
        Map<String, Integer> map = new HashMap<>(); // 计数
        PriorityQueue<String> pq = new PriorityQueue<>(new Comparator<String>() {
            public int compare(String x, String y) {
                if (map.get(x) == map.get(y)) {
                    return y.compareTo(x);
                }
                return map.get(x) - map.get(y);
            }
        }); // 维护前k个
        for (String word: words) {
            map.put(word, map.getOrDefault(word, 0) + 1);
        }
        for (String key: map.keySet()) {
            if (pq.size() < k) {
                pq.offer(key);
            } else if (
                (map.get(key) > map.get(pq.peek())) 
                 || (map.get(key) == map.get(pq.peek()) && key.compareTo(pq.peek()) < 0)
            ) {
                pq.poll();
                pq.offer(key);
            }
        }
        List<String> res = new ArrayList<>(); // 倒序插入list的技巧
        for (int i = 0; i < k; ++i) {
            res.add(0, pq.poll());
        }
        return res;
    }
}
```

## [1035. 不相交的线](https://leetcode-cn.com/problems/uncrossed-lines/)

> 数组

```java
class Solution {
    public int maxUncrossedLines(int[] nums1, int[] nums2) {
        // dp[i][j]表示nums1[i]和nums[j]的最大值
        // dp[i][j] = max(dp[i-1][j-1]+1,dp[i][j-1],dp[i-1][j])
        // dp[0][0]=0
        // dp[n][m]
        int n = nums1.length;
        int m = nums2.length;
        int[][] dp = new int[n + 1][m + 1];
        for (int i = 1; i <= n; ++i) {
            for (int j = 1; j <= m; ++j) {
                if (nums1[i - 1] == nums2[j - 1])
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                else
                    dp[i][j] = Math.max(dp[i][j - 1], dp[i - 1][j]);
            }
        }
        return dp[n][m];
    }
}
```

## [810. 黑板异或游戏](https://leetcode-cn.com/problems/chalkboard-xor-game/)

> 数学，异或

执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：38.1 MB, 在所有 Java 提交中击败了65.42%的用户

```java
class Solution {
    public boolean xorGame(int[] nums) {
        // 在偶数情况下不可能存在删除任意数，异或都为0的情况（数学反证可以证明），而且两轮之后又是偶数，立于不败之地
        if (nums.length % 2 == 0) {
            return true;
        }
        int xor = 0;
        for (int num : nums) {
            xor ^= num;
        }
        return xor == 0;
    }
}
```

## [664. 奇怪的打印机](https://leetcode-cn.com/problems/strange-printer/)

> 动态规划，字符串，区间DP

```java
class Solution {
    public int strangePrinter(String s) {
        // dp[i][j]表示i到j的最少打印次数
        // dp[i][j]=(s[i]=s[j]=>dp[i][j-1] || s[i]<>s[j]=>min(dp[i][k]+dp[k+1][j]))
        // dp[i][i]=1
        // dp[0][n-1]
        int n = s.length();
        int[][] dp = new int[n][n];
        // 区间遍历常用方法（保证动态规划的计算过程满足无后效性）
        for (int i = n - 1; i >= 0; --i) {
            dp[i][i] = 1;
            for (int j = i + 1; j < n; ++j) {
                if (s.charAt(i) == s.charAt(j))
                    dp[i][j] = dp[i][j - 1];
                else {
                    int minVal = Integer.MAX_VALUE;
                    for (int k = i; k < j; ++k) {
                        minVal = Math.min(minVal, dp[i][k] + dp[k + 1][j]);
                    }
                    dp[i][j] = minVal;
                }
            }
        }
        return dp[0][n - 1];
    }
}
```

## [1787. 使所有区间的异或结果为零](https://leetcode-cn.com/problems/make-the-xor-of-all-segments-equal-to-zero/)

> 动态规划，异或

TODO

```java
class Solution {
    public int minChanges(int[] nums, int k) {
        // dp[i][xor]表示修改第i组后结果为xor，0<=i<k，0<=xor<=1024
        // dp[i][xor]=min(dp[i-1][xor^x]+cnt-count[i,x])
        // dp[0][xor]=min(cnt-count[i,x])
        // dp[k-1][0]
        // 第一维用滚动数组优化
        int n = nums.length, m = 1024;
        int[] dp = new int[m];
        int min = Integer.MAX_VALUE; // 当前组的最小值
        for (int i = 0; i < k; ++i) {
            int cnt = 0; // 每组的个数
            int[] counter = new int[m]; // 组内每个数的个数
            for (int l = i; l < n; l += k, ++cnt)
                ++counter[nums[l]];
            if (i == 0) { // 首组初始化
                for (int j = 0; j < m; ++j) {
                    dp[j] = cnt - counter[j];
                    min = Math.min(min, dp[j]);
                }
            } else {
                int[] curDp = new int[m];
                int curMin = Integer.MAX_VALUE;
                for (int j = 0; j < m; ++j) {
                    curDp[j] = min + cnt; // 整列替换
                    for (int l = i; l < n; l += k) { // 某个数替换
                        curDp[j] = Math.min(curDp[j], dp[j ^ nums[l]] + cnt - counter[nums[l]]);
                    }
                    curMin = Math.min(curMin, curDp[j]);
                }
                dp = curDp;
                min = curMin;
            }
        }
        return dp[0];
    }
}
```

## [1190. 反转每对括号间的子串](https://leetcode-cn.com/problems/reverse-substrings-between-each-pair-of-parentheses/)

> 栈

用两个栈

执行用时：5 ms, 在所有 Java 提交中击败了53.18%的用户

内存消耗：38.3 MB, 在所有 Java 提交中击败了39.36%的用户

TODO 注意是不是要Last

```java
class Solution {
    public String reverseParentheses(String s) {
        LinkedList<Character> stack = new LinkedList<>();
        LinkedList<Character> tmp = new LinkedList<>();
        int n = s.length();
        if (n == 0)
            return "";
        for (char c: s.toCharArray()) {
            if (c == ')') {
                while (stack.peek() != '(') {
                    tmp.offerFirst(stack.poll());
                }
                stack.poll();
                while (!tmp.isEmpty()) {
                    stack.offerFirst(tmp.pollLast());
                }
                continue;
            }
            stack.offerFirst(c);
        }
        StringBuilder sb = new StringBuilder();
        while (!stack.isEmpty()) {
            sb.append(stack.pollLast());
        }
        return sb.toString();
    }
}
```

用一个栈 TODO

```java
class Solution {
    public String reverseParentheses(String s) {
        Deque<String> stack = new LinkedList<String>();
        StringBuffer sb = new StringBuffer();
        for (int i = 0; i < s.length(); i++) {
            char ch = s.charAt(i);
            if (ch == '(') {
                stack.push(sb.toString());
                sb.setLength(0);
            } else if (ch == ')') {
                sb.reverse();
                sb.insert(0, stack.pop());
            } else {
                sb.append(ch);
            }
        }
        return sb.toString();
    }
}
```

预处理括号 TODO 找到遍历的规律

```java
class Solution {
    public String reverseParentheses(String s) {
        int n = s.length();
        int[] pair = new int[n];
        Deque<Integer> stack = new LinkedList<Integer>();
        for (int i = 0; i < n; i++) {
            if (s.charAt(i) == '(') {
                stack.push(i);
            } else if (s.charAt(i) == ')') {
                int j = stack.pop();
                pair[i] = j;
                pair[j] = i;
            }
        }

        StringBuffer sb = new StringBuffer();
        int index = 0, step = 1;
        while (index < n) {
            if (s.charAt(index) == '(' || s.charAt(index) == ')') {
                index = pair[index];
                step = -step;
            } else {
                sb.append(s.charAt(index));
            }
            index += step;
        }
        return sb.toString();
    }
}
```

## [461. 汉明距离](https://leetcode-cn.com/problems/hamming-distance/)

> 位运算

```java
class Solution {
    public int hammingDistance(int x, int y) {
        int res = 0;
        for (int i = 0; i < 31; ++i) {
            if ((x & 1) != (y & 1))
                res++;
            x >>= 1;
            y >>= 1;
        }
        return res;
    }
}
```

利用异或

```java
class Solution {
    public int hammingDistance(int x, int y) {
        return Integer.bitCount(x ^ y);
    }
}
```

lowbit算法（x&=x-1去除最后一个1）

```java
class Solution {
    public int hammingDistance(int x, int y) {
        int s = x ^ y;
        int ans = 0;
        while (s != 0) {
            s &= s - 1;
            ans++;
        }
        return ans;
    }
}
```

## [477. 汉明距离总和](https://leetcode-cn.com/problems/total-hamming-distance/)

> 位运算

暴力

```java
class Solution {
    public int totalHammingDistance(int[] nums) {
        int n = nums.length;
        int sum = 0;
        for (int i = n - 1; i >= 0; --i) {
            for (int j = i + 1; j < n; ++j) {
                sum += Integer.bitCount(nums[i] ^ nums[j]);
            }
        }
        return sum;
    }
}
```

按位计算

执行用时：5 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：39.5 MB, 在所有 Java 提交中击败了48.59%的用户

```java
class Solution {
    public int totalHammingDistance(int[] nums) {
        int sum = 0;
        int n = nums.length;
        for (int i = 0; i < 30; ++i) { // 10^9只需要到29
            int one = 0;
            for (int num: nums) {
                one += (num >> i) & 1;
            }
            sum += one * (n - one);
        }
        return sum;
    }
}
```

## [1074. 元素和为目标值的子矩阵数量](https://leetcode-cn.com/problems/number-of-submatrices-that-sum-to-target/)

> 数组，前缀和

暴力前缀和

```java
class Solution {
    public int numSubmatrixSumTarget(int[][] matrix, int target) {
        int ans = 0;
        int m = matrix.length, n = matrix[0].length;
        int[][] preSum = new int[m + 1][n + 1];
        for (int i = 1; i <= m; ++i) {
            for (int j = 1; j <= n; ++j) {
                preSum[i][j] = preSum[i - 1][j] + preSum[i][j - 1] - preSum[i - 1][j - 1] + matrix[i - 1][j - 1];
            }
        }
        for (int x1 = m; x1 >= 1; --x1) {
            for (int x2 = x1; x2 <= m; ++x2) {
                for (int y1 = n; y1 >= 1; --y1) {
                    for (int y2 = y1; y2 <= n; ++y2) {
                        if (preSum[x2][y2] - preSum[x2][y1 - 1] - preSum[x1 - 1][y2] + preSum[x1 - 1][y1 - 1] == target)
                            ans++;
                    }
                }
            }
        }
        return ans;
    }
}
```

前缀和+哈希表

```java
class Solution {
    public int numSubmatrixSumTarget(int[][] matrix, int target) {
        int ans = 0;
        int m = matrix.length, n = matrix[0].length;
        boolean flag = m > n;
        int[] len = flag ? new int[]{n, m} : new int[]{m, n}; // 复杂度优化
        for (int i = 0; i < len[0]; ++i) { // 上界
            int[] flatSum = new int[len[1]];
            for (int j = i; j < len[0]; ++j) { // 下界
                for (int k = 0; k < len[1]; ++k) {
                    if (flag)
                        flatSum[k] += matrix[j][k];
                    else
                        flatSum[k] += matrix[k][j];
                }
                ans += countTarget(flatSum, target);
            }
        }
        return ans;
    }

    private int countTarget(int[] flatSum, int target) {
        int count = 0, pre = 0;
        Map<Integer,Integer> map = new HashMap<>();
        map.put(0, 1);
        for (int flatVal: flatSum) {
            pre += flatVal;
            if (map.containsKey(pre - target)) {
                count += map.get(pre - target);
            }
            map.put(pre, map.getOrDefault(pre, 0) + 1);
        }
        return count;
    }
}
```

## [231. 2的幂](https://leetcode-cn.com/problems/power-of-two/)

> 位运算，数学

```java
class Solution {
    public boolean isPowerOfTwo(int n) {
        return n > 0 && (n & (n - 1)) == 0; // 最低位的1变成0后直接为0说明只有一位是1
        // return n > 0 && Integer.bitCount(n) == 1;
        // return n > 0 && (n & -n) == n;
        // return n > 0 && (1 << 30) % n == 0;
    }
}
```

## [342. 4的幂](https://leetcode-cn.com/problems/power-of-four/)

> 位运算

```java
class Solution {
    public boolean isPowerOfFour(int n) {
        // 执行用时：1 ms, 在所有 Java 提交中击败了100.00%的用户
        // 内存消耗：35.5 MB, 在所有 Java 提交中击败了54.88%的用户
        return n > 0 && (n & (n - 1)) == 0 && ((n & 0x55555555) > 0);
        // return n > 0 && (n & (n - 1)) == 0 && (n & 0xaaaaaaaa) == 0;
        // return n > 0 && (n & (n - 1)) == 0 && n % 3 == 1;
    }
}
```

## [1744. 你能在你最喜欢的那天吃到你最喜欢的糖果吗？](https://leetcode-cn.com/problems/can-you-eat-your-favorite-candy-on-your-favorite-day/)

> 数组，数学，前缀和

```java
class Solution {
    public boolean[] canEat(int[] candiesCount, int[][] queries) {
        int n = candiesCount.length, m = queries.length;
        long[] pre = new long[n]; // 注意int溢出
        pre[0] = candiesCount[0];
        for (int i = 1; i < n; ++i) {
            pre[i] = pre[i - 1] + candiesCount[i];
        }
        boolean[] res = new boolean[m];
        for (int i = 0; i < m; ++i) {
            // 以吃到第favoriteType为准的有效区间[minVal, maxVal]
            int ft = queries[i][0];
            long minVal = ft == 0 ? 1 : pre[ft - 1] + 1; // 从第0天开始
            long maxVal = pre[ft];
            // 以吃法计算上下界，最小一天一个，最多一天dailyCap个，区间[minCount, maxCount]
            long minCount = queries[i][1] + 1; // 天数
            long maxCount = (long)minCount * queries[i][2];
            // 判断两个区间是否有交集
            res[i] = !(maxCount < minVal || minCount > maxVal);
        }
        return res;
    }
}
```

## [523. 连续的子数组和](https://leetcode-cn.com/problems/continuous-subarray-sum/)

> 数学，同余，前缀和，哈希表

优化的暴力法（反而快）

```java
class Solution {
    public boolean checkSubarraySum(int[] nums, int k) {
        for (int i = 0; i < nums.length - 1; i++) { // 提前结束
            if (nums[i] == 0 && nums[i + 1] == 0) {
                return true;
            }
        }
        for (int i = 0; i < nums.length; i++) {
            int sum = nums[i];
            for (int j = i + 1; j < nums.length; j++) {
                sum += nums[j];
                if (sum % k == 0) {
                    return true;
                }
            }
            if (sum < k) { // 提前结束
                break;
            }
        }
        return false;
    }
}
```

前缀和+哈希表+同余，TODO 挺适合面试的，诸多技巧

```java
class Solution {
    public boolean checkSubarraySum(int[] nums, int k) {
        // a - b = c , 如果c % k = 0， 则 a % k = b % k
        int n = nums.length;
        if (n == 1)
            return false;
        int preReminder = 0;
        Map<Integer, Integer> map = new HashMap<>(); // <余数，第一次出现的位置>
        map.put(0, -1); // 精髓，使得可以取到0
        for (int i = 0; i < n; ++i) {
            preReminder = (preReminder + nums[i]) % k;
            if (!map.containsKey(preReminder)) {
                map.put(preReminder, i);
            } else {
                int preIndex = map.get(preReminder);
                if (i - preIndex > 1)
                    return true;
            }
        }
        return false;
    }
}
```

## [525. 连续数组](https://leetcode-cn.com/problems/contiguous-array/)

> 前缀和，哈希表

```java
class Solution {
    public int findMaxLength(int[] nums) {
        int n = nums.length;
        int ans = 0;
        int delta = 0; // count(0)-count(1)
        Map<Integer, Integer> map = new HashMap<>(); // <delta, 最早出现的位置>
        map.put(0, -1);
        for (int i = 0; i < n; ++i) {
            delta += nums[i] == 0 ? 1 : -1;
            if (map.containsKey(delta)) { // 说明之间的delta=0即count(0)=count(1)
                ans = Math.max(ans, i - map.get(delta));
            } else {
                map.put(delta, i);
            }
        }
        return ans;
    }
}
```

## [1869. 哪种连续子字符串更长](https://leetcode-cn.com/problems/longer-contiguous-segments-of-ones-than-zeros/)

> 数组，双指针

```java
class Solution {
    public boolean checkZeroOnes(String s) {
        int count0 = 0, count1 = 0, maxCount0 = 0, maxCount1 = 0;
        char[] ca = s.toCharArray();
        if (ca[0] == '0') {
            count0 = 1;
            maxCount0 = 1;
        } else {
            count1 = 1;
            maxCount1 = 1;
        }
        for (int i = 1; i < ca.length; ++i) {
            if (ca[i - 1] == '0' && ca[i] == '1') {
                maxCount0 = Math.max(maxCount0, count0);
                count0 = 0;
                count1 = 1;
            } else if (ca[i - 1] == '1' && ca[i] == '1') {
                count1++;
            } else if (ca[i - 1] == '1' && ca[i] == '0') {
                maxCount1 = Math.max(maxCount1, count1);
                count1 = 0;
                count0 = 1;
            } else if (ca[i - 1] == '0' && ca[i] == '0') {
                count0++;
            }
        }
        maxCount0 = Math.max(maxCount0, count0);
        maxCount1 = Math.max(maxCount1, count1);
        return maxCount1 > maxCount0;
    }
}
```

简化版

```java
public boolean checkZeroOnes(String s){
    int len1 = 0, len0 = 0;
    int max1 = 0, max0 = 0;
    for (char c: s.toCharArray()) {
        if (c == '0') {
            len0++;
            len1 = 0;
        } else {
            len1++;
            len0 = 0;
        }
        max1 = Math.max(len1, max1);
        max0 = Math.max(len0, max0);
    }
    return max1 > max0;
}
```

## [203. 移除链表元素](https://leetcode-cn.com/problems/remove-linked-list-elements/)

> 链表，虚节点

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
    public ListNode removeElements(ListNode head, int val) {
        ListNode dummy = new ListNode();
        dummy.next = head;
        ListNode slow = dummy;
        ListNode fast = head;
        while (fast != null) {
            if (fast.val == val) {
                slow.next = fast.next;
                fast.next = null; // help GC
                fast = slow.next;
            } else {
                fast = fast.next;
                slow = slow.next;
            }
        }
        return dummy.next;
    }
}
```

递归 TODO

```java
class Solution {
    public ListNode removeElements(ListNode head, int val) {
        if (head == null) {
            return head;
        }
        head.next = removeElements(head.next, val);
        return head.val == val ? head.next : head;
    }
}
```

优化后的迭代 TODO

```java
class Solution {
    public ListNode removeElements(ListNode head, int val) {
        ListNode dummyHead = new ListNode(0);
        dummyHead.next = head;
        ListNode temp = dummyHead;
        while (temp.next != null) {
            if (temp.next.val == val) {
                temp.next = temp.next.next;
            } else {
                temp = temp.next;
            }
        }
        return dummyHead.next;
    }
}
```

## [474. 一和零](https://leetcode-cn.com/problems/ones-and-zeroes/)

> 动态规划，滚动数组

```java
class Solution {
    public int findMaxForm(String[] strs, int m, int n) {
        // 01背包问题变种
        // dp[s][i][j]表示在s个、0的容量i、1的容量j下的最大子集大小
        // dp[s][i][j]=1、dp[s-1][i][j]（容量不够i<zero|j<one）,2、max(dp[s-1][i][j],dp[s-1][i-zero][j-one]+1)（容量够）
        // dp[0][i][j]=0
        // dp[len][m][n];
        // len可以用滚动数组
        int len = strs.length;
        int[][] dp = new int[m + 1][n + 1];
        for (int s = 0; s < len; ++s) {
            int c0 = countZero(strs[s]);
            int c1 = strs[s].length() - c0;
            for (int i = m; i >= c0; --i) { // TODO 逆向循环，技巧，不影响下个更新
                for (int j = n; j >= c1; --j) {
                    dp[i][j] = Math.max(dp[i][j], dp[i - c0][j - c1] + 1);
                }
            }
        }
        return dp[m][n];
    }

    private int countZero(String str) {
        int cnt = 0;
        for (char c: str.toCharArray()) {
            if (c == '0')
                cnt++;
        }
        return cnt;
    }
}
```

## [494. 目标和](https://leetcode-cn.com/problems/target-sum/)

> 动态规划，背包问题，回溯

二维动态规划

执行用时：28 ms, 在所有 Java 提交中击败了44.25%的用户

内存消耗：37.9 MB, 在所有 Java 提交中击败了34.53%的用户

```java
class Solution {
    public int findTargetSumWays(int[] nums, int target) {
        // dp[i][j] 表示第i个target为(j-1000)的方法数 0<=i<=n,-1000<=target<=100
        // dp[i][j]=dp[i-1][j-nums[i]] + dp[i-1][j+nums[i]]
        // dp[0][1000]=1
        // dp[n][target+1000]
        int n = nums.length;
        int[][] dp = new int[n + 1][2001]; // [0,2000]表示[-1000,1000]
        dp[0][1000] = 1;
        for (int i = 1; i <= n; ++i) {
            for (int j = 0; j <= 2000; ++j) {
                int idx1 = j - nums[i - 1];
                int idx2 = j + nums[i - 1];
                if (idx1 >= 0 && idx1 <= 2000)
                    dp[i][j] += dp[i - 1][idx1];
                if (idx2 >= 0 && idx2 <= 2000)
                    dp[i][j] += dp[i - 1][idx2];
            }
        }
        return dp[n][target + 1000];
    }
}
```

TODO 官方方法是先转化在设计dp的，且可以优化

回溯

执行用时：691 ms, 在所有 Java 提交中击败了12.98%的用户

内存消耗：35.7 MB, 在所有 Java 提交中击败了95.22%的用户

```java
class Solution {
    public int findTargetSumWays(int[] nums, int t) {
        return dfs(nums, t, 0, 0);
    }
    int dfs(int[] nums, int t, int u, int cur) {
        if (u == nums.length) {
            return cur == t ? 1 : 0;
        }
        int left = dfs(nums, t, u + 1, cur + nums[u]);
        int right = dfs(nums, t, u + 1, cur - nums[u]);
        return left + right;
    }
}
```

## [1049. 最后一块石头的重量 II](https://leetcode-cn.com/problems/last-stone-weight-ii/)

> 动态规划，01背包

```java
class Solution {
    public int lastStoneWeightII(int[] stones) {
        // 01背包问题
        // 容量：C<=[sum/2]
        // 物品质量W=价值V=stones[i]
        // dp[i + 1][j] 第i步还剩j能凑出来true，不能false
        // i \in [0,n]; j \in [0,C]
        // ①dp[i + 1][j] = dp[i][j]                       IF stones[i] > j【i个石头装不下=>不装i】
        // ②dp[i + 1][j] = dp[i][j] V dp[i][j-stones[i])  IF stones[i] <= j【i个石头装得下=>不装i V 装i】
        // dp[0][0]=true;dp[0][>0]=false
        // dp[n][x]=true的最小x
        int sum = 0;
        for (int stone: stones) {
            sum += stone;
        }
        int cap = sum / 2;
        int n = stones.length;
        boolean[][] dp = new boolean[n + 1][cap + 1];
        dp[0][0] = true;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j <= cap; ++j) {
                dp[i + 1][j] = stones[i] > j ? dp[i][j] : (dp[i][j] || dp[i][j - stones[i]]);
            }
        }
        for (int i = cap; i >= 0; --i) {
            if (dp[n][i])
                return sum - 2 * i;
        }
        return 0;
    }
}
```

滚动数组

```java
class Solution {
    public int lastStoneWeightII(int[] stones) {
        int sum = 0;
        for (int weight : stones) {
            sum += weight;
        }
        int m = sum / 2;
        boolean[] dp = new boolean[m + 1];
        dp[0] = true;
        for (int weight : stones) {
            for (int j = m; j >= weight; --j) { // 注意是倒序
                dp[j] = dp[j] || dp[j - weight];
            }
        }
        for (int j = m;; --j) {
            if (dp[j]) {
                return sum - 2 * j;
            }
        }
    }
}
```

## [518. 零钱兑换 II](https://leetcode-cn.com/problems/coin-change-2/)

> 动态规划，完全背包问题

```java
class Solution {
    public int change(int amount, int[] coins) {
        // 完全背包问题，套用公式：外item内cap{dp[i]+=dp[i-w]}
        int n = coins.length;
        int[] dp = new int[amount + 1];
        dp[0] = 1;
        for (int coin: coins) {
            for (int i = coin; i <= amount; ++i)
                dp[i] += dp[i - coin];
        }
        return dp[amount];
    }
}
```

## [279. 完全平方数](https://leetcode-cn.com/problems/perfect-squares/)

> 动态规划，完全背包，数学

完全背包

```java
class Solution {
    public int numSquares(int n) {
        List<Integer> list = new ArrayList<>();
        for (int i = 1; ; ++i) {
            int d = i * i;
            if (d > n)
                break;
            list.add(d);
        }
        // 完全背包问题，套用 dp[i]=min(dp[i],dp[i-item]+1)
        int[] dp = new int[n + 1];
        Arrays.fill(dp, Integer.MAX_VALUE);
        dp[0] = 0;
        for (Integer num: list) {
            for (int i = num; i <= n; ++i) {
                dp[i] = Math.min(dp[i], dp[i - num] + 1);
            }
        }
        return dp[n];
    }
}
```

动态规划

```java
class Solution {
    public int numSquares(int n) {
        // dp[i]表示i的完全平方数最小数量
        // dp[i] = min(dp[i-j*j])+1 (j \in [1,sqrt(i)])
        // dp[0]=0
        // dp[n]
        int[] dp = new int[n + 1];
        for (int i = 1; i <= n; ++i) {
            int minn = Integer.MAX_VALUE;
            for (int j = 1; j <= (int)Math.sqrt(i); ++j) {
                minn = Math.min(minn, dp[i - j * j]);
            }
            dp[i] = minn + 1;
        }
        return dp[n];
    }
}
```

数学

```java
class Solution {
    public int numSquares(int n) {
        // 四平方法(分别判断1 4 2 3)
        if (isSequare(n)) {
            return 1;
        } else if (isFour(n)) {
            // 4^k*(8m+7)
            return 4;
        } else if (isTwo(n)){
            return 2;
        } else {
            return 3;
        }
    }

    private boolean isSequare(int x) {
        int tmp = (int)Math.sqrt(x);
        return tmp * tmp == x;
    }

    private boolean isFour(int x) {
        while (x % 4 == 0) {
            x /= 4;
        }
        return x % 8 == 7;
    }

    private boolean isTwo(int x) {
        for (int i = 1; i < Math.sqrt(x); ++i) {
            if (isSequare(x - i * i)) {
                return true;
            }
        }
        return false;
    }
}
```

## [1449. 数位成本和为目标值的最大数字](https://leetcode-cn.com/problems/form-largest-integer-with-digits-that-add-up-to-target/)

> 字符串，动态规划，完全背包

```java
class Solution {
    public String largestNumber(int[] cost, int target) {
        // 完全背包公式:dp[x]=Math.max(dp[x], dp[x-k]+1)
        int[] dp = new int[target + 1];
        Arrays.fill(dp, Integer.MIN_VALUE);
        dp[0] = 1;
        for (int c: cost) {
            for (int i = c; i <= target; ++i) {
                dp[i] = Math.max(dp[i], dp[i - c] + 1);
            }
        }
        if (dp[target] < 0) {
            return "0";
        }
        StringBuilder sb = new StringBuilder();
        int i = 8, j = target;
        while (i >= 0) {
            int c = cost[i];
            while (j >= c && dp[j] == dp[j - c] + 1) { // TODO
                sb.append(i + 1);
                j -= c;
            }
            --i;
        }
        return sb.toString();
    }
}
```

## [278. 第一个错误的版本](https://leetcode-cn.com/problems/first-bad-version/)

> 二分查找

```java
/* The isBadVersion API is defined in the parent class VersionControl.
      boolean isBadVersion(int version); */

public class Solution extends VersionControl {
    public int firstBadVersion(int n) {
        int l = 1, r = n;
        while (l < r) {
            int mid = l + (r - l) / 2;
            if (isBadVersion(mid)) {
                r = mid;
            } else {
                l = mid + 1;
            }
        }
        return l;
    }
}
```

## [374. 猜数字大小](https://leetcode-cn.com/problems/guess-number-higher-or-lower/)

> 二分查找

执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：35.1 MB, 在所有 Java 提交中击败了73.70%的用户

```java
/** 
 * Forward declaration of guess API.
 * @param  num   your guess
 * @return          -1 if num is lower than the guess number
 *                  1 if num is higher than the guess number
 *               otherwise return 0
 * int guess(int num);
 */

public class Solution extends GuessGame {
    private int l = 1, r = Integer.MAX_VALUE;

    public int guessNumber(int n) {
        int res = guess(n);
        if (res == 0)
            return n;
        else if (res == -1) {
            r = n;
            return guessNumber(l + (r - l) / 2);
        } else {
            l = n;
            return guessNumber(l + (r - l) / 2);
        }
    }
}
```

更符合题意的解法

```java
public class Solution extends GuessGame {
    public int guessNumber(int n) {
        int left = 1, right = n;
        while (left < right) { // 循环直至区间左右端点相同
            int mid = left + (right - left) / 2; // 防止计算时溢出
            if (guess(mid) <= 0) {
                right = mid; // 答案在区间 [left, mid] 中
            } else {
                left = mid + 1; // 答案在区间 [mid+1, right] 中
            }
        }
        // 此时有 left == right，区间缩为一个点，即为答案
        return left;
    }
}
```

## [852. 山脉数组的峰顶索引](https://leetcode-cn.com/problems/peak-index-in-a-mountain-array/)

## [剑指 Offer II 069. 山峰数组的顶部](https://leetcode-cn.com/problems/B1IidL/)

> 二分查找

```java
class Solution {
    public int peakIndexInMountainArray(int[] arr) {
        int l = 0, r = arr.length - 1;
        while (l < r) {
            int mid = l + (r - l) / 2;
            if (arr[mid - 1] < arr[mid] && arr[mid] > arr[mid + 1]) {
                return mid;
            } else if (arr[mid - 1] > arr[mid]) {
                r = mid;
            } else if (arr[mid] < arr[mid + 1]) {
                l = mid;
            }
        }
        return -1;
    }
}
```

## [877. 石子游戏](https://leetcode-cn.com/problems/stone-game/)

> 动态规划

```java
class Solution {
    public boolean stoneGame(int[] piles) {
        // dp[i][j]表示剩余从i到j时候两者石子之差最大值(i<=j)
        // dp[i][j]=max(piles[i] - dp[i + 1][j], piles[j] - dp[i][j - 1]) // 我这一轮比你多的减去你之后比我多的
        // dp[i][i]=piles[i] 只有一堆石子，必然差piles[i]
        // dp[0][n-1]
        int n = piles.length;
        int[][] dp = new int[n][n];
        for (int i = n - 1; i >= 0; --i) {
            dp[i][i] = piles[i];
            for (int j = i + 1; j < n; ++j) {
                dp[i][j] = Math.max(piles[i] - dp[i + 1][j], piles[j] - dp[i][j - 1]);
            }
        }
        return dp[0][n - 1] > 0;
    }
}
```

滚动数组优化

```java
class Solution {
    public boolean stoneGame(int[] piles) {
        // dp[i][j]表示剩余从i到j时候两者石子之差最大值(i<=j)
        // dp[i][j]=max(piles[i] - dp[i + 1][j], piles[j] - dp[i][j - 1]) // 我这一轮比你多的减去你之后比我多的
        // dp[i][i]=piles[i] 只有一堆石子，必然差piles[i]
        // dp[0][n-1]
        // 滚动数组优化
        int n = piles.length;
        int[] dp = new int[n];
        for (int i = n - 1; i >= 0; --i) {
            dp[i] = piles[i];
            for (int j = i + 1; j < n; ++j) {
                dp[j] = Math.max(piles[i] - dp[j], piles[j] - dp[j - 1]);
            }
        }
        return dp[n - 1] > 0;
    }
}
```

数学分析

先手必胜（奇偶位置分组，选择和大的一组一直取即可）

```java
class Solution {
    public boolean stoneGame(int[] piles) {
        return true;
    }
}
```

## [65. 有效数字](https://leetcode-cn.com/problems/valid-number/)

> 数学，字符串，有限状态机

注意split的转义和结尾是分隔符的情况，以及a==""陷阱

```java
class Solution {
    public boolean isNumber(String s) {
        String str = s.toLowerCase();
        String[] strs = str.split("e");
        if (strs.length == 2) {
            if (str.endsWith("e"))
                return false;
            boolean isEndInteger = checkIntegerOrDecimal(strs[1], true);
            if (!isEndInteger)
                return false;
            return checkIntegerOrDecimal(strs[0], false);
        } else if (strs.length == 1) {
            return checkIntegerOrDecimal(str, false);
        }
        return false;
    }

    private boolean checkIntegerOrDecimal(String s, boolean forceInteger) {
        if (s.equals(""))
            return false;
        if (s.startsWith("+") || s.startsWith("-")) {
            return forceInteger ? checkAllInteger(s.substring(1, s.length())) : checkIntegerOrDecimalPure(s.substring(1, s.length()));
        } else {
            return forceInteger ? checkAllInteger(s) : checkIntegerOrDecimalPure(s);
        }
    }

    private boolean checkIntegerOrDecimalPure(String s) {
        if (s.equals(""))
            return false;
        boolean isAllInteger = checkAllInteger(s);
        if (!isAllInteger) {
            return checkDecimal(s);
        } else {
            return true;
        }
    }

    private boolean checkDecimal(String s) {
        if (s.equals(""))
            return false;
        if (s.startsWith(".")) {
            return checkAllInteger(s.substring(1, s.length()));
        } else if (s.endsWith(".")) {
            return checkAllInteger(s.substring(0, s.length() - 1));
        } else {
            String[] strs = s.split("\\.");
            return strs.length == 2 && checkAllInteger(strs[0]) && checkAllInteger(strs[1]);
        }
    }

    private boolean checkAllInteger(String s) {
        if (s.equals(""))
            return false;
        for (char c: s.toCharArray()) {
            if (c > '9' || c < '0')
                return false;
        }
        return true;
    }
}
```

有限状态机 TODO

```java
class Solution {
    public int make(char c) {
        switch(c) {
            case ' ': return 0;
            case '+':
            case '-': return 1;
            case '.': return 3;
            case 'e': return 4;
            default:
                if(c >= 48 && c <= 57) return 2;
        }
        return -1;
    }

    public boolean isNumber(String s) {
        int state = 0;
        int finals = 0b101101000;
        int[][] transfer = new int[][]{{ 0, 1, 6, 2,-1},
                                       {-1,-1, 6, 2,-1},
                                       {-1,-1, 3,-1,-1},
                                       { 8,-1, 3,-1, 4},
                                       {-1, 7, 5,-1,-1},
                                       { 8,-1, 5,-1,-1},
                                       { 8,-1, 6, 3, 4},
                                       {-1,-1, 5,-1,-1},
                                       { 8,-1,-1,-1,-1}};
        char[] ss = s.toCharArray();
        for(int i = 0; i < ss.length; ++i) {
            int id = make(ss[i]);
            if (id < 0)
                return false;
            state = transfer[state][id];
            if (state < 0)
                return false;
        }
        return (finals & (1 << state)) > 0;
    }
}
```

## [483. 最小好进制](https://leetcode-cn.com/problems/smallest-good-base/)

> 数学，二分查找

TODO 数学推导，巧妙

```java
class Solution {
    public String smallestGoodBase(String n) {
        long num = Long.parseLong(n); // len下界=2，即11，此时：k=num-1【保底】
        int max = (int)(Math.log(num + 1) / Math.log(2)); // len上界=log_2(n+1)=代入最大值约等于60，此时：k=2
        for (int len = max; len >= 3; len--) {
            long k = (long)Math.pow(num, 1.0 / (len - 1)); // k < s√n < k+1
            long res = 0;
            for (int i = 0; i < len; i++)
                res = res * k + 1;
            if (res == num) // 验证k是否为整数
                return String.valueOf(k);
        }
        return String.valueOf(num - 1);
    }
}
```

TODO 可以用二分进行优化（实际不一定优化了，因为范围不大）

## [1239. 串联字符串的最大长度](https://leetcode-cn.com/problems/maximum-length-of-a-concatenated-string-with-unique-characters/)

> 位运算，回溯算法

```java
class Solution {
    private int max = 0;
    private List<Integer> bit;

    public int maxLength(List<String> arr) {
        bit = new ArrayList<>();
        bit.add(0);
        for (String str: arr) {
            int ans = 0;
            for (char c: str.toCharArray()) {
                int mask = 1 << (c - 'a');
                if ((mask & ans) > 0) {
                    ans = 0;
                    break;
                } else {
                    ans += mask;
                }
            }
            bit.add(ans);
        }
        backtrack(0, 0);
        return max;
    }

    private void backtrack(int cur, int pos) {
        if (pos == bit.size()) {
            max = Math.max(max, Integer.bitCount(cur));
            return;
        }
        if ((cur & bit.get(pos)) == 0) { // 可以选择，选择pos
            backtrack(cur | bit.get(pos), pos + 1);
        }
        // 不选pos
        backtrack(cur, pos + 1);
    }
}
```

## [1600. 皇位继承顺序](https://leetcode-cn.com/problems/throne-inheritance/)

> 多叉树，前序遍历，哈希表

哈希表

```java
class ThroneInheritance {
    private String kingName;
    private Map<String, List<String>> map;
    private Set<String> dead;

    public ThroneInheritance(String kingName) {
        this.kingName = kingName;
        map = new HashMap<>();
        dead = new HashSet<>();
    }

    public void birth(String parentName, String childName) {
        List<String> children = map.getOrDefault(parentName, new ArrayList<>());
        children.add(childName);
        map.put(parentName, children);
    }

    public void death(String name) {
        dead.add(name);
    }

    public List<String> getInheritanceOrder() {
        List<String> ans = new ArrayList<>();
        preorder(ans, kingName);
        return ans;
    }

    private void preorder(List<String> ans, String curName) {
        if (!dead.contains(curName)) {
            ans.add(curName);
        }
        List<String> children = map.getOrDefault(curName, new ArrayList<>());
        for (String child: children) {
            preorder(ans, child);
        }
    }
}

/**
 * Your ThroneInheritance object will be instantiated and called as such:
 * ThroneInheritance obj = new ThroneInheritance(kingName);
 * obj.birth(parentName,childName);
 * obj.death(name);
 * List<String> param_3 = obj.getInheritanceOrder();
 */
```

链表 TODO

```java
class ThroneInheritance {
    class Node {
        String name;
        Node next;
        Node last; // 记录最后一个儿子
        boolean isDeleted = false;
        Node (String _name) {
            name = _name;
        }
    }
    Map<String, Node> map = new HashMap<>();
    Node head = new Node(""), tail = new Node("");

    public ThroneInheritance(String name) {
        Node root = new Node(name);
        root.next = tail;
        head.next = root;
        map.put(name, root);
    }

    public void birth(String pname, String cname) {
        Node node = new Node(cname);
        map.put(cname, node);
        Node p = map.get(pname);
        Node tmp = p;
        while (tmp.last != null)
            tmp = tmp.last;
        node.next = tmp.next;
        tmp.next = node;
        p.last = node;
    }

    public void death(String name) {
        Node node = map.get(name);
        node.isDeleted = true;
    }

    public List<String> getInheritanceOrder() {
        List<String> ans = new ArrayList<>();
        Node tmp = head.next;
        while (tmp.next != null) {
            if (!tmp.isDeleted)
                ans.add(tmp.name);
            tmp = tmp.next;
        }
        return ans;
    }
}
```

## [401. 二进制手表](https://leetcode-cn.com/problems/binary-watch/)

> 位运算，回溯算法

```java
class Solution {
    public List<String> readBinaryWatch(int turnedOn) {
        List<String> res = new ArrayList<>();
        for (int i = 0; i <= 11; ++i) {
            for (int j = 0; j <= 59; ++j) {
                if (Integer.bitCount(i) + Integer.bitCount(j) == turnedOn) {
                    res.add(i + ":" + (j < 10 ? "0" : "") + j);
                }
            }
        }
        return res;
    }
}
```

## [剑指 Offer 38. 字符串的排列](https://leetcode-cn.com/problems/zi-fu-chuan-de-pai-lie-lcof/)

> 回溯算法

暴力

```java
class Solution {
    public String[] permutation(String s) {
        Set<String> lastSet = new HashSet<>();
        lastSet.add("");
        for (int i = 0; i < s.length(); ++i) {
            Set<String> curSet = new HashSet<>();
            for (String str: lastSet) {
                int n = str.length();
                for (int j = 0; j <= n; ++j)
                    curSet.add(str.substring(0, j) + s.charAt(i) + str.substring(j, n));
            }
            lastSet = curSet;
        }
        return lastSet.toArray(new String[0]);
    }
}
```

回溯（巧妙结合利用排序）

```java
class Solution {
    List<String> rec = new ArrayList<String>();
    boolean[] visit = new boolean[n];

    public String[] permutation(String s) {
        char[] arr = s.toCharArray();
        Arrays.sort(arr);
        StringBuffer perm = new StringBuffer();
        backtrack(arr, 0, s.length(), perm);
        return rec.toArray(new String[0]);
    }

    public void backtrack(char[] arr, int i, int n, StringBuffer perm) {
        if (i == n) {
            rec.add(perm.toString());
            return;
        }
        for (int j = 0; j < n; j++) {
            if (visit[j] || (j > 0 && !visit[j - 1] && arr[j - 1] == arr[j])) { // 排序后相邻值相同各取其一会有重复结果，所以不考虑直接跳过
                continue;
            }
            visit[j] = true;
            perm.append(arr[j]);
            backtrack(arr, i + 1, n, perm);
            perm.deleteCharAt(perm.length() - 1);
            visit[j] = false;
        }
    }
}
```

下一个全排列

```java
class Solution {
    public String[] permutation(String s) {
        List<String> rec = new ArrayList<>();
        char[] arr = s.toCharArray();
        Arrays.sort(arr);
        do {
            rec.add(new String(arr));
        } while(nextPerm(arr));
        return rec.toArray(new String[0]);
    }

    private boolean nextPerm(char[] arr) {
        // 计算"全排列下一个"四部曲
        // ① n-2从后往前找递减，记为i【左边的「较小数」】
        // ② n-1从后往前找第一个比arr[i]大的，记为j【最右的「较大数」】
        // ③ 交换i,j
        // ④ i+1~n逆序
        int n = arr.length, i = n - 2, j = n - 1;
        while (i >= 0 && arr[i] >= arr[i + 1]) i--;
        if (i < 0) return false; // 已经完全逆序
        while (j >= 0 && arr[i] >= arr[j]) j--;
        swap(arr, i, j);
        reverse(arr, i + 1);
        return true; // 未完全逆序
    }

    private void swap(char[] arr, int i, int j) {
        arr[i] = (char)(arr[i] ^ arr[j]);
        arr[j] = (char)(arr[i] ^ arr[j]);
        arr[i] = (char)(arr[i] ^ arr[j]);
    }

    private void reverse(char[] arr, int start) {
        for (int i = start, j = arr.length - 1; i < j; ++i, --j) {
            swap(arr, i, j);
        }
    }
}
```

## [剑指 Offer 15. 二进制中1的个数](https://leetcode-cn.com/problems/er-jin-zhi-zhong-1de-ge-shu-lcof/)

> 位运算

bitCount

```java
public class Solution {
    // you need to treat n as an unsigned value
    public int hammingWeight(int n) {
        return Integer.bitCount(n);
    }
}
```

n&(n-1)

```java
public class Solution {
    // you need to treat n as an unsigned value
    public int hammingWeight(int n) {
        int ans = 0;
        while (n != 0) {
            n &= n - 1;
            ans++;
        }
        return ans;
    }
}
```

## [149. 直线上最多的点数](https://leetcode-cn.com/problems/max-points-on-a-line/)

> 几何，哈希表，数学，GCD

```java
class Solution {
    public int maxPoints(int[][] points) {
        int n = points.length;
        if (n <= 2) // 1个或者2个点直接返回结果
            return n;
        int max = 0;
        // 斜率化简为最简分数a/b，a=ma/gcd(ma,mb)，b=mb/gcd(ma,mb)，记为key=a*20001+b
        for (int i = 0; i < n; ++i) {
            if (max >= n - i || max > n / 2) { // 前一个max已经超过了最多之后最多可能最大值了
                return max;
            }
            Map<Integer, Integer> map = new HashMap<>();
            int maxn = 0;
            for (int j = i + 1; j < n; ++j) { // 针对每个点求最大值
                int ma = points[j][1] - points[i][1];
                int mb = points[j][0] - points[i][0];
                int g = 1;
                if (ma == 0) // 平行于x轴
                    mb = 1;
                else if (mb == 0) // 平行于y轴
                    ma = 1;
                else if (ma < 0) {
                    ma = -ma;
                    mb = -mb;
                    g = gcd(Math.abs(ma), Math.abs(mb));
                } else {
                    g = gcd(Math.abs(ma), Math.abs(mb));
                }
                int key = ma / g * 20001 + mb / g; // 也可以用字符串记录
                int val = map.getOrDefault(key, 0) + 1;
                map.put(key, val);
                maxn = Math.max(maxn, val + 1);
            }
            max = Math.max(max, maxn);
        }
        return max;
    }

    private int gcd(int a, int b) {
        return b != 0 ? gcd(b, a % b) : a;
    }
}
```

## [752. 打开转盘锁](https://leetcode-cn.com/problems/open-the-lock/)

> BFS，数组，哈希表，字符串，迷宫，双向BFS，搜索，A*

单向BFS

```java
class Solution {
    public int openLock(String[] deadends, String target) {
        // 迷宫题目：
        // 1. 初始位置：'0000'
        // 2. 行走方向：4个拨盘向上向下拨，共8个方向
        // 3. 终止条件：搜索完成||遇到了target
        // 4. 继续条件：遇到障碍（set记录障碍）||搜索过（Set记录）
        if (target.equals("0000"))
            return 0;
        Set<String> deadendSet = new HashSet<>();
        for (String str: deadends) {
            if (str.equals("0000"))
                return -1;
            deadendSet.add(str);
        }
        Set<String> seen = new HashSet<>();
        seen.add("0000");
        Queue<String> q = new LinkedList<>();
        q.offer("0000");
        int level = 1;
        while (!q.isEmpty()) { // BFS模板
            int size = q.size();
            while (size-- > 0) {
                String str = q.poll();
                for (String next: getNext(str)) {
                    if (!deadendSet.contains(next) && !seen.contains(next)) {
                        if (next.equals(target))
                            return level;
                        seen.add(next);
                        q.offer(next);
                    }
                }
            }
            level++; // 每选择一次+1，第一次遇到即是最小
        }
        return -1;
    }

    private String[] getNext(String str) {
        int n = str.length();
        String[] ret = new String[8];
        int index = 0;
        for (int i = 0; i < 4; ++i) {
            char c = str.charAt(i);
            char[] arr0 = str.toCharArray();
            arr0[i] = c == '9' ? '0' : (char)(c + 1);
            ret[index++] = new String(arr0);
            char[] arr1 = str.toCharArray();
            arr1[i] = c == '0' ? '9' : (char)(c - 1);
            ret[index++] = new String(arr1);
        }
        return ret;
    }
}
```

双向BFS

```java
class Solution {
    int level = 1;
    Map<String, Integer> m1 = new HashMap<>();
    Map<String, Integer> m2 = new HashMap<>();
    Queue<String> q1 = new LinkedList<>();
    Queue<String> q2 = new LinkedList<>();
    Set<String> deadendSet = new HashSet<>();

    public int openLock(String[] deadends, String target) {
        if (target.equals("0000"))
            return 0;
        for (String str: deadends) {
            if (str.equals("0000"))
                return -1;
            deadendSet.add(str);
        }
        m1.put("0000", 0);
        m2.put(target, 0);
        q1.offer("0000");
        q2.offer(target);
        while (!q1.isEmpty() && !q2.isEmpty()) {
            int t = -1;
            if (q1.size() < q2.size()) {
                t = update(q1, m1, m2);
            } else {
                t = update(q2, m2, m1);
            }
            if (t != -1)
                return t;
        }
        return -1;
    }

    private int update(Queue<String> q, Map<String, Integer> cur, Map<String, Integer> other) {
        String str = q.poll();
        int step = cur.get(str);
        for (String next: getNext(str)) {
            if (!deadendSet.contains(next) && !cur.containsKey(next)) {
                if (other.containsKey(next)) {
                    return step + 1 + other.get(next);
                }
                cur.put(next, step + 1);
                q.offer(next);
            }
        }
        return -1;
    }

    private String[] getNext(String str) {
        int n = str.length();
        String[] ret = new String[8];
        int index = 0;
        for (int i = 0; i < 4; ++i) {
            char c = str.charAt(i);
            char[] arr0 = str.toCharArray();
            arr0[i] = c == '9' ? '0' : (char)(c + 1);
            ret[index++] = new String(arr0);
            char[] arr1 = str.toCharArray();
            arr1[i] = c == '0' ? '9' : (char)(c - 1);
            ret[index++] = new String(arr1);
        }
        return ret;
    }
}
```

AStar TODO

```java
class Solution {
    public int openLock(String[] deadends, String target) {
        if ("0000".equals(target))
            return 0;
        Set<String> dead = new HashSet<String>();
        for (String deadend : deadends) {
            dead.add(deadend);
        }
        if (dead.contains("0000")) {
            return -1;
        }
        PriorityQueue<AStar> pq = new PriorityQueue<AStar>((a, b) -> a.f - b.f);
        pq.offer(new AStar("0000", target, 0));
        Set<String> seen = new HashSet<String>();
        seen.add("0000");
        while (!pq.isEmpty()) {
            AStar node = pq.poll();
            for (String nextStatus : get(node.status)) {
                if (!seen.contains(nextStatus) && !dead.contains(nextStatus)) {
                    if (nextStatus.equals(target))
                        return node.g + 1;
                    pq.offer(new AStar(nextStatus, target, node.g + 1));
                    seen.add(nextStatus);
                }
            }
        }
        return -1;
    }

    // 枚举 status 通过一次旋转得到的数字
    public List<String> get(String status) {
        List<String> ret = new ArrayList<String>();
        char[] array = status.toCharArray();
        for (int i = 0; i < 4; ++i) {
            char x = array[i];
            array[i] = x == '0' ? '9' : (char) (x - 1);
            ret.add(new String(array));
            array[i] = x == '9' ? '0' : (char) (x + 1);
            ret.add(new String(array));
            array[i] = x;
        }
        return ret;
    }
}

class AStar {
    String status;
    int f, g, h;

    public AStar(String status, String target, int g) {
        this.status = status;
        this.g = g;
        this.h = getH(status, target);
        this.f = this.g + this.h;
    }

    // 计算启发函数
    public static int getH(String status, String target) {
        int ret = 0;
        for (int i = 0; i < 4; ++i) {
            int dist = Math.abs(status.charAt(i) - target.charAt(i));
            ret += Math.min(dist, 10 - dist);
        }
        return ret;
    }
}
```

IDA* TODO

## [773. 滑动谜题](https://leetcode-cn.com/problems/sliding-puzzle/)

> DFS，数组，矩阵，迷宫

```java
class Solution {
    private int[][] matrix = {{1, 3}, {0, 2, 4}, {1, 5}, {0, 4}, {1, 3, 5}, {2, 4}};

    public int slidingPuzzle(int[][] board) {
        // BFS:
        // 中间编码："123450"
        // 结束条件：q.isEmpty() || 找到谜底
        Queue<String> q = new LinkedList<>();
        Set<String> seen = new HashSet<>();
        String e = encode(board);
        if (e.equals("123450"))
            return 0;
        q.offer(e);
        seen.add(e);
        int level = 0;
        while (!q.isEmpty()) {
            ++level;
            int size = q.size();
            while (size-- > 0) {
                String cur = q.poll();
                for (String str: getNext(cur)) {
                    if (!seen.contains(str)) {
                        if (str.equals("123450")) {
                            return level;
                        }
                        q.offer(str);
                        seen.add(str);
                    }
                }
            }
        }
        return -1;
    }

    private void swap(char[] cs, int i, int j) {
        char temp = cs[i];
        cs[i] = cs[j];
        cs[j] = temp;
    }

    private List<String> getNext(String cur) {
        char[] cs = cur.toCharArray();
        int index = cur.indexOf("0");
        int[] directions = matrix[index];
        List<String> ret = new ArrayList<>();
        for (int i: directions) {
            swap(cs, index, i);
            ret.add(new String(cs));
            swap(cs, index, i);
        }
        return ret;
    }

    private String encode(int[][] board) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 3; ++j) {
                sb.append(board[i][j]);
            }
        }
        return sb.toString();
    }
}
```

AStar/康托展开 TODO

## [909. 蛇梯棋](https://leetcode-cn.com/problems/snakes-and-ladders/)

> BFS，数组，矩阵，迷宫

```java
class Solution {
    public int snakesAndLadders(int[][] board) {
        int n = board.length;
        int len = n * n + 1;
        int[] flat = new int[len];
        int cnt = 1;
        boolean direction = true; // 向右
        for (int i = n - 1; i >= 0; --i) { // 降维
            if (direction) {
                for (int j = 0; j < n; ++j)
                    flat[cnt++] = board[i][j];
            } else {
                for (int j = n - 1; j >= 0; --j)
                    flat[cnt++] = board[i][j];
            }
            direction = !direction;
        }
        int end = cnt - 1;
        Queue<Integer> q = new LinkedList<>();
        q.offer(1);
        boolean[] vis = new boolean[len];
        int level = 0;
        while (!q.isEmpty()) {
            ++level;
            int size = q.size();
            while (size-- > 0) {
                Integer cur = q.poll();
                for (Integer next: getNext(cur)) {
                    if (next > end)
                        break;
                    if (next == end)
                        return level;
                    if (!vis[next]) {
                        int b = flat[next];
                        int tmp = next;
                        if (b != -1) {
                            if (b == end)
                                return level;
                            tmp = b;
                        }
                        q.offer(tmp);
                        vis[next] = true; // 千万注意是next
                    }
                }
            }
        }
        return -1;
    }

    private List<Integer> getNext(Integer cur) {
        List<Integer> ret = new ArrayList<>();
        for (int i = 1; i <= 6; ++i) {
            ret.add(cur + i);
        }
        return ret;
    }
}
```

## [168. Excel表列名称](https://leetcode-cn.com/problems/excel-sheet-column-title/)

> 数学，字符串

执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：35 MB, 在所有 Java 提交中击败了99.76%的用户

```java
class Solution {
    public String convertToTitle(int columnNumber) {
        StringBuilder ret = new StringBuilder();
        while (columnNumber > 0) {
            char c = (char)('A' + (columnNumber - 1) % 26);
            ret.insert(0, c);
            columnNumber = (columnNumber - 1) / 26;
        }
        return ret.toString();
    }
}
```

## [剑指 Offer 37. 序列化二叉树](https://leetcode-cn.com/problems/xu-lie-hua-er-cha-shu-lcof/)

> 树，DFS，BFS，设计，字符串，二叉树

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
public class Codec {

    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
        if (root != null) {
            return root.val + "," + serialize(root.left) + "," + serialize(root.right);
        } else {
            return "#";
        }
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        List<String> dataList = new LinkedList<>(Arrays.asList(data.split(",")));
        return mDeserialize(dataList);
    }

    private TreeNode mDeserialize(List<String> dl) {
        if (dl.get(0).equals("#")) {
            dl.remove(0);
            return null;
        }
        TreeNode root = new TreeNode(Integer.parseInt(dl.get(0)));
        dl.remove(0);
        root.left = mDeserialize(dl);
        root.right = mDeserialize(dl);
        return root;
    }
}

// Your Codec object will be instantiated and called as such:
// Codec codec = new Codec();
// codec.deserialize(codec.serialize(root));
```

TODO LL(1)文法

## [LCP 07. 传递信息](https://leetcode-cn.com/problems/chuan-di-xin-xi/)

> DFS，BFS，图，动态规划

BFS

```java
class Solution {
    public int numWays(int n, int[][] relation, int k) {
        Map<Integer, Set<Integer>> map = new HashMap<>();
        for (int[] rel: relation) {
            Set<Integer> set;
            if (map.containsKey(rel[0])) {
                set = map.get(rel[0]);
            } else {
                set = new HashSet<>();
            }
            set.add(rel[1]);
            map.put(rel[0], set);
        }
        int cnt = 0;
        Queue<Integer> q = new LinkedList<>();
        q.offer(0);
        while (!q.isEmpty() && cnt < k) {
            cnt++;
            int size = q.size();
            while (size-- > 0) {
                Integer i = q.poll();
                if (map.containsKey(i)) {
                    for (Integer next: map.get(i)) {
                        q.offer(next);
                    }
                }
            }
        }
        int ret = 0;
        if (cnt == k) {
            while (!q.isEmpty()) {
                if (q.poll() == n - 1) {
                    ret++;
                }
            }
        }
        return ret;
    }
}
```

DFS，动态规划 TOOD

## [1833. 雪糕的最大数量](https://leetcode-cn.com/problems/maximum-ice-cream-bars/)

> 排序，贪心，计数数组

排序

```java
class Solution {
    public int maxIceCream(int[] costs, int coins) {
        int step = 0;
        Arrays.sort(costs);
        int tmp = 0;
        for (int cost: costs) {
            tmp += cost;
            if (tmp <= coins) {
                step++;
            } else {
                break;
            }
        }
        return step;
    }
}
```

计数排序

```java
class Solution {
    public int maxIceCream(int[] costs, int coins) {
        int[] freq = new int[100001];
        for (int coin: costs) {
            freq[coin]++;
        }
        int step = 0;
        for (int i = 0; i < 100001; ++i) {
            while (freq[i] > 0) {
                if (coins - i >= 0) {
                    freq[i]--;
                    coins -= i;
                    step++;
                } else {
                    return step;
                }
            }
        }
        return step;
    }
}
```

## [451. 根据字符出现频率排序](https://leetcode-cn.com/problems/sort-characters-by-frequency/)

> 哈希表，字符串，桶排序，计数，排序，堆，优先队列

哈希表+集合排序

```java
class Solution {
    public String frequencySort(String s) {
        Map<Character, Integer> freq = new HashMap<>();
        for (char c: s.toCharArray()) {
            freq.put(c, freq.getOrDefault(c, 0) + 1);
        }
        List<Character> list = new ArrayList<>(freq.keySet());
        Collections.sort(list, (x, y) -> freq.get(y) - freq.get(x));
        StringBuilder sb = new StringBuilder();
        for (Character c: list) {
            int cnt = freq.get(c);
            while (cnt-- > 0) {
                sb.append(c);
            }
        }
        return sb.toString();
    }
}
```

数组模拟

```java
class Solution {
    public String frequencySort(String s) {
        int[][] freq = new int[128][2];
        for (int i = 0; i < 128; ++i)
            freq[i][0] = i;
        for (char c: s.toCharArray()) {
            freq[c][1]++;
        }
        Arrays.sort(freq, (x, y) -> y[1] - x[1]);
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < 128; ++i) {
            if (freq[i][1] == 0)
                break;
            while (freq[i][1]-- > 0) {
                sb.append((char)freq[i][0]);
            }
        }
        return sb.toString();
    }
}
```

桶排序 TODO

## [645. 错误的集合](https://leetcode-cn.com/problems/set-mismatch/)

> 数组，哈希表，排序，位运算，频数数组

频数数组

```java
class Solution {
    public int[] findErrorNums(int[] nums) {
        int n = nums.length;
        int[] res = new int[2];
        int[] freq = new int[n + 1];
        for (int i: nums) {
            freq[i]++;
            if (freq[i] == 2)
                res[0] = i;
        }
        for (int i = 1; i <= n; ++i) {
            if (freq[i] == 0)
                res[1] = i;
        }
        return res;
    }
}
```

排序、位运算 TODO

## [726. 原子的数量](https://leetcode-cn.com/problems/number-of-atoms/)

> 栈，哈希表，字符串

```java
class Solution {
    public String countOfAtoms(String formula) {
        Deque<Map<String, Integer>> stack = new LinkedList<>(); // 栈中哈希表
        stack.push(new HashMap<>());
        char[] cs = formula.toCharArray();
        for (int i = 0; i < cs.length; ++i) {
            if (cs[i] == '(') {
                stack.push(new HashMap<>());
            } else if (cs[i] == ')') {
                int[] cnt = parseCnt(i + 1, cs);
                Map<String, Integer> tmpMap = stack.pop();
                Map<String, Integer> peekMap = stack.peek();
                for (String str: tmpMap.keySet()) {
                    peekMap.put(str, peekMap.getOrDefault(str, 0) + tmpMap.get(str) * cnt[0]);
                }
                i = cnt[1];
            } else {
                StringBuilder atomName = new StringBuilder();
                atomName.append(cs[i++]);
                int n = cs.length;
                while (i < n && isLower(cs[i])) {
                    atomName.append(cs[i++]);
                }
                String atom = atomName.toString();
                int[] cnt = parseCnt(i, cs);
                Map<String, Integer> peekMap = stack.peek();
                peekMap.put(atom, peekMap.getOrDefault(atom, 0) + cnt[0]);
                i = cnt[1];
            }
        }
        StringBuilder sb = new StringBuilder();
        TreeMap<String, Integer> finalMap = new TreeMap<>(stack.pop());
        for (String str: finalMap.keySet()) {
            int num = finalMap.get(str);
            sb.append(str).append(num == 1 ? "" : num);
        }
        return sb.toString();
    }

    private boolean isLower(char c) {
        return 'a' <= c && c <= 'z';
    }

    private int[] parseCnt(int i, char[] cs) {
        int n = cs.length;
        int[] res = {1, i};
        if (i >= n)
            return res;
        int tmp = 0;
        while (i < n && isNum(cs[i])) {
            tmp = tmp * 10 + (cs[i] - '0');
            i++;
        }
        res[0] = tmp == 0 ? 1 : tmp;
        res[1] = i - 1;
        return res;
    }

    private boolean isNum(char c) {
        return '0' <= c && c <= '9';
    }
}
```

## [1418. 点菜展示表](https://leetcode-cn.com/problems/display-table-of-food-orders-in-a-restaurant/)

> 数组，哈希表，字符串，有序集合，排序

```java
class Solution {
    public List<List<String>> displayTable(List<List<String>> orders) {
        Map<String, Integer> map = new HashMap<>();
        for (List<String> item: orders) {
            String key = item.get(1) + ";" + item.get(2);
            map.put(key, map.getOrDefault(key, 0) + 1);
        }
        List<List<String>> ret = new ArrayList<>();
        List<String> firstLine = new ArrayList<>();
        firstLine.add("Table");
        TreeSet<String> foods = new TreeSet<>();
        TreeSet<Integer> tables = new TreeSet<>();
        for (String key: map.keySet()) {
            String[] tmp = key.split(";");
            foods.add(tmp[1]);
            tables.add(Integer.parseInt(tmp[0]));
        }
        int foodSize = foods.size();
        firstLine.addAll(new ArrayList<>(foods));
        ret.add(firstLine);
        for (Integer i: tables) {
            List<String> line = new ArrayList<>();
            String tableNum = i + "";
            line.add(tableNum);
            for (int j = 0; j < foodSize; ++j) {
                String key = tableNum + ";" + firstLine.get(j + 1);
                line.add("" + map.getOrDefault(key, 0));
            }
            ret.add(line);
        }
        return ret;
    }
}
```

TODO 转化成Map<tableNumber,Map<菜名,数量>>

## [1711. 大餐计数](https://leetcode-cn.com/problems/count-good-meals/)

> 数组，哈希表

```java
class Solution {
    public int countPairs(int[] deliciousness) {
        int cnt = 0;
        Map<Integer, Integer> map = new HashMap<>(); // Key的出现次数
        int n = deliciousness.length;
        int maxVal = 0;
        for (int i: deliciousness) {
            maxVal = Math.max(maxVal, i);
        }
        int upper = maxVal * 2; // 和的上限
        for (int i: deliciousness) {
            for (int sum = 1; sum <= upper; sum <<= 1) { // 遍历可能的和
                cnt += map.getOrDefault(sum - i, 0); // 是否有，有几个
                cnt %= 1000000007; // 可以写成mod=(int)1e9+7;cnt>=mod则cnt-=mod
            }
            map.put(i, map.getOrDefault(i, 0) + 1);
        }
        return cnt;
    }
}
```

## [930. 和相同的二元子数组](https://leetcode-cn.com/problems/binary-subarrays-with-sum/)

> 数组，哈希表，前缀和，滑动窗口

```java
class Solution {
    public int numSubarraysWithSum(int[] nums, int goal) {
        int n = nums.length;
        int preSum = 0;
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);
        for (int i = 0; i < n; ++i) {
            preSum += nums[i];
            map.put(preSum, map.getOrDefault(preSum, 0) + 1);
        }
        int cnt = 0;
        for (int i = goal; i <= preSum; ++i) {
            if (map.containsKey(i) && map.containsKey(i - goal)) {
                int a = map.get(i), b = map.get(i - goal);
                cnt += goal == 0 ? (a * (a - 1) / 2) : a * b;
            }
        }
        return cnt;
    }
}
```

简化

```java
class Solution {
    public int numSubarraysWithSum(int[] nums, int goal) {
        Map<Integer, Integer> map = new HashMap<>();
        int cnt = 0, sum = 0;
        for (int i: nums) {
            map.put(sum, map.getOrDefault(sum, 0) + 1);
            sum += i;
            cnt += map.getOrDefault(sum - goal, 0);
        }
        return cnt;
    }
}
```

滑动窗口（前缀和是递增序列）

```java
class Solution {
    public int numSubarraysWithSum(int[] nums, int goal) {
        int cnt = 0, n = nums.length;
        int l1 = 0, l2 = 0, r = 0;
        int sum1 = 0, sum2 = 0;
        while (r < n) {
            sum1 += nums[r];
            sum2 += nums[r];
            while (l1 <= r && sum1 > goal)
                sum1 -= nums[l1++];
            while (l2 <= r && sum2 >= goal) // 注意等于
                sum2 -= nums[l2++];
            cnt += l2 - l1;
            r++;
        }
        return cnt;
    }
}
```

## [面试题 17.10. 主要元素](https://leetcode-cn.com/problems/find-majority-element-lcci/)

> 数组，计数，Boyer-Moore投票算法

哈希表

```java
class Solution {
    public int majorityElement(int[] nums) {
        int n = nums.length;
        int minNum = (n + 1 + (n & 1)) / 2;
        Map<Integer, Integer> map = new HashMap<>();
        for (int i: nums) {
            int tmp = map.getOrDefault(i, 0);
            if (tmp + 1 == minNum)
                return i;
            map.put(i, tmp + 1);
        }
        return -1;
    }
}
```

Boyer-Moore 投票算法

```java
class Solution {
    public int majorityElement(int[] nums) {
        int n = nums.length;
        int candidate = 0, cnt = 0;
        for (int i: nums) {
            if (i != candidate) {
                if (cnt == 0)
                    candidate = i;
                else
                    cnt--;
            } else {
                cnt++;
            }
        }
        cnt = 0;
        for (int i: nums) {
            if (i == candidate) {
                cnt++;
                if (cnt * 2 > n)
                    return candidate;
            }
        }
        return -1;
    }
}
```

## [981. 基于时间的键值存储](https://leetcode-cn.com/problems/time-based-key-value-store/)

> 设计，哈希表，字符串，二分查找

```java
class TimeMap {
    class Val {
        public String value;
        public int timestamp;
        Val(String value, int timestamp) {
            this.value = value;
            this.timestamp = timestamp;
        }
    }
    Map<String, List<Val>> map;

    /** Initialize your data structure here. */
    public TimeMap() {
        map = new HashMap<>();
    }

    public void set(String key, String value, int timestamp) {
        List<Val> list = map.getOrDefault(key, new ArrayList<>());
        list.add(new Val(value, timestamp));
        map.put(key, list);
    }

    public String get(String key, int timestamp) {
        if (map.containsKey(key)) {
            List<Val> list = map.get(key);
            int idx = bs(list, timestamp);
            return idx <= 0 ? "" : list.get(idx - 1).value;
        } else {
            return "";
        }
    }

    private int bs(List<Val> list, int timestamp) {
        int l = 0, r = list.size() - 1;
        if (timestamp < list.get(l).timestamp)
            return -1;
        if (timestamp >= list.get(r).timestamp)
            return r + 1;
        while (l < r) {
            int mid = l + (r - l) / 2;
            Val mv = list.get(mid);
            if (mv.timestamp <= timestamp) {
                l = mid + 1;
            } else {
                r = mid;
            }
        }
        return l;
    }
}

/**
 * Your TimeMap object will be instantiated and called as such:
 * TimeMap obj = new TimeMap();
 * obj.set(key,value,timestamp);
 * String param_2 = obj.get(key,timestamp);
 */
```

floorEntry

```java
class TimeMap {

    Map<String, TreeMap<Integer, String>> map;

    /** Initialize your data structure here. */
    public TimeMap() {
        map = new HashMap<>();
    }

    public void set(String key, String value, int timestamp) {
        TreeMap<Integer, String> tm = map.getOrDefault(key, new TreeMap<>());
        tm.put(timestamp, value);
        map.put(key, tm);
    }

    public String get(String key, int timestamp) {
        if (map.containsKey(key)) {
            TreeMap<Integer, String> tm = map.get(key);
            Map.Entry<Integer, String> me = tm.floorEntry(timestamp);
            return me == null ? "" : me.getValue();
        } else {
            return "";
        }
    }
}

/**
 * Your TimeMap object will be instantiated and called as such:
 * TimeMap obj = new TimeMap();
 * obj.set(key,value,timestamp);
 * String param_2 = obj.get(key,timestamp);
 */
```

## [274. H 指数](https://leetcode-cn.com/problems/h-index/)

> 数组，计数排序，排序

逆序

```java
class Solution {
    public int hIndex(int[] citations) {
        int[] reverseArr = Arrays.stream(citations).boxed().sorted((x, y) -> y - x).mapToInt(a -> a).toArray();
        int h = 0;
        for (int i = 0; i < reverseArr.length; ++i) {
            if (reverseArr[i] < i + 1)
                break;
            h++;
        }
        return h;
    }
}
```

正序

```java
class Solution {
    public int hIndex(int[] citations) {
        Arrays.sort(citations);
        int n = citations.length;
        int h = n;
        for (int i = 0; i < citations.length; ++i) {
            if (citations[i] < h)
                h--;
            else
                break;
        }
        return h;
    }
}
```

计数排序

```java
// 设计数数组counter[n+1]，从后向前累加论文数，如果大于i则找到h
class Solution {
    public int hIndex(int[] citations) {
        // 设计数数组counter[n+1]，从后向前累加论文数，如果大于i则找到h
        int n = citations.length;
        int[] counter = new int[n + 1];
        for (int i: citations) {
            if (i >= n)
                counter[n]++;
            else
                counter[i]++;
        }
        int tot = 0;
        for (int i = n; i >= 0; --i) {
            tot += counter[i];
            if (tot >= i)
                return i;
        }
        return 0;
    }
}
```

## [275. H 指数 II](https://leetcode-cn.com/problems/h-index-ii/)

> 数组，二分查找

执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：45.4 MB, 在所有 Java 提交中击败了20.66%的用户

```java
class Solution {
    public int hIndex(int[] citations) {
        int n = citations.length;
        int l = 0, r = n - 1;
        while (l < r) {
            int mid = l + (r - l) / 2;
            if (citations[mid] >= n - mid) {
                r = mid;
            } else {
                l = mid + 1;
            }
        }
        return citations[r] >= n - r ? n - r : 0;
    }
}
```

## [剑指 Offer 53 - I. 在排序数组中查找数字 I](https://leetcode-cn.com/problems/zai-pai-xu-shu-zu-zhong-cha-zhao-shu-zi-lcof/)

> 数组，二分查找

```java
class Solution {
    public int search(int[] nums, int target) {
        int left = bs(nums, target);
        if (left == -1)
            return 0;
        int cnt = 1;
        while (left < nums.length - 1 && nums[++left] == target)
            cnt++;
        return cnt;
    }

    private int bs(int[] nums, int target) {
        int l = 0, r = nums.length - 1;
        while (l <= r) {
            int mid = l + (r - l) / 2;
            if (nums[mid] < target) {
                l = mid + 1;
            } else {
                r = mid - 1;
            }
        }
        return l < nums.length && nums[l] == target ? l : -1;
    }
}
```

找左右端点

执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：40.7 MB, 在所有 Java 提交中击败了99.14%的用户

```java
class Solution {
    public int search(int[] nums, int target) {
        int l = bs(nums, target, true);
        if (l == -1)
            return 0;
        int r = bs(nums, target, false);
        if (r == -1)
            return 0;
        return r - l + 1;
    }

    private int bs(int[] nums, int target, boolean direction) {
        int l = 0, r = nums.length - 1;
        while (l <= r) {
            int mid = l + (r - l) / 2;
            if (nums[mid] < target) {
                l = mid + 1;
            } else if (nums[mid] > target){
                r = mid - 1;
            } else {
                if (direction)
                    r = mid - 1;
                else
                    l = mid + 1;
            }
        }
        if (direction)
            return l < nums.length && nums[l] == target ? l : -1;
        else
            return r >= 0 && nums[r] == target ? r : -1;
    }
}
```

## [剑指 Offer 42. 连续子数组的最大和](https://leetcode-cn.com/problems/lian-xu-zi-shu-zu-de-zui-da-he-lcof/)

## [53. 最大子序和](https://leetcode-cn.com/problems/maximum-subarray/)

> 数组，分治，动态规划

动态规划

```java
class Solution {
    public int maxSubArray(int[] nums) {
        // dp[i]表示第i个数结尾的连续子数组最大和
        // dp[i]=max(nums[i],dp[i-1]+nums[i])
        // dp[0]=0
        // dp[n]
        int n = nums.length;
        int dp = 0, max = nums[0];
        for (int i = 0; i < n; ++i) {
            dp = Math.max(nums[i], dp + nums[i]);
            max = Math.max(max, dp);
        }
        return max;
    }
}
```

分治

```java
class Solution {
    public class Status {
        public int ls, rs, is, ms;
        public Status(int ls, int rs, int is, int ms) {
            this.ls = ls;
            this.rs = rs;
            this.is = is;
            this.ms = ms;
        }
    }

    public int maxSubArray(int[] nums) {
        // ls左端点构成的子数组和，rs右端点构成的子数组和，is区间总和，ms区间内最大子区间和
        return getInfo(nums, 0, nums.length - 1).ms;
    }

    private Status getInfo(int[] nums, int l, int r) {
        if (l == r) {
            return new Status(nums[l], nums[l], nums[l], nums[l]);
        }
        int mid = l + (r - l) / 2;
        Status lStatus = getInfo(nums, l, mid);
        Status rStatus = getInfo(nums, mid + 1, r);
        return pushUp(lStatus, rStatus);
    }

    private Status pushUp(Status lStatus, Status rStatus) {
        int ls = Math.max(lStatus.ls, lStatus.is + rStatus.ls);
        int rs = Math.max(rStatus.rs, lStatus.rs + rStatus.is);
        int is = lStatus.is + rStatus.is;
        int ms = Math.max(lStatus.rs + rStatus.ls, Math.max(lStatus.ms, rStatus.ms));
        return new Status(ls, rs, is, ms);
    }
}
```

TODO 前缀和

## [面试题 10.02. 变位词组](https://leetcode-cn.com/problems/group-anagrams-lcci/)

> 哈希表，字符串，排序，数学

计数作为key

```java
class Solution {
    public List<List<String>> groupAnagrams(String[] strs) {
        Map<String, List<String>> map = new HashMap<>();
        for (String str: strs) {
            int[] cnt = new int[26];
            for (char c: str.toCharArray()) {
                cnt[c - 'a']++;
            }
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < 26; ++i) {
                if (cnt[i] > 0)
                    sb.append((char)'a' + i).append(cnt[i]);
            }
            String key = sb.toString();
            List<String> l = map.getOrDefault(key, new ArrayList<String>());
            l.add(str);
            map.put(key, l);
        }
        return new ArrayList<List<String>>(map.values());
    }
}
```

排序作为key

```java
class Solution {
    public List<List<String>> groupAnagrams(String[] strs) {
        Map<String, List<String>> map = new HashMap<>();
        for (String str: strs) {
            char[] ca = str.toCharArray();
            Arrays.sort(ca);
            String key = new String(ca);
            List<String> l = map.getOrDefault(key, new ArrayList<String>());
            l.add(str);
            map.put(key, l);
        }
        return new ArrayList<List<String>>(map.values());
    }
}
```

质数分解唯一性（会溢出，慎用）

```java
class Solution {
    static int[] nums = new int[26]; 
    static {
        for (int i = 2, idx = 0; idx != 26; i++) {
            boolean ok = true;
            for (int j = 2; j <= i / j; j++) {
                if (i % j == 0) {
                    ok = false;
                    break;
                } 
            }
            if (ok)
                nums[idx++] = i;
        }
    }
    public List<List<String>> groupAnagrams(String[] ss) {
        Map<Long, List<String>> map = new HashMap<>();
        for (String s : ss) {
            long cur = 1;
            for (char c : s.toCharArray()) {
                cur *= nums[c - 'a'];
            }
            List<String> list = map.getOrDefault(cur, new ArrayList<>());
            list.add(s);
            map.put(cur, list);
        }
        return new ArrayList<List<String>>(map.values());
    }
}
```

## [1838. 最高频元素的频数](https://leetcode-cn.com/problems/frequency-of-the-most-frequent-element/)

> 数组，滑动窗口

```java
class Solution {
    public int maxFrequency(int[] nums, int k) {
        // 画柱状图理解，用滑动数组
        Arrays.sort(nums);
        int n = nums.length;
        long total = 0;
        int l = 0, res = 1;
        for (int r = 1; r < n; ++r) {
            total += (long) (nums[r] - nums[r - 1]) * (r - l); // 右移r，需要加的值相当于叠加一层（长是r-l，高是nums[r]-nums[r-1]）
            while (total > k) { // 判断和是否超过限制
                total -= nums[r] - nums[l]; // 没有超过收缩左指针，更新和，减去右指针的值和左指针的差 
                ++l;
            }
            res = Math.max(res, r - l + 1);
        }
        return res;
    }
}
```

## [1877. 数组中最大数对和的最小值](https://leetcode-cn.com/problems/minimize-maximum-pair-sum-in-array/)

> 贪心，数组，双指针，排序

```java
class Solution {
    public int minPairSum(int[] nums) {
        int n = nums.length;
        Arrays.sort(nums);
        int ret = 0;
        for (int i = 0; i < n / 2; ++i) {
            ret = Math.max(ret, nums[i] + nums[n - i - 1]);
        }
        return ret;
    }
}
```

## [剑指 Offer 52. 两个链表的第一个公共节点](https://leetcode-cn.com/problems/liang-ge-lian-biao-de-di-yi-ge-gong-gong-jie-dian-lcof/)

> 链表，双指针，哈希表

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
        ListNode tA = headA, tB = headB;
        boolean flagA = false, flagB = false;
        while (true) {
            if (tA == tB)
                return tA;
            if (tA == null) {
                if (!flagA) {
                    tA = headB;
                    flagA = true;
                } else {
                    return null;
                }
            } else {
                tA = tA.next;
            }
            if (tB == null) {
                if (!flagB) {
                    tB = headA;
                    flagB = true;
                } else {
                    return null;
                }
            } else {
                tB = tB.next;
            }
        }

    }
}
```

简单版本 TODO

```java
public class Solution {
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        if (headA == null || headB == null) {
            return null;
        }
        ListNode pA = headA, pB = headB;
        while (pA != pB) { // 双null也是相等，妙
            pA = pA == null ? headB : pA.next;
            pB = pB == null ? headA : pB.next;
        }
        return pA;
    }
}
```

## [138. 复制带随机指针的链表](https://leetcode-cn.com/problems/copy-list-with-random-pointer/)

> 哈希表，链表

执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：37.6 MB, 在所有 Java 提交中击败了94.74%的用户

```java
/*
// Definition for a Node.
class Node {
    int val;
    Node next;
    Node random;

    public Node(int val) {
        this.val = val;
        this.next = null;
        this.random = null;
    }
}
*/

class Solution {
    public Node copyRandomList(Node head) {
        if (head == null)
            return null;
        Map<Node, Node> map = new HashMap<>();
        Node po = head, pn = null;
        Node newHead = null;
        while (po != null) {
            if (pn == null) {
                pn = new Node(po.val);
                newHead = pn;
            } else {
                pn.next = new Node(po.val);
                pn = pn.next;
            }
            map.put(po, pn);
            po = po.next;
        }
        po = head;
        pn = newHead;
        while (po != null) {
            if (po.random != null)
                pn.random = map.get(po.random);
            po = po.next;
            pn = pn.next;
        }
        return newHead;
    }
}
```

哈希+回溯

利用递归可以只循环一次

执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：38.2 MB, 在所有 Java 提交中击败了40.88%的用户

```java
class Solution {
    Map<Node, Node> cache = new HashMap<>();

    public Node copyRandomList(Node head) {
        if (head == null)
            return null;
        if (!cache.containsKey(head)) {
            Node nHead = new Node(head.val);
            cache.put(head, nHead);
            nHead.next = copyRandomList(head.next);
            nHead.random = copyRandomList(head.random);
        }
        return cache.get(head);
    }
}
```

迭代 + 节点拆分：

利用复制相同节点链到原始链表中可以不用哈希表

执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：37.6 MB, 在所有 Java 提交中击败了95.13%的用户

```java
class Solution {
    public Node copyRandomList(Node head) {
        if (head == null)
            return null;
        for (Node p = head; p != null; p = p.next) {
            Node tmp = p.next;
            p.next = new Node(p.val);
            p.next.next = tmp;
            p = p.next;
        }
        for (Node p = head; p != null; p = p.next.next) {
            p.next.random = p.random == null ? null : p.random.next;
        }
        Node newHead = head.next;
        for (Node p = head; p != null; p = p.next) {
            Node pn = p.next;
            p.next = p.next.next;
            pn.next = pn.next == null ? null : pn.next.next;
        }
        return newHead;
    }
}
```

## [1893. 检查是否区域内所有整数都被覆盖](https://leetcode-cn.com/problems/check-if-all-the-integers-in-a-range-are-covered/)

> 数组，哈希表，差分数组，前缀和

哈希表

执行用时：3 ms, 在所有 Java 提交中击败了19.54%的用户

内存消耗：37.6 MB, 在所有 Java 提交中击败了68.38%的用户

```java
class Solution {
    public boolean isCovered(int[][] ranges, int left, int right) {
        Set<Integer> set = new HashSet<>();
        for (int[] range: ranges)
            for (int i = range[0]; i <= range[1]; ++i)
                set.add(i);
        for (int i = left; i <= right; ++i) {
            if (!set.contains(i))
                return false;
        }
        return true;
    }
}
```

差分数组

```java
class Solution {
    public boolean isCovered(int[][] ranges, int left, int right) {
        int[] diff = new int[52];
        // 对差分数组进行处理
        for(int i = 0; i < ranges.length; i++){
            diff[ranges[i][0]]++;
            diff[ranges[i][1] + 1]--;
        }
        // 根据差分数组处理前缀和，为理解方便单独定义sum，可以原地做
        int[] sum = new int[52];
        for(int i = 1; i <= 51; i++){
            sum[i] = sum[i - 1] + diff[i];
        }
        // 从left到right判断是否满足sum > 0
        for(int i = left; i <= right; i++){
            if(sum[i] <= 0)
                return false;
        }
        return true;
    }
}
```

树状数组/线段树 TODO

## [1736. 替换隐藏数字得到的最晚时间](https://leetcode-cn.com/problems/latest-time-by-replacing-hidden-digits/)

> 字符串

```java
class Solution {
    public String maximumTime(String time) {
        char[] cs = time.toCharArray();
        if (cs[0] == '?')
            cs[0] = ('4' <= cs[1] && cs[1] <= '9') ? '1' : '2';
        if (cs[1] == '?')
            cs[1] = (cs[0] == '2') ? '3' : '9';
        if (cs[3] == '?')
            cs[3] = '5';
        if (cs[4] == '?')
            cs[4] = '9';
        return new String(cs);
    }
}
```

## [1743. 从相邻元素对还原数组](https://leetcode-cn.com/problems/restore-the-array-from-adjacent-pairs/)

> 数组，哈希表

```java
class Solution {
    private static int MAX = 100001;

    public int[] restoreArray(int[][] adjacentPairs) {
        Map<Integer, int[]> map = new HashMap<>();
        for (int[] a: adjacentPairs) {
            int[] pair;
            if (map.containsKey(a[0])) {
                pair = map.get(a[0]);
                pair[1] = a[1];
            } else {
                pair = new int[2];
                pair[0] = a[1];
                pair[1] = MAX;
            }
            map.put(a[0], pair);
            if (map.containsKey(a[1])) {
                pair = map.get(a[1]);
                pair[1] = a[0];
            } else {
                pair = new int[2];
                pair[0] = a[0];
                pair[1] = MAX;
            }
            map.put(a[1], pair);
        }
        int start = 0;
        for (Map.Entry<Integer, int[]> entry: map.entrySet()) {
            int[] pair = entry.getValue();
            if (pair[1] == MAX)
                start = entry.getKey();
        }
        int n = map.keySet().size();
        int[] ret = new int[n];
        if (n <= 1)
            return ret;
        ret[0] = start;
        ret[1] = map.get(start)[0];
        for (int i = 2, j = ret[1]; i < n; ++i) {
            int[] tmp = map.get(j);
            if (tmp[1] == MAX) {
                ret[i] = tmp[0];
                break;
            }
            int[] arr = map.get(j);
            ret[i] = arr[0] == ret[i - 2] ? arr[1] : arr[0];
            j = ret[i];
        }
        return ret;
    }
}
```

## [671. 二叉树中第二小的节点](https://leetcode-cn.com/problems/second-minimum-node-in-a-binary-tree/)

> 树，DFS，二叉树

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
    public int findSecondMinimumValue(TreeNode root) {
        return dfs(root, root.val);
    }

    private int dfs(TreeNode root, int min) {
        if (root.left == null) {
            return -1;
        }
        if (root.left.val == min && root.right.val == min) {
            int d1 = dfs(root.left, min);
            int d2 = dfs(root.right, min);
            if (d1 == -1)
                return d2;
            if (d2 == -1)
                return d1;
            return Math.min(d1, d2);
        }
        if (root.left.val == min) {
            int left = dfs(root.left, min);
            if (left == -1)
                return root.right.val;
            else
                return Math.min(left, root.right.val);
        }
        if (root.right.val == min) {
            int right = dfs(root.right, min);
            if (right == -1)
                return root.left.val;
            else
                return Math.min(right, root.left.val);
        }
        return -1;
    }
}
```

无返回值简化版

```java
class Solution {
    int ans = -1;
    public int findSecondMinimumValue(TreeNode root) {
        dfs(root, root.val);
        return ans;
    }
    void dfs(TreeNode root, int cur) {
        if (root == null) return ;
        if (root.val != cur) {
            if (ans == -1)
                ans = root.val;
            else
                ans = Math.min(ans, root.val);
            return;
        }
        dfs(root.left, cur);
        dfs(root.right, cur);
    }
}
```

## [863. 二叉树中所有距离为 K 的结点](https://leetcode-cn.com/problems/all-nodes-distance-k-in-binary-tree/)

> 树，二叉树，DFS

两遍DFS

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
    Map<Integer, TreeNode> parentMap = new HashMap<>(); // <node的值,node的父节点>
    List<Integer> ret = new ArrayList<>();

    public List<Integer> distanceK(TreeNode root, TreeNode target, int k) {
        dfsRoot(root); // 建前向索引
        dfsTarget(target, null, 0, k);
        return ret;
    }

    private void dfsRoot(TreeNode root) {
        if (root.left != null) {
            parentMap.put(root.left.val, root);
            dfsRoot(root.left);
        }
        if (root.right != null) {
            parentMap.put(root.right.val, root);
            dfsRoot(root.right);
        }
    }

    private void dfsTarget(TreeNode curNode, TreeNode fromNode, int cur, int k) {
        if (curNode == fromNode) { // TODO fromNode是精髓
            return;
        }
        if (cur == k) {
            ret.add(curNode.val);
        }
        if (curNode.left != null) {
            dfsTarget(curNode.left, curNode, cur + 1, k);
        }
        if (curNode.right != null) {
            dfsTarget(curNode.right, curNode, cur + 1, k);
        }
        dfsTarget(parentMap.get(curNode.val), curNode, cur + 1, k);
    }
}
```

## [1104. 二叉树寻路](https://leetcode-cn.com/problems/path-in-zigzag-labelled-binary-tree/)

> 树，数学，二叉树

执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：35.5 MB, 在所有 Java 提交中击败了99.12%的用户

```java
class Solution {
    public List<Integer> pathInZigZagTree(int label) {
        List<Integer> ret = new ArrayList<>();
        ret.add(label);
        int level = getLevel(label);
        while (label > 1) {
            label = (1 << level) + (1 << level - 1) - 1 - label / 2;
            ret.add(label);
            level--;
        }
        Collections.reverse(ret);
        return ret;
    }

    private int getLevel(int label) {
        int cnt = 0;
        while (label > 0) {
            label >>= 1;
            cnt++;
        }
        return cnt - 1;
    }
}
```

## [171. Excel 表列序号](https://leetcode-cn.com/problems/excel-sheet-column-number/)

> 数字，字符串

```java
class Solution {
    public int titleToNumber(String columnTitle) {
        int res = 0;
        for (int i = 0; i < columnTitle.length(); ++i) {
            res = (columnTitle.charAt(i) - 'A' + 1) + res * 26;
        }
        return res;
    }
}
```

## [1337. 矩阵中战斗力最弱的 K 行](https://leetcode-cn.com/problems/the-k-weakest-rows-in-a-matrix/)

> 数组，二分查找，矩阵，排序，堆

```java
class Solution {
    public int[] kWeakestRows(int[][] mat, int k) {
        int m = mat.length, n = mat[0].length;
        int[] ret = new int[k];
        int cnt = 0;
        boolean flag = false;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                if (mat[j][i] == 0 && (i == 0 || mat[j][0] != -1)) {
                    mat[j][0] = -1;
                    ret[cnt++] = j;
                    if (cnt == k) {
                        flag = true;
                        break;
                    }
                }
            }
            if (flag) {
                break;
            }
        }
        for (int i = 0; i < m; ++i) {
            if (mat[i][0] != -1) {
                if (cnt == k) {
                    break;
                }
                ret[cnt++] = i;
            }
        }
        return ret;
    }
}
```

TODO 二分，优先队列等

## [581. 最短无序连续子数组](https://leetcode-cn.com/problems/shortest-unsorted-continuous-subarray/)

> 排序，双指针，数组

排序法

```java
class Solution {
    public int findUnsortedSubarray(int[] nums) {
        int[] arr = nums.clone(); // 拷贝数组的方法，另一种是System.arraycopy(A,0,B,0,n)
        Arrays.sort(arr);
        int n = nums.length, l = 0, r = n - 1;
        while (l <= r && nums[l] == arr[l]) l++;
        while (l <= r && nums[r] == arr[r]) r--;
        return r - l + 1;
    }
}
```

TODO 双指针法

## [1137. 第 N 个泰波那契数](https://leetcode-cn.com/problems/n-th-tribonacci-number/)

> 记忆化搜索，数学，动态规划，滚动数组

滚动数组

```java
class Solution {
    public int tribonacci(int n) {
        int[] ans = {0, 1, 1, 2};
        if (n <= 3) {
            return ans[n];
        }
        int ret = 0;
        for (int i = 4; i <= n; ++i) {
            int index = i % 4;
            ans[index] = ans[(index + 1) % 4] + ans[(index + 2) % 4] + ans[(index + 3) % 4];
            ret = ans[index];
        }
        return ret;
    }
}
```

矩阵快速幂 TODO

## [413. 等差数列划分](https://leetcode-cn.com/problems/arithmetic-slices/)

> 数组，动态规划

执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：36.1 MB, 在所有 Java 提交中击败了74.44%的用户

```java
class Solution {
    public int numberOfArithmeticSlices(int[] nums) {
        int res = 0;
        int n = nums.length;
        if (n < 3)
            return 0;
        int tmp = 2;
        for (int i = 2; i < n; ++i) {
            if (nums[i] - nums[i - 1] == nums[i - 1] - nums[i - 2]) {
                tmp++;
            } else {
                if (tmp != 2) {
                    res += (tmp - 1) * (tmp - 2) / 2;
                    tmp = 2;
                }
            }
        }
        if (tmp != 2)
            res += (tmp - 1) * (tmp - 2) / 2;
        return res;
    }
}
```

简短写法

```java
class Solution {
    public int numberOfArithmeticSlices(int[] nums) {
        int n = nums.length, l = 0, ans = 0;
        for(int i = 2; i < n; i++) {
            if (nums[i] - nums[i - 1] == nums[i - 1] - nums[i-2])
                ans += ++l;
            else
                l = 0;
        }
        return ans;
    }
}
```

## [446. 等差数列划分 II - 子序列](https://leetcode-cn.com/problems/arithmetic-slices-ii-subsequence/)

> 数组，动态规划，序列DP

Map数组

```java
class Solution {
    public int numberOfArithmeticSlices(int[] nums) {
        // dp[i][d]表示以i为尾项，公差为d的序列的数目
        // dp[i][d]=sum(dp[j][d]+1)，j<i&&dp[i]-dp[j]=d(即i可以直接接在j后面构成等差数列)
        // dp[i][d]=0;
        // res = 所有dp的和
        // 优化：因为第二维范围太大稀疏，所以用map来代替
        int n = nums.length;
        int res = 0;
        Map<Long, Integer>[] dp = new Map[n]; // 妙 <子序列差值，子序列数列>
        for (int i = 0; i < n; ++i) {
            dp[i] = new HashMap<>();
        }
        for (int i = 1; i < n; ++i) {
            for (int j = 0; j < i; ++j) { // 遍历i之前的
                long d = 1L * nums[i] - nums[j];
                int dp_j_d = dp[j].getOrDefault(d, 0); // dp[j][d]
                res += dp_j_d;
                dp[i].put(d, dp[i].getOrDefault(d, 0) + dp_j_d + 1);
            }
        }
        return res;
    }
}
```

记录末两位数

```java
class Solution {
    public int numberOfArithmeticSlices(int[] nums) {
        // dp[i][j]表示nums[i],num[j]为末尾两个数组成的序列的个数
        // dp[j][k]=sum(dp[i][j]+1)，i<j<k&&dp[j]-dp[i]=dp[k]-dp[j]，即nums[i]，nums[j]，num[k]组成连续等差数列
        // dp[i][j]=0;
        // res = 所有dp的和
        int n = nums.length;
        if (n < 3)
            return 0;
        int res = 0;
        int[][] dp = new int[n][n];
        // 可以缓存所有的相同值的位置，方便找k
        Map<Integer, List<Integer>> map = new HashMap<>(); // <数的值，数的位置>
        for (int i = 0; i < n; ++i) {
            List<Integer> list = map.getOrDefault(nums[i], new ArrayList<>());
            list.add(i);
            map.put(nums[i], list);
        }
        for (int k = 1; k < n; ++k) {
            for (int j = 0; j < k; ++j) {
                long iVal = 2L * nums[j] - nums[k];
                if (iVal > Integer.MAX_VALUE || iVal < Integer.MIN_VALUE) // 超出范围的不考虑
                    continue;
                if (map.containsKey((int)iVal)) { // 注意强转
                    List<Integer> allI = map.get((int)iVal);
                    for (int i: allI) {
                        if (i < j) {
                            dp[j][k] += dp[i][j] + 1;
                        }
                    }
                }
                res += dp[j][k];
            }
        }
        return res;
    }
}
```

## [1929. 数组串联](https://leetcode-cn.com/problems/concatenation-of-array/)

> 数组

```java
class Solution {
    public int[] getConcatenation(int[] nums) {
        int n = nums.length;
        int[] ans = new int[2 * n];
        for (int i = 0; i < n; ++i) {
            ans[i] = nums[i];
            ans[n + i] = nums[i];
        }
        return ans;
    }
}
```

arraycopy函数

```java
class Solution {
    public int[] getConcatenation(int[] nums) {
        int n = nums.length;
        int[] ans = new int[2 * n];
        System.arraycopy(nums, 0, ans, 0, n);
        System.arraycopy(nums, 0, ans, n, n);
        return ans;
    }
}
```

## [1583. 统计不开心的朋友](https://leetcode-cn.com/problems/count-unhappy-friends/)

> 数组，模拟

两个Map模拟一下

```java
class Solution {
    public int unhappyFriends(int n, int[][] preferences, int[][] pairs) {
        Map<String, Integer> prefCache = new HashMap<>();
        Map<Integer, Integer> pairCache = new HashMap<>();
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n - 1; ++j) {
                int friend = preferences[i][j];
                prefCache.put(i + "," + friend, j);
            }
        }
        for (int[] pair: pairs) {
            pairCache.put(pair[0], pair[1]);
            pairCache.put(pair[1], pair[0]);
        }
        int ans = 0;
        for (int x = 0; x < n; ++x) {
            int y = pairCache.get(x);
            int yi = prefCache.get(x + "," + y);
            for (int ui = 0; ui < yi; ++ui) { // 更亲密的朋友们
                int u = preferences[x][ui];
                int score = prefCache.get(u + "," + x);
                int v = pairCache.get(u);
                int score2 = prefCache.get(u + "," + v);
                if (score < score2) {
                    ans++;
                    break;
                }
            }
        }
        return ans;
    }
}
```

TODO 其实用数组就行

## [551. 学生出勤记录 I](https://leetcode-cn.com/problems/student-attendance-record-i/)

> 字符串

```java
class Solution {
    public boolean checkRecord(String s) {
        int countA = 0, countL = 0;
        for (int i = 0; i < s.length(); ++i) {
            if (s.charAt(i) == 'A') {
                countA++;
                if (countA > 1) {
                    return false;
                }
            }
            if (s.charAt(i) == 'L') {
                countL++;
                if (countL >= 3)
                    return false;
            } else {
                countL = 0;
            }
        }
        return true;
    }
}
```

## [345. 反转字符串中的元音字母](https://leetcode-cn.com/problems/reverse-vowels-of-a-string/)

> 双指针，字符串

```java
class Solution {
    public String reverseVowels(String s) {
        int l = 0, r = s.length() - 1;
        char[] chars = s.toCharArray();
        char[] allVowel = {'a', 'e', 'i', 'o', 'u'};
        Set<Character> cs = new HashSet<>();
        for (char c: allVowel) {
            cs.add(c);
            cs.add(Character.toUpperCase(c));
        }
        while (l < r) {
            while (l < r && !cs.contains(s.charAt(l))) l++;
            while (r > l && !cs.contains(s.charAt(r))) r--;
            swap(chars, l, r);
            l++;
            r--;
        }
        return new String(chars);
    }

    private void swap(char[] chars, int l, int r) {
        char tmp = chars[l];
        chars[l] = chars[r];
        chars[r] = tmp;
    }
}
```

## [541. 反转字符串 II](https://leetcode-cn.com/problems/reverse-string-ii/)

> 双指针，字符串

```java
class Solution {
    public String reverseStr(String s, int k) {
        int kk = 2 * k;
        int l = 0, r = kk - 1;
        int n = s.length();
        char[] cs = s.toCharArray();
        while (r < n) {
            reverse(cs, l, l + k - 1);
            l += kk;
            r = l + kk - 1;
        }
        if (l + k - 1 < n)
            reverse(cs, l, l + k - 1);
        else
            reverse(cs, l, n - 1);
        return new String(cs);
    }

    private void reverse(char[] cs, int l, int r) {
        while (l < r) {
            swap(cs, l++, r--);
        }
    }

    private void swap(char[] cs, int a, int b) {
        char tmp = cs[a];
        cs[a] = cs[b];
        cs[b] = tmp;
    }
}
```

简洁版

```java
class Solution {
    public String reverseStr(String s, int k) {
        int n = s.length();
        char[] arr = s.toCharArray();
        for (int i = 0; i < n; i += 2 * k) {
            reverse(arr, i, Math.min(i + k, n) - 1);
        }
        return new String(arr);
    }

    public void reverse(char[] arr, int left, int right) {
        while (left < right) {
            char temp = arr[left];
            arr[left] = arr[right];
            arr[right] = temp;
            left++;
            right--;
        }
    }
}
```

## [443. 压缩字符串](https://leetcode-cn.com/problems/string-compression/)

> 双指针，字符串

双指针原地算法 TODO可以优化

```java
class Solution {
    public int compress(char[] chars) {
        int l = 0, r = 1;
        int cnt = 1;
        int n = chars.length;
        while (r < n) {
            if (chars[r] == chars[r - 1]) { // 相等
                cnt++;
            } else if (cnt != 1) { // 不相等且之前好几个
                l = moveChar(l, chars[r - 1], chars, cnt + "") + 1;
                cnt = 1;
            } else { // 不相等且前面有一个
                chars[l] = chars[r - 1];
                l++;
            }
            r++;
        }
        if (cnt != 1) {
            l = moveChar(l, chars[r - 1], chars, cnt + "") + 1;
        } else {
            chars[l] = chars[r - 1];
            l++;
        }
        return l;
    }

    private int moveChar(int l, char x, char[] chars, String count) {
        char[] nChar = count.toCharArray();
        int i = 0;
        while (i < nChar.length) {
            chars[l + i + 1] = nChar[i];
            i++;
        }
        chars[l] = x;
        l += nChar.length;
        return l;
    }
}
```

## [789. 逃脱阻碍者](https://leetcode-cn.com/problems/escape-the-ghosts/)

> 数组，数学，脑筋急转弯，曼哈顿距离

```java
class Solution {
    public boolean escapeGhosts(int[][] ghosts, int[] target) {
        int[] start = {0, 0};
        int targetDistance = getManhattanDistance(start, target);
        for (int i = 0; i < ghosts.length; ++i) {
            if (getManhattanDistance(ghosts[i], target) <= targetDistance)
                return false;
        }
        return true;
    }

    private int getManhattanDistance(int[] start, int[] pos) {
        return Math.abs(pos[0] - start[0]) + Math.abs(pos[1] - start[1]);
    }
}
```

## [1646. 获取生成数组中的最大值](https://leetcode-cn.com/problems/get-maximum-in-generated-array/)

> 数组，模拟

执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：35.1 MB, 在所有 Java 提交中击败了82.52%的用户

```java
class Solution {
    public int getMaximumGenerated(int n) {
        if (n <= 1) {
            return n;
        }
        int[] nums = new int[n + 1];
        nums[1] = 1;
        int max = 0;
        for (int i = 2; i <= n; ++i) {
            if ((i & 1) == 0) {
                nums[i] = nums[i / 2];
            } else {
                nums[i] = nums[(i - 1) / 2] + nums[(i - 1) / 2 + 1];
            }
            max = Math.max(max, nums[i]);
        }
        return max;
    }
}
```

合并（性能不高）

```java
class Solution {
    public int getMaximumGenerated(int n) {
        if (n == 0) {
            return 0;
        }
        int[] nums = new int[n + 1];
        nums[1] = 1;
        for (int i = 2; i <= n; ++i) {
            nums[i] = nums[i / 2] + i % 2 * nums[i / 2 + 1];
        }
        return Arrays.stream(nums).max().getAsInt();
    }
}
```

## [787. K 站中转内最便宜的航班](https://leetcode-cn.com/problems/cheapest-flights-within-k-stops/)

> 动态规划，有向图

```java
class Solution {
    public int findCheapestPrice(int n, int[][] flights, int src, int dst, int k) {
        // dp[t][i]表示恰好t次到i最小花销
        // dp[t][i]=min(dp[t-1][j]+cost[j][i]) j->i
        // dp[0][i]=INF(dp[0][dst]=0)其他点直接到dst的距离为无穷大，dst上的点到dst距离为0
        // 结果为min(dp[1][dst], dp[2][dst],...,dp[k][dst])
        final int INF = 10000 * 101 + 1;
        int min = Integer.MAX_VALUE;
        int[][] dp = new int[k + 2][n]; // 注意k表示中转，k+1才表示步数
        for (int t = 0; t < k + 2; ++t) {
            Arrays.fill(dp[t], INF);
        }
        dp[0][src] = 0; // 0步，自己到自己是0
        for (int t = 1; t < k + 2; ++t) {
            for (int[] flight: flights) {
                int i = flight[1], j = flight[0], cost = flight[2];
                dp[t][i] = Math.min(dp[t][i], dp[t - 1][j] + cost);
            }
        }
        int ans = INF;
        for (int t = 1; t < k + 2; ++t) {
            ans = Math.min(ans, dp[t][dst]);
        }
        return ans == INF ? -1 : ans;
    }
}
```

## [797. 所有可能的路径](https://leetcode-cn.com/problems/all-paths-from-source-to-target/)

> 广度优先搜索，深度优先搜索，图，回溯

BFS

```java
class Solution {
    private List<List<Integer>> ret;
    private int n;

    public List<List<Integer>> allPathsSourceTarget(int[][] graph) {
        n = graph.length;
        ret = new ArrayList<>();
        List<Integer> first = new ArrayList<>();
        first.add(0);
        bfs(graph, 0, first);
        return ret;
    }

    private void bfs(int[][] graph, int cur, List<Integer> list) {
        if (cur == n - 1) {
            ret.add(list);
            return;
        }
        if (graph[cur].length == 0) {
            return;
        }
        for(int i: graph[cur]) {
            List<Integer> newList = deepcopy(list);
            newList.add(i);
            bfs(graph, i, newList);
        }
    }

    private List<Integer> deepcopy(List<Integer> sourceList) {
        List<Integer> destList = new ArrayList<>();
        for (Integer i: sourceList) {
            destList.add(i);
        }
        return destList;
    }
}
```

DFS

```java
class Solution {
    private List<List<Integer>> ret = new ArrayList<>();
    private Deque<Integer> stack = new LinkedList<>();

    public List<List<Integer>> allPathsSourceTarget(int[][] graph) {
        stack.offerLast(0);
        dfs(graph, 0, graph.length - 1);
        return ret;
    }

    private void dfs(int[][] graph, int cur, int target) {
        if (cur == target) {
            ret.add(new ArrayList<>(stack));
            return;
        }
        for (int i: graph[cur]) {
            stack.offerLast(i);
            dfs(graph, i, target);
            stack.pollLast();
        }
    }
}
```

## [881. 救生艇](https://leetcode-cn.com/problems/boats-to-save-people/)

> 贪心，数组，排序，双指针

```java
class Solution {
    public int numRescueBoats(int[] people, int limit) {
        int n = people.length;
        int ans = n;
        Arrays.sort(people);
        int l = 0;
        for (int r = n - 1; r > l; --r) {
            if (limit - people[l] >= people[r]) {
                l++;
                ans--;
            }
        }
        return ans;
    }
}
```

另一种写法

```java
class Solution {
    public int numRescueBoats(int[] people, int limit) {
        int n = people.length;
        int ans = n;
        Arrays.sort(people);
        int l = 0, r = n - 1;
        while (l < r) {
            if (limit - people[l] >= people[r]) {
                l++;
                ans--;
            }
            r--;
        }
        return ans;
    }
}
```

## [182. 查找重复的电子邮箱](https://leetcode-cn.com/problems/duplicate-emails/)

> 数据库

```sql
SELECT Email FROM Person GROUP BY Email HAVING COUNT(Email) > 1
```

## [94. 二叉树的中序遍历](https://leetcode-cn.com/problems/binary-tree-inorder-traversal/)

> 栈，树，深度优先搜索，二叉树

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
    private List<Integer> ret = new ArrayList<>();

    public List<Integer> inorderTraversal(TreeNode root) {
        dfs(root);
        return ret;
    }

    private void dfs(TreeNode root) {
        if (root == null) {
            return;
        }
        dfs(root.left);
        ret.add(root.val);
        dfs(root.right);
    }
}
```

迭代Deque，Morris 中序遍历 TODO

## [1920. 基于排列构建数组](https://leetcode-cn.com/problems/build-array-from-permutation/)

> 数组，模拟

```java
class Solution {
    public int[] buildArray(int[] nums) {
        int n = nums.length;
        int[] ans = new int[n];
        for (int i = 0; i < n; ++i) {
            ans[i] = nums[nums[i]];
        }
        return ans;
    }
}
```

## [1863. 找出所有子集的异或总和再求和](https://leetcode-cn.com/problems/sum-of-all-subset-xor-totals/)

> 位运算，数组，回溯

list添加

```java
class Solution {
    public int subsetXORSum(int[] nums) {
        ArrayList<Integer> list = new ArrayList<>();
        list.add(0);
        for (int i: nums) {
            ArrayList<Integer> list2 = new ArrayList<>();
            for (int j: list) {
                list2.add(j ^ i);
            }
            list.addAll(list2);
        }
        return list.stream().reduce(Integer::sum).orElse(0);
    }
}
```

位运算

```java
class Solution {
    public int subsetXORSum(int[] nums) {
        int n = nums.length;
        int cur = (1 << n) - 1;
        int ans = 0;
        for (int i = cur; i > 0; --i) {
            int tmp = 0;
            for (int j = 0; j < n; ++j) {
                if ((i & (1 << j)) > 0) {
                    tmp ^= nums[j];
                }
            }
            ans += tmp;
        }
        return ans;
    }
}
```

数学，按位考虑(O(n)) TODO

```java
class Solution {
    public int subsetXORSum(int[] nums) {
        int tmp = 0;
        for (int i: nums)
            tmp |= i;
        return tmp << (nums.length - 1);
    }
}
```

## [1913. 两个数对之间的最大乘积差](https://leetcode-cn.com/problems/maximum-product-difference-between-two-pairs/)

> 数组，排序，贪心

```java
class Solution {
    public int maxProductDifference(int[] nums) {
        int tmp[] = new int[4];
        System.arraycopy(nums, 0, tmp, 0, 4);
        Arrays.sort(tmp);
        int n = nums.length;
        for (int i = 4; i < n; ++i) {
            if (nums[i] > tmp[3]) {
                tmp[2] = tmp[3];
                tmp[3] = nums[i];
            } else if (nums[i] < tmp[0]) {
                tmp[1] = tmp[0];
                tmp[0] = nums[i];
            } else if (nums[i] > tmp[2]) {
                tmp[2] = nums[i];
            } else if (nums[i] < tmp[1]) {
                tmp[1] = nums[i];
            }
        }
        return tmp[2] * tmp[3] - tmp[0] * tmp[1];
    }
}
```

效率稍高

```java
class Solution {
    public int maxProductDifference(int[] nums) {
        int n = nums.length;
        // 数组中最大的两个值
        int mx1 = Math.max(nums[0], nums[1]);
        int mx2 = Math.min(nums[0], nums[1]);
        // 数组中最小的两个值
        int mn1 = mx2;
        int mn2 = mx1;
        for (int i = 2; i < n; ++i){
            int tmp = nums[i];
            if (tmp > mx1){
                mx2 = mx1;
                mx1 = tmp;
            }
            else if (tmp > mx2){
                mx2 = tmp;
            }
            if (tmp < mn1){
                mn2 = mn1;
                mn1 = tmp;
            }
            else if (tmp < mn2){
                mn2 = tmp;
            }
        }
        return (mx1 * mx2) - (mn1 * mn2);
    }
}
```

## [528. 按权重随机选择](https://leetcode-cn.com/problems/random-pick-with-weight/)

> 数学，二分查找，随机化，前缀和

```java
class Solution {
    private int[] pSum;
    private int n;

    public Solution(int[] w) {
        n = w.length;
        pSum = new int[n];
        pSum[0] = w[0];
        for (int i = 1; i < n; ++i) {
            pSum[i] = pSum[i - 1] + w[i];
        }
    }

    public int pickIndex() {
        int x = (int)(Math.random() * pSum[n - 1]) + 1;
        return bs(x);
    }

    private int bs(int target) {
        int l = 0, r = n - 1;
        while (l < r) { // 注意边界
            int mid = l + (r - l) / 2;
            if (pSum[mid] > target) {
                r = mid;
            } else if (pSum[mid] < target) {
                l = mid + 1;
            } else {
                return mid;
            }
        }
        return l;
    }
}

/**
 * Your Solution object will be instantiated and called as such:
 * Solution obj = new Solution(w);
 * int param_1 = obj.pickIndex();
 */
```

## [1109. 航班预订统计](https://leetcode-cn.com/problems/corporate-flight-bookings/)

> 数组，差分数组

暴力

```java
class Solution {
    public int[] corpFlightBookings(int[][] bookings, int n) {
        int[] res = new int[n];
        for (int[] booking: bookings) {
            for (int i = booking[0]; i <= booking[1]; ++i) {
                res[i - 1] += booking[2];
            }
        }
        return res;
    }
}
```

差分数组

```java
class Solution {
    public int[] corpFlightBookings(int[][] bookings, int n) {
        int[] arr = new int[n];
        for (int[] booking: bookings) {
            arr[booking[0] - 1] += booking[2];
            if (booking[1] < n)
                arr[booking[1]] += -booking[2];
        }
        int[] res = new int[n];
        res[0] = arr[0];
        for (int i = 1; i < n; ++i) {
            res[i] = res[i - 1] + arr[i];
        }
        return res;
    }
}
```

## [165. 比较版本号](https://leetcode-cn.com/problems/compare-version-numbers/)

> 双指针，字符串

字符串分割

```java
class Solution {
    public int compareVersion(String version1, String version2) {
        String[] vs1 = version1.split("\\.");
        String[] vs2 = version2.split("\\.");
        if (vs1.length < vs2.length) return -compareVersion(version2, version1);
        for (int i = 0; i < vs1.length; ++i) {
            int a = Integer.parseInt(vs1[i]);
            int b = vs2.length <= i ? 0 : Integer.parseInt(vs2[i]);
            if (a != b) return a > b ? 1 : -1;
        }
        return 0;
    }
}
```

TODO 双指针

## [剑指 Offer 22. 链表中倒数第k个节点](https://leetcode-cn.com/problems/lian-biao-zhong-dao-shu-di-kge-jie-dian-lcof/)

> 链表，双指针

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
    public ListNode getKthFromEnd(ListNode head, int k) {
        ListNode fast = head, slow = head;
        while (k-- > 0) {
            fast = fast.next;
        }
        while (fast != null) {
            fast = fast.next;
            slow = slow.next;
        }
        return slow;
    }
}
```

TODO 顺序查找

## [面试题 17.14. 最小K个数](https://leetcode-cn.com/problems/smallest-k-lcci/)

> 堆，优先队列，数组，排序

优先队列（小根堆）

```java
class Solution {
    public int[] smallestK(int[] arr, int k) {
        PriorityQueue<Integer> pq = new PriorityQueue<>();
        int currentMax = Integer.MIN_VALUE;
        for (int i: arr) {
            pq.offer(i);
        }
        int[] ret = new int[k];
        for (int i = 0; i < k; ++i) {
            ret[i] = pq.poll();
        }
        return ret;
    }
}
```

排序

```java
class Solution {
    public int[] smallestK(int[] arr, int k) {
        Arrays.sort(arr);
        int[] ret = new int[k];
        System.arraycopy(arr, 0, ret, 0, k);
        return ret;
    }
}
```

有限队列（大根堆）

```java
class Solution {
    public int[] smallestK(int[] arr, int k) {
        PriorityQueue<Integer> pq = new PriorityQueue<>((x, y) -> y - x);
        int[] ret = new int[k];
        if (k == 0)
            return ret;
        for (int i: arr) {
            if (pq.size() >= k) {
                if (pq.peek() > i) {
                    pq.poll();
                    pq.offer(i);
                }
            } else {
                pq.offer(i);
            }
        }
        for (int i = 0; i < k; ++i)
            ret[i] = pq.poll();
        return ret;
    }
}
```

TODO 快速选择

## [剑指 Offer 10- I. 斐波那契数列](https://leetcode-cn.com/problems/fei-bo-na-qi-shu-lie-lcof/)

> 动态规划，数学，矩阵快速幂，打表

动态规划

```java
class Solution {
    private static final int MOD = 1000000007;

    public int fib(int n) {
        if (n <= 1)
            return n;
        int ans = 1, fn1 = 1, fn2 = 0;
        for (int i = 2; i <= n; ++i) {
            ans = (fn1 + fn2) % MOD; // 提前求模
            fn2 = fn1;
            fn1 = ans;
        }
        return ans;
    }
}
```

打表

```java
class Solution {
    static int mod = (int)1e9+7; // 注意这种写法
    static int N = 110;
    static int[] cache = new int[N];
    static { // 静态块可以用来打表
        cache[1] = 1;
        for (int i = 2; i < N; i++) {
            cache[i] = cache[i - 1] + cache[i - 2];
            cache[i] %= mod;
        }
    }
    public int fib(int n) {
        return cache[n];
    }
}
```

矩阵快速幂 TODO

```java
class Solution {
    static final int MOD = (int)1e9+7;

    public int fib(int n) {
        if (n < 2)
            return n;
        int[][] q = {{1, 1}, {1, 0}};
        int[][] res = pow(q, n - 1);
        return res[0][0];
    }

    public int[][] pow(int[][] a, int n) {
        int[][] ret = {{1, 0}, {0, 1}};
        while (n > 0) {
            if ((n & 1) == 1) { // 幂二进制为1的时候才乘
                ret = multiply(ret, a);
            }
            n >>= 1;
            a = multiply(a, a);
        }
        return ret;
    }

    public int[][] multiply(int[][] a, int[][] b) {
        int[][] c = new int[2][2];
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                c[i][j] = (int) (((long) a[i][0] * b[0][j] + (long) a[i][1] * b[1][j]) % MOD);
            }
        }
        return c;
    }
}
```

## [470. 用 Rand7() 实现 Rand10()](https://leetcode-cn.com/problems/implement-rand10-using-rand7/)

> 数学，拒绝采样，概率统计，随机化

拒绝采样

```java
/**
 * The rand7() API is already defined in the parent class SolBase.
 * public int rand7();
 * @return a random integer in the range 1 to 7
 */
 // 次数期望=每次的次数*(拒绝概率^0+拒绝概率^1+...+拒绝概率^∞)=每次的次数/(1-拒绝概率)
 // 通过不同位构造独立两个分布扩大同概率的数区间，然后用拒绝采样筛选
class Solution extends SolBase {
    public int rand10() {
        while (true) {
            int tmp = (rand7() - 1) * 7 + (rand7() - 1); // 转化为2位7进制
            if (tmp > 0 && tmp <= 10) // 拒绝
                return tmp;
        }
    }
}
```

减小while次数

```java
class Solution extends SolBase {
    public int rand10() {
        while (true) {
            int tmp = (rand7() - 1) * 7 + (rand7() - 1); // 转化为2位7进制
            if (tmp > 0 && tmp <= 40) // 拒绝
                return tmp % 10 + 1;
        }
    }
}
```

二进制映射+拒绝采样

```java
class Solution extends SolBase {
    public int rand10() {
        while (true) {
            int x = rand7() * 10 + rand7(); // 转化为2位十进制一共49中组合，取40个四个一组映射到1-10
            if (x == 11 || x == 12 || x == 13 || x == 14)
                return 1;
            else if (x == 15 || x == 16 || x == 17 || x == 21)
                return 2;
            else if (x == 22 || x == 23 || x == 24 || x == 25)
                return 3;
            else if (x == 26 || x == 27 || x == 31 || x == 32)
                return 4;
            else if (x == 33 || x == 34 || x == 35 || x == 36)
                return 5;
            else if (x == 37 || x == 41 || x == 42 || x == 43)
                return 6;
            else if (x == 44 || x == 45 || x == 46 || x == 47)
                return 7;
            else if (x == 51 || x == 52 || x == 53 || x == 54)
                return 8;
            else if (x == 55 || x == 56 || x == 57 || x == 61)
                return 9;
            else if (x == 62 || x == 63 || x == 64 || x == 65)
                return 10;
        }
    }
}
```

## [704. 二分查找](https://leetcode-cn.com/problems/binary-search/)

> 数组，二分查找

```java
class Solution {
    public int search(int[] nums, int target) {
        int n = nums.length;
        int l = 0, r = n - 1;
        while (l <= r) {
            int mid = l + (r - l) / 2;
            if (nums[mid] == target) {
                return mid;
            } else if (nums[mid] < target) {
                l = mid + 1;
            } else {
                r = mid - 1;
            }
        }
        return -1;
        // return l < n && nums[l] == target ? l : -1;
    }
}
```

## [1221. 分割平衡字符串](https://leetcode-cn.com/problems/split-a-string-in-balanced-strings/)

> 贪心，字符串，计数

```java
class Solution {
    public int balancedStringSplit(String s) {
        int res = 0, tmp = 0;
        for (char c: s.toCharArray()) {
            tmp = c == 'R' ? tmp + 1 : tmp - 1;
            if(tmp == 0) {
                res++;
            }
        }
        return res;
    }
}
```

## [68. 文本左右对齐](https://leetcode-cn.com/problems/text-justification/)

> 字符串，模拟

执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：36.5 MB, 在所有 Java 提交中击败了93.17%的用户

```java
class Solution {
    public List<String> fullJustify(String[] words, int maxWidth) {
        int rest = maxWidth; // 每一行带至少一个空格的剩余的长度
        int blankNum = maxWidth; // 每一行空格的数量
        List<String> strs = new ArrayList<>(); // 每一行的临时队列
        int index = 0, n = words.length;
        List<String> res = new ArrayList<>();
        while (index < n) {
            String curStr = words[index]; // 当前字符串
            int curLen = curStr.length(); // 当前字符串长度
            if (rest < curLen) { // 剩下的位置不够放
                // 将之前的处理为字符串
                res.add(convert(strs, blankNum, maxWidth, false));
                // 预备下一行
                strs.clear();
                rest = maxWidth;
                blankNum = maxWidth;
            }
            rest -= curLen + 1;
            blankNum -= curLen;
            strs.add(curStr);
            index++;
        }
        if (strs.size() > 0)
            res.add(convert(strs, blankNum, maxWidth, true));
        return res;
    }

    private String convert(List<String> strs, int blankNum, int maxWidth, boolean isLast) {
        StringBuilder sb = new StringBuilder();
        int strNum = strs.size();
        if (isLast || strNum == 1) { // 最后一行的处理或这一行只有一个单词
            for (int i = 0; i < strNum; ++i) {
                if (i == strNum - 1) {
                    sb.append(strs.get(i));
                } else {
                    sb.append(strs.get(i)).append(' ');
                }
            }
            int restLen = maxWidth - sb.length();
            for (int i = 0; i < restLen; ++i) {
                sb.append(' ');
            }
        } else { // 非最后一行处理
            int blankAvg = blankNum / (strNum - 1); // 每个单词间最少隔几个
            int restBlank = blankNum % (strNum - 1); // 有几个是要多加空格的
            for (int i = 0; i < strNum; ++i) {
                sb.append(strs.get(i));
                if (i != strNum - 1) {
                    for (int j = 0; j < blankAvg; ++j) {
                        sb.append(' ');
                    }
                    if (restBlank > 0) {
                        restBlank--;
                        sb.append(' ');
                    }
                }
            }
        }
        return sb.toString();
    }
}
```

## [1894. 找到需要补充粉笔的学生编号](https://leetcode-cn.com/problems/find-the-student-that-will-replace-the-chalk/)

> 模拟

模拟

```java
class Solution {
    public int chalkReplacer(int[] chalk, int k) {
        int i = 0, n = chalk.length;
        while (true) {
            if (k < chalk[i]) break;
            k -= chalk[i];
            i++;
            if (i == n) i = 0;
        }
        return i;
    }
}
```

优化模拟

```java
class Solution {
    public int chalkReplacer(int[] chalk, int k) {
        int i = 0, n = chalk.length;
        long sum = 0;
        for (int j: chalk) {
            sum += j;
        }
        k %= sum; // 取模
        while (true) {
            if (k < chalk[i]) break;
            k -= chalk[i];
            i++;
        }
        return i;
    }
}
```

前缀和+二分查找

```java
class Solution {
    public int chalkReplacer(int[] chalk, int k) {
        int n = chalk.length, sum = 0;
        if (chalk[0] > k) return 0;
        for (int i = 1; i < n; ++i) { // 求前缀和
            chalk[i] += chalk[i - 1];
            if (chalk[i] > k)
                return i;
        }
        k %= chalk[n - 1];
        return bs(chalk, k);
    }

    private int bs(int[] arr, int k) {
        int l = 0, r = arr.length - 1; // [l,r]
        while (l < r) { // [l,r],不平衡查找(mid,mid+1),l==r时无效,所以用<
            int mid = l + (r - l) / 2;
            if (arr[mid] > k) {
                r = mid;
            } else {
                l = mid + 1;
            }
        }
        return l;
    }
}
```

## [600. 不含连续1的非负整数](https://leetcode-cn.com/problems/non-negative-integers-without-consecutive-ones/)

> 动态规划，二进制，01字典树，数位DP

```java
class Solution {
    public int findIntegers(int n) {
        // dp[i]表示二进制为100...0(i位)时包含的非连续1的个数
        // dp[i]=dp[i-1]+dp[i-2]斐波那契数列
        // dp[0]=1,dp[1]=2(0,1);dp[2]=3(0,1,10);dp[3]=5(0,1,10,100,101)
        int[] dp = new int[32];
        dp[0] = 1;
        dp[1] = 2;
        dp[2] = 3;
        for (int i = 3; i < 32; ++i) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        int cnt = 0;
        String str = convertStr(n);
        int len = str.length();
        // 1001100(1000000,1000,100(x))
        for (int i = 0; i < len; ++i) {
            if (str.charAt(i) == '0')
                continue;
            cnt += dp[len - i - 1];
            if(i != 0 && str.charAt(i - 1) == '1') // 往后不满足
                return cnt;
        }
        return cnt + 1;
    }

    private String convertStr(int x) {
        StringBuilder sb = new StringBuilder();
        while (x > 0) {
            sb.insert(0, x & 1);
            x >>= 1;
        }
        return sb.toString();
    }
}
```

字典树TODO

## [678. 有效的括号字符串](https://leetcode-cn.com/problems/valid-parenthesis-string/)

> 栈，贪心，字符串，动态规划

动态规划（字符串常用dp_i_j）

```java
class Solution {
    public boolean checkValidString(String s) {
        // dp[i][j]表示i到j是否为有效字段
        // dp[i][j]=true(i<j且满足要求)；true(j-i>=2=>dp[i+1][j-1]或存在dp[i][k]且dp[k+1][j]=true)
        // dp[i][i]=true("*");dp[i-1][i]=true("()","*)","(*","**")
        // dp[0][n-1]
        int n = s.length();
        boolean[][] dp = new boolean[n][n];
        // 初始化1个
        for (int i = 0; i < n; ++i) {
            if (s.charAt(i) == '*') {
                dp[i][i] = true;
            }
        }
        // 初始化2个
        for (int i = 1; i < n; ++i) {
            char lc = s.charAt(i - 1), rc = s.charAt(i);
            dp[i - 1][i] = (lc == '(' || lc == '*') && (rc == ')' || rc == '*');
        }
        // 3个及以上
        for (int i = n - 3; i >= 0; --i) {
            char lc = s.charAt(i);
            for (int j = i + 2; j < n; ++j) {
                char rc = s.charAt(j);
                if ((lc == '(' || lc == '*') && (rc == ')' || rc == '*')) // 两侧满足
                    dp[i][j] = dp[i + 1][j - 1];
                for (int k = i; k < j && !dp[i][j]; ++k) // 分成两段都满足
                    dp[i][j] = dp[i][k] && dp[k + 1][j];
            }
        }
        return dp[0][n - 1];
    }
}
```

栈

```java
class Solution {
    public boolean checkValidString(String s) {
        // 栈：左括号栈+星号栈
        // c为(或者*则入栈下标，c为)则优先匹配(，没有则匹配*
        // 结束后左括号栈和星号栈分别弹栈，左小标小于*则匹配成功，最后看左括号栈是否为空，为空则为true，反之false
        Deque<Integer> leftStack = new LinkedList<>();
        Deque<Integer> starStack = new LinkedList<>();
        for (int i = 0; i < s.length(); ++i) {
            char c = s.charAt(i);
            if (c == '(')
                leftStack.push(i);
            else if (c == '*')
                starStack.push(i);
            else {
                if (!leftStack.isEmpty())
                    leftStack.pop();
                else if (!starStack.isEmpty())
                    starStack.pop();
                else
                    return false;
            }
        }
        while (!starStack.isEmpty()) {
            if (leftStack.isEmpty())
                return true;
            else {
                int li = leftStack.pop();
                int si = starStack.pop();
                if (li < si)
                    continue;
                else
                    return false;
            }
        }
        return leftStack.isEmpty();
    }
}
```

贪心

```java
class Solution {
    public boolean checkValidString(String s) {
        // 贪心：[l,r]表示未匹配的左括号的上下限
        // 遇到(: l+1,r+1
        // 遇到): l-1,r-1
        // 遇到*: l-1,r+1
        // 退出情况: r<0（右括号过多，无法拯救）返回false；l为非负
        // 最后l必须为0则true
        int l = 0, r = 0;
        for (char c: s.toCharArray()) {
            if (c == '(') {
                l++;
                r++;
            } else if (c == ')') {
                l = l - 1 < 0 ? 0 : l - 1;
                r--;
                if (r < 0)
                    return false;
            } else {
                l = l - 1 < 0 ? 0 : l - 1;
                r++;
            }
        }
        return l == 0;
    }
}
```

## [447. 回旋镖的数量](https://leetcode-cn.com/problems/number-of-boomerangs/)

> 数组，哈希表，数学，排列组合

```java
class Solution {
    public int numberOfBoomerangs(int[][] points) {
        int n = points.length;
        int res = 0;
        for (int i = 0; i < n; ++i) { // 以每个点为中心
            Map<Integer, Integer> cache = new HashMap<>();
            int[] pointi = points[i];
            for (int j = 0; j < n; ++j) {
                if (i == j) continue;
                int[] pointj = points[j];
                int dis = getDis(pointi, pointj);
                cache.put(dis, cache.getOrDefault(dis, 0) + 1); // 记录距离相等的数量
            }
            for (int value: cache.values()) {
                if (value > 1)
                    res += value * (value - 1); // A(m,2)=m*(m-1)
            }
        }
        return res;
    }

    private int getDis(int[] a, int[] b) {
        return (b[0] - a[0]) * (b[0] - a[0]) + (b[1] - a[1]) * (b[1] - a[1]);
    }
}
```

## [524. 通过删除字母匹配到字典里最长单词](https://leetcode-cn.com/problems/longest-word-in-dictionary-through-deleting/)

> 数组，双指针，字符串，排序，动态规划

双指针

```java
class Solution {
    public String findLongestWord(String s, List<String> dictionary) {
        String ret = "";
        for (String str: dictionary) {
            int n = str.length(), max = ret.length();
            if (n < max)
                continue;
            int p1 = 0, p2 = 0;
            while (p1 < s.length() && p2 < n) {
                if (s.charAt(p1) == str.charAt(p2))
                    p2++;
                p1++;
            }
            if (p2 == n)
                if (n > max || n == max && str.compareTo(ret) < 0)
                    ret = str;
        }
        return ret;
    }
}
```

动态规划优化

```java
class Solution {
    public String findLongestWord(String s, List<String> dictionary) {
        // 非常巧的处理方法
        // dp[i][j]表示s中从i位置从左到右找到j字符的最近位置
        // dp[i][j]=i(s.charAt(i)==j);dp[i][j]=dp[i+1][j](s.charAt(i)<>j)
        // dp[n][j]=n
        int n = s.length();
        int[][] dp = new int[n + 1][26];
        Arrays.fill(dp[n], n);
        for (int i = n - 1; i >= 0; --i) {
            char c = s.charAt(i);
            for (int j = 0; j < 26; ++j) {
                if (c == (char)('a' + j)) {
                    dp[i][j] = i;
                } else {
                    dp[i][j] = dp[i + 1][j];
                }
            }
        }
        String ret = "";
        for (String str: dictionary) {
            int j = 0;
            int slen = str.length();
            boolean flag = true;
            for (int i = 0; i < slen; ++i) {
                char c = str.charAt(i);
                int find = dp[j][c - 'a'];
                if (find == n) { // 找不到
                    flag = false;
                    break;
                }
                j = find + 1;
            }
            if (flag) {
                if (slen > ret.length() || slen == ret.length() && str.compareTo(ret) < 0) {
                    ret = str;
                }
            }
        }
        return ret;
    }
}
```

## [175. 组合两个表](https://leetcode-cn.com/problems/combine-two-tables/)

> 数据库

```sql
select p.FirstName, p.LastName, a.City, a.State from Person as p left join Address as a on p.PersonId = a.PersonId
```

## [1979. 找出数组的最大公约数](https://leetcode-cn.com/problems/find-greatest-common-divisor-of-array/)

> 数组，数学，最大公约数

```java
class Solution {
    public int findGCD(int[] nums) {
        int max = nums[0], min = nums[0];
        for (int i: nums) {
            max = Math.max(max, i);
            min = Math.min(min, i);
        }
        return gcd(max, min);
    }

    private int gcd(int a, int b) {
        return b != 0 ? gcd(b, a % b) : a;
    }
}
```

## [212. 单词搜索 II](https://leetcode-cn.com/problems/word-search-ii/)

> 字典树，数组，字符串，回溯，矩阵

TODO

```java
class Solution {
    int[][] dirs = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

    public List<String> findWords(char[][] board, String[] words) {
        // 插入字典树
        Trie trie = new Trie();
        for (String word : words) {
            trie.insert(word);
        }
        // 结果集去重
        Set<String> ans = new HashSet<String>();
        // 以每个坐标作为起点进行DFS
        for (int i = 0; i < board.length; ++i) {
            for (int j = 0; j < board[0].length; ++j) {
                dfs(board, trie, i, j, ans);
            }
        }
        return new ArrayList<String>(ans);
    }

    public void dfs(char[][] board, Trie cur, int i, int j, Set<String> ans) {
        if (!cur.children.containsKey(board[i][j])) {
            return;
        }
        char c = board[i][j];
        cur = cur.children.get(c);
        if (cur.word != "") {
            ans.add(cur.word);
        }
        board[i][j] = '#';
        // 每个方向往下递归
        for (int[] dir : dirs) {
            int ni = i + dir[0], nj = j + dir[1];
            if (ni >= 0 && ni < board.length && nj >= 0 && nj < board[0].length) { // 非边界
                dfs(board, cur, ni, nj, ans);
            }
        }
        // 回溯
        board[i][j] = c;
    }
}

class Trie {
    String word; // 精髓，判断是否取到
    Map<Character, Trie> children;

    public Trie() {
        this.word = "";
        this.children = new HashMap<Character, Trie>();
    }

    public void insert(String word) {
        Trie cur = this;
        for (int i = 0; i < word.length(); ++i) {
            char c = word.charAt(i);
            if (!cur.children.containsKey(c)) {
                cur.children.put(c, new Trie());
            }
            cur = cur.children.get(c);
        }
        cur.word = word;
    }
}
```

## [36. 有效的数独](https://leetcode-cn.com/problems/valid-sudoku/)

> 数组，哈希表，矩阵，位运算

```java
class Solution {
    public boolean isValidSudoku(char[][] board) {
        // check line
        for (int i = 0; i < 9; ++i) {
            if (!check(board[i])) return false;
        }
        // check row
        for (int j = 0; j < 9; ++j) {
            char[] row = new char[9];
            for (int i = 0; i < 9; ++i) {
                row[i] = board[i][j];
            }
            if (!check(row)) return false;
        }
        // check box
        int[][] dirs = {{-1, -1}, {-1, 0}, {-1, 1},
                        {0, -1}, {0, 0}, {0, 1},
                        {1, -1}, {1, 0}, {1, 1}};
        for (int i = 1; i < 8; i += 3) {
            for (int j = 1; j < 8; j += 3) {
                char[] box = new char[9];
                int index = 0;
                for (int[] dir: dirs) {
                    box[index++] = board[i + dir[0]][j + dir[1]];
                }
                if (!check(box)) return false;
            }
        }
        return true;
    }

    private boolean check(char[] chars) {
        int[] cs = new int[9];
        for (int j = 0; j < 9; ++j) {
            char c = chars[j];
            if (c == '.')
                continue;
            if (cs[c - '1'] == 0) {
                cs[c - '1'] = 1;
            } else {
                return false;
            }
        }
        return true;
    }
}
```

一次遍历 boolean

```java
class Solution {
    public boolean isValidSudoku(char[][] board) {
        boolean[][] row = new boolean[10][10];
        boolean[][] col = new boolean[10][10];
        boolean[][] box = new boolean[10][10];        
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                int c = board[i][j];
                if (c == '.') continue;
                int u = c - '0';
                int idx = i / 3 * 3 + j / 3; // [i,j]->boxId
                if (row[i][u] || col[j][u] || box[idx][u]) return false;
                row[i][u] = col[j][u] = box[idx][u] = true;
            }
        }
        return true;
    }
}
```

一次遍历 int

```java
class Solution {
    public boolean isValidSudoku(char[][] board) {
        int[][] rows = new int[9][9];
        int[][] columns = new int[9][9];
        int[][][] boxs = new int[3][3][9];
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                char c = board[i][j];
                if (c != '.') {
                    int index = c - '0' - 1;
                    rows[i][index]++;
                    columns[j][index]++;
                    boxs[i / 3][j / 3][index]++;
                    if (rows[i][index] > 1 || columns[j][index] > 1 || boxs[i / 3][j / 3][index] > 1) {
                        return false;
                    }
                }
            }
        }
        return true;
    }
}
```

位运算

```java
class Solution {
    public boolean isValidSudoku(char[][] board) {
        int[] row = new int[10], col = new int[10], box = new int[10];
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                char c = board[i][j];
                if (c == '.') continue;
                int u = c - '0';
                int idx = i / 3 * 3 + j / 3;
                if ((((row[i] >> u) & 1) == 1) || (((col[j] >> u) & 1) == 1) || (((box[idx] >> u) & 1) == 1)) return false;
                row[i] |= (1 << u);
                col[j] |= (1 << u);
                box[idx] |= (1 << u);
            }
        }
        return true;
    }
}
```

## [292. Nim 游戏](https://leetcode-cn.com/problems/nim-game/)

> 脑筋急转弯，数学，博弈

```java
class Solution {
    public boolean canWinNim(int n) {
        // 1-3√，4个先手必输，5-7√，8×
        // 博弈论题目要么是特定情况下先手必赢，要么是分先手,后手进行状态转移
        return (n & 3) != 0;
    }
}
```

## [650. 只有两个键的键盘](https://leetcode-cn.com/problems/2-keys-keyboard/)

> 数学，动态规划

动态规划

```java
class Solution {
    public int minSteps(int n) {
        // dp[i]表示第i个数最少操作次数
        // dp[i]=min(dp[j]+i/j)（i % j == 0)
        // dp[1]=0
        // dp[n]
        if (n == 1) return 0;
        int[] dp = new int[n + 1];
        Arrays.fill(dp, Integer.MAX_VALUE);
        dp[1] = 0;
        for (int i = 2; i <= n; ++i) {
            List<Integer> apprs = getApprList(i);
            for (int j: apprs) {
                dp[i] = Math.min(dp[i], dp[j] + i / j);
            }
        }
        return dp[n];
    }

    private List<Integer> getApprList(int x) {
        List<Integer> ret = new ArrayList<>();
        int a = x - 1;
        while (a > 0) {
            if (x % a == 0) {
                ret.add(a);
            }
            a--;
        }
        return ret;
    }
}
```

动态规划（优化）

```java
lass Solution {
    public int minSteps(int n) {
        // dp[i]表示第i个数最少操作次数
        // dp[i]=min(dp[j]+i/j)（i % j == 0)
        // dp[1]=0
        // dp[n]
        int[] dp = new int[n + 1];
        for (int i = 2; i <= n; ++i) {
            dp[i] = Integer.MAX_VALUE;
            for (int j = 1; j * j <= i; ++j) {
                if (i % j == 0) {
                    dp[i] = Math.min(dp[i], dp[j] + i / j);
                    dp[i] = Math.min(dp[i], dp[i / j] + j);
                }
            }
        }
        return dp[n];
    }
}
```

质因数分解 TODO

```java
class Solution {
    public int minSteps(int n) {
        int ans = 0;
        for (int i = 2; i * i <= n; ++i) {
            while (n % i == 0) {
                n /= i;
                ans += i;
            }
        }
        if (n > 1) {
            ans += n;
        }
        return ans;
    }
}
```

## [673. 最长递增子序列的个数](https://leetcode-cn.com/problems/number-of-longest-increasing-subsequence/)

> 树状数组，线段树，数组，动态规划，贪心，前缀和，二分查找，LIS，CDQ

动态规划

```java
class Solution {
    public int findNumberOfLIS(int[] nums) {
        // dp[i]表示以i结尾的最长序列长度
        // dp[i]=max(dp[j]+1/0)(j<i)
        // dp[i]=1
        // count(max(dp[i]))
        int n = nums.length;
        int[] dp = new int[n];
        int[] count = new int[n];
        Arrays.fill(dp, 1);
        count[0] = 1;
        int max = 1;
        for (int i = 1; i < n; ++i) {
            for (int j = 0; j < i; ++j) {
                if (nums[i] > nums[j]) {
                    dp[i] = Math.max(dp[i], dp[j] + 1);
                }
            }
            for (int j = 0; j < i; ++j) {
                if (nums[i] > nums[j] && dp[j] == dp[i] - 1) {
                    count[i] += count[j];
                }
            }
            if (count[i] == 0) { // 孤立的自己成一个
                count[i] = 1;
            }
            max = Math.max(max, dp[i]);
        }
        int ans = 0;
        for (int i = 0; i < n; ++i) {
            if (dp[i] == max) {
                ans += count[i];
            }
        }
        return ans;
    }
}
```

动态规划（一次遍历）

```java
class Solution {
    public int findNumberOfLIS(int[] nums) {
        // dp[i]表示以i结尾的最长序列长度
        // dp[i]=max(dp[j]+1/0)(j<i)
        // dp[i]=1
        // count(max(dp[i]))
        int n = nums.length, max = 0, ans = 0;
        int[] dp = new int[n];
        int[] cnt = new int[n];
        for (int i = 0; i < n; ++i) {
            dp[i] = 1;
            cnt[i] = 1;
            for (int j = 0; j < i; ++j) {
                if (nums[i] > nums[j]) {
                    if (dp[j] + 1 > dp[i]) {
                        dp[i] = dp[j] + 1;
                        cnt[i] = cnt[j]; // 找到更大的长度，重置计数
                    } else if (dp[j] + 1 == dp[i]) {
                        cnt[i] += cnt[j]; // 就是最大值，累加计数
                    }
                }
            }
            if (dp[i] > max) { // 找到全局更大的长度
                max = dp[i]; // 更新全局值
                ans = cnt[i]; // 重置计数
            } else if (dp[i] == max) {
                ans += cnt[i]; // 累加计数
            }
        }
        return ans;
    }
}
```

树状数组 TODO

贪心 + 前缀和 + 二分查找 TODO

## [709. 转换成小写字母](https://leetcode-cn.com/problems/to-lower-case/)

> 字符串

```java
class Solution {
    public String toLowerCase(String s) {
        return s.toLowerCase();
    }
}
```

位运算技巧

```java
class Solution {
    public String toLowerCase(String s) {
        StringBuilder sb = new StringBuilder();
        for (char c: s.toCharArray()) {
            sb.append((char)(65 <= c && c <= 90 ? (c | 32) : c));
        }
        return sb.toString();
    }
}
```

## [58. 最后一个单词的长度](https://leetcode-cn.com/problems/length-of-last-word/)

> 字符串

```java
class Solution {
    public int lengthOfLastWord(String s) {
        int i = s.length() - 1;
        while (s.charAt(i) == ' ') i--; // 去除空格
        int ans = 0;
        while (i >= 0 && s.charAt(i) != ' ') { // 计数
            ans++;
            i--;
        }
        return ans;
    }
}
```

## [725. 分隔链表](https://leetcode-cn.com/problems/split-linked-list-in-parts/)

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
    private ListNode[] ret;
    private int index = 0;

    public ListNode[] splitListToParts(ListNode head, int k) {
        int n = 0;
        ListNode p = head;
        while (p != null) {
            n++;
            p = p.next;
        }
        int s = n / k; // 较小长度
        int r = n % k; // 较大长度(s+1)的个数
        ret = new ListNode[k];
        cut(k - r, s, cut(r, s + 1, head)); // 注意细节
        return ret;
    }

    private ListNode cut(int cnt, int len, ListNode head) {
        while (cnt > 0) {
            ListNode start = head;
            ListNode end = null;
            if (start != null) {
                int curLen = len; // 注意细节
                while (curLen > 0) {
                    if (curLen == 1) {
                        end = head;
                    }
                    head = head.next;
                    curLen--;
                }
                if (end != null)
                    end.next = null;
            }
            ret[index++] = start;
            cnt--;
        }
        return head;
    }
}
```

## [326. 3的幂](https://leetcode-cn.com/problems/power-of-three/)

> 递归，数学，循环

循环

执行用时：15 ms, 在所有 Java 提交中击败了93.08%的用户

内存消耗：38.1 MB, 在所有 Java 提交中击败了77.74%的用户

```java
class Solution {
    public boolean isPowerOfThree(int n) {
        while (n > 0) {
            if (n % 3 == 0) {
                n /= 3;
            } else {
                return n == 1;
            }
        }
        return false;
    }
}
```

递归

执行用时：15 ms, 在所有 Java 提交中击败了93.08%的用户

内存消耗：38.4 MB, 在所有 Java 提交中击败了5.35%的用户

```java
class Solution {
    public boolean isPowerOfThree(int n) {
        if (n < 3)
            return n == 1;
        if (n % 3 == 0)
            return isPowerOfThree(n / 3);
        return false;
    }
}
```

取巧

最大的 33 的幂为 3^{19} = 1162261467，判断n是否为其约数即可

执行用时：15 ms, 在所有 Java 提交中击败了93.08%的用户

内存消耗：38.3 MB, 在所有 Java 提交中击败了31.92%的用户

```java
class Solution {
    public boolean isPowerOfThree(int n) {
        return n > 0 && 1162261467 % n == 0;
    }
}
```

对数运算

执行用时：15 ms, 在所有 Java 提交中击败了93.08%的用户

内存消耗：38.3 MB, 在所有 Java 提交中击败了25.24%的用户

```java
class Solution {
    public boolean isPowerOfThree(int n) {
        double x = Math.log(n) / Math.log(3);
        return Math.abs(x - Math.round(x)) < Math.pow(10, -14);
    }
}
```

## [430. 扁平化多级双向链表](https://leetcode-cn.com/problems/flatten-a-multilevel-doubly-linked-list/)

> 深度优先搜索，链表，双向链表，栈，回溯

执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：36.5 MB, 在所有 Java 提交中击败了44.48%的用户

```java
/*
// Definition for a Node.
class Node {
    public int val;
    public Node prev;
    public Node next;
    public Node child;
};
*/

class Solution {
    public Node flatten(Node head) {
        if (head == null)
            return null;
        Deque<Node> stack = new LinkedList<>();
        Node cur = null;
        stack.push(head);
        while (!stack.isEmpty()) {
            Node p = stack.pop();
            if (cur == null) {
                cur = p;
            } else {
                cur.next = p;
                p.prev = cur;
                cur = cur.next;
            }
            while (cur != null && cur.next != null && cur.child == null) { // 往右走
                cur = cur.next;
            }
            if (cur != null && cur.child != null) {
                if (cur.next != null) {
                    stack.push(cur.next);
                    cur.next.prev = null;
                }
                cur.next = cur.child;
                cur.child.prev = cur;
                stack.push(cur.child);
                cur.child = null;
            }
        }
        return head;
    }
}
```

## [583. 两个字符串的删除操作](https://leetcode-cn.com/problems/delete-operation-for-two-strings/)

> 字符串，动态规划，最长公共子串

执行用时：6 ms, 在所有 Java 提交中击败了94.95%的用户

内存消耗：39.2 MB, 在所有 Java 提交中击败了16.28%的用户

```java
class Solution {
    public int minDistance(String word1, String word2) {
        // dp[i][j]表示1串i和2串j的最长公共子串的长度
        // dp[i][j] = max(dp[i-1][j],dp[i][j-1],dp[i-1][j-1]+1(相等))
        // dp[i][j]=0
        // dp[n][m]
        int n = word1.length(), m = word2.length();
        int[][] dp = new int[n + 1][m + 1];
        for (int i = 1; i <= n; ++i) {
            char c1 = word1.charAt(i - 1);
            for (int j = 1; j <= m; ++j) {
                char c2 = word2.charAt(j - 1);
                dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                if (c1 == c2) dp[i][j] = Math.max(dp[i][j], dp[i - 1][j - 1] + 1);
            }
        }
        int maxLen = dp[n][m];
        return n + m - 2 * maxLen;
    }
}
```

直接用动态规划TODO

## [371. 两整数之和](https://leetcode-cn.com/problems/sum-of-two-integers/)

> 位运算，数学

```java
class Solution {
    public int getSum(int a, int b) {
        while (b != 0) {
            int carry = (a & b) << 1; // 所有位计算进位
            a ^= b; // 所有位计算当前位
            b = carry; // 获取进位
        }
        return a;
    }
}
```

递归写法

```java
class Solution {
    public int getSum(int a, int b) {
        return b == 0 ? a : getSum(a ^ b, (a & b) << 1);
    }
}
```

## [639. 解码方法 II](https://leetcode-cn.com/problems/decode-ways-ii/)

> 字符串，动态规划，分情况讨论DP，滚动数组

```java
class Solution {
    private static final int MOD = 1000000007;

    public int numDecodings(String s) {
        // dp[i]表示前i个数目
        // dp[i]=a*dp[i-1](自己单独编码)+b*dp[i-2](和前者一起编码)(依据s[i],s[i-1],s[i-2]的值来确定a,b)
        // a = c == '0' ? 0 : (c == '*' ? 9 : 1)
        // b = 根据c前一个和c来决定
        // dp[0]=1(dp_i_1)空字符串假定为1(因为要乘)
        // dp[n]
        int n = s.length();
        long dp_i = 0, dp_i_1 = 1, dp_i_2 = 0;
        for (int i = 1; i <= n; ++i) {
            char c1 = s.charAt(i - 1);
            // 自己单独编码
            dp_i = dp_i_1 * (c1 == '0' ? 0 : (c1 == '*' ? 9 : 1)) % MOD;
            // 和前者一起编码
            if (i > 1) {
                dp_i = (dp_i + dp_i_2 * getCount(s.charAt(i - 2), c1)) % MOD; 
            }
            // 轮动
            dp_i_2 = dp_i_1;
            dp_i_1 = dp_i;
        }
        return (int)dp_i;
    }

    private int getCount(char c2, char c1) {
        // ** -> 15
        // *[] -> *0~*6:2,*7~*9:1
        // []* -> 1*:10,2*:7,其他*:0
        // [][] -> 10~26:1
        if (c2 == '*' && c1 == '*')
            return 15;
        else if (c2 == '*' && c1 != '*')
            return c1 <= '6' ? 2 : 1;
        else if (c2 != '*' && c1 == '*')
            return c2 == '1' ? 9 : (c2 == '2' ? 6 : 0);
        else
            return c2 == '1' ? 1 : (c2 == '2' ? (c1 <= '6' ? 1 : 0) : 0);
    }
}
```

## [437. 路径总和 III](https://leetcode-cn.com/problems/path-sum-iii/)

> 树，深度优先搜索，递归，前缀和，回溯

双重递归

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
    private int cnt = 0;

    public int pathSum(TreeNode root, int targetSum) {
        if (root == null)
            return 0;
        helper(root, targetSum); // 第二层递归表示对每个节点搜索，满足条件+1
        pathSum(root.left, targetSum); // 第一层递归表示遍历所有节点作为起点
        pathSum(root.right, targetSum);
        return cnt;
    }

    private void helper(TreeNode root, int targetSum) {
        if (root == null)
            return;
        if (root.val == targetSum)
            cnt++;
        int delta = targetSum - root.val;
        helper(root.left, delta);
        helper(root.right, delta);
    }
}
```

前缀和

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
    public int pathSum(TreeNode root, int targetSum) {
        Map<Long, Integer> map = new HashMap<>(); // <前缀和,个数>
        map.put(0L, 1); // TODO
        return dfs(root, map, 0, targetSum);
    }

    private int dfs(TreeNode root, Map<Long, Integer> map, long curSum, int targetSum) {
        if (root == null)
            return 0;
        curSum += root.val;
        int ans = map.getOrDefault(curSum - targetSum, 0);
        map.put(curSum, map.getOrDefault(curSum, 0) + 1);
        ans += dfs(root.left, map, curSum, targetSum);
        ans += dfs(root.right, map, curSum, targetSum);
        map.put(curSum, map.getOrDefault(curSum, 0) - 1); // 回溯
        return ans;
    }
}
```

## [517. 超级洗衣机](https://leetcode-cn.com/problems/super-washing-machines/)

> 数组，贪心，前缀和，脑筋急转弯

考虑左右过路

```java
class Solution {
    public int findMinMoves(int[] machines) {
        // 分析每个位置，过路的最大值即为结果，结果=MAX(左边往右边搬运+右边往左边搬运)
        // 或者思考为当前位置需要向左匀和向右匀（一样的求法）
        // int sum = Arrays.stream(machines).sum();
        int sum = 0, n = machines.length;
        for (int machine: machines)
            sum += machine;
        if (sum % n != 0)
            return -1;
        int avg = sum / n, max = 0;
        int l = 0, r = sum;
        for (int i = 0; i < n; ++i) {
            r -= machines[i];
            max = Math.max(max, Math.max(0, i * avg - l) + Math.max(0, (n - i - 1) * avg - r));
            l += machines[i];
        }
        return max;
    }
}
```

前缀和TODO

## [223. 矩形面积](https://leetcode-cn.com/problems/rectangle-area/)

> 几何，数学

执行用时：2 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：37.6 MB, 在所有 Java 提交中击败了82.14%的用户

好理解

```java
class Solution {
    public int computeArea(int ax1, int ay1, int ax2, int ay2, int bx1, int by1, int bx2, int by2) {
        return (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - 
        getCover(ax1, ax2, bx1, bx2) * getCover(ay1, ay2, by1, by2);
    }

    private int getCover(int a, int b, int c, int d) {
        if (b <= c || d <= a)
            return 0;
        if (c <= a && b <= d)
            return b - a;
        if (a <= c && d <= b)
            return d - c;
        if (c < a)
            return d - a;
        return b - c;
    }
}
```

更简化

```java
class Solution {
    public int computeArea(int ax1, int ay1, int ax2, int ay2, int bx1, int by1, int bx2, int by2) {
        return (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - (Math.max(Math.min(ax2, bx2) - Math.max(ax1, bx1), 0) * Math.max(Math.min(ay2, by2) - Math.max(ay1, by1), 0));
    }
}
```

## [1436. 旅行终点站](https://leetcode-cn.com/problems/destination-city/)

> 哈希表，字符串

入度出度

```java
class Solution {
    public String destCity(List<List<String>> paths) {
        Map<String, Integer> map = new HashMap<>();
        for (List<String> path: paths) {
            map.put(path.get(0), map.getOrDefault(path.get(0), 0) + 1);
            map.put(path.get(1), map.getOrDefault(path.get(1), 0) - 1);
        }
        for (String str: map.keySet()) {
            if (map.get(str) == -1)
                return str;
        }
        return "";
    }
}
```

路径压缩

执行用时：2 ms, 在所有 Java 提交中击败了94.53%的用户

内存消耗：38 MB, 在所有 Java 提交中击败了62.24%的用户

```java
class Solution {
    public String destCity(List<List<String>> paths) {
        Map<String, String> map = new HashMap<>();
        for (List<String> path: paths) {
            map.put(path.get(0), path.get(1));
        }
        String cur = paths.get(0).get(0);
        while (map.containsKey(cur)) {
            cur = map.get(cur);
        }
        return cur;
    }
}
```

键不存在

```java
class Solution {
    public String destCity(List<List<String>> paths) {
        Set<String> citiesA = new HashSet<String>();
        for (List<String> path : paths) {
            citiesA.add(path.get(0));
        }
        for (List<String> path : paths) {
            if (!citiesA.contains(path.get(1))) {
                return path.get(1);
            }
        }
        return "";
    }
}
```

## [405. 数字转换为十六进制数](https://leetcode-cn.com/problems/convert-a-number-to-hexadecimal/)

> 位运算，数学

```java
class Solution {
    public String toHex(int num) {
        if (num == 0)
            return "0";
        char[] cs = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};
        StringBuilder sb = new StringBuilder();
        long myNum = num;
        if (myNum < 0)
            myNum += (long)1 << 32; // 补码进制求法！TODO
        while (myNum > 0) {
            sb.insert(0, cs[(int)(myNum % 16)]);
            myNum /= 16;
        }
        return sb.toString();
    }
}
```

位运算+分组换算

```java
class Solution {
    public String toHex(int num) {
        if (num == 0)
            return "0";
        // 32位，4个1组，共8组
        StringBuilder sb = new StringBuilder();
        for (int i = 7; i >= 0; --i) {
            int digit = (num >> (i * 4)) & 0xf; // 取最后四位
            if (sb.length() > 0 || digit > 0) {
                sb.append((char)(digit < 10 ? digit + '0' : digit - 10 + 'a'));
            }
        }
        return sb.toString();
    }
}
```

## [166. 分数到小数](https://leetcode-cn.com/problems/fraction-to-recurring-decimal/)

> 哈希表，数学，字符串

执行用时：1 ms, 在所有 Java 提交中击败了99.95%的用户

内存消耗：35.7 MB, 在所有 Java 提交中击败了82.12%的用户

```java
class Solution {
    public String fractionToDecimal(int numerator, int denominator) {
        StringBuilder sb = new StringBuilder();
        if (numerator < 0 && denominator > 0 || numerator > 0 && denominator < 0) // 也可以 if(num<0^den<0)
            sb.append("-");
        long nu = Math.abs((long)numerator); // 注意溢出和提前转为long
        long de = Math.abs((long)denominator);
        sb.append(nu / de);
        long rest = nu % de;
        if (rest == 0)
            return sb.toString();
        sb.append(".");
        int curIndex = sb.length();
        Map<Long, Integer> map = new HashMap<>(); // <余数，初始位置>
        while (true) {
            rest *= 10;
            sb.append(rest / de);
            rest %= de;
            if (rest == 0)
                return sb.toString();
            if (map.containsKey(rest)) {
                sb.insert(map.get(rest), "(");
                sb.append(")");
                return sb.toString();
            }
            map.put(rest, ++curIndex);
        }
    }
}
```

## [482. 密钥格式化](https://leetcode-cn.com/problems/license-key-formatting/)

> 字符串

```java
class Solution {
    public String licenseKeyFormatting(String s, int k) {
        int n = s.length();
        StringBuilder sb = new StringBuilder();
        int cnt = 0;
        for (int i = n - 1; i >= 0; --i) {
            if (s.charAt(i) != '-') {
                sb.append(Character.toUpperCase(s.charAt(i)));
                cnt++;
            }
            if (cnt == k) {
                cnt = 0;
                sb.append("-");
            }
        }
        if (sb.length() > 0 && sb.charAt(sb.length() - 1) == '-')
            sb.deleteCharAt(sb.length() - 1);
        return sb.reverse().toString();
    }
}
```

## [284. 顶端迭代器](https://leetcode-cn.com/problems/peeking-iterator/)

> 设计，数组，迭代器

```java
// Java Iterator interface reference:
// https://docs.oracle.com/javase/8/docs/api/java/util/Iterator.html

class PeekingIterator implements Iterator<Integer> {
    private List<Integer> nums;
    private int ptr = 0;
    private int size = 0;

    public PeekingIterator(Iterator<Integer> iterator) {
        // initialize any member here.
        nums = new ArrayList<>();
        while (iterator.hasNext()) {
            nums.add(iterator.next());
        }
        size = nums.size();
    }

    // Returns the next element in the iteration without advancing the iterator.
    public Integer peek() {
        return nums.get(ptr);
    }

    // hasNext() and next() should behave the same as in the Iterator interface.
    // Override them if needed.
    @Override
    public Integer next() {
        Integer ret = null;
        if (ptr < size) {
            ret = nums.get(ptr);
        }
        ptr++;
        return ret;
    }

    @Override
    public boolean hasNext() {
        if (ptr >= size) {
            return false;
        } else {
            return true;
        }
    }
}
```

不用数组

```java
class PeekingIterator implements Iterator<Integer> {
    private Iterator<Integer> iterator;
    private Integer cur;

    public PeekingIterator(Iterator<Integer> iterator) {
        this.iterator = iterator;
        cur = iterator.next();
    }

    public Integer peek() {
        return cur;
    }

    @Override
    public Integer next() {
        Integer ret = cur;
        cur = iterator.hasNext() ? iterator.next() : null;
        return ret;
    }

    @Override
    public boolean hasNext() {
        return cur != null;
    }
}
```

## [414. 第三大的数](https://leetcode-cn.com/problems/third-maximum-number/)

> 数组，排序

```java
class Solution {
    public int thirdMax(int[] nums) {
        long[] max = new long[3]; // 注意溢出，也可以用a,b,c
        Arrays.fill(max, Long.MIN_VALUE); // 也可以用null表示最小值
        for (int num: nums) {
            if (num == max[0] || num == max[1] || num == max[2])
                continue;
            if (num > max[0]) {
                max[2] = max[1];
                max[1] = max[0];
                max[0] = num;
            } else if (num > max[1]) {
                max[2] = max[1];
                max[1] = num;
            } else if (num > max[2]) {
                max[2] = num;
            }
        }
        if (max[2] == Long.MIN_VALUE || max[1] == Long.MIN_VALUE)
            return (int)max[0];
        return (int)max[2];
    }
}
```

TreeSet TODO

```java
class Solution {
    public int thirdMax(int[] nums) {
        TreeSet<Integer> s = new TreeSet<Integer>();
        for (int num : nums) {
            s.add(num);
            if (s.size() > 3) {
                s.remove(s.first()); // 注意从小到大
            }
        }
        return s.size() == 3 ? s.first() : s.last();
    }
}
```

排序 TODO

## [434. 字符串中的单词数](https://leetcode-cn.com/problems/number-of-segments-in-a-string/)

> 字符串

```java
class Solution {
    public int countSegments(String s) {
        s = s.replaceAll("[ ]+", " ").trim();
        if (s.length() == 0 || s.equals(" "))
            return 0;
        int ans = 0;
        for (char c: s.toCharArray()) {
            if (c == ' ')
                ans++;
        }
        return ans + 1;
    }
}
```

简化

```java
class Solution {
    public int countSegments(String s) {
        int ans = 0;
        for (int i = 0; i < s.length(); i++) {
            if ((i == 0 || s.charAt(i - 1) == ' ') && s.charAt(i) != ' ')
                ans++;
        }
        return ans;
    }
}
```

## [187. 重复的DNA序列](https://leetcode-cn.com/problems/repeated-dna-sequences/)

> 位运算，哈希表，字符串，滑动窗口，哈希函数，滚动哈希

stringbuilder

```java
class Solution {
    public List<String> findRepeatedDnaSequences(String s) {
        int n = s.length();
        Set<String> ret = new HashSet<>();
        if (n > 10) {
            Set<String> tmpSet = new HashSet<>();
            StringBuilder cur = new StringBuilder();
            for (int i = 10; i < n; ++i) {
                cur.deleteCharAt(i - 10);
                cur.append(s.charAt(i));
                if (tmpSet.contains(cur.toString())) {
                    ret.add(cur.toString());
                } else {
                    tmpSet.add(cur.toString());
                }
            }
        }
        return new ArrayList<>(ret);
    }
}
```

substring

```java
class Solution {
    public List<String> findRepeatedDnaSequences(String s) {
        int n = s.length();
        Set<String> ret = new HashSet<>();
        if (n > 10) {
            Set<String> tmpSet = new HashSet<>();
            String cur = new String();
            for (int i = 0; i <= n - 10; ++i) {
                cur = s.substring(i, i + 10);
                if (tmpSet.contains(cur)) {
                    ret.add(cur);
                } else {
                    tmpSet.add(cur);
                }
            }
        }
        return new ArrayList<>(ret);
    }
}
```

位运算 TODO（哈希只有Integer作为key才能保证严格O(1)取到值）

```java
class Solution {
    Map<Character, Integer> bin = new HashMap<Character, Integer>() {{
        put('A', 0); // 00
        put('C', 1); // 01
        put('G', 2); // 10
        put('T', 3); // 11
    }};
    public List<String> findRepeatedDnaSequences(String s) {
        List<String> ans = new ArrayList<String>();
        int n = s.length();
        if (n <= 10) {
            return ans;
        }
        int x = 0;
        for (int i = 0; i < 9; ++i) {
            x = (x << 2) | bin.get(s.charAt(i));
        }
        Map<Integer, Integer> cnt = new HashMap<Integer, Integer>();
        for (int i = 0; i <= n - 10; ++i) {
            x = ((x << 2) | bin.get(s.charAt(i + 9))) & ((1 << 20) - 1);
            cnt.put(x, cnt.getOrDefault(x, 0) + 1);
            if (cnt.get(x) == 2) {
                ans.add(s.substring(i, i + 10));
            }
        }
        return ans;
    }
}
```

## [441. 排列硬币](https://leetcode-cn.com/problems/arranging-coins/)

> 数学，二分查找

执行用时：1 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：35.5 MB, 在所有 Java 提交中击败了73%的用户

```java
class Solution {
    public int arrangeCoins(int n) {
        long candidate = (long)Math.sqrt(2 * (long)n);
        return (int)(candidate * (candidate + 1) <= 2 * (long)n ? candidate : candidate - 1);
    }
}
```

用二元方程公式 TODO

```java
class Solution {
    public int arrangeCoins(int n) {
        return (int)((Math.sqrt(1 + 8.0 * n) - 1) / 2);
    }
}
```

二分 TODO

## [273. 整数转换英文表示](https://leetcode-cn.com/problems/integer-to-english-words/)

> 递归，数学，字符串

```java
class Solution {
    private static final String[] en = {"One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen", "Twenty"};

    private static final String[] en2 = {"Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"};

    public String numberToWords(int num) {
        if (num == 0)
            return "Zero";
        int n = (num + "").length();
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < 12 - n; ++i)
            sb.append("0");
        sb.append(num);
        StringBuilder ans = new StringBuilder();
        for (int i = 0; i < 12; i += 3) {
            if (sb.substring(i, i + 3).equals("000"))
                continue;
            StringBuilder threeSb = getThree(sb.charAt(i), sb.charAt(i + 1), sb.charAt(i + 2));
            if (threeSb.length() > 0)
                ans.append(" ").append(threeSb).append(" ").append(getBMT(i));
        }
        return ans.toString().trim();
    }

    private StringBuilder getThree(char c1, char c2, char c3) {
        StringBuilder sb = new StringBuilder();
        if (c1 != '0') {
            sb.append(en[c1 - '0' - 1]).append(" ").append("Hundred");
        }
        if (c2 != '0' || c3 != '0') {
            if (c1 != '0')
                sb.append(" ");
            int tmp = (c2 - '0') * 10 + (c3 - '0');
            if (tmp <= 20) {
                sb.append(en[tmp - 1]);
            } else if (c3 == '0') {
                sb.append(en2[c2 - '0' - 2]);
            } else {
                sb.append(en2[c2 - '0' - 2]).append(" ").append(en[c3 - '0' - 1]);
            }
        }
        return sb;
    }

    private String getBMT(int idx) {
        switch(idx) {
            case 0:
                return "Billion";
            case 3:
                return "Million";
            case 6:
                return "Thousand";
        }
        return "";
    }
}
```

递归TODO

## [29. 两数相除](https://leetcode-cn.com/problems/divide-two-integers/)

> 倍增法，数学，二分查找

```java
class Solution {
    public int divide(int dividend, int divisor) {
        if (dividend == 0) return 0;
        long d1, d2;
        boolean flag = false;
        if (dividend > 0 && divisor < 0) {
            d1 = (long)dividend;
            d2 = -(long)divisor;
        } else if (dividend < 0 && divisor > 0) {
            d1 = -(long)dividend;
            d2 = (long)divisor;
        } else {
            d1 = Math.abs((long)dividend);
            d2 = Math.abs((long)divisor);
            flag = true;
        }
        long ans = 0;
        while (d1 >= d2) {
            if (d2 == 1) {
                ans = d1;
                break;
            }
            if (d2 == 2) {
                ans = d1 >> 1;
                break;
            }
            d1 -= d2;
            ans++;
        }
        ans = flag ? ans : -ans;
        return ans >= 2147483648L ? 2147483647 : (int)ans;
    }
}
```

倍增+无long TODO

```java
class Solution {
    // 都映射到负数，因为负数表示范围大
    int MIN = Integer.MIN_VALUE, MAX = Integer.MAX_VALUE;
    int LIMIT = -1073741824; // MIN 的一半
    public int divide(int a, int b) {
        if (a == MIN && b == -1) return MAX;
        boolean flag = false;
        if ((a ^ b) < 0) flag = true;
        if (a > 0) a = -a;
        if (b > 0) b = -b;
        int ans = 0;
        while (a <= b) { // -a >= -b 倍增法，用b倍增来试探到a
            int c = b, d = -1;
            while (c >= LIMIT && d >= LIMIT && c >= a - c){
                c += c;
                d += d;
            }
            a -= c; // 更新a到a-nb继续试探
            ans += d;
        }
        return flag ? ans : -ans;
    }
}
```

## [412. Fizz Buzz](https://leetcode-cn.com/problems/fizz-buzz/)

> 数学，字符串，模拟

```java
class Solution {
    public List<String> fizzBuzz(int n) {
        List<String> ans = new ArrayList<>();
        for (int i = 1; i <= n; ++i) {
            StringBuilder sb = new StringBuilder();
            if (i % 3 == 0)
                sb.append("Fizz");
            if (i % 5 == 0)
                sb.append("Buzz");
            if (sb.length() == 0)
                sb.append(i);
            ans.add(sb.toString());
        }
        return ans;
    }
}
```

## [125. 验证回文串](https://leetcode-cn.com/problems/valid-palindrome/)

> 双指针，字符串

```java
class Solution {
    public boolean isPalindrome(String s) {
        int n = s.length();
        int l = 0, r = n - 1;
        while (l < r) {
            while (l < n && !Character.isLetterOrDigit(s.charAt(l))) l++;
            while (r >= 0 && !Character.isLetterOrDigit(s.charAt(r))) r--;
            if (l < r) {
                if (Character.toLowerCase(s.charAt(l)) != Character.toLowerCase(s.charAt(r))) return false;
                l++;
                r--;
            }
        }
        return true;
    }
}
```

## [21. 合并两个有序链表](https://leetcode-cn.com/problems/merge-two-sorted-lists/)

> 链表，递归，迭代

迭代

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
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        ListNode dummy = new ListNode();
        ListNode p = dummy, p1 = l1, p2 = l2;
        while (p1 != null && p2 != null) {
            if (p1.val <= p2.val) {
                p.next = p1;
                p1 = p1.next;
            } else {
                p.next = p2;
                p2 = p2.next;
            }
            p = p.next;
        }
        p.next = p1 != null ? p1 : p2;
        return dummy.next;
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
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if (l1 == null)
            return l2;
        if (l2 == null)
            return l1;
        if (l1.val <= l2.val) {
            l1.next = mergeTwoLists(l1.next, l2);
            return l1;
        } else {
            l2.next = mergeTwoLists(l1, l2.next);
            return l2;
        }
    }
}
```

## [230. 二叉搜索树中第K小的元素](https://leetcode-cn.com/problems/kth-smallest-element-in-a-bst/)

> 树，深度优先搜索，二叉搜索树，二叉树，中序遍历

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
    private int mk;
    private List<Integer> list;

    public int kthSmallest(TreeNode root, int k) {
        mk = k;
        list = new ArrayList<>();
        dfs(root);
        return list.get(k - 1);
    }

    private void dfs(TreeNode root) {
        if (root == null) {
            return;
        }
        dfs(root.left);
        list.add(root.val);
        if (list.size() >= mk) {
            return;
        }
        dfs(root.right);
    }
}
```

其他优化，AVL TODO

## [476. 数字的补数](https://leetcode-cn.com/problems/number-complement/)

> 位运算

执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：35.2 MB, 在所有 Java 提交中击败了57.76%的用户

```java
class Solution {
    public int findComplement(int num) {
        return (1 << Integer.toBinaryString(num).length()) - 1 - num;
    }
}
```

## [211. 添加与搜索单词 - 数据结构设计](https://leetcode-cn.com/problems/design-add-and-search-words-data-structure/)

> 深度优先搜索，设计，字典树，字符串

执行用时：39 ms, 在所有 Java 提交中击败了81.63%的用户

内存消耗：49 MB, 在所有 Java 提交中击败了60.17%的用户

```java
class WordDictionary {
    class Trie {
        private boolean end;
        private Trie[] children;
        public Trie () {
            children = new Trie[26];
            end = false;
        }
        public boolean getEnd() {
            return this.end;
        }
        public Trie[] getChildren() {
            return this.children;
        }
    }

    private Trie words;

    public WordDictionary() {
        words = new Trie();
    }

    public void addWord(String word) {
        Trie p = words;
        for (char c: word.toCharArray()) {
            if (p.children[c - 'a'] == null) {
                p.children[c - 'a'] = new Trie();
            }
            p = p.children[c - 'a'];
        }
        p.end = true;
    }

    public boolean search(String word) {
        return dfs(word, 0, words);
    }

    private boolean dfs(String word, int index, Trie p) {
        if (p == null) {
            return false;
        }
        if (index == word.length()) {
            return p.getEnd();
        }
        char c = word.charAt(index);
        if (c != '.') {
            Trie next = p.getChildren()[c - 'a'];
            if (next == null)
                return false;
            return dfs(word, index + 1, next);
        } else {
            for (int i = 0; i < 26; ++i) {
                if (p != null && dfs(word, index + 1, p.getChildren()[i])) {
                    return true;
                }
            }
            return false;
        }
    }
}

/**
 * Your WordDictionary object will be instantiated and called as such:
 * WordDictionary obj = new WordDictionary();
 * obj.addWord(word);
 * boolean param_2 = obj.search(word);
 */
```

## [453. 最小操作次数使数组元素相等](https://leetcode-cn.com/problems/minimum-moves-to-equal-array-elements/)

> 数组，数学，脑筋急转弯

```java
class Solution {
    public int minMoves(int[] nums) {
        int min = Arrays.stream(nums).min().getAsInt();
        int ans = 0;
        for (int num: nums) {
            ans += num - min;
        }
        return ans;
    }
}
```

## [2011. 执行操作后的变量值](https://leetcode-cn.com/problems/final-value-of-variable-after-performing-operations/)

> 数组，字符串，模拟

```java
class Solution {
    public int finalValueAfterOperations(String[] operations) {
        int X = 0;
        for (String op: operations) {
            if (op.charAt(1) == '+') {
                X++;
            } else {
                X--;
            }
        }
        return X;
    }
}
```

## [229. 求众数 II](https://leetcode-cn.com/problems/majority-element-ii/)

> 数组，哈希表，计数，排序，摩尔投票

哈希表

```java
class Solution {
    public List<Integer> majorityElement(int[] nums) {
        int n = nums.length;
        Set<Integer> ret = new HashSet<>();
        Map<Integer, Integer> map = new HashMap<>();
        for (int num: nums) {
            map.put(num, map.getOrDefault(num, 0) + 1);
            if (map.get(num) > n / 3) {
                ret.add(num);
                if (ret.size() >= 2)
                    break;
            }
        }
        return new ArrayList(ret);
    }
}
```

摩尔投票

```java
class Solution {
    public List<Integer> majorityElement(int[] nums) {
        // x是候选之一，该候选+1；x不是候选，都-1，如果存在候选是0的，则替代为(x,1)
        int n = nums.length;
        int a = 0, b = 0;
        int ca = 0, cb = 0;
        for (int i: nums) {
            if (i == a || i == b) { // 有相等的，把相等的+1
                if (i == a) ca++;
                else if (i == b) cb++;
            } else { // 三者不同
                if (ca > 0 && cb > 0) { // 都大于0，则相互抵消
                    ca--;
                    cb--;
                } else if (ca == 0) { // 替代a
                    a = i;
                    ca = 1;
                } else if (cb == 0) { // 替代b
                    b = i;
                    cb = 1;
                }
            }
        }
        // 验证
        ca = 0;
        cb = 0;
        for (int num: nums) {
            if (a == num) ca++;
            else if (b == num) cb++;
        }
        List<Integer> ret = new ArrayList<>();
        if (ca > n / 3) ret.add(a);
        if (cb > n / 3) ret.add(b);
        return ret;
    }
}
```

## [492. 构造矩形](https://leetcode-cn.com/problems/construct-the-rectangle/)

> 数学

```java
class Solution {
    public int[] constructRectangle(int area) {
        int candidate = (int)Math.sqrt(area);
        while (area % candidate != 0) {
            candidate--;
        }
        return new int[]{area / candidate, candidate};
    }
}
```

## [496. 下一个更大元素 I](https://leetcode-cn.com/problems/next-greater-element-i/)

> 栈，数组，哈希表，单调栈

```java
class Solution {
    public int[] nextGreaterElement(int[] nums1, int[] nums2) {
        // 下一个更大=>单调栈 TODO 熟记
        HashMap<Integer, Integer> map = new HashMap<>();
        Deque<Integer> stack = new ArrayDeque<>();
        for (int i = nums2.length - 1; i >= 0; --i) {
            int num = nums2[i];
            while (!stack.isEmpty() && stack.peek() <= num) {
                stack.pop();
            }
            map.put(num, stack.isEmpty() ? -1 : stack.peek());
            stack.push(num);
        }
        int n1 = nums1.length;
        int[] ret = new int[n1];
        for (int i = 0; i < n1; ++i) {
            ret[i] = map.get(nums1[i]);
        }
        return ret;
    }
}
```

## [869. 重新排序得到 2 的幂](https://leetcode-cn.com/problems/reordered-power-of-2/)

> 数学，计数，枚举，排序

```java
class Solution {
    public boolean reorderedPowerOf2(int n) {
        Set<String> set = new HashSet<>();
        int i = 1;
        while (i <= 1000000000) {
            set.add(getFingerPrint(i));
            i *= 2;
        }
        return set.contains(getFingerPrint(n));
    }

    private String getFingerPrint(int x) {
        int[] number = new int[10];
        String xStr = x + "";
        for (char c: xStr.toCharArray()) {
            number[c - '0']++;
        }
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < 10; ++i) {
            sb.append(number[i]);
        }
        return sb.toString();
    }
}
```

优化版 TODO

```java
class Solution {
    Set<String> powerOf2Digits = new HashSet<String>();

    public boolean reorderedPowerOf2(int n) {
        init();
        return powerOf2Digits.contains(countDigits(n));
    }

    public void init() {
        for (int n = 1; n <= 1e9; n <<= 1) {
            powerOf2Digits.add(countDigits(n));
        }
    }

    public String countDigits(int n) {
        char[] cnt = new char[10];
        while (n > 0) {
            ++cnt[n % 10];
            n /= 10;
        }
        return new String(cnt);
    }
}
```

## [335. 路径交叉](https://leetcode-cn.com/problems/self-crossing/)

> 几何，数组，数学

```java
class Solution {
    public boolean isSelfCrossing(int[] distance) {
        int n = distance.length;
        if (n < 4) return false;
        for (int i = 3; i < n; ++i) {
            if (distance[i] >= distance[i - 2] && distance[i - 1] <= distance[i - 3]) return true;
            if (i >= 4 && distance[i - 1] == distance[i - 3] && distance[i] + distance[i - 4] >= distance[i - 2]) return true;
            if (i >= 5 && distance[i] + distance[i - 4] >= distance[i - 2] && distance[i - 1] + distance[i - 5] >= distance[i - 3] && distance[i - 1] <= distance[i - 3] && distance[i - 2] > distance[i - 4]) return true;
        }
        return false;
    }
}
```

## [260. 只出现一次的数字 III](https://leetcode-cn.com/problems/single-number-iii/)

> 位运算，数组

```java
class Solution {
    public int[] singleNumber(int[] nums) {
        int sum = 0;
        for (int num: nums) {
            sum ^= num;
        }
        // 求最低位的1（哪个是1的位都行），防止溢出 -Integer.MIN_VALUE会溢出
        int lowbit = sum == Integer.MIN_VALUE ? sum : sum & -sum;
        int a = 0, b = 0;
        for (int num: nums) {
            // 分组
            if ((num & lowbit) == 0) {
                a ^= num;
            } else {
                b ^= num;
            }
        }
        return new int[]{a, b};
    }
}
```

## [500. 键盘行](https://leetcode-cn.com/problems/keyboard-row/)

> 数组，哈希表，字符串

```java
class Solution {
    private Map<Character, Integer> map = new HashMap<>();
    private String[] lines = {"qwertyuiopQWERTYUIOP", "asdfghjklASDFGHJKL", "zxcvbnmZXCVBNM"};

    public String[] findWords(String[] words) {
        List<String> ret = new ArrayList<>();
        for (int i = 0; i < 3; ++i) {
            for (char c: lines[i].toCharArray()) {
                map.put(c, i);
            }
        }
        for (String word: words) {
            if (check(word)) {
                ret.add(word);
            }
        }
        return ret.toArray(new String[ret.size()]);
    }

    private boolean check(String word) {
        int tmp = -1;
        for (char c: word.toCharArray()) {
            if (tmp == -1)
                tmp = map.get(c);
            else if (tmp != map.get(c))
                return false;
        }
        return true;
    }
}
```

TODO 可以记录每个字符的位置"12210111011122000010020202"

## [575. 分糖果](https://leetcode-cn.com/problems/distribute-candies/)

> 数组，哈希表

```java
class Solution {
    public int distributeCandies(int[] candyType) {
        Set<Integer> set = new HashSet<>();
        for (int type: candyType) {
            set.add(type);
        }
        return Math.min(candyType.length / 2, set.size());
    }
}
```

## [35. 搜索插入位置](https://leetcode-cn.com/problems/search-insert-position/)

> 数组，二分查找

```java
class Solution {
    public int searchInsert(int[] nums, int target) {
        int n = nums.length;
        int l = 0, r = n - 1;
        while (l < r) {
            int mid = l + (r - l) / 2;
            if (nums[mid] == target) {
                return mid;
            } else if (nums[mid] > target) {
                r = mid;
            } else {
                l = mid + 1;
            }
        }
        return nums[r] >= target ? r : r + 1;
    }
}
```

## [407. 接雨水 II](https://leetcode-cn.com/problems/trapping-rain-water-ii/)

> 广度优先搜索，最小堆，优先队列，数组，矩阵，Dijkstra

```java
class Solution {
    public int trapRainWater(int[][] heightMap) {
        int m = heightMap.length;
        int n = heightMap[0].length;
        boolean[][] vis = new boolean[m][n];
        if (m <= 2 || n <= 2)
            return 0;
        PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> a[1] - b[1]); // <当前坐标,接水后的高度值>
        // 初始化，边缘接水后高度就是木块高度
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i == 0 || j == 0 || i == m - 1 || j == n - 1) {
                    pq.offer(new int[]{i * n + j, heightMap[i][j]});
                    vis[i][j] = true;
                }
            }
        }
        int[][] directions = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
        int res = 0;
        // 从边缘向内
        while (!pq.isEmpty()) {
            // 取出最小值
            int[] cur = pq.poll();
            int curHeight = cur[1];
            for (int[] direction: directions) {
                int nx = cur[0] / n + direction[0];
                int ny = cur[0] % n + direction[1];
                if (nx >= 0 && nx < m && ny >= 0 && ny < n && !vis[nx][ny]) {
                    if (curHeight > heightMap[nx][ny]) {
                        // 可以接水
                        res += curHeight - heightMap[nx][ny];
                    }
                    pq.offer(new int[]{nx * n + ny, Math.max(heightMap[nx][ny], curHeight)});
                    vis[nx][ny] = true;
                }
            }
        }
        return res;
    }
}
```

Dijkstra,BFS TODO

## [367. 有效的完全平方数](https://leetcode-cn.com/problems/valid-perfect-square/)

> 数学，二分查找，牛顿法

执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：35 MB, 在所有 Java 提交中击败了89.56%的用户

```java
class Solution {
    public boolean isPerfectSquare(int num) {
        int r = 46340, l = 1;
        while (l < r) {
            int mid = l + (r - l) / 2;
            if (mid * mid == num) {
                return true;
            } else if (mid * mid < num) {
                l = mid + 1;
            } else {
                r = mid - 1;
            }
        }
        return l * l == num;
    }
}
```

牛顿法 TODO

## [1218. 最长定差子序列](https://leetcode-cn.com/problems/longest-arithmetic-subsequence-of-given-difference/)

> 数组，哈希表，动态规划

```java
class Solution {
    public int longestSubsequence(int[] arr, int difference) {
        // dp[i]表示以值i为结尾的最长结果
        // dp[i]=dp[i-diff]+1
        // dp[i]=0
        // max(dp[k])
        int max = 0;
        Map<Integer, Integer> map = new HashMap<>();
        for (int i: arr) {
            map.put(i, map.getOrDefault(i - difference, 0) + 1);
            max = Math.max(max, map.get(i));
        }
        return max;
    }
}
```

## [268. 丢失的数字](https://leetcode-cn.com/problems/missing-number/)

> 位运算，数组，哈希表，数学，排序

执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：38.8 MB, 在所有 Java 提交中击败了48.47%的用户

```java
class Solution {
    public int missingNumber(int[] nums) {
        int n = nums.length;
        int sum = n * (n + 1) / 2;
        int total = 0;
        for (int num: nums) {
            total += num;
        }
        return sum - total;
    }
}
```

异或位运算

执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：39 MB, 在所有 Java 提交中击败了12.06%的用户

```java
class Solution {
    public int missingNumber(int[] nums) {
        int xor = 0;
        for (int num: nums) {
            xor ^= num;
        }
        int n = nums.length;
        // 后面添加0~n的每个数进行异或
        for (int i = 0; i <= n; ++i) {
            xor ^= i;
        }
        return xor;
    }
}
```

## [598. 范围求和 II](https://leetcode-cn.com/problems/range-addition-ii/)

> 数组，数学

```java
class Solution {
    public int maxCount(int m, int n, int[][] ops) {
        int mina = m, minb = n;
        for (int[] op: ops) {
            mina = Math.min(mina, op[0]);
            minb = Math.min(minb, op[1]);
        }
        return mina * minb;
    }
}
```

## [299. 猜数字游戏](https://leetcode-cn.com/problems/bulls-and-cows/)

> 哈希表，字符串，计数

执行用时：1 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：36.8 MB, 在所有 Java 提交中击败了99.11%的用户

```java
class Solution {
    public String getHint(String secret, String guess) {
        int a = 0, b = 0;
        int[] counta = new int[10];
        int[] countb = new int[10];
        for (int i = 0; i < secret.length(); ++i) {
            char secretChar = secret.charAt(i);
            char guessChar = guess.charAt(i);
            if (secretChar == guessChar) {
                a++;
            } else {
                counta[secretChar - '0']++;
                countb[guessChar - '0']++;
            }
        }
        for (int i = 0; i <= 9; ++i) {
            b += Math.min(counta[i], countb[i]);
        }
        StringBuilder sb = new StringBuilder();
        return sb.append(a).append('A').append(b).append('B').toString();
    }
}
```

## [495. 提莫攻击](https://leetcode-cn.com/problems/teemo-attacking/)

> 数组，模拟

```java
class Solution {
    public int findPoisonedDuration(int[] timeSeries, int duration) {
        int res = 0;
        for (int i = 1; i < timeSeries.length; ++i) {
            int next = timeSeries[i - 1] + duration - 1;
            if (next < timeSeries[i]) {
                res += duration;
            } else {
                res += timeSeries[i] - timeSeries[i - 1];
            }
        }
        res += duration;
        return res;
    }
}
```

## [375. 猜数字大小 II](https://leetcode-cn.com/problems/guess-number-higher-or-lower-ii/)

> 数学，动态规划，博弈

```java
class Solution {
    public int getMoneyAmount(int n) {
        // dp[i][j]表示[i,j]的最小金额
        // dp[i][j]=min(k+max(dp[i][k-1],dp[k+1][j]))
        // dp[i][i]=0
        // dp[1][n]
        // TODO 注意边界情况
        int[][] dp = new int[n + 1][n + 1];
        for (int i = n - 1; i >= 1; --i) {
            for (int j = i + 1; j <= n; ++j) {
                int min = Integer.MAX_VALUE;
                for (int k = i; k < j; ++k) {
                    min = Math.min(min, k + Math.max(dp[i][k - 1], dp[k + 1][j]));
                }
                dp[i][j] = min;
            }
        }
        return dp[1][n];
    }
}
```

## [520. 检测大写字母](https://leetcode-cn.com/problems/detect-capital/)

> 字符串

```java
class Solution {
    public boolean detectCapitalUse(String word) {
        if (word.length() == 1)
            return true;
        int mode = 0; // 0 都小写 1 首字母大写 2 都大写
        if (Character.isUpperCase(word.charAt(0))) {
            if (Character.isUpperCase(word.charAt(1)))
                mode = 2;
            else
                mode = 1;
        } else {
            if (Character.isUpperCase(word.charAt(1)))
                return false;
        }
        for (int i = 2; i < word.length(); ++i) {
            if (mode != 2 && Character.isUpperCase(word.charAt(i)) || mode == 2 && Character.isLowerCase(word.charAt(i)))
                return false;
        }
        return true;
    }
}
```

简化 TODO

```java
class Solution {
    public boolean detectCapitalUse(String word) {
        // 若第 1 个字母为小写，则需额外判断第 2 个字母是否为小写
        if (word.length() >= 2 && Character.isLowerCase(word.charAt(0)) && Character.isUpperCase(word.charAt(1))) {
            return false;
        }

        // 无论第 1 个字母是否大写，其他字母必须与第 2 个字母的大小写相同
        for (int i = 2; i < word.length(); ++i) {
            if (Character.isLowerCase(word.charAt(i)) ^ Character.isLowerCase(word.charAt(1))) {
                return false;
            }
        }
        return true;
    }
}
```

## [319. 灯泡开关](https://leetcode-cn.com/problems/bulb-switcher/)

> 脑筋急转弯，数学

```java
class Solution {
    public int bulbSwitch(int n) {
        return (int)Math.sqrt(n + 0.5);
    }
}
```

## [318. 最大单词长度乘积](https://leetcode-cn.com/problems/maximum-product-of-word-lengths/)

> 位运算，数组，字符串

```java
class Solution {
    public int maxProduct(String[] words) {
        int n = words.length;
        int[] masks = new int[n];
        for (int i = 0; i < n; ++i) {
            String word = words[i];
            for (char c: word.toCharArray()) {
                masks[i] |= (1 << (c - 'a'));
            }
        }
        int max = 0;
        for (int i = 0; i < n - 1; ++i) {
            for (int j = i + 1; j < n; ++j) {
                if ((masks[i] & masks[j]) == 0)
                    max = Math.max(max, words[i].length() * words[j].length());
            }
        }
        return max;
    }
}
```

## [563. 二叉树的坡度](https://leetcode-cn.com/problems/binary-tree-tilt/)

> 树，深度优先搜索，二叉树

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
    private int res = 0;

    public int findTilt(TreeNode root) {
        dfs(root);
        return res;
    }

    private int dfs(TreeNode root) {
        if (root != null) {
            int ls = dfs(root.left);
            int rs = dfs(root.right);
            res += Math.abs(ls - rs);
            return ls + rs + root.val;
        } else {
            return 0;
        }
    }
}
```

## [397. 整数替换](https://leetcode-cn.com/problems/integer-replacement/)

> 贪心，位运算，记忆化搜索，动态规划

执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：35.2 MB, 在所有 Java 提交中击败了56.80%的用户

```java
class Solution {
    public int integerReplacement(int n) {
        // 贪心，让末尾的0尽可能多（x01->x0，x11->x00），这样能够尽快右移（n/=2）
        int res = 0;
        while (n > 1) {
            if (n == 3) { // 特殊情况
                res += 2;
                return res;
            }
            if ((n & 1) == 1) { // XXXXX1
                if ((n & 2) == 0) { // XXXX01
                    n >>= 1;
                } else { // XXXX11
                    n = (n >> 1) + 1;
                }
                res += 2;
            } else { // XXXXX0
                n >>= 1;
                res++;
            }
        }
        return res;
    }
}
```

动态规划（递归），记忆化搜索，DFS，BFS TODO

## [594. 最长和谐子序列](https://leetcode-cn.com/problems/longest-harmonious-subsequence/)

> 数组，哈希表，快排

哈希表

```java
class Solution {
    public int findLHS(int[] nums) {
        Map<Integer, Integer> map = new HashMap<>();
        int max = 0;
        for (int num: nums) {
            int cnum = map.getOrDefault(num, 0) + 1;
            map.put(num, cnum);
            if (map.containsKey(num - 1)) {
                max = Math.max(max, cnum + map.get(num - 1));
            }
            if (map.containsKey(num + 1)) {
                max = Math.max(max, cnum + map.get(num + 1));
            }
        }
        return max;
    }
}
```

排序+滑动窗口 TODO

## [559. N 叉树的最大深度](https://leetcode-cn.com/problems/maximum-depth-of-n-ary-tree/)

> 树，深度优先搜索，广度优先搜索

```java
/*
// Definition for a Node.
class Node {
    public int val;
    public List<Node> children;

    public Node() {}

    public Node(int _val) {
        val = _val;
    }

    public Node(int _val, List<Node> _children) {
        val = _val;
        children = _children;
    }
};
*/

class Solution {
    private int max = 0;

    public int maxDepth(Node root) {
        dfs(root, 0);
        return max;
    }

    private void dfs(Node root, int curDepth) {
        if (root == null) {
            max = Math.max(max, curDepth);
            return;
        }
        if (root.children == null || root.children.size() == 0) {
            max = Math.max(max, curDepth + 1);
            return;
        }
        for (Node node: root.children) {
            dfs(node, curDepth + 1);
        }
    }
}
```

## [384. 打乱数组](https://leetcode-cn.com/problems/shuffle-an-array/)

> 数组，数学，随机化

Fisher-Yates 洗牌算法

```java
class Solution {
    private int[] origin;
    private int[] nums;
    private static Random rand = new Random();

    public Solution(int[] nums) {
        int n = nums.length;
        origin = new int[n];
        System.arraycopy(nums, 0, origin, 0, n);
        this.nums = nums;
    }

    public int[] reset() {
        return origin;
    }

    public int[] shuffle() {
        int length = this.nums.length;
        for (int i = length; i > 0; i--) {
            swap(this.nums, rand.nextInt(i), i - 1);
        }
        return this.nums;
    }

    private void swap(int[] a, int i, int j) {
        int tmp = a[i];
        a[i] = a[j];
        a[j] = tmp;
    }
}

/**
 * Your Solution object will be instantiated and called as such:
 * Solution obj = new Solution(nums);
 * int[] param_1 = obj.reset();
 * int[] param_2 = obj.shuffle();
 */
```

## [859. 亲密字符串](https://leetcode-cn.com/problems/buddy-strings/)

> 哈希表，字符串

```java
class Solution {
    public boolean buddyStrings(String s, String goal) {
        if (s.length() != goal.length()) return false;
        if (s.equals(goal)) {
            char[] cs = new char[26];
            for (char c: s.toCharArray()) {
                if (cs[c - 'a'] > 0) {
                    return true;
                } else {
                    cs[c - 'a']++;
                }
            }
            return false;
        }
        int n = s.length();
        int state = 0;
        char a = '1', b = '1';
        for (int i = 0; i < n; ++i) {
            if (s.charAt(i) != goal.charAt(i)) {
                if (state == 0) {
                    a = s.charAt(i);
                    b = goal.charAt(i);
                    state = 1;
                } else if (state == 1) {
                    if (a == goal.charAt(i) && b == s.charAt(i)) {
                        state = 2;
                    } else {
                        return false;
                    }
                } else {
                    return false;
                }
            }
        }
        return state == 2;
    }
}
```

## [423. 从英文中重建数字](https://leetcode-cn.com/problems/reconstruct-original-digits-from-english/)

> 哈希表，数学，字符串

执行用时：3 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：39 MB, 在所有 Java 提交中击败了47.17%的用户

```java
class Solution {
    public String originalDigits(String s) {
        // "zero"z, "one"o, "two"w, "three"h, "four"u, 
        // "five"v, "six"x, "seven"s, "eight"g, "nine"
        // x排除six6 -> s排除seven7 -> v排除five5
        // z排除zero0 ->> o排除one1
        // w排除two2 ->> o排除one1
        // u排除four4 ->> o排除one1
        // g排除eight8 -> h排除three3
        // 最后还有i就是9
        int[] cc = new int[26];
        for (char c: s.toCharArray()) {
            cc[c - 'a']++;
        }
        int[] nc = new int[10];
        char[] chars = {'z', 'w', 'u', 'g', 'x', 's', 'v', 'h', 'o', 'i'};
        int[] nums = {0, 2, 4, 8, 6, 7, 5, 3, 1, 9};
        char[][] rests = {{'e', 'r', 'o'}, {'t', 'o'}, {'f', 'o', 'r'}, {'e', 'i', 'h', 't'}, 
        {'s', 'i'}, {'e', 'v', 'n'}, {'f', 'i', 'e'}, {'t', 'r', 'e'}, {'n', 'e'}, {}};
        for (int i = 0; i < 10; ++i) {
            int cnt = cc[chars[i] - 'a'];
            if (cnt > 0) {
                nc[nums[i]] = cnt;
                for (char c: rests[i]) {
                    cc[c - 'a'] -= cnt;
                }
            }
        }
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < 10; ++i) {
            while (nc[i] > 0) {
                sb.append(i);
                nc[i]--;
            }
        }
        return sb.toString();
    }
}
```

## [458. 可怜的小猪](https://leetcode-cn.com/problems/poor-pigs/)

> 数学，动态规划，组合数学，香农极限，信息熵

```java
class Solution {
    public int poorPigs(int buckets, int minutesToDie, int minutesToTest) {
        // k*log(n+1)>=log(buckets) 猪数量*每头猪携带的信息量【极限码率】>=桶含有1瓶毒药的信息量
        // 轮数^实验体数 >= 桶数
        int num = minutesToTest / minutesToDie + 1;
        return (int)Math.ceil(Math.log(buckets) / Math.log(num));
    }
}
```

进制推导、动态规划 TODO 

## [700. 二叉搜索树中的搜索](https://leetcode-cn.com/problems/search-in-a-binary-search-tree/)

> 树，二叉搜索树，二叉树

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
    public TreeNode searchBST(TreeNode root, int val) {
        if (root == null) {
            return null;
        }
        if (root.val == val) {
            return root;
        } else if (root.val > val) {
            return searchBST(root.left, val);
        } else {
            return searchBST(root.right, val);
        }
    }
}
```

迭代 TODO

## [438. 找到字符串中所有字母异位词](https://leetcode-cn.com/problems/find-all-anagrams-in-a-string/)

> 哈希表，字符串，滑动窗口

```java
class Solution {
    public List<Integer> findAnagrams(String s, String p) {
        List<Integer> ret = new ArrayList<>();
        if (s.length() < p.length()) return ret;
        int[] cnum = new int[26], tmp = new int[26];
        int n = p.length(), m = s.length();
        for (int i = 0; i < n; ++i) {
            cnum[p.charAt(i) - 'a']++;
            tmp[s.charAt(i) - 'a']++;
        }
        for (int i = 0; i <= m - n; ++i) {
            if (Arrays.equals(tmp, cnum))
                ret.add(i);
            if (i == m - n)
                break;
            tmp[s.charAt(i) - 'a']--;
            tmp[s.charAt(i + n) - 'a']++;
        }
        return ret;
    }
}
```

## [519. 随机翻转矩阵](https://leetcode-cn.com/problems/random-flip-matrix/)

> 哈希表，数学，随机化，水塘抽样

```java
class Solution {
    private int m, n, total;
    private Random random = new Random();
    private Map<Integer, Integer> map = new HashMap<>();

    public Solution(int m, int n) {
        this.m = m;
        this.n = n;
        total = m * n;
    }

    public int[] flip() {
        int ri = random.nextInt(total); // [0,total)随机位置
        total--;
        int idx = map.getOrDefault(ri, ri); // 该位置上的数字，初始为值为位置
        map.put(ri, map.getOrDefault(total, total)); // TODO 精髓，该位置为当前total位置的值（把后面置换到前面）
        return new int[]{idx / n, idx % n};
    }

    public void reset() {
        total = this.m * this.n;
        map.clear();
    }
}

/**
 * Your Solution object will be instantiated and called as such:
 * Solution obj = new Solution(m, n);
 * int[] param_1 = obj.flip();
 * obj.reset();
 */
```

## [786. 第 K 个最小的素数分数](https://leetcode-cn.com/problems/k-th-smallest-prime-fraction/)

> 数组，二分查找，堆，优先队列，排序

```java
class Solution {
    public int[] kthSmallestPrimeFraction(int[] arr, int k) {
        List<int[]> list = new ArrayList<>();
        int n = arr.length;
        for (int i = 0; i < n - 1; ++i) {
            for (int j = i + 1; j < n; ++j) {
                list.add(new int[]{arr[i], arr[j]});
            }
        }
        Collections.sort(list, (x, y) -> (x[0] * y[1] - x[1] * y[0]));
        return list.get(k - 1);
    }
}
```

TODO 优先队列/二分查找+双指针

## [400. 第 N 位数字](https://leetcode-cn.com/problems/nth-digit/)

> 数学，二分查找

```java
class Solution {
    public int findNthDigit(int n) {
        // 求d位和d位有几个
        int d = 1, count = 9;
        while (n > (long)d * count) {
            n -= d * count;
            d++;
            count *= 10;
        }
        int index = n - 1;
        // d:2 count:90 rest:n
        int start = (int)Math.pow(10, d - 1); // start~end
        int end = start + index / d; // 从start数n%d个数即是end
        int digit = index % d; // end的第几位
        String str = end + "";
        return str.charAt(digit) - '0';
    }
}
```

二分查找TODO

## [1446. 连续字符](https://leetcode-cn.com/problems/consecutive-characters/)

> 字符串

```java
class Solution {
    public int maxPower(String s) {
        int max = 1, tmp = 1;
        for (int i = 1; i < s.length(); ++i) {
            if (s.charAt(i) == s.charAt(i - 1)) {
                tmp++;
            } else {
                max = Math.max(max, tmp);
                tmp = 1;
            }
        }
        return Math.max(max, tmp);
    }
}
```

## [506. 相对名次](https://leetcode-cn.com/problems/relative-ranks/)

> 数组，排序，堆，优先队列

```java
class Solution {
    public String[] findRelativeRanks(int[] score) {
        int n = score.length;
        int[][] scoreMap = new int[n][2];
        for (int i = 0; i < n; ++i) {
            scoreMap[i][0] = score[i];
            scoreMap[i][1] = i;
        }
        Arrays.sort(scoreMap, (x, y) -> {
            return y[0] - x[0];
        });
        String[] res = new String[n];
        for (int i = 0; i < n; ++i) {
            String str = "";
            if (i == 0) str = "Gold Medal";
            else if (i == 1) str = "Silver Medal";
            else if (i == 2) str = "Bronze Medal";
            else str = Integer.toString(i + 1);
            res[scoreMap[i][1]] = str;
        }
        return res;
    }
}
```

flat

```java
class Solution {
    public String[] findRelativeRanks(int[] score) {
        int n = score.length;
        int max = 0;
        for (int i = 0; i < n; ++i) {
            max = Math.max(max, score[i]);
        }
        int[] map = new int[max + 1];
        for (int i = 0; i < n; ++i) {
            map[score[i]] = i + 1;
        }
        int cnt = 1;
        String[] res = new String[n];
        for (int i = max; i >= 0; --i) {
            if (map[i] != 0) {
                switch(cnt) {
                    case 1:
                        res[map[i] - 1] = "Gold Medal";
                        break;
                    case 2:
                        res[map[i] - 1] = "Silver Medal";
                        break;
                    case 3:
                        res[map[i] - 1] = "Bronze Medal";
                        break;
                    default:
                        res[map[i] - 1] = Integer.toString(cnt);
                        break;
                }
                cnt++;
            }
        }
        return res;
    }
}
```

## [1005. K 次取反后最大化的数组和](https://leetcode-cn.com/problems/maximize-sum-of-array-after-k-negations/)

> 贪心，数组，排序

```java
class Solution {
    public int largestSumAfterKNegations(int[] nums, int k) {
        Arrays.sort(nums);
        int res = 0;
        for (int i = 0; i < nums.length; ++i) {
            if (nums[i] < 0 && k > 0) {
                k--;
                res += -nums[i];
            } else if (nums[i] < 0 && k >= 0) {
                res += nums[i];
            } else {
                if (k > 0) {
                    if ((k & 1) == 1) {
                        if (i > 0) {
                            if (-nums[i] > nums[i - 1]) {
                                res += -nums[i];
                            } else {
                                res += nums[i] + 2 * nums[i - 1] ;
                            }
                        } else {
                            res += -nums[i];
                        }
                    } else {
                        res += nums[i];
                    }
                    k = 0;
                } else {
                    res += nums[i];
                }
            }
        }
        if (k > 0 && (k & 1) == 1) {
            res += 2 * nums[nums.length - 1];
        }
        return res;
    }
}
```

## [383. 赎金信](https://leetcode-cn.com/problems/ransom-note/)

> 哈希表，字符串，计数

```java
class Solution {
    public boolean canConstruct(String ransomNote, String magazine) {
        int[] charNum = new int[26];
        for (char c: magazine.toCharArray()) {
            charNum[c - 'a']++;
        }
        for (char c: ransomNote.toCharArray()) {
            if (charNum[c - 'a'] <= 0) return false;
            charNum[c - 'a']--;
        }
        return true;
    }
}
```

## [372. 超级次方](https://leetcode-cn.com/problems/super-pow/)

> 快速幂，秦九昭算法，数学

```java
class Solution {
    private int MOD = 1337;

    public int superPow(int a, int[] b) {
        int ans = 1;
        for (int x: b) {
            ans = (int)((long)pow(ans, 10) * pow(a, x) % MOD);
        }
        return ans;
    }

    // 快速幂 x^y
    private int pow(int x, int y) {
        int res = 1;
        while (y > 0) {
            if ((y & 1) == 1) {
                res = (int)((long)res * x % MOD);
            }
            x = (int)((long)x * x % MOD);
            y >>= 1;
        }
        return res;
    }
}
```

## [50. Pow(x, n)](https://leetcode-cn.com/problems/powx-n/)

> 快速幂，数学，递归

```java
class Solution {
    public double myPow(double x, int n) {
        if (x == 0) return 0;
        if (x == 1d) return 1;
        if (x == -1d) return (n & 1) == 1 ? -1 : 1;
        if (n == 0 || x == 1d) return 1;
        if (n == 1) return x;
        if (n < 0)
            if (n == -2147483648) return 0;
            else return 1d / myPow(x, -n);
        double ans = 1d;
        while (n > 0) {
            if ((n & 1) == 1) {
                ans *= x;
            }
            x *= x;
            n >>= 1;
        }
        return ans;
    }
}
```

## [1816. 截断句子](https://leetcode-cn.com/problems/truncate-sentence/)

> 数组，字符串

执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：36.3 MB, 在所有 Java 提交中击败了90.95%的用户

```java
class Solution {
    public String truncateSentence(String s, int k) {
        for (int i = 0; i < s.length(); ++i) {
            if (s.charAt(i) == ' ') {
                k--;
                if (k == 0) {
                    return s.substring(0, i);
                }
            }
        }
        return s;
    }
}
```

## [1034. 边界着色](https://leetcode-cn.com/problems/coloring-a-border/)

> 深度优先搜索，广度优先搜索，数组，矩阵

```java
class Solution {
    private int[][] directions = new int[][]{{0, 1}, {1, 0}, {-1, 0}, {0, -1}};
    private boolean[][] visited;
    private int m, n;
    private List<int[]> borders = new ArrayList<>();

    public int[][] colorBorder(int[][] grid, int row, int col, int color) {
        int pointColor = grid[row][col];
        m = grid.length;
        n = grid[0].length;
        visited = new boolean[m][n];
        dfs(grid, row, col, pointColor);
        for (int[] border: borders) {
            grid[border[0]][border[1]] = color;
        }
        return grid;
    }

    private void dfs(int[][] grid, int x, int y, int color) {
        boolean isBorder = false;
        for (int[] direction: directions) {
            int xx = x + direction[0], yy = y + direction[1];
            if (xx >= m || yy >= n || xx < 0 || yy < 0 || grid[xx][yy] != color) { // 四周有出界的
                isBorder = true;
            } else if (!visited[xx][yy]) { // 没出界且未访问
                visited[xx][yy] = true;
                dfs(grid, xx, yy, color);
            }
        }
        if (isBorder) {
            borders.add(new int[]{x, y});
        }
    }
}
```

BFS TODO

## [689. 三个无重叠子数组的最大和](https://leetcode-cn.com/problems/maximum-sum-of-3-non-overlapping-subarrays/)

> 滑动数组，动态规划

```java
class Solution {
    public int[] maxSumOfThreeSubarrays(int[] nums, int k) {
        // 三个片段同时滑动
        int[] res = new int[3];
        int sum1 = 0, sum2 = 0, sum3 = 0;
        int max1 = 0, max12 = 0, max123 = 0;
        int max1Idx = 0, max12Idx1 = 0, max12Idx2 = 0;
        int n = nums.length;
        for (int i = k * 2; i < n; ++i) { // i表示第三个片段的右端点
            sum1 += nums[i - 2 * k];
            sum2 += nums[i - k];
            sum3 += nums[i];
            if (i >= k * 3 - 1) {
                if (sum1 > max1) {
                    max1 = sum1;
                    max1Idx = i - k * 3 + 1;
                }
                if (max1 + sum2 > max12) {
                    max12 = max1 + sum2;
                    max12Idx1 = max1Idx;
                    max12Idx2 = i - k * 2 + 1;
                }
                if (max12 + sum3 > max123) {
                    max123 = max12 + sum3;
                    res[0] = max12Idx1;
                    res[1] = max12Idx2;
                    res[2] = i - k + 1;
                }
                sum1 -= nums[i - k * 3 + 1];
                sum2 -= nums[i - k * 2 + 1];
                sum3 -= nums[i - k + 1];
            }
        }
        return res;
    }
}
```

## [794. 有效的井字游戏](https://leetcode-cn.com/problems/valid-tic-tac-toe-state/)

> 数组，字符串

执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：35.8 MB, 在所有 Java 提交中击败了33.14%的用户

```java
class Solution {
    public boolean validTicTacToe(String[] board) {
        int cx = 0, co = 0, xline = 0, oline = 0;
        for (String b: board) {
            if (b.equals("XXX")) {
                xline++;
                if (xline + oline > 1) return false;
            }
            if (b.equals("OOO")) {
                oline++;
                if (xline + oline > 1) return false;
            }
            for (char c: b.toCharArray()) {
                if (c == 'O') co++;
                else if (c == 'X') cx++;
            }
        }
        for (int i = 0; i < 3; ++i) {
            char c = board[0].charAt(i);
            if (c == 'X' && board[1].charAt(i) == 'X' && board[2].charAt(i) == 'X')
                xline++;
            if (c == 'O' && board[1].charAt(i) == 'O' && board[2].charAt(i) == 'O')
                oline++;
        }
        if (board[0].charAt(0) == 'X' && board[1].charAt(1) == 'X' && board[2].charAt(2) == 'X')
            xline++;
        if (board[2].charAt(0) == 'X' && board[1].charAt(1) == 'X' && board[0].charAt(2) == 'X')
            xline++;
        if (board[0].charAt(0) == 'O' && board[1].charAt(1) == 'O' && board[2].charAt(2) == 'O')
            oline++;
        if (board[2].charAt(0) == 'O' && board[1].charAt(1) == 'O' && board[0].charAt(2) == 'O')
            oline++;
        return 
            (cx == 5 && co == 4 && oline == 0) ||  // 数量超过1的必然是5个X和4个O且O不能连线
            (oline + xline == 0 && (cx == co || cx == co + 1)) || // 数量为0，则X和O相等或者X比O多一个都满足要求
            (oline == 1 && xline == 0 && cx == co) || // 数量为1（O为1，X为0），则X和O必须相等 
            (xline == 1 && oline == 0 && cx == co + 1); // 数量为1（O为0，X为1），则X必须比O多1
    }
}
```

## [748. 最短补全词](https://leetcode-cn.com/problems/shortest-completing-word/)

> 数组，哈希表，字符串

执行用时：2 ms, 在所有 Java 提交中击败了99.84%的用户

内存消耗：38.6 MB, 在所有 Java 提交中击败了78.42%的用户

```java
class Solution {
    public String shortestCompletingWord(String licensePlate, String[] words) {
        int[] cc = covert(licensePlate, false);
        int minLen = Integer.MAX_VALUE;
        String res = "";
        for (String word: words) {
            if (word.length() < minLen && isContains(covert(word, true), cc)) {
                minLen = word.length();
                res = word;
            }
        }
        return res;
    }

    private boolean isContains(int[] a, int[] b) {
        for (int i = 0; i < 26; ++i) {
            if (a[i] - b[i] < 0)
                return false;
        }
        return true;
    }

    private int[] covert(String str, boolean isWord) {
        int[] cc = new int[26];
        for (char c: str.toCharArray()) {
            if (!isWord) {
                if (Character.isLetter(c)) {
                    char lc = Character.toLowerCase(c);
                    cc[lc - 'a']++;
                }
            } else {
                cc[c - 'a']++;
            }
        }
        return cc;
    }
}
```

## [911. 在线选举](https://leetcode-cn.com/problems/online-election/)

> 设计，数组，哈希表，二分查找

```java
class TopVotedCandidate {
    private Map<Integer, Integer> votes = new HashMap<>();
    private List<Integer> tops = new ArrayList<>();
    private int maxVotePerson = -1; // 当前最高票的位置
    private int[] times;

    public TopVotedCandidate(int[] persons, int[] times) {
        this.times = times;
        votes.put(-1, -1); // 记录-1位置票数为-1
        for (int i = 0; i < persons.length; ++i) {
            int p = persons[i];
            int curVotes = votes.getOrDefault(p, 0) + 1; // person当前票数
            votes.put(p, curVotes); // 记录当前时刻person的票数
            if (curVotes >= votes.get(maxVotePerson)) { // 超过了当前最高票
                maxVotePerson = p; // 记录当前时刻最高票人选
            }
            tops.add(maxVotePerson); // 记录当前时刻最高票人选
        }
    }

    public int q(int t) {
        // 二分查找t的位置 TODO 比较特殊的二分查找，需要注意
        int l = 0, r = times.length - 1;
        while (l < r) {
            int mid = l + (r - l) / 2;
            if (times[mid] == t) {
                return tops.get(mid);
            } else if (times[mid] > t) {
                r = mid - 1;
            } else {
                l = mid + 1;
            }
        }
        if (times[l] > t) return tops.get(l - 1);
        return tops.get(l);
    }
}

/**
 * Your TopVotedCandidate object will be instantiated and called as such:
 * TopVotedCandidate obj = new TopVotedCandidate(persons, times);
 * int param_1 = obj.q(t);
 */
```

## [100. 相同的树](https://leetcode-cn.com/problems/same-tree/)

> 树，深度优先搜索，广度优先搜索，二叉树，递归

执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：35.6 MB, 在所有 Java 提交中击败了82.15%的用户

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
    public boolean isSameTree(TreeNode p, TreeNode q) {
        if (p == null && q == null)
            return true;
        if (p == null && q != null || p != null && q == null)
            return false;
        return p.val == q.val && isSameTree(p.left, q.left) && isSameTree(p.right, q.right);
    }
}
```

## [807. 保持城市天际线](https://leetcode-cn.com/problems/max-increase-to-keep-city-skyline/)

> 贪心，数组，矩阵

执行用时：1 ms, 在所有 Java 提交中击败了83.37%的用户

内存消耗：37.9 MB, 在所有 Java 提交中击败了94.69%的用户

```java
class Solution {
    public int maxIncreaseKeepingSkyline(int[][] grid) {
        int n = grid.length;
        int[] maxLine = new int[n]; // 每一行的最大值
        int[] maxRow = new int[n]; // 每一列的最大值
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                maxLine[i] = Math.max(maxLine[i], grid[i][j]);
                maxRow[i] = Math.max(maxRow[i], grid[j][i]);
            }
        }
        int ans = 0;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                ans += Math.min(maxLine[i], maxRow[j]) - grid[i][j];
            }
        }
        return ans;
    }
}
```

## [1610. 可见点的最大数目](https://leetcode-cn.com/problems/maximum-number-of-visible-points/)

> 几何，数组，数学，排序，滑动窗口

TODO eps

```java
class Solution {
    public int visiblePoints(List<List<Integer>> points, int angle, List<Integer> location) {
        double eps = 1e-9; // double求相等的误差允许范围
        int posx = location.get(0), posy = location.get(1);
        int sameCnt = 0;
        List<Double> angles = new ArrayList<>();
        for (List<Integer> point: points) {
            int x = point.get(0);
            int y = point.get(1);
            if (x == posx && y == posy) {
                sameCnt++;
                continue;
            }
            angles.add(Math.atan2(posx - x, posy - y));
        }
        Collections.sort(angles);
        // 滑动区间（循环数组查找最长连续段，扩展数组为2倍）[i,j],i从0到n，j从0到2n
        int max = 0;
        double rangeAngle = angle * Math.PI / 180;
        int n = angles.size(), doublen = 2 * n;
        for (int i = 0; i < n; ++i) {
            angles.add(angles.get(i) + 2 * Math.PI);
        }
        int i = 0, j = 0;
        while (j < doublen) {
            while (i < j && angles.get(j) - angles.get(i) > rangeAngle + eps)
                i++; // 收缩
            max = Math.max(max, j - i + 1);
            j++;
        }
        return max + sameCnt;
    }
}
```

## [1518. 换酒问题](https://leetcode-cn.com/problems/water-bottles/)

> 数学，模拟

```java
class Solution {
    public int numWaterBottles(int numBottles, int numExchange) {
        int ret = numBottles;
        while (numBottles >= numExchange) {
            int full = numBottles / numExchange; // 新的酒数量
            ret += full; // 喝掉
            numBottles = full + numBottles % numExchange; // 更新空酒瓶（喝掉后的瓶和本来就空的瓶）
        }
        return ret;
    }
}
```

## [419. 甲板上的战舰](https://leetcode-cn.com/problems/battleships-in-a-board/)

> 深度优先搜索，数组，矩阵

```java
class Solution {
    public int countBattleships(char[][] board) {
        int res = 0, m = board.length, n = board[0].length;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (board[i][j] == 'X' && 
                (i == 0 || i > 0 && board[i - 1][j] != 'X') &&
                (j == 0 || j > 0 && board[i][j - 1] != 'X')) {
                    res++;
                }
            }
        }
        return res;
    }
}
```

## [997. 找到小镇的法官](https://leetcode-cn.com/problems/find-the-town-judge/)

> 图，数组，哈希表

```java
class Solution {
    public int findJudge(int n, int[][] trust) {
        if (n == 1 && trust.length == 0)
            return 1;
        Map<Integer, Integer> trustMap = new HashMap<>();
        Map<Integer, Integer> cntMap = new HashMap<>();
        for (int[] t: trust) {
            trustMap.put(t[0], t[1]);
            cntMap.put(t[1], cntMap.getOrDefault(t[1], 0) + 1);
        }
        for (int x: cntMap.keySet()) {
            if (cntMap.get(x) == n - 1 && trustMap.get(x) == null) {
                return x;
            }
        }
        return -1;
    }
}
```

入度和出度

执行用时：2 ms, 在所有 Java 提交中击败了99.21%的用户

内存消耗：45.8 MB, 在所有 Java 提交中击败了83.14%的用户

```java
class Solution {
    public int findJudge(int n, int[][] trust) {
        int[] inDegrees = new int[n + 1];
        int[] outDegrees = new int[n + 1];
        for (int[] t: trust) {
            inDegrees[t[1]]++;
            outDegrees[t[0]]++;
        }
        for (int i = 1; i <= n; ++i) {
            if (inDegrees[i] == n - 1 && outDegrees[i] == 0) {
                return i;
            }
        }
        return -1;
    }
}
```

## [1154. 一年中的第几天](https://leetcode-cn.com/problems/day-of-the-year/)

> 数学，字符串

```java
class Solution {
    public int dayOfYear(String date) {
        int[] days = new int[]{0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335};
        String[] strs = date.split("-");
        int year = Integer.parseInt(strs[0]);
        int mouth = Integer.parseInt(strs[1]);
        int day = Integer.parseInt(strs[2]);
        if (mouth <= 2 || year % 400 == 0 || (year % 4 == 0 && year % 100 != 0)) { // TODO 闰年判断
            return days[mouth - 1] + day;
        } else {
            return days[mouth - 1] + day - 1;
        }
    }
}
```

## [686. 重复叠加字符串匹配](https://leetcode-cn.com/problems/repeated-string-match/)

> 字符串，字符串匹配

找上下界

```java
class Solution {
    public int repeatedStringMatch(String a, String b) {
        // len(a) * n（下界） >= 匹配 len(b)
        // len(a) * (n+1) (上界) 匹配 len(b)
        int lena = a.length(), lenb = b.length(), cnt = 0;
        StringBuilder sb = new StringBuilder();
        while (lena * ++cnt < lenb) {
            sb.append(a);
        }
        sb.append(a);
        if (sb.toString().indexOf(b) != -1) {
            return cnt;
        }
        sb.append(a);
        return sb.toString().indexOf(b) != -1 ? cnt + 1 : -1;
    }
}
```

KMP TODO

## [1705. 吃苹果的最大数目](https://leetcode-cn.com/problems/maximum-number-of-eaten-apples/)

> 贪心，数组，优先队列

```java
class Solution {
    public int eatenApples(int[] apples, int[] days) {
        // PriorityQueue<[过期日,数量]> 优先取过期日小的（最快过期的一个）[小顶堆]
        int n = apples.length;
        PriorityQueue<int[]> q = new PriorityQueue<>((x, y) -> x[0] - y[0]);
        int i = 0, ans = 0;
        while (i < n || !q.isEmpty()) {
            while (!q.isEmpty() && q.peek()[0] < i) { // 排除已经腐烂的
                q.poll();
            }
            if (i < n && apples[i] > 0) { // 当天有苹果，将苹果加入
                q.offer(new int[]{i + days[i] - 1, apples[i]});
            }
            if (!q.isEmpty()) { // 有没过期的
                int[] top = q.poll();
                if (--top[1] > 0 && top[0] > i) { // 之后仍然足够
                    q.offer(top);
                }
                ans++;
            }
            i++;
        }
        return ans;
    }
}
```

## [1609. 奇偶树](https://leetcode-cn.com/problems/even-odd-tree/)

> 树，广度优先搜索，二叉树，层序遍历

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
    public boolean isEvenOddTree(TreeNode root) {
        Deque<TreeNode> q = new LinkedList<>();
        q.push(root);
        boolean isOddLevel = false; // 是否奇数层
        while (!q.isEmpty()) {
            int size = q.size(); // 这一层有几个
            int last = isOddLevel ? 1000001 : 0;
            while (--size >= 0) {
                TreeNode cur = q.pollFirst();
                if (
                    isOddLevel && ((cur.val & 1) == 1 || cur.val >= last) ||
                    !isOddLevel && ((cur.val & 1) == 0 || cur.val <= last)
                )
                    return false;
                last = cur.val;
                if (cur.left != null) q.offerLast(cur.left);
                if (cur.right != null) q.offerLast(cur.right);
            }
            isOddLevel = !isOddLevel;
        }
        return true;
    }
}
```

## [1078. Bigram 分词](https://leetcode-cn.com/problems/occurrences-after-bigram/)

> 字符串

```java
class Solution {
    public String[] findOcurrences(String text, String first, String second) {
        List<String> ws = new ArrayList<>();
        String[] words = text.split(" ");
        for (int i = 0; i < words.length - 2; ++i) {
            if (first.equals(words[i]) && second.equals(words[i + 1])) {
                ws.add(words[i + 2]);
            }
        }
        return ws.toArray(new String[0]);
    }
}
```

## [825. 适龄的朋友](https://leetcode-cn.com/problems/friends-of-appropriate-ages/)

> 数组，双指针，前缀和，双轴排序，桶排序

排序+双指针

```java
class Solution {
    public int numFriendRequests(int[] ages) {
        int ans = 0, n = ages.length;
        Arrays.sort(ages);
        int l = 0, r = 0;
        for (int age: ages) {
            if (age < 15) continue;
            while (ages[l] <= 0.5 * age + 7) l++;
            while (r < n && ages[r] <= age) r++;
            ans += r - l - 1; // 去掉自己
        }
        return ans;
    }
}
```

桶排序+前缀和

```java
class Solution {
    public int numFriendRequests(int[] ages) {
        int[] cnt = new int[121]; // 桶数组
        for (int age: ages) {
            cnt[age]++;
        }
        int[] pre = new int[121]; // 前缀和
        for (int i = 1; i < 121; ++i) {
            pre[i] = pre[i - 1] + cnt[i];
        }
        int ans = 0;
        for (int i = 15; i < 121; ++i) { // i表示年龄
            if (cnt[i] > 0) {
                int l = (int)(0.5 * i + 8) - 1;
                ans += (pre[i] - pre[l] - 1) * cnt[i];
            }
        }
        return ans;
    }
}
```

## [472. 连接词](https://leetcode-cn.com/problems/concatenated-words/)

> 深度优先搜索，字典树，数组，字符串，记忆化

```java
class Solution {
    class Trie {
        TrieNode root;

        class TrieNode {
            TrieNode[] child = new TrieNode[26];
            boolean isEnd = false;
            String word = "";
        }

        public Trie() {
            root = new TrieNode();
        }

        public void insert(String word) {
            TrieNode p = root;
            for (int i = 0; i < word.length(); ++i) {
                int u = word.charAt(i) - 'a';
                if (p.child[u] == null) {
                    p.child[u] = new TrieNode();
                }
                p = p.child[u];
            }
            p.isEnd = true;
            p.word = word;
        }

        public boolean dfs(String word, int start) {
            if (word.length() == start) {
                return true;
            }
            TrieNode p = root;
            for (int i = start; i < word.length(); i++) {
                int u = word.charAt(i) - 'a';
                if (p.child[u] == null) {
                    return false;
                }
                p = p.child[u];
                if (p.isEnd && dfs(word, i + 1)) {
                    return true;
                }
            }
            return false;
        }
    }

    public List<String> findAllConcatenatedWordsInADict(String[] words) {
        Trie t = new Trie();
        Arrays.sort(words, (x, y) -> x.length() - y.length());
        t.insert(words[0]);
        List<String> ret = new ArrayList<>();
        for (int i = 1; i < words.length; ++i) {
            if (words[i].length() == 0) continue;
            if (t.dfs(words[i], 0)) {
                ret.add(words[i]);
            } else {
                t.insert(words[i]);
            }
        }
        return ret;
    }
}
```

TODO 记忆化

## [1995. 统计特殊四元组](https://leetcode-cn.com/problems/count-special-quadruplets/)

> 数组，枚举，哈希

```java
class Solution {
    public int countQuadruplets(int[] nums) {
        int ans = 0;
        Map<Integer, Integer> map = new HashMap<>();
        int n = nums.length;
        // nums[a] + nums[b] = nums[d] - nums[c];
        for (int b = n - 3; b >= 1; --b) {
            for (int d = b + 2; d < n; ++d) {
                map.put(nums[d] - nums[b + 1], map.getOrDefault(nums[d] - nums[b + 1], 0) + 1);
            }
            for (int a = 0; a < b; ++a) {
                ans += map.getOrDefault(nums[a] + nums[b], 0);
            }
        }
        return ans;
    }
}
```

TODO 组合优化，维度背包

## [846. 一手顺子](https://leetcode-cn.com/problems/hand-of-straights/)

> 贪心，数组，哈希表，排序

执行用时：13 ms, 在所有 Java 提交中击败了95.15%的用户

内存消耗：39.7 MB, 在所有 Java 提交中击败了40.30%的用户

```java
class Solution {
    public boolean isNStraightHand(int[] hand, int groupSize) {
        int n = hand.length;
        if (n % groupSize != 0)
            return false;
        Arrays.sort(hand);
        ArrayList<int[]> cnts = new ArrayList<>();
        int x = hand[0];
        int cnt = 1;
        for (int i = 1; i < n; ++i) {
            if (hand[i] != x) {
                cnts.add(new int[]{x, cnt});
                x = hand[i];
                cnt = 1;
            } else {
                cnt++;
            }
        }
        cnts.add(new int[]{x, cnt});
        int l = 0, r = 0;
        while (r < cnts.size()) {
            int counter = groupSize;
            boolean isStart = true;
            while (counter > 0) {
                if (r >= cnts.size() || cnts.get(r)[1] == 0 || !isStart && r > 0 && cnts.get(r)[0] != cnts.get(r - 1)[0] + 1)
                    return false;
                isStart = false;
                cnts.get(r)[1]--;
                r++;
                counter--;
            }
            while (l < cnts.size() && cnts.get(l)[1] == 0) l++;
            r = l;
        }
        return true;
    }
}
```

本题以下：1296，1296用了更简单的方法

## [1296. 划分数组为连续数字的集合](https://leetcode-cn.com/problems/divide-array-in-sets-of-k-consecutive-numbers/)

> 贪心，数组，哈希表，排序

```java
class Solution {
    public boolean isPossibleDivide(int[] nums, int k) {
        if (nums.length % k != 0) return false;
        Arrays.sort(nums);
        Map<Integer, Integer> map = new HashMap<>();
        for (int i: nums) {
            map.put(i, map.getOrDefault(i, 0) + 1);
        }
        for (int i: nums) {
            if (!map.containsKey(i)) continue;
            for (int j = 0; j < k; ++j) {
                int cur = i + j;
                if (!map.containsKey(cur))
                    return false;
                int newCur = map.get(cur) - 1;
                if (newCur == 0) {
                    map.remove(cur);
                } else {
                    map.put(cur, newCur);
                }
            }
        }
        return true;
    }
}
```

## [507. 完美数](https://leetcode-cn.com/problems/perfect-number/)

> 数学，枚举

用i<=num/i代替sqrt

执行用时：1 ms, 在所有 Java 提交中击败了92.45%的用户

内存消耗：35.4 MB, 在所有 Java 提交中击败了10.46%的用户

```java
class Solution {
    public boolean checkPerfectNumber(int num) {
        if (num == 1) return false;
        int sum = 1;
        for (int i = 2; i <= num / i; ++i) {
            sum += (num % i == 0) ? ((i * i == num) ? i : (i + num / i)) : 0;
        }
        return sum == num;
    }
}
```

TODO 欧几里得-欧拉定理

## [2022. 将一维数组转变成二维数组](https://leetcode-cn.com/problems/convert-1d-array-into-2d-array/)

> 数组，矩阵，模拟

```java
class Solution {
    public int[][] construct2DArray(int[] original, int m, int n) {
        int len = original.length;
        if (len != m * n) return new int[0][0]; // int[0][]也可
        int[][] ans = new int[m][n];
        for (int i = 0; i < len; ++i) {
            ans[i / n][i % n] = original[i]; // TODO System.arraycopy
        }
        return ans;
    }
}
```

## [390. 消除游戏](https://leetcode-cn.com/problems/elimination-game/)

> 数学，等差数列，类约瑟夫环

等差数列

```java
class Solution {
    public int lastRemaining(int n) {
        int start = 1, step = 1, cnt = n, k = 0;
        while (cnt > 1) {
            if ((k & 1) == 0) { // 偶数轮次向右，start是前一轮start的step后
                start += step;
            } else { // 奇数轮次向左，剩余个数为偶数则start不变，奇数则start是前一轮start的step后
                start = (cnt & 1) == 0 ? start : start + step;
            }
            cnt >>= 1; // 个数减半
            step <<= 1; // 步长增大1倍
            k++; // 轮次加1
        }
        return start;
    }
}
```

约瑟夫环递推法 TODO

## [1185. 一周中的第几天](https://leetcode-cn.com/problems/day-of-the-week/)

> 数学，日期

```java
class Solution {
    private int[] md = new int[]{31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    private String[] week = new String[]{"Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"};
    public String dayOfTheWeek(int day, int month, int year) {
        int d = 4;
        for (int i = 1971; i < year; ++i) {
            if (i % 400 == 0 || i % 100 != 0 && i % 4 == 0) {
                d += 366;
            } else {
                d += 365;
            }
        }
        for (int i = 0; i < month - 1; ++i) {
            d += md[i];
            if (i == 2 && (year % 400 == 0 || year % 100 != 0 && year % 4 == 0))
                d += 1;
        }
        d += day;
        return week[d % 7];
    }
}
```

## [1576. 替换所有的问号](https://leetcode-cn.com/problems/replace-all-s-to-avoid-consecutive-repeating-characters/)

> 字符串

```java
class Solution {
    public String modifyString(String s) {
        char[] cs = s.toCharArray();
        int n = cs.length;
        for (int i = 0; i < n; ++i) {
            if (cs[i] == '?') {
                char left = 'd', right = 'd';
                if (i >= 1) {
                    left = cs[i - 1];
                }
                if (i < n - 1) {
                    right = cs[i + 1];
                }
                char x = 'a';
                if (x == left || x == right) {
                    x = 'b';
                    if (x == left || x == right) {
                        x = 'c';
                    }
                }
                cs[i] = x;
            }
        }
        return new String(cs);
    }
}
```

## [71. 简化路径](https://leetcode-cn.com/problems/simplify-path/)

> 栈，字符串

执行用时：3 ms, 在所有 Java 提交中击败了92.76%的用户

内存消耗：38.5 MB, 在所有 Java 提交中击败了63.73%的用户

```java
class Solution {
    public String simplifyPath(String path) {
        Deque<String> stack = new LinkedList<>();
        StringBuilder sb = new StringBuilder();
        path = path + "/";
        for (char c: path.toCharArray()) {
            if(c == '/') {
                if (sb.length() != 0 && !sb.toString().equals("..") && !sb.toString().equals(".")) {
                    stack.push(sb.toString());
                } else if (sb.length() != 0 && sb.toString().equals("..")) {
                    if (!stack.isEmpty())
                        stack.pop();
                }
                sb.setLength(0);
            } else {
                sb.append(c);
            }
        }
        StringBuilder res = new StringBuilder();
        while (!stack.isEmpty()) {
            res.insert(0, stack.pop()).insert(0, '/');
        }
        return res.length() == 0 ? "/" : res.toString();
    }
}
```

## [104. 二叉树的最大深度](https://leetcode-cn.com/problems/maximum-depth-of-binary-tree/)

> 树，深度优先搜索，广度优先搜索，二叉树

执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户

内存消耗：38.4 MB, 在所有 Java 提交中击败了30.98%的用户

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
    public int maxDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        return Math.max(maxDepth(root.left), maxDepth(root.right)) + 1;
    }
}
```

## [89. 格雷编码](https://leetcode-cn.com/problems/gray-code/)

> 位运算，数学，回溯

对称生成 0->1->1(1)->1(0)->1(10)->1(11)->1(01)->1(00)->...

执行用时：7 ms, 在所有 Java 提交中击败了66.30%的用户

内存消耗：45.6 MB, 在所有 Java 提交中击败了51.02%的用户

```java
class Solution {
    public List<Integer> grayCode(int n) {
        List<Integer> res = new ArrayList<>();
        res.add(0);
        res.add(1);
        int width = 1;
        while (width < n) {
            for (int i = res.size() - 1; i >= 0; --i) {
                res.add(res.get(i) | (1 << width));
            }
            width++;
        }
        return res;
    }
}
```

也可以反过来

```java
class Solution {
    public List<Integer> grayCode(int n) {
        List<Integer> res = new ArrayList<>();
        res.add(0);
        while (n-- > 0) {
            for (int i = res.size() - 1; i >= 0; --i) {
                res.add(res.get(i) | (1 << n));
            }
        }
        return res;
    }
}
```

TODO 二进制数转格雷码

执行用时：4 ms, 在所有 Java 提交中击败了96.71%的用户

内存消耗：45.4 MB, 在所有 Java 提交中击败了68.50%的用户

```java
class Solution {
    public List<Integer> grayCode(int n) {
        List<Integer> ret = new ArrayList<Integer>();
        for (int i = 0; i < 1 << n; i++) {
            ret.add((i >> 1) ^ i);
        }
        return ret;
    }
}
```

## [1629. 按键持续时间最长的键](https://leetcode-cn.com/problems/slowest-key/)

> 数组，字符串

```java
class Solution {
    public char slowestKey(int[] releaseTimes, String keysPressed) {
        char res = keysPressed.charAt(0);
        int max = releaseTimes[0];
        int[] cnt = new int[26];
        cnt[res - 'a'] = max;
        for (int i = 1; i < keysPressed.length(); ++i) {
            char c = keysPressed.charAt(i);
            int offset = c - 'a';
            int time = releaseTimes[i] - releaseTimes[i - 1];
            if (time == max && c > res) {
                res = c;
            } else if (time > max) {
                max = time;
                res = c;
            }
        }
        return res;
    }
}
```

## [306. 累加数](https://leetcode-cn.com/problems/additive-number/)

> 字符串，枚举，回溯，高精度，深度优先搜索，剪枝

```java
class Solution {
    public boolean isAdditiveNumber(String num) {
        // 确定前两个数的长度[1,(n-1)/2][1,(n-1)/2]
        int n = num.length();
        int half = (n - 1) / 2;
        for (int i = 1; i <= half; ++i) {
            for (int j = 1; i + j < n; ++j) {
                if (num.charAt(i) == '0' && j > 1)
                    continue;
                if (check(num, 0, i, i + j))
                    return true;
            }
        }
        return false;
    }

    private boolean check(String num, int a, int b, int c) {
        if (c >= num.length()) return false;
        String cur1 = num.substring(a, b);
        String cur2 = num.substring(b, c);
        if (cur1.startsWith("0") && !cur1.equals("0") || cur2.startsWith("0") && !cur2.equals("0"))
            return false;
        String sum = add(cur1, cur2);
        int d = c + sum.length();
        if (d > num.length()) return false;
        if (num.substring(c, d).equals(sum)) {
            if (d == num.length()) return true;
            return check(num, b, c, d);
        } else {
            return false;
        }
    }

    private String add(String a, String b) {
        int carry = 0, lena = a.length(), lenb = b.length(), max = Math.max(lena, lenb);
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < max; ++i) {
            int aa = 0, bb = 0;
            if (i < lena) {
                aa = a.charAt(lena - 1 - i) - '0';
            }
            if (i < lenb) {
                bb = b.charAt(lenb - 1 - i) - '0';
            }
            int sum = aa + bb + carry;
            if (sum >= 10) {
                carry = 1;
                sb.insert(0, sum - 10);
            } else {
                carry = 0;
                sb.insert(0, sum);
            }
        }
        if (carry == 1) sb.insert(0, 1);
        return sb.toString();
    }
}
```

## [334. 递增的三元子序列](https://leetcode-cn.com/problems/increasing-triplet-subsequence/)

> 贪心，数组

```java
class Solution {
    public boolean increasingTriplet(int[] nums) {
        int first = nums[0], second = Integer.MAX_VALUE;
        for (int i = 1; i < nums.length; ++i) {
            if (nums[i] > second) {
                return true;
            } else if (nums[i] > first) {
                second = nums[i];
            } else {
                first = nums[i];
            }
        }
        return false;
    }
}
```
TODO 最长上升子序列LIS（贪心+二分）；序列DP超时；左右最小值数组

## [747. 至少是其他数字两倍的最大数](https://leetcode-cn.com/problems/largest-number-at-least-twice-of-others/)

> 数组，排序

```java
class Solution {
    public int dominantIndex(int[] nums) {
        int max = -1, nextMax = -1, idx = -1;
        for (int i = 0; i < nums.length; ++i) {
            if (nums[i] > max) {
                nextMax = max;
                max = nums[i];
                idx = i;
            } else if (nums[i] > nextMax) {
                nextMax = nums[i];
            }
        }
        return max >= nextMax * 2 ? idx : -1;
    }
}
```

## [1716. 计算力扣银行的钱](https://leetcode-cn.com/problems/calculate-money-in-leetcode-bank/)
> 
```java
class Solution {
    public int totalMoney(int n) {
        // 所有完整周钱
        int x = n / 7;
        int firstWeek = 28;
        int lastWeek = firstWeek + 7 * (x - 1);
        int week = (firstWeek + lastWeek) * x / 2;
        // 剩下的
        int y = n % 7;
        int firstDay = 1 + x;
        int lastDay = firstDay + y - 1;
        int day = (firstDay + lastDay) * y / 2;
        return week + day;
    }
}
```

## [382. 链表随机节点](https://leetcode-cn.com/problems/linked-list-random-node/)

> 蓄水池抽样，链表，数学，随机化

```java
class Solution {
    private List<Integer> list = new ArrayList<>();
    private Random rand = new Random();;

    public Solution(ListNode head) {
        while (head != null) {
            list.add(head.val);
            head = head.next;
        }
    }
    
    public int getRandom() {
        return list.get(rand.nextInt(list.size()));
    }
}
```
未知数量的随机可以用蓄水池/水塘抽样
```java
class Solution {
    ListNode head;
    Random random;

    public Solution(ListNode head) {
        this.head = head;
        random = new Random();
    }

    public int getRandom() {
        int i = 1, ans = 0;
        for (ListNode p = head; p != null; p = p.next) {
            if (random.nextInt(i) == 0) { // 1/i 的概率选中（替换为答案）
                ans = p.val;
            }
            ++i;
        }
        return ans;
    }
}
```

## [539. 最小时间差](https://leetcode-cn.com/problems/minimum-time-difference/)

> 数组，数学，鸽巢原理，字符串，排序

```java
class Solution {
    public int findMinDifference(List<String> timePoints) {
        int n = timePoints.size();
        if (n > 1440) return 0;
        Collections.sort(timePoints);
        int min = 1440;
        for (int i = 1; i < n; ++i) {
            if (timePoints.get(i).equals(timePoints.get(i - 1))) return 0;
            min = Math.min(min, getIntTime(timePoints.get(i)) - getIntTime(timePoints.get(i - 1)));
        }
        min = Math.min(min, 1440 - getIntTime(timePoints.get(n - 1)) + getIntTime(timePoints.get(0)));
        return min;
    }

    private int getIntTime(String str) {
        String[] ss = str.split(":");
        return Integer.parseInt(ss[0]) * 60 + Integer.parseInt(ss[1]);
    }
}
```

## [219. 存在重复元素 II](https://leetcode-cn.com/problems/contains-duplicate-ii/)

> 数组，哈希表，滑动窗口

执行用时：17 ms, 在所有 Java 提交中击败了93.76%的用户
内存消耗：47.4 MB, 在所有 Java 提交中击败了73.06%的用户

```java
class Solution {
    public boolean containsNearbyDuplicate(int[] nums, int k) {
        int n = nums.length;
        if (k >= n) {
            Set tmp = new HashSet<>();
            for (int num: nums) {
                tmp.add(num);
            }
            return tmp.size() != n;
        }
        Set<Integer> set = new HashSet<>();
        for (int i = 0; i <= k; ++i) {
            set.add(nums[i]);
        }
        if (set.size() < k + 1) return true;
        for (int i = k + 1; i < n; ++i) {
            set.remove(nums[i - k - 1]);
            set.add(nums[i]);
            if (set.size() < k + 1)
                return true;
        }
        return false;
    }
}
```
TODO 滑动窗口，哈希表

## [1332. 删除回文子序列](https://leetcode-cn.com/problems/remove-palindromic-subsequences/)

> 双指针，字符串，回文串

```java
class Solution {
    public int removePalindromeSub(String s) {
        // 只有本身就是回文串（1次）和2次两种可能
        int n = s.length();
        for (int i = 0; i < n / 2; ++i) {
            if (s.charAt(i) != s.charAt(n - i - 1)) {
                return 2;
            }
        }
        return 1;
    }
}
```

## [2034. 股票价格波动](https://leetcode-cn.com/problems/stock-price-fluctuation/submissions/)

> 设计，哈希表，数据流，有序列表，堆，优先队列

```java
class StockPrice {
    private int currentTime;
    private Map<Integer, Integer> map; // <时间,价格>
    private TreeMap<Integer, Integer> treeMap; // <价格,时间个数>

    public StockPrice() {
        map = new HashMap<>();
        treeMap = new TreeMap<>();
    }
    
    public void update(int timestamp, int price) {
        currentTime = Math.max(currentTime, timestamp);
        if (map.containsKey(timestamp)) {
            int old = map.get(timestamp);
            int cnt = treeMap.get(old);
            if (cnt == 1) {
                treeMap.remove(old);
            } else {
                treeMap.put(old, cnt - 1);
            }
        }
        map.put(timestamp, price);
        treeMap.put(price, treeMap.getOrDefault(price, 0) + 1);
    }
    
    public int current() {
        return map.get(currentTime);
    }
    
    public int maximum() {
        return treeMap.lastKey();
    }
    
    public int minimum() {
        return treeMap.firstKey();
    }
}
```
## [2013. 检测正方形](https://leetcode-cn.com/problems/detect-squares/)

> 数组，哈希，设计，计数

```java
class DetectSquares {
    Map<Integer, List<Integer>> lineMap;
    Map<String, Integer> map;

    public DetectSquares() {
        lineMap = new HashMap<>();
        map = new HashMap<>();
    }
    
    public void add(int[] point) {
        int x = point[0], y = point[1];
        String str = x + "," + y;
        map.put(str, map.getOrDefault(str, 0) + 1);
        if (lineMap.containsKey(x)) {
            List<Integer> list = lineMap.get(x);
            list.add(y);
        } else {
            List<Integer> list = new ArrayList<>();
            list.add(y);
            lineMap.put(x, list);
        }
    }
    
    public int count(int[] point) {
        int ret = 0;
        int x = point[0], y = point[1];
        if (lineMap.containsKey(x)) {
            for (Integer y2: lineMap.get(x)) {
                if (y != y2) {
                    int delta = Math.abs(y - y2);
                    // 左边
                    String l1Str = (x - delta) + "," + y;
                    String l2Str = (x - delta) + "," + y2;
                    ret += map.getOrDefault(l1Str, 0) * map.getOrDefault(l2Str, 0);
                    // 右边
                    String r1Str = (x + delta) + "," + y;
                    String r2Str = (x + delta) + "," + y2;
                    int[] r1 = new int[]{x + delta, y};
                    int[] r2 = new int[]{x + delta, y2};
                    ret += map.getOrDefault(r1Str, 0) * map.getOrDefault(r2Str, 0);
                }
            }
        }
        return ret;
    }
}

/**
 * Your DetectSquares object will be instantiated and called as such:
 * DetectSquares obj = new DetectSquares();
 * obj.add(point);
 * int param_2 = obj.count(point);
 */
```
TODO Map<x, Map<y, 个数>>更简单

## [884. 两句话中的不常见单词](https://leetcode-cn.com/problems/uncommon-words-from-two-sentences/)

> 哈希表，字符串

```java
class Solution {
    public String[] uncommonFromSentences(String s1, String s2) {
        StringBuilder sb = new StringBuilder();
        sb.append(s1).append(" ").append(s2);
        String[] strs = sb.toString().split(" ");
        HashMap<String, Integer> map = new HashMap();
        for (String str: strs) {
            map.put(str, map.getOrDefault(str, 0) + 1);
        }
        List<String> list = new ArrayList<>();
        for (String str: map.keySet()) {
            if (map.get(str) == 1)
                list.add(str);
        }
        return list.toArray(new String[0]);
    }
}
```

## [181. 超过经理收入的员工](https://leetcode-cn.com/problems/employees-earning-more-than-their-managers/)

> 数据库，自关联查询

```sql
SELECT e.Name AS Employee FROM Employee e LEFT JOIN Employee m ON m.Id = e.ManagerId WHERE e.Salary > m.Salary
```

## [183. 从不订购的客户](https://leetcode-cn.com/problems/customers-who-never-order/)

> 数据库

子查询
```sql
SELECT Name AS Customers FROM Customers WHERE Id NOT IN (SELECT CustomerId FROM Orders);
```
关联查询
```sql
ELECT c.Name AS Customers FROM Customers c LEFT JOIN Orders o ON o.CustomerId = c.Id WHERE o.CustomerId IS NULL
```

## [1763. 最长的美好子字符串](https://leetcode-cn.com/problems/longest-nice-substring/)

> 位运算，哈希表，字符串，滑动窗口

```java
class Solution {
    public String longestNiceSubstring(String s) {
        int n = s.length();
        String ret = "";
        int max = 0;
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                String str = s.substring(i, j + 1);
                int len = check(str);
                if (len > max) {
                    max = len;
                    ret = str;
                }
            }
        }
        return ret;
    }

    private int check(String str) {
        boolean[] cs = new boolean[64];
        for (char c: str.toCharArray()) {
            if ('A' <= c && c <= 'Z' && cs[c - 'A' + 32] == false) cs[c - 'A' + 32] = true;
            if ('a' <= c && c <= 'z' && cs[c - 'a'] == false) cs[c - 'a'] = true;
        }
        for (int i = 0; i < 32; ++i) {
            if (cs[i] != cs[i + 32])
                return 0;
        }
        return str.length();
    }
}
```

TODO 分治，滑动窗口，位运算

## [2000. 反转单词前缀](https://leetcode-cn.com/problems/reverse-prefix-of-word/)

> 双指针，字符串

```java
class Solution {
    public String reversePrefix(String word, char ch) {
        int index = word.indexOf(ch);
        if (index == -1) return word;
        StringBuilder sb = new StringBuilder(word.substring(0, index + 1));
        return sb.reverse().toString() + word.substring(index + 1, word.length());
    }
}
```
直接用字符数组双指针
```java
class Solution {
    public String reversePrefix(String word, char ch) {
        int index = word.indexOf(ch);
        if (index == -1) return word;
        char[] cs = word.toCharArray();
        int left = 0, right = index;
        while (left < right) {
            char tmp = cs[left];
            cs[left] = cs[right];
            cs[right] = tmp;
            left++;
            right--;
        }
        return new String(cs);
    }
}
```

## [1725. 可以形成最大正方形的矩形数目](https://leetcode-cn.com/problems/number-of-rectangles-that-can-form-the-largest-square/)

> 数组

```java
class Solution {
    public int countGoodRectangles(int[][] rectangles) {
        int cnt = 0, maxLen = 0;
        for (int[] rec: rectangles) {
            int len = Math.min(rec[0], rec[1]);
            if (len > maxLen) {
                maxLen = len;
                cnt = 1;
            } else if (len == maxLen) {
                cnt++;
            }
        }
        return cnt;
    }
}
```

## [1748. 唯一元素的和](https://leetcode-cn.com/problems/sum-of-unique-elements/)

> 数组，哈希表，计数

```java
class Solution {
    public int sumOfUnique(int[] nums) {
        int[] cnt = new int[100];
        for (int i: nums) {
            cnt[i - 1]++;
        }
        int ans = 0;
        for (int i = 0; i < 100; ++i) {
            if (cnt[i] == 1) {
                ans += i + 1;
            }
        }
        return ans;
    }
}
```

## [1405. 最长快乐字符串](https://leetcode-cn.com/problems/longest-happy-string/)

> 贪心，字符串，优先队列

```java
class Solution {
    public String longestDiverseString(int a, int b, int c) {
        StringBuilder sb = new StringBuilder();
        // 每次从最大开始尝试加，加到不能加加次大
        PriorityQueue<int[]> q = new PriorityQueue<>((x, y) -> y[1] - x[1]);
        if (a != 0) q.add(new int[]{0, a});
        if (b != 0) q.add(new int[]{1, b});
        if (c != 0) q.add(new int[]{2, c});
        while (!q.isEmpty()) {
            int[] max = q.poll();
            int n = sb.length();
            if (n >= 2 && sb.charAt(n - 1) == max[0] + 'a' && sb.charAt(n - 2) == max[0] + 'a') {
                // 不能再加了，加次大
                if (q.isEmpty()) break;
                int[] medium = q.poll();
                sb.append((char)(medium[0] + 'a'));
                if (--medium[1] > 0) q.add(medium);
                q.add(max);
            } else {
                // 可以加
                sb.append((char)(max[0] + 'a'));
                if (--max[1] > 0) q.add(max);
            }
        }
        return sb.toString();
    }
}
```

## [2006. 差的绝对值为 K 的数对数目](https://leetcode-cn.com/problems/count-number-of-pairs-with-absolute-difference-k/)

> 数组，哈希表，计数

```java
class Solution {
    public int countKDifference(int[] nums, int k) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int num: nums) {
            map.put(num, map.getOrDefault(num, 0) + 1);
        }
        int res = 0;
        for (int i: map.keySet()) {
            res += map.getOrDefault(i, 0) * (map.getOrDefault(i + k, 0) + map.getOrDefault(i - k, 0));
            map.put(i, 0);
        }
        return res;
    }
}
```

TODO 一次遍历

## [1447. 最简分数](https://leetcode-cn.com/problems/simplified-fractions/)

> 数学，字符串，数论

执行用时：12 ms, 在所有 Java 提交中击败了100.00%的用户
内存消耗：42.1 MB, 在所有 Java 提交中击败了14.90%的用户
```java
class Solution {
    public List<String> simplifiedFractions(int n) {
        List<String> res = new ArrayList<>();
        dfs(n, res);
        return res;
    }
    
    private void dfs(int n, List<String> list) {
        if (n == 1) {
            return;
        }
        if (n == 2) {
            list.add("1/2");
            return;
        }
        for (int i = 1; i < n; ++i) {
            if (gcd(n, i) == 1) {
                list.add(i + "/" + n);
            }
        }
        dfs(n - 1, list);
    }

    private int gcd(int a, int b) {
        return b > 0 ? gcd(b, a % b) : a;
    }
}
```
TODO GCD的求解方法：欧几里得算法、更相减损法、stein算法

## [1984. 学生分数的最小差值](https://leetcode-cn.com/problems/minimum-difference-between-highest-and-lowest-of-k-scores/)

> 数组，排序，滑动窗口

执行用时：4 ms, 在所有 Java 提交中击败了100.00%的用户
内存消耗：41.7 MB, 在所有 Java 提交中击败了5.17%的用户
```java
class Solution {
    public int minimumDifference(int[] nums, int k) {
        Arrays.sort(nums);
        int res = Integer.MAX_VALUE;
        for (int i = k - 1; i < nums.length; ++i) {
            res = Math.min(res, nums[i] - nums[i - k + 1]);
        }
        return res;
    }
}
```

## [1020. 飞地的数量](https://leetcode-cn.com/problems/number-of-enclaves/)

> 广度优先搜索，数组，矩阵，并查集，多源BFS

```java
class Solution {
    private int total = 0, m = 0, n = 0;

    public int numEnclaves(int[][] grid) {
        m = grid.length;
        n = grid[0].length;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (grid[i][j] == 1) total += 1;
            }
        }
        boolean[][] vis = new boolean[m][n];
        for (int i = 0; i < m; ++i) {
            bfs(grid, i, 0, vis);
            bfs(grid, i, n - 1, vis);
        }
        for (int j = 0; j < n; ++j) {
            bfs(grid, 0, j, vis);
            bfs(grid, m - 1, j, vis);
        }
        return total;
    }

    private void bfs(int[][] grid, int i, int j, boolean[][] vis) {
        if (i < 0 || i >= m || j < 0 || j >= n || vis[i][j] || grid[i][j] == 0) {
            return;
        }
        vis[i][j] = true;
        total -= 1;
        int[][] directions = new int[][]{{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        for (int[] d: directions) {
            bfs(grid, i + d[0], j + d[1], vis);
        }
    }
}
```
TODO 并查集

## [1189. “气球” 的最大数量](https://leetcode-cn.com/problems/maximum-number-of-balloons/)

> 哈希表，字符串，计数

执行用时：1 ms, 在所有 Java 提交中击败了100.00%的用户
内存消耗：39.4 MB, 在所有 Java 提交中击败了12.81%的用户
```java
class Solution {
    public int maxNumberOfBalloons(String text) {
        int[] cnt = new int[26];
        for (char c: text.toCharArray()) {
            cnt[c - 'a']++;
        }
        int res = Integer.MAX_VALUE;
        for (int i = 0; i < 26; ++i) {
            if (i == 0 || i == 1 || i == 13) {
                res = Math.min(res, cnt[i]);
            } else if (i == 11 || i == 14) {
                res = Math.min(res, cnt[i] / 2);
            }
        }
        return res;
    }
}
```

## [540. 有序数组中的单一元素](https://leetcode-cn.com/problems/single-element-in-a-sorted-array/)

> 数组，二分查找

```java
class Solution {
    public int singleNonDuplicate(int[] nums) {
        int n = nums.length, l = 0, r = n - 1;
        while (l <= r) {
            int mid = l + (r - l) / 2;
            if (mid == l || mid == 0 || mid == n - 1) return nums[mid];
            if (nums[mid] == nums[mid - 1]) {
                if (((mid - l) & 1) != 0) l = mid + 1;
                else r = mid;
            } else if (nums[mid] == nums[mid + 1]) {
                if (((r - mid) & 1) != 0) r = mid - 1;
                else l = mid;
            } else {
                return nums[mid];
            }
        }
        return nums[l];
    }
}
```
TODO 异或/偶数位优化

## [1380. 矩阵中的幸运数](https://leetcode-cn.com/problems/lucky-numbers-in-a-matrix/)

> 数组，矩阵

```java
class Solution {
    public List<Integer> luckyNumbers (int[][] matrix) {
        int m = matrix.length, n = matrix[0].length;
        Set<Integer> mins = new HashSet<>();
        for (int i = 0; i < m; ++ i) {
            int min = Integer.MAX_VALUE;
            for (int j = 0; j < n; ++j) {
                min = Math.min(min, matrix[i][j]);
            }
            mins.add(min);
        }
        Set<Integer> maxs = new HashSet<>();
        for (int j = 0; j < n; ++ j) {
            int max = 0;
            for (int i = 0; i < m; ++i) {
                max = Math.max(max, matrix[i][j]);
            }
            maxs.add(max);
        }
        Set<Integer> resSet = new HashSet<>();
        resSet.addAll(mins);
        resSet.retainAll(maxs);
        return new ArrayList<>(resSet);
    }
}
```
TODO 直接用数组预处理即可

## [688. 骑士在棋盘上的概率](https://leetcode-cn.com/problems/knight-probability-in-chessboard/)

> 动态规划，三维DP

```java
class Solution {
    public double knightProbability(int n, int k, int row, int column) {
        // dp[k][i][j]表示k步从i,j出发的概率
        // dp[k][i][j]=∑dp[k-1][i+x][j+y]*1/8,(x,y)是8个方向
        // dp[0][i][j]=1
        // dp[k][row][column]
        int[][] directions = new int[][]{{2, 1}, {2, -1}, {1, 2}, {1, -2}, {-2, -1}, {-2, 1}, {-1, -2}, {-1, 2}};
        double[][][] dp = new double[k + 1][n][n];
        for (int s = 0; s <= k; ++s) {
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    if (s == 0) {
                        dp[0][i][j] = 1;
                    } else {
                        for (int[] d: directions) {
                            int xi = i + d[0], yi = j + d[1];
                            if (xi >= 0 && xi < n && yi >= 0 && yi < n) {
                                dp[s][i][j] += dp[s - 1][xi][yi] / 8.0;
                            }
                        }
                    }
                }
            }
        }
        return dp[k][row][column];
    }
}
```

## [1791. 找出星型图的中心节点](https://leetcode-cn.com/problems/find-center-of-star-graph/)

> 图

```java
class Solution {
    public int findCenter(int[][] edges) {
        if (edges[0][0] == edges[1][0] || edges[0][0] == edges[1][1]) return edges[0][0];
        else return edges[0][1];
    }
}
```

## [969. 煎饼排序](https://leetcode-cn.com/problems/pancake-sorting/)

> 贪心，数组，双指针，排序，煎饼排序

```java
class Solution {
    public List<Integer> pancakeSort(int[] arr) {
        List<Integer> res = new ArrayList<>();
        int n = arr.length;
        for (int i = n; i > 0; --i) {
            int j = 0, max = 0, maxIdx = 0;
            for (; j < i; ++j) {
                if (arr[j] > max) {
                    max = arr[j];
                    maxIdx = j;
                }
            }
            reverse(arr, maxIdx);
            res.add(maxIdx + 1);
            reverse(arr, i - 1);
            res.add(i);
        }
        return res;
    }

    private void reverse(int[] arr, int r) {
        int l = 0;
        while (l < r) {
            int tmp = arr[r];
            arr[r] = arr[l];
            arr[l] = tmp;
            l++;
            r--;
        }
    }
}
```

## [2114. 句子中的最多单词数](https://leetcode-cn.com/problems/maximum-number-of-words-found-in-sentences/)

> 数组，字符串

```java
class Solution {
    public int mostWordsFound(String[] sentences) {
        int most = 0;
        for (String sen: sentences) {
            most = Math.max(most, sen.split(" ").length);
        }
        return most;
    }
}
```

## [917. 仅仅反转字母](https://leetcode-cn.com/problems/reverse-only-letters/)

> 双指针，字符串

```java
class Solution {
    public String reverseOnlyLetters(String s) {
        int l = 0, r = s.length() - 1, n = s.length();
        char[] arr = s.toCharArray();
        while (l < r) {
            while (l < n && !Character.isLetter(s.charAt(l))) l++;
            while (r >= 0 && !Character.isLetter(s.charAt(r))) r--;
            if (l < r) {
                swap(arr, l, r);
                l++;
                r--;
            } else {
                break;
            }
        }
        return new String(arr);
    }

    private void swap(char[] arr, int l, int r) {
        char tmp = arr[l];
        arr[l] = arr[r];
        arr[r] = tmp;
    }
}
```

## [1706. 球会落何处](https://leetcode-cn.com/problems/where-will-the-ball-fall/)

> 深度优先搜索，数组，动态规划，矩阵，模拟

```java
class Solution {
    public int[] findBall(int[][] grid) {
        int m = grid.length, n = grid[0].length; // m i row, n j column
        int[] res = new int[n];
        for (int j = 0; j < n; ++j) {
            int x = j;
            for (int i = 0; i < m; ++i) {
                if (grid[i][x] == 1) {
                    if (x + 1 == n || x + 1 < n && grid[i][x + 1] == -1) {
                        res[j] = -1;
                        break;
                    } else {
                        x++;
                    }
                } else {
                    if (x == 0 || x >= 0 && grid[i][x - 1] == 1) {
                        res[j] = -1;
                        break;
                    } else {
                        x--;
                    }
                }
                res[j] = x;
            }
        }
        return res;
    }
}
```

## [537. 复数乘法](https://leetcode-cn.com/problems/complex-number-multiplication/)

> 数学，字符串，模拟

```java
class Solution {
    public String complexNumberMultiply(String num1, String num2) {
        int[] ab = new int[4], cd = new int[4];
        ab = parse(num1);
        cd = parse(num2);
        StringBuilder sb = new StringBuilder();
        sb.append(ab[0] * cd[0] - ab[1] * cd[1]).append('+').append(ab[0] * cd[1] + ab[1] * cd[0]).append('i');
        return sb.toString();
    }

    private int[] parse(String num) {
        int[] ret = new int[2];
        String[] ns = num.split("\\+");
        ret[0] = Integer.parseInt(ns[0]);
        ret[1] = Integer.parseInt(ns[1].split("i")[0]);
        return ret;
    }
}
```

## [2016. 增量元素之间的最大差值](https://leetcode-cn.com/problems/maximum-difference-between-increasing-elements/)

> 数组

暴力
```java
class Solution {
    public int maximumDifference(int[] nums) {
        int n = nums.length;
        int res = -1;
        for (int i = 0; i < n - 1; ++i) {
            for (int j = i + 1; j < n; ++j) {
                if (nums[j] > nums[i]) {
                    res = Math.max(res, nums[j] - nums[i]);
                }
            }
        }
        return res;
    }
}
```
前缀最小值
```java
class Solution {
    public int maximumDifference(int[] nums) {
        int n = nums.length, res = -1, min = nums[0];
        for (int i = 0; i < n; ++i) {
            if (nums[i] > min) {
                res = Math.max(res, nums[i] - min);
            } else if (nums[i] < min) {
                min = nums[i];
            }
        }
        return res;
    }
}
```

## [553. 最优除法](https://leetcode-cn.com/problems/optimal-division/)

> 数组，数学，脑筋急转弯

```java
class Solution {
    public String optimalDivision(int[] nums) {
        // 均保留第一个数，后续的加上括号
        int n = nums.length;
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < n; ++i) {
            if (i == 0) {
                sb.append(nums[i]);
                continue;
            } else if (i == 1 && n > 2) {
                sb.append("/(").append(nums[i]);
            } else {
                sb.append('/').append(nums[i]);
            }
            if (i == n - 1 && n > 2) sb.append(')');
        }
        return sb.toString();
    }
}
```

## [2160. 拆分数位后四位数字的最小和](https://leetcode-cn.com/problems/minimum-sum-of-four-digit-number-after-splitting-digits/)

> 贪心，数学，排序

```java
class Solution {
    public int minimumSum(int num) {
        int[] nums = new int[4];
        nums[0] = num % 10;
        nums[1] = (num / 10) % 10;
        nums[2] = (num / 100) % 10;
        nums[3] = (num / 1000) % 10;
        Arrays.sort(nums);
        return (nums[0] + nums[1]) * 10 + nums[2] + nums[3];
    }
}
```
直接用字符数组
执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户
内存消耗：38.5 MB, 在所有 Java 提交中击败了19.28%的用户
```java
class Solution {
    public int minimumSum(int num) {
        char[] cs = String.valueOf(num).toCharArray();
        Arrays.sort(cs);
        return (cs[0] + cs[1]) * 10 + cs[2] + cs[3] - 1056;
    }
}
```

## [521. 最长特殊序列 Ⅰ](https://leetcode-cn.com/problems/longest-uncommon-subsequence-i/)

> 字符串，脑筋急转弯

```java
class Solution {
    public int findLUSlength(String a, String b) {
        if (a.equals(b)) return -1;
        return Math.max(a.length(), b.length());
    }
}
```

## [2100. 适合打劫银行的日子](https://leetcode-cn.com/problems/find-good-days-to-rob-the-bank/)

> 数组，动态规划，前缀和

执行用时：6 ms, 在所有 Java 提交中击败了92.53%的用户
内存消耗：60.5 MB, 在所有 Java 提交中击败了5.13%的用户
```java
class Solution {
    public List<Integer> goodDaysToRobBank(int[] security, int time) {
        int n = security.length;
        List<Integer> ret = new ArrayList<>();
        if (time == 0) {
            for (int i = 0; i < n; ++i) {
                ret.add(i);
            }
            return ret;
        }
        boolean[] goodDays = new boolean[n];
        int cnt = 0;
        for (int i = 1; i < n; ++i) {
            if (security[i] <= security[i - 1]) {
                cnt++;
                if (cnt >= time)
                    goodDays[i] = true;
            } else {
                cnt = 0;
            }
        }
        cnt = 0;
        for (int i = n - 2; i >= 0; --i) {
            if (security[i] <= security[i + 1]) {
                cnt++;
                if (cnt >= time && goodDays[i]) {
                    ret.add(i);
                }
            } else {
                cnt = 0;
            }
        }
        return ret;
    }
}
```

## [504. 七进制数](https://leetcode-cn.com/problems/base-7/)

> 数学

```java
class Solution {
    public String convertToBase7(int num) {
        StringBuilder sb = new StringBuilder();
        if (num < 0) {
            sb.append("-").append(convertToBase7(-num));
            return sb.toString();
        }
        if (num == 0) {
            return "0";
        }
        while (num > 0) {
            sb.append(num % 7);
            num /= 7;
        }
        return sb.reverse().toString();
    }
}
```

## [2055. 蜡烛之间的盘子](https://leetcode-cn.com/problems/plates-between-candles/)

> 数组，字符串，前缀和

```java
class Solution {
    public int[] platesBetweenCandles(String s, int[][] queries) {
        int n = s.length();
        int[] preStar = new int[n + 1], ll = new int[n], rl = new int[n];
        preStar[0] = s.charAt(0) == '*' ? 1 : 0;
        ll[0] = s.charAt(0) == '*' ? Integer.MAX_VALUE : 0;
        for (int i = 1; i <= n; ++i) {
            preStar[i] = s.charAt(i - 1) == '*' ? preStar[i - 1] + 1 : preStar[i - 1];
            if (i < n) {
                if (s.charAt(i) == '|') {
                    ll[i] = 0;
                } else {
                    if (ll[i - 1] == Integer.MAX_VALUE) ll[i] = Integer.MAX_VALUE;
                    else ll[i] = ll[i - 1] + 1;
                }
            }
        }
        rl[n - 1] = s.charAt(n - 1) == '*' ? Integer.MAX_VALUE : 0;
        for (int i = n - 2; i >= 0; --i) {
            if (s.charAt(i) == '|') {
                rl[i] = 0;
            } else {
                if (rl[i + 1] == Integer.MAX_VALUE) rl[i] = Integer.MAX_VALUE;
                else rl[i] = rl[i + 1] + 1;
            }
        }
        int m = queries.length;
        int[] ret = new int[m];
        for (int i = 0; i < m; ++i) {
            int l = queries[i][0], r = queries[i][1];
            if (rl[l] == Integer.MAX_VALUE || ll[r] == Integer.MAX_VALUE) {
                ret[i] = 0;
            } else {
                ret[i] = preStar[r + 1] - preStar[l] - rl[l] - ll[r];
                ret[i] = ret[i] <= 0 ? 0 : ret[i];
            }
        }
        return ret;
    }
}
```
TODO 小技巧优化

## [798. 得分最高的最小轮调](https://leetcode-cn.com/problems/smallest-rotation-with-highest-score/)

> 数组，差分数组，前缀和

执行用时：7 ms, 在所有 Java 提交中击败了32.86%的用户
内存消耗：56.5 MB, 在所有 Java 提交中击败了5.71%的用户
```java
class Solution {
    public int bestRotation(int[] nums) {
        // 两种情况：翻后面ni=i+(n-k)[k>i] 或者 翻前面ni=i-k[k<=i]
        // 且满足：n>ni>=nums[i]
        // 求得k取值的两个可能的区间[0,n)∩[i+1,i+n-nums[i]]或者[0,n)∩[0,i-nums[i]]，记为[a,b]||[0,c]
        int n = nums.length;
        int[] delta = new int[n];
        for (int i = 0; i < n; ++i) {
            int a = i + 1;
            int b = Math.min(n - 1, i + n - nums[i]);
            int c = Math.min(n - 1, i - nums[i]);
            if (a <= b) {
                delta[a]++;
                if (b + 1 < n)
                    delta[b + 1]--;
            }
            if (c >= 0 && c < a) {
                delta[0]++;
                if (c + 1 < n)
                    delta[c + 1]--;
            }
        }
        int ret = 0, max = 0, cur = 0;
        for (int i = 1; i < n; ++i) {
            cur += delta[i];
            if (cur > max) {
                max = cur;
                ret = i;
            }
        }
        return ret;
    }
}
```

## [589. N 叉树的前序遍历](https://leetcode-cn.com/problems/n-ary-tree-preorder-traversal/)

> 栈，树，深度优先搜索

```java
/*
// Definition for a Node.
class Node {
    public int val;
    public List<Node> children;

    public Node() {}

    public Node(int _val) {
        val = _val;
    }

    public Node(int _val, List<Node> _children) {
        val = _val;
        children = _children;
    }
};
*/

class Solution {
    public List<Integer> preorder(Node root) {
        List<Integer> ret = new ArrayList<>();
        dfs(ret, root);
        return ret;
    }

    private void dfs(List<Integer> list, Node root) {
        if (root == null) {
            return;
        }
        list.add(root.val);
        List<Node> ch = root.children;
        for (int i = 0; i < ch.size(); ++i) {
            dfs(list, ch.get(i));
        }
    }
}
```

## [590. N 叉树的后序遍历](https://leetcode-cn.com/problems/n-ary-tree-postorder-traversal/)

> 栈，树，深度优先搜索

```java
/*
// Definition for a Node.
class Node {
    public int val;
    public List<Node> children;

    public Node() {}

    public Node(int _val) {
        val = _val;
    }

    public Node(int _val, List<Node> _children) {
        val = _val;
        children = _children;
    }
};
*/

class Solution {
    public List<Integer> postorder(Node root) {
        List<Integer> ret = new ArrayList<>();
        dfs(ret, root);
        return ret;
    }

    private void dfs(List<Integer> list, Node root) {
        if (root == null) {
            return;
        }
        for (Node node: root.children) {
            dfs(list, node);
        }
        list.add(root.val);
    }
}
```

## [393. UTF-8 编码验证](https://leetcode-cn.com/problems/utf-8-validation/)

> 位运算，数组

执行用时：1 ms, 在所有 Java 提交中击败了100.00%的用户
内存消耗：41.6 MB, 在所有 Java 提交中击败了40.81%的用户
```java
class Solution {
    public boolean validUtf8(int[] data) {
        int n = data.length, i = 0;
        while (i < n) {
            int d = data[i];
            if ((d & (1 << 7)) > 0) {
                int cnt = 8;
                for (int j = 6; j >= 0; --j) {
                    if ((d & (1 << j)) == 0) {
                        cnt = 7 - j;
                        break;
                    }
                }
                if (cnt == 1 || cnt > 4 || cnt > n - i) {
                    return false;
                }
                for (int k = 0; k < cnt - 1; ++k) {
                    i++;
                    if ((data[i] & (1 << 7)) == 0 || (data[i] & (1 << 6)) > 0) {
                        return false;
                    }
                }
            }
            i++;
        }
        return true;
    }
}
```

## [599. 两个列表的最小索引总和](https://leetcode-cn.com/problems/minimum-index-sum-of-two-lists/)

> 数组，哈希表，字符串

```java
class Solution {
    public String[] findRestaurant(String[] list1, String[] list2) {
        int min = Integer.MAX_VALUE;
        List<String> list = new ArrayList<>();
        for (int i = 0; i < list1.length; ++i) {
            for (int j = 0; j < list2.length; ++j) {
                if (list1[i].equals(list2[j])) {
                    if (i + j < min) {
                        min = i + j;
                        list.clear();
                    }
                    if (i + j <= min) {
                        list.add(list1[i]);
                    }
                }
            }
        }
        return list.toArray(new String[list.size()]);
    }
}
```
TODO hashmap

## [2044. 统计按位或能得到最大值的子集数目](https://leetcode-cn.com/problems/count-number-of-maximum-bitwise-or-subsets/)

> 位运算，数组，回溯

```java
class Solution {
    private int cnt = 0, max = 0;
    private int[] _nums;

    public int countMaxOrSubsets(int[] nums) {
        _nums = nums;
        dfs(0, 0); // 当前位置，状态
        return cnt;
    }

    private void dfs(int idx, int val) {
        if (idx == _nums.length) {
            if (val > max) {
                max = val;
                cnt = 1;
            } else if (val == max) {
                cnt++;
            }
            return;
        }
        dfs(idx + 1, val);
        dfs(idx + 1, val | _nums[idx]);
    }
}
```

## [720. 词典中最长的单词](https://leetcode-cn.com/problems/longest-word-in-dictionary/)

> 字典树，数组，字符串，哈希

```java
class Solution {
    private String res = "";

    class Trie {
        TrieNode root;

        class TrieNode {
            String word = "";
            TrieNode[] children = new TrieNode[26];
        }

        public Trie() {
            root = new TrieNode();
        }

        public void insert(String s) {
            TrieNode p = root;
            for (int i = 0; i < s.length(); ++i) {
                int u = s.charAt(i) - 'a';
                if (p.children[u] == null) {
                    p.children[u] = new TrieNode();
                }
                p = p.children[u];
            }
            p.word = s;
        }

        public void dfs(TrieNode root) {
            if (root == null || root.word.length() == 0) {
                return;
            }
            if (root.word.length() > res.length()) {
                res = root.word;
            } else if (root.word.length() == res.length()) {
                res = res.compareTo(root.word) <= 0 ? res : root.word;
            }
            for (int i = 0; i < 26; ++i) {
                if (root.children[i] != null) {
                    dfs(root.children[i]);
                }
            }
        }
    }

    public String longestWord(String[] words) {
        Trie trie = new Trie();
        for (String word: words) {
            trie.insert(word);
        }
        for (int i = 0; i < 26; ++i) {
            trie.dfs(trie.root.children[i]);
        }
        return res;
    }
}
```
TODO 哈希

## [2043. 简易银行系统](https://leetcode-cn.com/problems/simple-bank-system/)

> 设计，数组，哈希，模拟

```java
class Bank {
    private long[] balance;

    public Bank(long[] balance) {
        this.balance = balance;
    }
    
    public boolean transfer(int account1, int account2, long money) {
        if (account1 <= 0 || account1 > balance.length || account2 <= 0 || account2 > balance.length) {
            return false;
        }
        if (balance[account1 - 1] >= money) {
            balance[account1 - 1] -= money;
            balance[account2 - 1] += money;
            return true;
        } else {
            return false;
        }
    }
    
    public boolean deposit(int account, long money) {
        if (account <= 0 || account > balance.length) {
            return false;
        }
        balance[account - 1] += money;
        return true;
    }
    
    public boolean withdraw(int account, long money) {
        if (account <= 0 || account > balance.length) {
            return false;
        }
        if (balance[account - 1] >= money) {
            balance[account - 1] -= money;
            return true;
        } else {
            return false;
        }
    }
}

/**
 * Your Bank object will be instantiated and called as such:
 * Bank obj = new Bank(balance);
 * boolean param_1 = obj.transfer(account1,account2,money);
 * boolean param_2 = obj.deposit(account,money);
 * boolean param_3 = obj.withdraw(account,money);
 */
```

## [606. 根据二叉树创建字符串](https://leetcode-cn.com/problems/construct-string-from-binary-tree/)

> 二叉树，深度优先搜索，树，字符串

执行用时：1 ms, 在所有 Java 提交中击败了100.00%的用户
内存消耗：41.1 MB, 在所有 Java 提交中击败了33.42%的用户
```java
class Solution {
    private StringBuilder sb;

    public String tree2str(TreeNode root) {
        sb = new StringBuilder();
        dfs(root);
        return sb.toString();
    }

    private void dfs(TreeNode root) {
        if (root == null) {
            return;
        }
        sb.append(root.val);
        if (root.left != null) {
            sb.append('(');
            dfs(root.left);
            sb.append(')');
        }
        if (root.right != null) {
            if (root.left == null) {
                sb.append("()");
            }
            sb.append('(');
            dfs(root.right);
            sb.append(')');
        }
    }
}
```

## [2039. 网络空闲的时刻](https://leetcode-cn.com/problems/the-time-when-the-network-becomes-idle/)

> 深度优先搜索，图，数组

```java
class Solution {
    public int networkBecomesIdle(int[][] edges, int[] patience) {
        // 构建邻接表
        int n = patience.length;
        List<Integer>[] adj = new List[n];
        // 初始化
        for (int i = 0; i < n; ++i) {
            adj[i] = new ArrayList<>();
        }
        for (int[] edge: edges) {
            adj[edge[0]].add(edge[1]);
            adj[edge[1]].add(edge[0]);
        }
        // BFS
        boolean[] visited = new boolean[n];
        Queue<Integer> q = new ArrayDeque<>();
        // 推入初始节点
        visited[0] = true;
        q.offer(0);
        int dist = 1;
        int ans = 0;
        while (!q.isEmpty()) {
            int size = q.size();
            for (int i = 0; i < size; ++i) {
                int cur = q.poll();
                for (int j: adj[cur]) {
                    if (visited[j]) {
                        continue;
                    }
                    int time = patience[j] * ((dist * 2 - 1) / patience[j]) + 2 * dist + 1;
                    ans = Math.max(ans, time);
                    q.offer(j);
                    visited[j] = true;
                }
            }
            // 深度+1
            dist++;
        }
        return ans;
    }
}
```
TODO 用数组实现邻接表（链式前向星）+static优化

## [653. 两数之和 IV - 输入 BST](https://leetcode-cn.com/problems/two-sum-iv-input-is-a-bst/)

> 树，深度优先搜索，广度优先搜索，二叉搜索树，双指针

```java
class Solution {
    private Set<Integer> set;

    public boolean findTarget(TreeNode root, int k) {
        set = new HashSet<>();
        dfs(root);
        for (int x: set) {
            if (k - x != x && set.contains(k - x)) {
                return true;
            }
        }
        return false;
    }

    private void dfs(TreeNode root) {
        if (root == null) {
            return;
        }
        set.add(root.val);
        dfs(root.left);
        dfs(root.right);
    }
}
```
优化
```java
class Solution {
    private Set<Integer> set = new HashSet<>();

    public boolean findTarget(TreeNode root, int k) {
        if (root == null) return false;
        if (set.contains(k - root.val)) return true;
        set.add(root.val);
        return findTarget(root.left, k) || findTarget(root.right, k);
    }
}
```
TODO BST上双指针

## [2038. 如果相邻两个颜色均相同则删除当前颜色](https://leetcode-cn.com/problems/remove-colored-pieces-if-both-neighbors-are-the-same-color/)

> 脑筋急转弯，博弈，贪心，数学，字符串

```java
class Solution {
    public boolean winnerOfGame(String colors) {
        // 求除了2端之外的可删除的A的数和B的数，谁大则谁赢
        int n = colors.length(), cnt = 0;
        char[] cs = colors.toCharArray();
        for (int i = 1; i < n - 1; ++i) {
            if (cs[i - 1] == cs[i] && cs[i] == cs[i + 1]) {
                cnt += cs[i] == 'A' ? 1 : -1;
            }
        }
        return cnt > 0;
    }
}
```

## [440. 字典序的第K小数字](https://leetcode-cn.com/problems/k-th-smallest-in-lexicographical-order/)

> 字典树

会超时
```java
class Solution {
    public int findKthNumber(int n, int k) {
        PriorityQueue<String> pq = new PriorityQueue<>(k, Comparator.reverseOrder());
        for (int i = 1; i <= k; ++i) {
            pq.offer(String.valueOf(i));
        }
        for (int i = k + 1; i <= n; ++i) {
            String x = String.valueOf(i);
            if (x.compareTo(pq.peek()) < 0) {
                pq.poll();
                pq.offer(x);
            }
        }
        return Integer.parseInt(pq.poll());
    }
}
```
字典树
```java
class Solution {
    // O(log^2(n))/O(1)
    public int findKthNumber(int n, int k) {
        // 字典树，getStep
        int cur = 1;
        k--;
        while (k > 0) {
            // 获取当前cur节点的子节点总个数
            int step = getStep(cur, n);
            if (step <= k) {
                // 在右边节点找
                cur++;
                // 直接跳过step步
                k -= step;
            } else {
                // 从子节点最左边找
                cur *= 10;
                // 跳过一步
                k--;
            }
        }
        return cur;
    }

    private int getStep(int cur, int n) {
        // cur下层范围为[cur * 10, cur * 10 + 9]，再下层依次类推，右侧被n限制住
        int res = 0;
        long left = cur, right = cur;
        while (left <= n) {
            res += Math.min(right, n) - left + 1;
            left *= 10;
            right = right * 10 + 9;
        }
        return res;
    }
}
```

## [661. 图片平滑器](https://leetcode-cn.com/problems/image-smoother/)

> 数组，矩阵

```java
class Solution {
    public int[][] imageSmoother(int[][] img) {
        int m = img.length, n = img[0].length;
        int[][] res = new int[m][n];
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                int cnt = 1, sum = img[i][j];
                int[][] dirs = new int[][]{{0, 1}, {0, -1}, {-1, 1}, {-1, 0}, {-1, -1}, {1, 1}, {1, 0}, {1, -1}};
                for (int k = 0; k < dirs.length; ++k) {
                    int x = i + dirs[k][0], y = j + dirs[k][1];
                    if (x >= 0 && y >= 0 && x < m && y < n) {
                        sum += img[x][y];
                        cnt++;
                    }
                }
                res[i][j] = sum / cnt;
            }
        }
        return res;
    }
}
```

## [101. 对称二叉树](https://leetcode-cn.com/problems/symmetric-tree/)

> 树，深度优先搜索，二叉树

```java
class Solution {
    public boolean isSymmetric(TreeNode root) {
        return dfs(root.left, root.right);
    }

    private boolean dfs(TreeNode l, TreeNode r) {
        if (l == null && r == null) {
            return true;
        } else if (l == null || r == null) {
            return false;
        }
        return l.val == r.val && dfs(l.right, r.left) && dfs(l.left, r.right);
    }
}
```

## [2028. 找出缺失的观测数据](https://leetcode-cn.com/problems/find-missing-observations/)

> 数组，数学，模拟

```java
执行用时：2 ms, 在所有 Java 提交中击败了100.00%的用户
内存消耗：59 MB, 在所有 Java 提交中击败了24.32%的用户
class Solution {
    public int[] missingRolls(int[] rolls, int mean, int n) {
        int mSum = 0;
        for (int i = 0; i < rolls.length; ++i) {
            mSum += rolls[i];
        }
        int nSum = (rolls.length + n) * mean - mSum;
        int avg = nSum / n;
        if (avg < 1 || avg > 6 || (avg == 6 && nSum % n != 0)) {
            return new int[0];
        }
        int rest = nSum - avg * n;
        int[] res = new int[n];
        for (int i = 0; i < rest; ++i) {
            res[i] = avg + 1;
        }
        for (int i = rest; i < n; ++i) {
            res[i] = avg;
        }
        return res;
    }
}
```
写法更简单
```java
class Solution {
    public int[] missingRolls(int[] rolls, int mean, int n) {
        // 求总和
        int sum = (rolls.length + n) * mean;
        // 求前n的和
        for (int i: rolls) {
            sum -= i;
        }
        // 前n的和不满足题意
        if (n > sum || sum > 6 * n) {
            return new int[0];
        }
        // 用平均值打底，然后余下的值前几项再+1
        int[] res = new int[n];
        int avg = sum / n, point = sum % n;
        for (int i = 0; i < n; ++i) {
            res[i] = avg + (i < point ? 1 : 0);
        }
        return res;
    }
}
```

## [693. 交替位二进制数](https://leetcode-cn.com/problems/binary-number-with-alternating-bits/)

> 位运算
执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户
内存消耗：38.4 MB, 在所有 Java 提交中击败了17.60%的用户
```java
class Solution {
    public boolean hasAlternatingBits(int n) {
        boolean flag = (n & 1) == 1;
        n >>= 1;
        while (n > 0) {
            if (((n & 1) == 1) == flag) {
                return false;
            }
            flag = !flag;
            n >>= 1;
        }
        return true;
    }
}
```
位运算技巧
```java
class Solution {
    public boolean hasAlternatingBits(int n) {
        int x = n ^ (n >> 1);
        return (x & (x + 1)) == 0;
    }
}
```

## [2024. 考试的最大困扰度](https://leetcode-cn.com/problems/maximize-the-confusion-of-an-exam/)

> 字符串，二分查找，前缀和，滑动窗口

```java
class Solution {
    public int maxConsecutiveAnswers(String answerKey, int k) {
        // 滑动窗口 T则求窗口内F的数量<=k满足题意，否则不满足，滑动
        int n = answerKey.length();
        int maxLen = 0;
        char[] cs = new char[]{'T', 'F'};
        for (int i = 0; i < 2; ++i) {
            int l = 0, r = 0, curLen = 0;
            while (r < n) {
                // 更新r状态
                if (answerKey.charAt(r) == cs[i]) {
                    curLen++;
                }
                while (curLen > k) {
                    // 更新l状态
                    if (answerKey.charAt(l) == cs[i]) {
                        curLen--;
                    }
                    l++;
                }
                // 更新最值
                maxLen = Math.max(maxLen, r - l + 1);
                r++;
            }
        }
        return maxLen;
    }
}
```

## [728. 自除数](https://leetcode-cn.com/problems/self-dividing-numbers/)

> 模拟

```java
class Solution {
    public List<Integer> selfDividingNumbers(int left, int right) {
        List<Integer> list = new ArrayList<>();
        for (int i = left; i <= right; ++i) {
            if (check(i)) {
                list.add(i);
            }
        }
        return list;
    }

    private boolean check(int i) {
        int o = i;
        while (i > 0) {
            int x = i % 10;
            if (x == 0 || o % x != 0) {
                return false;
            }
            i /= 10;
        }
        return true;
    }
}
```

## [954. 二倍数对数组](https://leetcode-cn.com/problems/array-of-doubled-pairs/)

> 贪心，数组，哈希表，排序

执行用时：22 ms, 在所有 Java 提交中击败了92.78%的用户
内存消耗：49.1 MB, 在所有 Java 提交中击败了12.88%
```java
class Solution {
    public boolean canReorderDoubled(int[] arr) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int x: arr) {
            map.put(x, map.getOrDefault(x, 0) + 1);
        }
        List<Integer> list = new ArrayList<>(map.keySet());
        Collections.sort(list, (x, y) -> Math.abs(x) - Math.abs(y));
        for (int i: list) {
            int m = i * 2;
            if (map.getOrDefault(m, 0) >= map.get(i)) {
                map.put(m, map.getOrDefault(m, 0) - map.get(i));
            } else {
                return false;
            }
        }
        return true;
    }
}
```

## [744. 寻找比目标字母大的最小字母](https://leetcode-cn.com/problems/find-smallest-letter-greater-than-target/)

> 数组，二分查找

```java
class Solution {
    public char nextGreatestLetter(char[] letters, char target) {
        int l = 0, r = letters.length - 1;
        while (l < r) {
            int mid = l + (r - l) / 2;
            if (letters[mid] > target) {
                r = mid - 1;
            } else {
                l = mid + 1;
            }
        }
        if (letters[l] > target) {
            return letters[l];
        } else {
            return l + 1 < letters.length ? letters[l + 1] : letters[0];
        }
    }
}
```

## [762. 二进制表示中质数个计算置位](https://leetcode-cn.com/problems/prime-number-of-set-bits-in-binary-representation/)

> 位运算，数学

```java
class Solution {
    public int countPrimeSetBits(int left, int right) {
        int ans = 0;
        for (int i = left; i <= right; ++i) {
            if (isPrime(Integer.bitCount(i))) {
                ans++;
            }
        }
        return ans;
    }

    private boolean isPrime(int x) {
        if (x < 2) return false;
        for (int i = 2; i * i <= x; ++i) {
            if (x % i == 0) {
                return false;
            }
        }
        return true;
    }
}
```
TODO 直接利用有限质数判断
```java
class Solution {
    public int countPrimeSetBits(int left, int right) {
        int ans = 0;
        for (int x = left; x <= right; ++x) {
            // 20位以内质数：2,3,5,7,11,13,17,19->(10100010100010101100)2
            if (((1 << Integer.bitCount(x)) & 665772) != 0) {
                ++ans;
            }
        }
        return ans;
    }
}
```

## [796. 旋转字符串](https://leetcode-cn.com/problems/rotate-string/)

> 字符串，字符串匹配

执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户
内存消耗：38.7 MB, 在所有 Java 提交中击败了76.45%的用户
```java
class Solution {
    public boolean rotateString(String s, String goal) {
        int n = s.length();
        if (n != goal.length()) return false;
        for (int i = 0; i < n; ++i) {
            if (goal.equals(s.substring(i, n) + s.substring(0, i))) return true;
        }
        return false;
    }
}
```
s+s的巧妙解法 TODO
```java
class Solution {
    public boolean rotateString(String s, String goal) {
        StringBuilder sb = new StringBuilder();
        sb.append(s).append(s);
        return goal.length() == s.length() && sb.toString().contains(goal);
    }
}
```

## [429. N 叉树的层序遍历](https://leetcode-cn.com/problems/n-ary-tree-level-order-traversal/)

> 数，广度优先搜索

```java
class Solution {
    public List<List<Integer>> levelOrder(Node root) {
        List<List<Integer>> res = new ArrayList<>();
        if (root == null) return res;
        Deque<Node> q = new ArrayDeque<>();
        q.offer(root);
        while (!q.isEmpty()) {
            List<Integer> tmp = new ArrayList<>();
            int len = q.size();
            for (int i = 0; i < len; ++i) {
                Node node = q.poll();
                tmp.add(node.val);
                for (Node child: node.children) {
                    q.offer(child);
                }
            }
            res.add(tmp);
        }
        return res;
    }
}
```
TODO dfs
```java
class Solution {
    public List<List<Integer>> levelOrder(Node root) {
        List<List<Integer>> res = new ArrayList<>();
        dfs(root, 0, res);
        return res;
    }

    public void dfs(Node root, int deep, List<List<Integer>> res){
        if(root == null) return;
        deep++;
        if(deep > res.size()){
            res.add(new ArrayList<>());
        }
        res.get(deep - 1).add(root.val);
        for(Node child: root.children){
            dfs(child, deep, res);
        }
    }
}
```

## [780. 到达终点](https://leetcode-cn.com/problems/reaching-points/)

> 数学，脑筋急转弯

```java
class Solution {
    public boolean reachingPoints(int sx, int sy, int tx, int ty) {
        // 从(tx, ty)反推到(sx, sy)
        while (tx > sx && ty > sy) {
            // 类似辗转相除反向消减，用大数不断减小数
            if (ty > tx) ty %= tx;
            else tx %= ty;
        }
        // 减完后比原数还小
        if (tx < sx || ty < sy) return false;
        return tx == sx ? (ty - sy) % tx == 0 : (tx - sx) % ty == 0;
    }
}
```

## [804. 唯一摩尔斯密码词](https://leetcode-cn.com/problems/unique-morse-code-words/)

> 数组，哈希表，字符串

```java
class Solution {
    public int uniqueMorseRepresentations(String[] words) {
        Set<String> translations = new HashSet<>();
        String[] codes = new String[]{".-","-...","-.-.","-..",".","..-.","--.","....","..",".---","-.-",".-..","--","-.","---",".--.","--.-",".-.","...","-","..-","...-",".--","-..-","-.--","--.."};
        for (String word: words) {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < word.length(); ++i) {
                sb.append(codes[word.charAt(i) - 'a']);
            }
            translations.add(sb.toString());
        }
        return translations.size();
    }
}
```

## [176. 第二高的薪水](https://leetcode-cn.com/problems/second-highest-salary/)

> 数据库, IFNULL, LIMIT

```sql
SELECT IFNULL((SELECT DISTINCT salary FROM Employee ORDER BY salary DESC LIMIT 1, 1), NULL) AS SecondHighestSalary
```

## [177. 第N高的薪水](https://leetcode-cn.com/problems/nth-highest-salary/)

> 数据库, IFNULL, LIMIT，函数

执行用时：371 ms, 在所有 MySQL 提交中击败了82.24%的用户
内存消耗：0 B, 在所有 MySQL 提交中击败了100.00%的用户
```sql
CREATE FUNCTION getNthHighestSalary(N INT) RETURNS INT
BEGIN
  SET N = N - 1;
  RETURN (
    SELECT IFNULL((SELECT DISTINCT salary FROM Employee ORDER BY salary DESC LIMIT N,1), NULL)
  );
END
```

## [178. 分数排名](https://leetcode-cn.com/problems/rank-scores/)

> 数据库，窗口函数

```sql
SELECT score, DENSE_RANK() OVER (ORDER BY score DESC) `rank` FROM Scores;
```

## [193. 有效电话号码](https://leetcode-cn.com/problems/valid-phone-numbers/)

> Shell，正则表达式

```bash
grep -P '^(\(\d{3}\) |\d{3}-)\d{3}-\d{4}$' file.txt
```

## [357. 统计各位数字都不同的数字个数](https://leetcode-cn.com/problems/count-numbers-with-unique-digits/submissions/)

> 数学，排列组合

执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户
内存消耗：37.4 MB, 在所有 Java 提交中击败了81.34%的用户
```java
class Solution {
    public int countNumbersWithUniqueDigits(int n) {
        // 0位:1; 1位:9; 2位:9*9; 3位:9*9*8
        int ans = 1;
        for (int i = 0; i < n; ++i) {
            int tmp = 9;
            for (int j = 0; j < i; ++j) {
                tmp *= 9 - j;
            }
            ans += tmp;
        }
        return ans;
    }
}
```
TODO 数位DP

## [806. 写字符串需要的行数](https://leetcode-cn.com/problems/number-of-lines-to-write-string/)

> 数组，字符串

模拟
```java
class Solution {
    public int[] numberOfLines(int[] widths, String s) {
        int ans = 0, lines = 1;
        for (int i = 0; i < s.length(); ++i) {
            int cur = widths[s.charAt(i) - 'a'];
            ans += cur;
            if (ans > 100) {
                ans = cur;
                lines++;
            }
        }
        return new int[]{lines, ans};
    }
}
```

## [196. 删除重复的电子邮箱](https://leetcode-cn.com/problems/delete-duplicate-emails/)

> 数据库, DELETE

自连接（没索引效率低）
```sql
DELETE p1 FROM Person p1, Person p2 WHERE p1.email = p2.email AND p1.id > p2.id;
```
子查询
```sql
DELETE FROM Person WHERE id NOT IN (SELECT * FROM (SELECT MIN(id) FROM Person GROUP BY Email) AS t)
```

## [380. O(1) 时间插入、删除和获取随机元素](https://leetcode-cn.com/problems/insert-delete-getrandom-o1/)

> 设计，数组，哈希表，数学，随机化

```java
class RandomizedSet {
    private List<Integer> list;
    private Map<Integer, Integer> map;
    private Random r;

    public RandomizedSet() {
        list = new ArrayList<>();
        map = new HashMap<>();
        r = new Random();
    }
    
    public boolean insert(int val) {
        if (map.containsKey(val)) {
            return false;
        }
        int size = list.size();
        map.put(val, size);
        list.add(val);
        return true;
    }
    
    public boolean remove(int val) {
        if (map.containsKey(val)) {
            int idx = map.get(val);
            int last = list.get(list.size() - 1);
            map.put(last, idx);
            list.set(idx, last);
            list.remove(list.size() - 1);
            map.remove(val);
            return true;
        }
        return false;
    }
    
    public int getRandom() {
        return list.get(r.nextInt(list.size()));
    }
}
```

## [197. 上升的温度](https://leetcode-cn.com/problems/rising-temperature/)

> 数据库，非等值连接

```sql
SELECT t1.id FROM Weather t1, Weather t2 WHERE t1.Temperature > t2.Temperature AND DATEDIFF(t1.recordDate, t2.recordDate) = 1
```

## [1141. 查询近30天活跃用户数](https://leetcode-cn.com/problems/user-activity-for-the-past-30-days-i/)

> 数据库，聚合

```sql
SELECT activity_date AS day, COUNT(DISTINCT user_id) AS active_users FROM Activity WHERE DATEDIFF('2019-07-27', activity_date) < 30 GROUP BY activity_date
```

## [1148. 文章浏览 I](https://leetcode-cn.com/problems/article-views-i/submissions/)

> 数据库

```sql
SELECT DISTINCT author_id AS id FROM Views WHERE author_id = viewer_id ORDER BY author_id
```

## [385. 迷你语法分析器](https://leetcode-cn.com/problems/mini-parser/)

> 栈，深度优先搜索，字符串

```java
class Solution {
    public NestedInteger deserialize(String s) {
        if (!s.startsWith("[")) {
            return new NestedInteger(Integer.parseInt(s));
        } else {
            NestedInteger ni = new NestedInteger();
            String tmp = s.substring(1, s.length() - 1);
            StringBuilder sb = new StringBuilder();
            int flag = 0;
            for (int i = 0; i < tmp.length(); ++i) {
                char c = tmp.charAt(i);
                if (c == ',') {
                    if (sb.length() > 0 && flag == 0) {
                        // 前方是整数，传下去
                        ni.add(deserialize(sb.toString()));
                        sb = new StringBuilder();
                    } else if (sb.length() == 0) {
                        // 前方刚刚清空sb，直接跳过
                        continue;
                    } else {
                        // 半路
                        sb.append(c);
                    }
                } else {
                    sb.append(c);
                    if (c == '[') {
                        flag++;
                    } else if (c == ']') {
                        flag--;
                        if (flag == 0) {
                            ni.add(deserialize(sb.toString()));
                            sb = new StringBuilder();
                        }
                    }
                }
            }
            if (sb.length() > 0) {
                ni.add(deserialize(sb.toString()));
            }
            return ni;
        }
    }
}
```

## [479. 最大回文数乘积](https://leetcode-cn.com/problems/largest-palindrome-product/)

> 数学

```java
class Solution {
    public int largestPalindrome(int n) {
        // 回文数最多2n位，枚举前n位，然后看结果是否可以分解
        if (n == 1) return 9;
        int ans = 0;
        int upper = (int)Math.pow(10, n) - 1; // 上界
        for (int i = upper; i > 0; --i) {
            // 构造回文数
            long t = i;
            int y = i;
            while (y > 0) {
                t = t * 10 + y % 10;
                y /= 10;
            }
            // 寻找这个回文数可能的因子，正好从upper开始，注意是long
            for (long j = upper; j * j >= t; --j) {
                if (t % j == 0) {
                    ans = (int)(t % 1337);
                    break;
                }
            }
            if (ans != 0) {
                break;
            }
        }
        return ans;
    }
}
```

## [819. 最常见的单词](https://leetcode-cn.com/problems/most-common-word/)

> 哈希表，字符串，计数

执行用时：6 ms, 在所有 Java 提交中击败了78.94%的用户
内存消耗：41 MB, 在所有 Java 提交中击败了76.86%的用户
```java
class Solution {
    public String mostCommonWord(String paragraph, String[] banned) {
        String res = "";
        int max = 0;
        StringBuilder sb = new StringBuilder();
        Map<String, Integer> map = new HashMap<>();
        Set<String> bannedWords = new HashSet<>();
        for (String b: banned) {
            bannedWords.add(b);
        }
        paragraph += " ";
        for (int i = 0; i < paragraph.length(); ++i) {
            if (!Character.isLetter(paragraph.charAt(i))) {
                if (sb.length() > 0) {
                    String key = sb.toString();
                    if (!bannedWords.contains(key)) {
                        int val = map.getOrDefault(key, 0) + 1;
                        map.put(key, val);
                        if (val > max) {
                            max = val;
                            res = key;
                        }
                    }
                    sb = new StringBuilder();
                }
            } else {
                sb.append(Character.toLowerCase(paragraph.charAt(i)));
            }
        }
        return res;
    }
}
```

## [595. 大的国家](https://leetcode-cn.com/problems/big-countries/)

> 数据库

```java
SELECT name, population, area FROM World WHERE area >= 3000000 OR population >= 25000000;
```

## [195. 第十行](https://leetcode-cn.com/problems/tenth-line/)

> Shell

```sh
cat file.txt | tail -n +10 | head -n 1 # 或者tail -n +10 file.txt | head -1
sed -n '10,1p' file.txt # 或者sed -n 10p file.txt
awk 'NR==10' file.txt
```

## [386. 字典序排数](https://leetcode-cn.com/problems/lexicographical-numbers/)

> 深度优先搜索，字典树

PriorityQueue不满足O(1)但是能过
```java
class Solution {
    public List<Integer> lexicalOrder(int n) {
        PriorityQueue<Integer> pq = new PriorityQueue<>((x, y) -> {
            return String.valueOf(x).compareTo(String.valueOf(y));
        });
        for (int i = 1; i <= n; ++i) {
            pq.offer(i);
        }
        List<Integer> ret = new ArrayList<>();
        while (!pq.isEmpty()) {
            ret.add(pq.poll());
        }
        return ret;
    }
}
```
DFS
```java
class Solution {
    public List<Integer> lexicalOrder(int n) {
        // 下一个
        // 先尝试cur*10，如果cur*10<=n，则next=cur*10
        // 否则cur%10==9||cur+1>n，回溯cur=cur/10
        // cur++
        // 加了n个数则停止
        List<Integer> list = new ArrayList<>();
        int cur = 1;
        for (int i = 0; i < n; ++i) {
            list.add(cur);
            if (cur * 10 <= n) {
                cur *= 10;
            } else {
                // 最后一位走到头了
                while (cur % 10 == 9 || cur + 1 > n)
                    cur /= 10;
                cur++;
            }
        }
        return list;
    }
}
```
DFS2
```java
class Solution {
    private List<Integer> res;

    public List<Integer> lexicalOrder(int n) {
        res = new ArrayList<>(n);
        for(int i = 1; i <= 9; i++){
            dfs(i, n);
        }
        return res;
    }

    public void dfs(int start, int n){
        if (start > n) return;
        res.add(start);
        for (int i = 0; i <= 9; i++){
            if (start * 10 + i > n) break;
            dfs(start * 10 + i, n);
        }
    }
}
```

## [821. 字符的最短距离](https://leetcode-cn.com/problems/shortest-distance-to-a-character/)

> 数组，双指针，字符串

执行用时：1 ms, 在所有 Java 提交中击败了98.08%的用户
内存消耗：41.3 MB, 在所有 Java 提交中击败了47.38%的用户
```java
class Solution {
    public int[] shortestToChar(String s, char c) {
        int n = s.length();
        int[] res = new int[n];
        for (int i = 0; i < n; ++i) {
            if (s.charAt(i) == c) {
                res[i] = 0;
            } else {
                if (i == 0) res[i] = Integer.MAX_VALUE;
                else res[i] = res[i - 1] == Integer.MAX_VALUE ? Integer.MAX_VALUE : res[i - 1] + 1;
            }
        }
        for (int i = n - 1; i >= 0; --i) {
            if (s.charAt(i) == c) {
                res[i] = 0;
            } else {
                if (i == n - 1) res[i] = Math.min(res[i], Integer.MAX_VALUE);
                else res[i] = Math.min(res[i], res[i + 1] + 1);
            }
        }
        return res;
    }
}
```

## [824. 山羊拉丁文](https://leetcode-cn.com/problems/goat-latin/)

> 字符串

```java
class Solution {
    private static Character[] CHARS = new Character[]{'a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U'};

    public String toGoatLatin(String sentence) {
        int idx = 1;
        boolean start = true;
        String end = "";
        StringBuilder sb = new StringBuilder();
        Set<Character> set = new HashSet<>() {{
            add('a');add('e');add('i');add('o');add('u');add('A');add('E');add('I');add('O');add('U');
        }};
        for (int i = 0; i < sentence.length(); ++i) {
            char c = sentence.charAt(i);
            if (i == sentence.length() - 1) {
                sb.append(c);
                if (i == 0 || sentence.charAt(i - 1) == ' ') {
                    sb.append("ma");
                } else {
                    sb.append(end);
                }
                for (int j = 0; j < idx; ++j) {
                    sb.append('a');
                }
                break;
            }
            if (start) {
                if (set.contains(c)) {
                    end = "ma";
                    sb.append(c);
                } else {
                    end = c + "ma";
                }
                start = false;
            } else if (c == ' ') {
                sb.append(end);
                for (int j = 0; j < idx; ++j) {
                    sb.append('a');
                }
                sb.append(c);
                idx++;
                start = true;
                end = "";
            } else {
                sb.append(c);
                start = false;
            }
        }
        return sb.toString();
    }
}
```

## [868. 二进制间距](https://leetcode-cn.com/problems/binary-gap/submissions/)

> 位运算，数学

复杂了
执行用时：1 ms, 在所有 Java 提交中击败了33.33%的用户
内存消耗：38.5 MB, 在所有 Java 提交中击败了37.79%的用户
```java
class Solution {
    private static HashMap<Long, Integer> map = new HashMap<>();

    static {
        int i = 1;
        long res = 1;
        while (res < 10e9) {
            map.put(res, i);
            res <<= 1;
            i++;
        }
    }

    public int binaryGap(int n) {
        int gap = 0, last = 0;
        while (n > 0) {
            int x = map.get((long)(n & -n));
            if (last != 0) {
                gap = Math.max(gap, x - last);
            }
            last = x;
            n &= (n - 1);
        }
        return gap;
    }
}
```
执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户
内存消耗：38.4 MB, 在所有 Java 提交中击败了51.97%的用户
```java
class Solution {
    public int binaryGap(int n) {
        int gap = 0, last = 0, i = 1;
        while (n > 0) {
            if ((n & 1) == 1) {
                if (last != 0) {
                    gap = Math.max(gap, i - last);
                }
                last = i;
            }
            i++;
            n >>= 1;
        }
        return gap;
    }
}
```

## [398. 随机数索引](https://leetcode-cn.com/problems/random-pick-index/)

> 水塘抽样，哈希表，数学，随机化

哈希表+随机（空间复杂度高）
```java
class Solution {
    private HashMap<Integer, List<Integer>> map;
    private Random r;

    public Solution(int[] nums) {
        r = new Random();
        map = new HashMap<>();
        for (int i = 0; i < nums.length; ++i) {
            map.putIfAbsent(nums[i], new ArrayList<Integer>());
            map.get(nums[i]).add(i);
        }
    }
    
    public int pick(int target) {
        List<Integer> tmp = map.get(target);
        return tmp.get(r.nextInt(tmp.size()));
    }
}
```
蓄水池抽样（不定长数据流）
```java
class Solution {
    private int[] _nums;
    private Random r;

    public Solution(int[] nums) {
        this._nums = nums;
        r = new Random();
    }
    
    public int pick(int target) {
        int res = 0;
        for (int i = 0, cnt = 0; i < _nums.length; ++i) {
            if (target == _nums[i]) {
                // 找到时，选择概率为0~cnt
                cnt++;
                if (r.nextInt(cnt) == 0) res = i;
            }
        }
        return res;
    }
}
```

## [883. 三维形体投影面积](https://leetcode-cn.com/problems/projection-area-of-3d-shapes/)

> 几何，数组，数学，矩阵

```java
class Solution {
    public int projectionArea(int[][] grid) {
        int n = grid.length, top = 0, front = 0, side = 0;
        for (int i = 0; i < n; ++i) {
            int lineMax = 0, rowMax = 0;
            for (int j = 0; j < n; ++j) {
                if (grid[i][j] != 0) top++;
                lineMax = Math.max(lineMax, grid[i][j]);
                rowMax = Math.max(rowMax, grid[j][i]);
            }
            side += lineMax;
            front += rowMax;
        }
        return top + front + side;
    }
}
```

## [417. 太平洋大西洋水流问题](https://leetcode-cn.com/problems/pacific-atlantic-water-flow/submissions/)

> DFS，BFS，数组，矩阵

```java
class Solution {
    private List<List<Integer>> po = new ArrayList<>(), ao = new ArrayList<>();
    private int m, n;

    public List<List<Integer>> pacificAtlantic(int[][] heights) {
        // 反向dfs
        m = heights.length;
        n = heights[0].length;
        boolean isPo = true;
        boolean[][] visited = new boolean[m][n];
        for (int i = 0; i < m; ++i) {
            dfs(heights, i, 0, visited, isPo);
        }
        for (int j = 1; j < n; ++j) {
            dfs(heights, 0, j, visited, isPo);
        }
        visited = new boolean[m][n];
        for (int i = 0; i < m; ++i) {
            dfs(heights, i, n - 1, visited, !isPo);
        }
        for (int j = 0; j < n - 1; ++j) {
            dfs(heights, m - 1, j, visited, !isPo);
        }
        // 比较两个list取出都有的
        List<List<Integer>> ret = new ArrayList<>();
        Set<String> cache = new HashSet<>();
        for (int i = 0; i < po.size(); ++i) {
            for (int j = 0; j < ao.size(); ++j) {
                List<Integer> pol = po.get(i);
                List<Integer> aol = ao.get(j);
                if (pol.get(0).equals(aol.get(0)) && pol.get(1).equals(aol.get(1))) {
                    String tmp = pol.get(0) + "," + pol.get(1);
                    if (!cache.contains(tmp)) {
                        cache.add(tmp);
                        ret.add(pol);
                    }
                }
            }
        }
        return ret;
    }

    private void dfs(int[][] heights, int x, int y, boolean[][] visited, boolean isPo) {
        visited[x][y] = true;
        if (isPo) {
            po.add(new ArrayList<>(){{add(x);add(y);}});
        } else {
            ao.add(new ArrayList<>(){{add(x);add(y);}});
        }
        int[][] directions = new int[][]{{0, 1}, {0, -1}, {-1, 0}, {1, 0}};
        for (int[] d: directions) {
            int nx = x + d[0], ny = y + d[1];
            if (nx >= 0 && nx < m && ny >= 0 && ny < n && !visited[nx][ny] && heights[nx][ny] >= heights[x][y]) {
                dfs(heights, nx, ny, visited, isPo);
            }
        }
    }
}
```
TODO 只用boolean数组
```java
class Solution {
    static int[][] dirs = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    int[][] heights;
    int m, n;

    public List<List<Integer>> pacificAtlantic(int[][] heights) {
        this.heights = heights;
        this.m = heights.length;
        this.n = heights[0].length;
        boolean[][] pacific = new boolean[m][n];
        boolean[][] atlantic = new boolean[m][n];
        for (int i = 0; i < m; i++) {
            dfs(i, 0, pacific);
        }
        for (int j = 1; j < n; j++) {
            dfs(0, j, pacific);
        }
        for (int i = 0; i < m; i++) {
            dfs(i, n - 1, atlantic);
        }
        for (int j = 0; j < n - 1; j++) {
            dfs(m - 1, j, atlantic);
        }
        List<List<Integer>> result = new ArrayList<List<Integer>>();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (pacific[i][j] && atlantic[i][j]) {
                    List<Integer> cell = new ArrayList<Integer>();
                    cell.add(i);
                    cell.add(j);
                    result.add(cell);
                }
            }
        }
        return result;
    }

    public void dfs(int row, int col, boolean[][] ocean) {
        if (ocean[row][col]) {
            return;
        }
        ocean[row][col] = true;
        for (int[] dir : dirs) {
            int newRow = row + dir[0], newCol = col + dir[1];
            if (newRow >= 0 && newRow < m && newCol >= 0 && newCol < n && heights[newRow][newCol] >= heights[row][col]) {
                dfs(newRow, newCol, ocean);
            }
        }
    }
}
```

## [905. 按奇偶排序数组](https://leetcode-cn.com/problems/sort-array-by-parity/)

> 数组，双指针，排序

执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户
内存消耗：42 MB, 在所有 Java 提交中击败了69.49%的用户
```java
class Solution {
    public int[] sortArrayByParity(int[] nums) {
        int last = -1;
        for (int i = 0; i < nums.length; ++i) {
            if ((nums[i] & 1) == 0) {
                if (++last <= i) {
                    swap(nums, last, i);
                }
            }
        }
        return nums;
    }

    private void swap(int[] num, int x, int y) {
        int tmp = num[x];
        num[x] = num[y];
        num[y] = tmp;
    }
}
```

## [427. 建立四叉树](https://leetcode-cn.com/problems/construct-quad-tree/)

> 树，数组，分治，矩阵

```java
class Solution {
    public Node construct(int[][] grid) {
        int n = grid.length;
        if (n == 1) {
            return new Node(grid[0][0] == 1, true);
        }
        return helper(grid, 0, 0, n);
    }

    private Node helper(int[][] grid, int x, int y, int w) {
        if (w == 1) {
            return new Node(grid[x][y] == 1, true);
        }
        boolean same = true;
        for (int i = x; i < x + w; ++i) {
            for (int j = y; j < y + w; ++j) {
                if (grid[i][j] != grid[x][y]) {
                    same = false;
                    break;
                }
            }
            if (!same) break;
        }
        if (same) {
            return new Node(grid[x][y] == 1, true);
        }
        int nw = w >> 1;
        Node n1 = helper(grid, x, y, nw); // topLeft
        Node n2 = helper(grid, x, y + nw, nw); // topRight
        Node n3 = helper(grid, x + nw, y, nw); // bottomLeft
        Node n4 = helper(grid, x + nw, y + nw, nw); // bottomRight
        return new Node(true, false, n1, n2, n3, n4);
    }
}
```

## [908. 最小差值 I](https://leetcode-cn.com/problems/smallest-range-i/submissions/)

> 数组，数学

```java
class Solution {
    public int smallestRangeI(int[] nums, int k) {
        // 最低点的最大值，最高点的最小值
        int min = Integer.MAX_VALUE, max = Integer.MIN_VALUE;
        for (int num: nums) {
            min = Math.min(min, num + k);
            max = Math.max(max, num - k);
        }
        return Math.max(0, max - min);
    }
}
```

## [1305. 两棵二叉搜索树中的所有元素](https://leetcode-cn.com/problems/all-elements-in-two-binary-search-trees/submissions/)

> 树，深度优先搜索，二叉搜索树，二叉树，排序

```java
class Solution {
    private List<Integer> list1 = new ArrayList<>();
    private List<Integer> list2 = new ArrayList<>();
    private List<Integer> ret = new ArrayList<>();

    public List<Integer> getAllElements(TreeNode root1, TreeNode root2) {
        dfs(root1, list1);
        dfs(root2, list2);
        merge();
        return ret;
    }

    private void dfs(TreeNode root, List<Integer> list) {
        if (root == null) {
            return;
        }
        dfs(root.left, list);
        list.add(root.val);
        dfs(root.right, list);
    }

    private void merge() {
        int i = 0, j = 0, n1 = list1.size(), n2 = list2.size();
        while (i < n1 && j < n2) {
            while (i < n1 && j < n2 && list1.get(i) >= list2.get(j)) {
                ret.add(list2.get(j++));
            }
            while (i < n1 && j < n2 && list1.get(i) <= list2.get(j)) {
                ret.add(list1.get(i++));
            }
        }
        if (i < n1) {
            for (int k = i; k < n1; ++k) ret.add(list1.get(k));
        }
        if (j < n2) {
            for (int k = j; k < n2; ++k) ret.add(list2.get(k));
        }
    }
}
```

## [591. 标签验证器](https://leetcode-cn.com/problems/tag-validator/submissions/)

> 栈，字符串

通过'<'开始进行条件判断，获取到的有效标签进行出入站操作，CDATA特殊处理
注意以下两个用例是不合法的
1.不能开新标签，整个字符串只能包含一套嵌套的标签，即
"<A></A><B></B>"不合法
"<A><B></B></A>"合法
2.不在<TAG>标签内的CDATA不合法，即
"<![CDATA[wahaha]]]>"不合法
"<DIV><![CDATA[wahaha]]]></DIV>"合法

执行用时：1 ms, 在所有 Java 提交中击败了100.00%的用户
内存消耗：39.6 MB, 在所有 Java 提交中击败了65.28%的用户
```java
class Solution {
    public boolean isValid(String code) {
        Deque<String> tags = new ArrayDeque<>();
        int i = 0, n = code.length();
        while (i < n) {
            if (code.charAt(i) == '<') {
                // i + 1开始是标签
                if (i == n - 1) return false;
                else {
                    if (code.charAt(i + 1) == '/') {
                        // i+2开始是闭合标签
                        // 找到右>位置
                        int j = code.indexOf('>', i + 2);
                        // TAG_NAME长度范围不对
                        if (j < 0 || j - i > 11 || j - i <= 2) return false;
                        String tag = code.substring(i + 2, j);
                        // 栈不匹配
                        if (tags.isEmpty() || !tags.peek().equals(tag)) return false;
                        // 匹配成功出栈
                        tags.pop();
                        // 更新下标
                        i = j + 1;
                    } else if (code.charAt(i + 1) == '!') {
                        // i+2开始是CDATA
                        // 必须在标签内
                        if (tags.isEmpty()) return false;
                        // 判别[i+2,i+9)为[CDATA[
                        if (i + 9 > n) return false;
                        String test = code.substring(i + 2, i + 9);
                        if (!test.equals("[CDATA[")) return false;
                        // 找到右]]>位置
                        int j = code.indexOf("]]>", i + 9);
                        if (j < 0) return false;
                        // 更新下标
                        i = j + 3;
                    } else {
                        // i+1开始是起始标签
                        // 不能新开标签
                        if (i != 0 && tags.isEmpty()) return false;
                        // 找到右>位置
                        int j = code.indexOf('>', i + 1);
                        // TAG_NAME长度范围不对
                        if (j < 0 || j - i > 10 || j - i <= 1) return false;
                        String tag = code.substring(i + 1, j);
                        if (!checkUpper(tag)) return false;
                        // 合法标签入栈
                        tags.push(tag);
                        // 更新小标
                        i = j + 1;
                    }
                }
            } else {
                // 非标签
                if (tags.isEmpty()) return false;
                i++;
            }
        }
        return tags.isEmpty();
    }

    private boolean checkUpper(String str) {
        for (char c: str.toCharArray()) {
            if (!Character.isUpperCase(c))
                return false;
        }
        return true;
    }
}
```

## [2243. 计算字符串的数字和](https://leetcode-cn.com/problems/calculate-digit-sum-of-a-string/)

> 字符串，模拟

```java
class Solution {
    public String digitSum(String s, int k) {
        if (k > s.length()) return s;
        while (s.length() > k) {
            StringBuilder sb = new StringBuilder();
            int i = 0;
            while (i + k < s.length()) {
                int sum = 0;
                for (int j = 0; j < k; ++j) {
                    sum += s.charAt(i + j) - '0';
                }
                sb.append(String.valueOf(sum));
                i += k;
            }
            if (i < s.length()) {
                int sum = 0;
                for (int j = i; j < s.length(); ++j) {
                    sum += s.charAt(j) - '0';
                }
                sb.append(String.valueOf(sum));
            }
            s = sb.toString();
        }
        return s;
    }
}
```
TODO 递归

## [2244. 完成所有任务需要的最少轮数](https://leetcode-cn.com/problems/minimum-rounds-to-complete-all-tasks/)

> 贪心，数组，哈希表，计数

```java
class Solution {
    public int minimumRounds(int[] tasks) {
        Arrays.sort(tasks);
        int ans = 0, cnt = 1, n = tasks.length;
        if (n == 1) return -1;
        for (int i = 1; i < n; ++i) {
            if (tasks[i] == tasks[i - 1]) {
                cnt++;
                if (i == n - 1) ans += cnt / 3 + (cnt % 3 == 0 ? 0 : 1);
            } else {
                if (i == n - 1) return -1;
                if (cnt == 1) return -1;
                ans += cnt / 3 + (cnt % 3 == 0 ? 0 : 1);
                cnt = 1;
            }
        }
        return ans;
    }
}
```

## [2245. 转角路径的乘积中最多能有几个尾随零](https://leetcode-cn.com/problems/maximum-trailing-zeros-in-a-cornered-path/)

> 数组，矩阵，前缀和

```java
class Solution {
    public int maxTrailingZeros(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        int[][][] map = new int[m][n][2];
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                map[i][j] = count25(grid[i][j]);
            }
        }
        int ret = 0;
        int[][][][] cache = new int[m + 2][n + 2][4][2]; // [x][y][上左下右][2数量/5数量]
        for (int i = 1; i < m + 1; ++i) {
            for (int j = 1; j < n + 1; ++j) {
                cache[i][j][0][0] = cache[i - 1][j][0][0] + map[i - 1][j - 1][0];
                cache[i][j][1][0] = cache[i][j - 1][1][0] + map[i - 1][j - 1][0];
                cache[i][j][0][1] = cache[i - 1][j][0][1] + map[i - 1][j - 1][1];
                cache[i][j][1][1] = cache[i][j - 1][1][1] + map[i - 1][j - 1][1];
            }
        }
        for (int i = m; i > 0; --i) {
            for (int j = n; j > 0; --j) {
                cache[i][j][2][0] = cache[i + 1][j][2][0] + map[i - 1][j - 1][0];
                cache[i][j][3][0] = cache[i][j + 1][3][0] + map[i - 1][j - 1][0];
                cache[i][j][2][1] = cache[i + 1][j][2][1] + map[i - 1][j - 1][1];
                cache[i][j][3][1] = cache[i][j + 1][3][1] + map[i - 1][j - 1][1];
            }
        }
        for (int i = 1; i < m + 1; ++i) {
            for (int j = 1; j < n + 1; ++j) {
                int[] tmp = new int[4];
                tmp[0] = Math.min(cache[i][j][0][0] + cache[i][j][1][0] - map[i - 1][j - 1][0], cache[i][j][0][1] + cache[i][j][1][1] - map[i - 1][j - 1][1]);
                tmp[1] = Math.min(cache[i][j][0][0] + cache[i][j][3][0] - map[i - 1][j - 1][0], cache[i][j][0][1] + cache[i][j][3][1] - map[i - 1][j - 1][1]);
                tmp[2] = Math.min(cache[i][j][1][0] + cache[i][j][2][0] - map[i - 1][j - 1][0], cache[i][j][1][1] + cache[i][j][2][1] - map[i - 1][j - 1][1]);
                tmp[3] = Math.min(cache[i][j][2][0] + cache[i][j][3][0] - map[i - 1][j - 1][0], cache[i][j][2][1] + cache[i][j][3][1] - map[i - 1][j - 1][1]);
                ret = Math.max(ret, Arrays.stream(tmp).max().getAsInt());
            }
        }
        return ret;
    }
    
    private int[] count25(int x) {
        int a = x, ans1 = 0;
        while (a > 0) {
            if (a % 2 == 0) ans1++;
            else break;
            a /= 2;
        }
        int b = x, ans2 = 0;
        while (b > 0) {
            if (b % 5 == 0) ans2++;
            else break;
            b /= 5;
        }
        return new int[]{ans1, ans2};
    }
}
```
TODO 枚举最长的L型即可，采用前缀和优化

## [937. 重新排列日志文件](https://leetcode-cn.com/problems/reorder-data-in-log-files/)

> 数组，字符串，排序

执行用时：3 ms, 在所有 Java 提交中击败了95.54%的用户
内存消耗：41.8 MB, 在所有 Java 提交中击败了47.44%的用户
```java
class Solution {
    public String[] reorderLogFiles(String[] logs) {
        Arrays.sort(logs, (str1, str2) -> {
            int si1 = str1.indexOf(' ');
            int si2 = str2.indexOf(' ');
            String id1 = str1.substring(0, si1);
            String content1 = str1.substring(si1 + 1);
            String id2 = str2.substring(0, si2);
            String content2 = str2.substring(si2 + 1);
            if (Character.isDigit(content1.charAt(0)) && Character.isLetter(content2.charAt(0))) {
                // 第一个数字，第二个字母，交换
                return 1;
            }
            if (Character.isDigit(content1.charAt(0)) && Character.isDigit(content2.charAt(0))) {
                // 两个都是数字，什么都不做
                return 0;
            }
            if (Character.isLetter(content1.charAt(0)) && Character.isLetter(content2.charAt(0))) {
                // 两个都是字母
                if (content1.compareTo(content2) > 0) {
                    // 第一个内容大，交换
                    return 1;
                } else if (content1.compareTo(content2) == 0) {
                    // 内容一样大，按标识符顺序排序
                    return id1.compareTo(id2);
                }
            }
            // 其他情况，升序排列
            return -1;
        });
        return logs;
    }
}
```

## [1823. 找出游戏的获胜者](https://leetcode-cn.com/problems/find-the-winner-of-the-circular-game/)

> 递归，队列，数组，数学，模拟，约瑟夫环

模拟
```java
class Solution {
    class Node {
        public int val;
        public Node next, last;

        public Node(int val) {
            this.val = val;
        }
    }

    public int findTheWinner(int n, int k) {
        if (k == 1) return n;
        Node root = new Node(1), p = root;
        for (int i = 2; i <= n; ++i) {
            Node node = new Node(i);
            node.last = p;
            p.next = node;
            p = p.next;
        }
        p.next = root;
        root.last = p;
        p = root;
        int cnt = k - 1;
        while (p.next != p) {
            while(p.next != p && cnt > 1) {
                p = p.next;
                cnt--;
            }
            p.next = p.next.next;
            p = p.next;
            cnt = k - 1;
        }
        return p.val;
    }
}
```
递归
```java
class Solution {
    // 从n个人，隔k个杀转化为n-1个人，隔k个杀（往前推k个）
    public int findTheWinner(int n, int k) {
        if (n == 1) return 1;
        int ans = (findTheWinner(n - 1, k) + k) % n;
        return ans == 0 ? n : ans;
    }
}
```
TODO 迭代
```java
class Solution {
    public int findTheWinner(int n, int k) {
        int id = 0;
        for (int i = 2; i <= n; ++i) {
            id = (id + k) % i;
        }
        return id + 1;
    }
}
```

## [713. 乘积小于 K 的子数组](https://leetcode-cn.com/problems/subarray-product-less-than-k/)

> 数组，滑动窗口

```java
class Solution {
    public int numSubarrayProductLessThanK(int[] nums, int k) {
        if (k <= 1) return 0;
        int ans = 0, mul = 1, l = 0, r = 0;
        while (r < nums.length) {
            // 更新状态
            mul *= nums[r];
            while (l <= r && mul >= k) { // 错误
                mul /= nums[l];
                l++;
            }
            ans += r - l + 1; // 更新结果
            r++;
        }
        return ans;
    }
}
```
TODO 二分查找

## [933. 最近的请求次数](https://leetcode-cn.com/problems/number-of-recent-calls/)

> 设计，队列，数据流

```java
class RecentCounter {
    private List<Integer> pingTimes;

    public RecentCounter() {
        pingTimes = new ArrayList<>();
    }
    
    public int ping(int t) {
        pingTimes.add(t);
        int ans = 0;
        int end = Math.max(0, t - 3000);
        for (int i = pingTimes.size() - 1; i >= 0 && pingTimes.get(i) >= t - 3000; --i) {
            ans++;
        }
        return ans;
    }
}
```
优化
```java
class RecentCounter {
    private int[] times = new int[10005];
    private int l = 0, r = 0;

    public RecentCounter() {
    }
    
    public int ping(int t) {
        times[r++] = t;
        while (times[l] < t - 3000) l++;
        return r - l;
    }
}
```

## [433. 最小基因变化](https://leetcode-cn.com/problems/minimum-genetic-mutation/)

> 广度优先搜索，哈希表，字符串

```java
class Solution {
    public int minMutation(String start, String end, String[] bank) {
        Set<String> bankSet = new HashSet<>();
        for (String b: bank) {
            bankSet.add(b);
        }
        if (!bankSet.contains(end)) return -1;
        int step = 0;
        Deque<String> q = new ArrayDeque<>();
        q.push(start);
        char[] cc = new char[]{'A', 'C', 'G', 'T'};
        while (!q.isEmpty()) {
            int size = q.size();
            boolean isAllFailed = true;
            for (int i = 0; i < size; ++i) {
                String str = q.pop();
                for (int j = 0; j < 8; ++j) {
                    for (int k = 0; k < 4; ++k) {
                        StringBuilder sb = new StringBuilder(str);
                        if (sb.charAt(j) != cc[k]) {
                            sb.setCharAt(j, cc[k]);
                            String tmp = sb.toString();
                            if (tmp.equals(end)) {
                                return step + 1;
                            } else if (bankSet.contains(tmp)) {
                                q.push(tmp);
                                isAllFailed = false;
                                bankSet.remove(tmp);
                            }
                        }
                    }
                }
            }
            if (isAllFailed) return -1;
            step++;
        }
        return step;
    }
}
```
TODO 预处理优化，双向BFS，建图，A*

## [442. 数组中重复的数据](https://leetcode-cn.com/problems/find-all-duplicates-in-an-array/)

> 数组，哈希表

```java
class Solution {
    public List<Integer> findDuplicates(int[] nums) {
        Map<Integer, Integer> map = new HashMap<>();
        Set<Integer> ret = new HashSet<>();
        for (int num: nums) {
            map.put(num, map.getOrDefault(num, 0) + 1);
        }
        for (int i: map.keySet()) {
            if (map.get(i) == 2) {
                ret.add(i);
            }
        }
        return new ArrayList<>(ret);
    }
}
```
交换
```java
class Solution {
    public List<Integer> findDuplicates(int[] nums) {
        List<Integer> ret = new ArrayList<>();
        int n = nums.length;
        for (int i = 0; i < n; ++i) {
            while (nums[i] != nums[nums[i] - 1]) { // 当前i位置不是应该的数
                swap(nums, i, nums[i] - 1);
            }
        }
        for (int i = 0; i < n; ++i) {
            if (nums[i] - 1 != i) {
                ret.add(nums[i]);
            }
        }
        return ret;
    }

    private void swap(int[] nums, int a, int b) {
        int tmp = nums[a];
        nums[a] = nums[b];
        nums[b] = tmp;
    }
}
```
TODO 使用负号
```java
class Solution {
    public List<Integer> findDuplicates(int[] nums) {
        int n = nums.length;
        List<Integer> ans = new ArrayList<Integer>();
        for (int i = 0; i < n; ++i) {
            int x = Math.abs(nums[i]);
            if (nums[x - 1] > 0) {
                nums[x - 1] = -nums[x - 1];
            } else {
                ans.add(x);
            }
        }
        return ans;
    }
}
```

## [942. 增减字符串匹配](https://leetcode.cn/problems/di-string-match/)

> 贪心，数组，数学，双指针，字符串，脑筋急转弯

```java
class Solution {
    public int[] diStringMatch(String s) {
        int n = s.length();
        Map<Integer, List<Integer>> map = new HashMap<>(); // <前面有几个相等的,位置>
        char[] cs = s.toCharArray();
        int cnt = 0;
        List<Integer> list0 = map.getOrDefault(cnt, new ArrayList<>());
        list0.add(0);
        map.put(0, list0);
        for (int i = 0; i < n; ++i) {
            if (s.charAt(i) == 'I') cnt++;
            else cnt--;
            List<Integer> list = map.getOrDefault(cnt, new ArrayList<>());
            list.add(i + 1);
            map.put(cnt, list);
        }
        int[] ret = new int[n + 1];
        List<Integer> keys = new ArrayList<>(map.keySet());
        Collections.sort(keys);
        int tmp = 0;
        for (int key: keys) {
            int size = map.get(key).size();
            for (int i = 0; i < size; ++i) {
                ret[map.get(key).get(i)] = tmp++;
            }
        }
        return ret;
    }
}
```
贪心
```java
class Solution {
    public int[] diStringMatch(String s) {
        // 每次都贪心，去掉后子问题都一样
        int l = 0, h = s.length(), n = h;
        int[] ret = new int[n + 1];
        for (int i = 0; i < n; ++i) {
            ret[i] = s.charAt(i) == 'I' ? l++ : h--;
        }
        ret[n] = l;
        return ret;
    }
}
```

## [1859. 将句子排序](https://leetcode.cn/problems/sorting-the-sentence/)

> 字符串，排序

```java
class Solution {
    public String sortSentence(String s) {
        String[] splitStrs = s.split(" ");
        Arrays.sort(splitStrs, (s1, s2) -> {
            return s1.charAt(s1.length() - 1) - s2.charAt(s2.length() - 1);
        });
        StringBuilder sb = new StringBuilder();
        for (String str: splitStrs) {
            sb.append(str.substring(0, str.length() - 1)).append(" ");
        }
        return sb.toString().trim();
    }
}
```
TODO 可以不排序

## [449. 序列化和反序列化二叉搜索树](https://leetcode.cn/problems/serialize-and-deserialize-bst/submissions/)

> 树，深度优先搜索，广度优先搜索，设计，二叉搜索树，字符串，二叉树

```java
public class Codec {
    List<String> list = new ArrayList<>();

    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
        preorder(root);
        return String.join(",", list);
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        String[] dataStrs = data.split(",");
        Deque<String> datas = new ArrayDeque<>();
        for (String str: dataStrs) {
            datas.offerLast(str);
        }
        return createTree(datas);
    }

    /* 先序遍历 */
    private void preorder(TreeNode root) {
        if (root == null) {
            list.add("#");
            return;
        }
        list.add(String.valueOf(root.val));
        preorder(root.left);
        preorder(root.right);
    }

    private TreeNode createTree(Deque<String> datas) {
        if (datas.isEmpty()) return null;
        String valStr = datas.pollFirst();
        if (!valStr.equals("#")) {
            TreeNode node = new TreeNode(Integer.parseInt(valStr));
            node.left = createTree(datas);
            node.right = createTree(datas);
            return node;
        }
        return null;
    }
}
```
根据有序性优化
```java
public class Codec {
    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
        List<Integer> list = new ArrayList<>();
        preorder(root, list);
        String str = list.toString();
        return str.substring(1, str.length() - 1);
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        if (data.length() == 0) return null;
        String[] dataStrs = data.split(", ");
        Deque<Integer> datas = new ArrayDeque<>();
        for (String str: dataStrs) {
            datas.offerLast(Integer.parseInt(str));
        }
        return construct(0, Integer.MAX_VALUE, datas);
    }

    /* 先序遍历 */
    private void preorder(TreeNode root, List<Integer> list) {
        if (root == null) return;
        list.add(root.val);
        preorder(root.left, list);
        preorder(root.right, list);
    }

    private TreeNode construct(int lower, int upper, Deque<Integer> datas) {
        if (datas.isEmpty() || datas.peekFirst() < lower || datas.peekFirst() > upper) return null;
        int val = datas.pollFirst();
        TreeNode node = new TreeNode(val);
        node.left = construct(lower, val, datas);
        node.right = construct(val, upper, datas);
        return node;
    }
}
```

## [297. 二叉树的序列化与反序列化](https://leetcode.cn/problems/serialize-and-deserialize-binary-tree/)

> 数，深度优先搜索，广度优先搜索，设计，字符串，二叉树

```java
public class Codec {
    List<String> list = new ArrayList<>();

    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
        preorder(root);
        return String.join(",", list);
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        String[] dataStrs = data.split(",");
        Deque<String> datas = new ArrayDeque<>();
        for (String str: dataStrs) {
            datas.offerLast(str);
        }
        return createTree(datas);
    }

    /* 先序遍历 */
    private void preorder(TreeNode root) {
        if (root == null) {
            list.add("#");
            return;
        }
        list.add(String.valueOf(root.val));
        preorder(root.left);
        preorder(root.right);
    }

    private TreeNode createTree(Deque<String> datas) {
        if (datas.isEmpty()) return null;
        String valStr = datas.pollFirst();
        if (!valStr.equals("#")) {
            TreeNode node = new TreeNode(Integer.parseInt(valStr));
            node.left = createTree(datas);
            node.right = createTree(datas);
            return node;
        }
        return null;
    }
}
```
TODO 括号表示编码 + 递归下降解码

## [944. 删列造序](https://leetcode.cn/problems/delete-columns-to-make-sorted/submissions/)

> 数组，字符串

```java
class Solution {
    public int minDeletionSize(String[] strs) {
        int n = strs[0].length();
        int delCnt = 0;
        for (int i = 0; i < n; ++i) {
            for (int j = 1; j < strs.length; ++j) {
                if (strs[j].charAt(i) < strs[j - 1].charAt(i)) {
                    delCnt++;
                    break;
                }
            }
        }
        return delCnt;
    }
}
```

## [面试题 01.05. 一次编辑](https://leetcode.cn/problems/one-away-lcci/)

> 双指针，字符串

执行用时：1 ms, 在所有 Java 提交中击败了97.69%的用户
内存消耗：41.3 MB, 在所有 Java 提交中击败了48.91%的用户
```java
class Solution {
    public boolean oneEditAway(String first, String second) {
        if (first.equals(second)) return true;
        int fn = first.length(), sn = second.length();
        if (Math.abs(fn - sn) > 1) return false;
        if (fn == sn) {
            // 相等，某一个字符替换即可
            boolean flag = false;
            for (int i = 0; i < fn; ++i) {
                if (first.charAt(i) != second.charAt(i)) {
                    if (flag) return false;
                    else flag = true;
                }
            }
        } else {
            // 相差1个，长的字符串需要删除1位
            boolean snSmaller = fn > sn;
            boolean flag = false;
            for (int i = 0, j = 0; i < fn && j < sn;) {
                if (first.charAt(i) != second.charAt(j)) {
                    if (flag) {
                        return false;
                    } else {
                        if (snSmaller) i++;
                        else j++;
                        flag = true;
                    }
                } else {
                    i++;
                    j++;
                }
            }
        }
        return true;
    }
}
```

## [812. 最大三角形面积](https://leetcode.cn/problems/largest-triangle-area/)

> 几何，数学，数组

暴力
```java
class Solution {
    public double largestTriangleArea(int[][] points) {
        // S = 1/2(x1y2-x2y1 + x2y3-x3y2 + x3y1-x1y3)
        double max = 0d;
        int n = points.length;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    if (i != j && i != k) {
                        int x1 = points[i][0], y1 = points[i][1];
                        int x2 = points[j][0], y2 = points[j][1];
                        int x3 = points[k][0], y3 = points[k][1];
                        max = Math.max(max, x1 * y2 - x2 * y1 + x2 * y3 - x3 * y2 + x3 * y1 - x1 * y3);
                    }
                }
            }
        }
        return max / 2;
    }
}
```
TODO 凸包优化

## [面试题 04.06. 后继者](https://leetcode.cn/problems/successor-lcci/)

> 树，深度优先搜索，二叉搜索树，二叉树，BST

```java
class Solution {
    private TreeNode ret;
    private boolean next = false;;

    public TreeNode inorderSuccessor(TreeNode root, TreeNode p) {
        dfs(root, p);
        return ret;
    }

    private void dfs(TreeNode root, TreeNode p) {
        if (root == null) return;
        dfs(root.left, p);
        if (next) {
            ret = root;
            next = false;
            return;
        }
        if (root == p) {
            next = true;
        }
        dfs(root.right, p);
    }
}
```
依据排序特点优化
执行用时：2 ms, 在所有 Java 提交中击败了100.00%的用户
内存消耗：42 MB, 在所有 Java 提交中击败了79.51%的用户
```java
class Solution {
    private TreeNode ret;

    public TreeNode inorderSuccessor(TreeNode root, TreeNode p) {
        dfs(root, p);
        return ret;
    }

    private void dfs(TreeNode node, TreeNode p) {
        if (node == null || p == null) return;
        if (node.val > p.val) {
            // 必然是本节点或者本节点的左子树下
            ret = node;
            dfs(node.left, p);
        } else {
            // 必然在本节点右子树下
            dfs(node.right, p);
        }
    }
}
```
TODO 精简版
```java
class Solution {
    public TreeNode inorderSuccessor(TreeNode root, TreeNode p) {
        if (root == null) return null;
        if (root.val <= p.val) return inorderSuccessor(root.right, p);
        TreeNode ans = inorderSuccessor(root.left, p);
        return ans == null ? root : ans;
    }
}
```

## [953. 验证外星语词典](https://leetcode.cn/problems/verifying-an-alien-dictionary/)

> 数组，哈希表，字符串

```java
class Solution {
    public boolean isAlienSorted(String[] words, String order) {
        Map<Character, Integer> map = new HashMap<>();
        for (int i = 0; i < order.length(); ++i) {
            map.put(order.charAt(i), i);
        }
        for (int i = 1; i < words.length; ++i) {
            if (!check(words[i - 1], words[i], map)) return false;
        }
        return true;
    }

    // a,b按照字典序a<=b true;a>b false
    private boolean check(String a, String b, Map<Character, Integer> map) {
        int na = a.length(), nb = b.length();
        for (int i = 0; i < na && i < nb; ++i) {
            if (map.get(a.charAt(i)) > map.get(b.charAt(i))) return false;
            else if (map.get(a.charAt(i)) < map.get(b.charAt(i))) return true;
        }
        return na <= nb;
    }
}
```

## [668. 乘法表中第k小的数](https://leetcode.cn/problems/kth-smallest-number-in-multiplication-table/)

> 二分查找

暴力（超时）
```java
class Solution {
    public int findKthNumber(int m, int n, int k) {
        PriorityQueue<Integer> pq = new PriorityQueue<>(k, (x, y) -> y - x);
        for (int i = 1; i <= n; ++i) {
            for (int j = 1; j <= m && j <= k / i; ++j) {
                int tmp = i * j;
                if (pq.size() == k && tmp < pq.peek()) {
                    pq.poll();
                    pq.offer(tmp);
                } else if (pq.size() < k) {
                    pq.offer(tmp);
                }
            }
        }
        return pq.peek();
    }
}
```
二分
```java
class Solution {
    public int findKthNumber(int m, int n, int k) {
        // [1,mn]二分查找
        // 乘法表中某个数x的位次时：\sum[1..m](min(x/i, n))【每一行计数后求和】
        // 进一步简化前几行可能都取，x/n*n+\sum[x/n+1..m](x/i)
        int l = 1, r = m * n;
        while (l < r) {
            int mid = l + (r - l) / 2;
            int pos = mid / n * n;
            for (int i = mid / n + 1; i <= m; ++i) {
                pos += mid / i;
            }
            if (pos < k) {
                l = mid + 1;
            } else {
                r = mid;
            }
        }
        return l;
    }
}
```

## [462. 最少移动次数使数组元素相等 II](https://leetcode.cn/problems/minimum-moves-to-equal-array-elements-ii/)

> 数组，数学，排序

前缀和
```java
class Solution {
    public int minMoves2(int[] nums) {
        Arrays.sort(nums);
        int n = nums.length;
        long[] lsum = new long[n], rsum = new long[n];
        for (int i = 1; i < n; ++i) {
            lsum[i] = lsum[i - 1] + (nums[i] - nums[i - 1]) * i;
        }
        for (int i = n - 2; i >= 0; --i) {
            rsum[i] = rsum[i + 1] + (nums[i + 1] - nums[i]) * (n - i - 1);
        }
        long ans = Long.MAX_VALUE;
        for (int i = 0; i < n; ++i) {
            ans = Math.min(ans, lsum[i] + rsum[i]);
        }
        return (int)ans;
    }
}
```
取中位数时最小
```java
class Solution {
    public int minMoves2(int[] nums) {
        Arrays.sort(nums);
        int n = nums.length, x = nums[n / 2], ans = 0;
        for (int i = 0; i < n; ++i) {
            ans += Math.abs(x - nums[i]);
        }
        return ans;
    }
}
```
TODO 不排序快速选择算法求中位数

## [436. 寻找右区间](https://leetcode.cn/problems/find-right-interval/)

> 数组，二分查找，排序

```java
class Solution {
    class Interval {
        public int start;
        public int end;
        public int idx;
        public Interval(int start, int end, int idx) {
            this.start = start;
            this.end = end;
            this.idx = idx;
        }
    }

    public int[] findRightInterval(int[][] intervals) {
        int n = intervals.length;
        Interval[] its = new Interval[n];
        for (int i = 0; i < n; ++i) {
            its[i] = new Interval(intervals[i][0], intervals[i][1], i);
        }
        Arrays.sort(its, (a, b) -> {
            return a.start - b.start;
        });
        int[] next = new int[n];
        for (int i = 0; i < n; ++i) {
            next[its[i].idx] = -1;
            int j = i;
            while (j < n) {
                if (its[j].start >= its[i].end) {
                    next[its[i].idx] = its[j].idx;
                    break;
                }
                j++;
            }
        }
        return next;
    }
}
```
双指针
```java
class Solution {
    public int[] findRightInterval(int[][] intervals) {
        int n = intervals.length;
        int[][] ss = new int[n][2];
        for (int i = 0; i < n; ++i) {
            ss[i] = new int[]{intervals[i][0], i};
        }
        Arrays.sort(ss, (a, b) -> a[0] - b[0]);
        int[] ret = new int[n];
        for (int i = 0; i < n; ++i) {
            int target = intervals[i][1];
            int l = 0, r = n - 1;
            while(l < r) {
                int mid = l + (r - l) / 2;
                if (ss[mid][0] >= target) r = mid;
                else l = mid + 1;
            }
            ret[i] = l < n && ss[l][0] >= target ? ss[l][1] : -1;
        }
        return ret;
    }
}
```

## [961. 在长度 2N 的数组中找出重复 N 次的元素](https://leetcode.cn/problems/n-repeated-element-in-size-2n-array/)

> 数组，哈希表

排序
```java
class Solution {
    public int repeatedNTimes(int[] nums) {
        Arrays.sort(nums);
        for (int i = 1; i < nums.length; ++i) {
            if (nums[i] == nums[i - 1]) return nums[i];
        }
        return -1;
    }
}
```
计数数组
```java
class Solution {
    public int repeatedNTimes(int[] nums) {
        int[] cnt = new int[10001];
        for (int i = 0; i < nums.length; ++i) {
            if (cnt[nums[i]] != 0) return nums[i];
            else cnt[nums[i]]++;
        }
        return -1;
    }
}
```
哈希表
执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户
内存消耗：41.8 MB, 在所有 Java 提交中击败了74.56%的用户
```java
class Solution {
    public int repeatedNTimes(int[] nums) {
        HashSet<Integer> set = new HashSet<>();
        for (int num: nums) {
            if (set.contains(num)) return num;
            else set.add(num);
        }
        return -1;
    }
}
```
数学
执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户
内存消耗：41.9 MB, 在所有 Java 提交中击败了64.29%的用户
```java
class Solution {
    public int repeatedNTimes(int[] nums) {
        int n = nums.length;
        for (int gap = 1; gap <= 3; ++gap) {
            for (int i = 0; i + gap < n; ++i) {
                if (nums[i] == nums[i + gap]) return nums[i];
            }
        }
        return -1;
    }
}
```
随机选择
执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户
内存消耗：42.1 MB, 在所有 Java 提交中击败了46.58%的用户
```java
class Solution {
    public int repeatedNTimes(int[] nums) {
        int n = nums.length;
        Random r = new Random();
        while(true) {
            int i = r.nextInt(n), j = r.nextInt(n);
            if (i != j && nums[i] == nums[j]) return nums[i];
        }
    }
}
```

## [464. 我能赢吗](https://leetcode.cn/problems/can-i-win/)

> 位运算，记忆化搜索，数学，动态规划，状态压缩，博弈

```java
class Solution {
    Map<Integer, Boolean> mem = new HashMap<>();

    public boolean canIWin(int maxChoosableInteger, int desiredTotal) {
        // dfs搜索，最多20位，所以可以用状压，Integer转二进制表示选择状态，记忆化搜索
        if ((1 + maxChoosableInteger) * (maxChoosableInteger - 1) < desiredTotal) return false;
        if (desiredTotal <= maxChoosableInteger) return true;
        return dfs(maxChoosableInteger, desiredTotal, 0, 0);
    }

    // 选择状态，当前累积的值
    private boolean dfs(int mci, int dt, int status, int cur) {
        if (!mem.containsKey(status)) {
            boolean res = false;
            for (int i = 0; i < mci; ++i) {
                if (((status >> i) & 1) == 0) { // 第i个数没有用
                    if (cur + i + 1 >= dt) { // 选完就赢了
                        res = true;
                        break;
                    }
                    if (!dfs(mci, dt, status | (1 << i), cur + i + 1)) { // 下一个人必输
                        res = true;
                        break;
                    }
                }
            }
            mem.put(status, res);
        }
        return mem.get(status);
    }
}
```

## [965. 单值二叉树](https://leetcode.cn/problems/univalued-binary-tree/)

> 树，深度优先搜索，广度优先搜索，二叉树

```java
class Solution {
    public boolean isUnivalTree(TreeNode root) {
        return dfs(root.left, root.val) && dfs(root.right, root.val);
    }

    private boolean dfs(TreeNode root, int val) {
        if (root == null) return true;
        if (root.val != val) return false;
        return dfs(root.left, val) && dfs(root.right, val);
    }
}
```

## [467. 环绕字符串中唯一的子字符串](https://leetcode.cn/problems/unique-substrings-in-wraparound-string/)

> 字符串，动态规划

```java
class Solution {
    public int findSubstringInWraproundString(String p) {
        // cnt[c]表示以c字符结尾的最大长度
        // cnt[c]=max(所有位置上为c的最大长度)
        int cur = 1;
        int[] cnt = new int[26];
        cnt[p.charAt(0) - 'a'] = 1;
        for (int i = 1; i < p.length(); ++i) {
            char c = p.charAt(i);
            char cl = p.charAt(i - 1);
            if (c == cl + 1 || c == 'a' && cl == 'z') {
                cur++;
            } else {
                cur = 1;
            }
            cnt[c - 'a'] = Math.max(cnt[c - 'a'], cur);
        }
        int ans = 0;
        for (int i = 0; i < 26; ++i) {
            ans += cnt[i];
        }
        return ans;
    }
}
```

## [699. 掉落的方块](https://leetcode.cn/problems/falling-squares/)

> 数组，有序集合，线段树

```java
class Solution {
    public List<Integer> fallingSquares(int[][] positions) {
        List<Integer> list = new ArrayList<>();
        for (int i = 0; i < positions.length; ++i) {
            int il = positions[i][0], ir = positions[i][0] + positions[i][1] - 1;
            int maxHeight = positions[i][1];
            for (int j = 0; j < i; ++j) {
                int jl = positions[j][0], jr = positions[j][0] + positions[j][1] - 1;
                // i方块和j方块有交集，i叠在j上面
                if (il <= jr && ir >= jl) {
                    maxHeight = Math.max(maxHeight, list.get(j) + positions[i][1]);
                }
            }
            list.add(maxHeight);
        }
        for (int i = 1; i < list.size(); ++i) {
            list.set(i, Math.max(list.get(i), list.get(i - 1)));
        }
        return list;
    }
}
```
TODO 有序集合，线段树

## [23. 合并K个升序链表](https://leetcode.cn/problems/merge-k-sorted-lists/)
## [剑指 Offer II 078. 合并排序链表](https://leetcode.cn/problems/vvXgSW/)

> 链表，分治，堆，归并排序

法一：连续归并
O(k^2n)
O(1)
```java
class Solution {
    public ListNode mergeKLists(ListNode[] lists) {
        ListNode ret = null;
        for (int i = 0; i < lists.length; i++) {
            ret = merge(ret, lists[i]);
        }
        return ret;
    }

    private ListNode merge(ListNode a, ListNode b) {
        if (a == null || b == null) {
            return a == null ? b : a;
        }
        ListNode dummy = new ListNode(), p = dummy, p1 = a, p2 = b;
        while (p1 != null && p2 != null) {
            if (p1.val < p2.val) {
                p.next = p1;
                p1 = p1.next;
            } else {
                p.next = p2;
                p2 = p2.next;
            }
            p = p.next;
        }
        p.next = (p1 != null ? p1 : p2);
        return dummy.next;
    }
}
```
法二：分治归并
O(klogkn)
O(logk)
```java
class Solution {
    public ListNode mergeKLists(ListNode[] lists) {
        return merge(lists, 0, lists.length - 1);
    }

    private ListNode merge(ListNode[] lists, int l, int r) {
        if (l == r) return lists[l];
        if (l > r) return null;
        int mid = (l + r) >> 1;
        return mergeTwo(merge(lists, l, mid), merge(lists, mid + 1, r));
    }

    private ListNode mergeTwo(ListNode a, ListNode b) {
        if (a == null || b == null) {
            return a == null ? b : a;
        }
        ListNode dummy = new ListNode(), p = dummy, p1 = a, p2 = b;
        while (p1 != null && p2 != null) {
            if (p1.val < p2.val) {
                p.next = p1;
                p1 = p1.next;
            } else {
                p.next = p2;
                p2 = p2.next;
            }
            p = p.next;
        }
        p.next = (p1 != null ? p1 : p2);
        return dummy.next;
    }
}
```
优先队列
O(klogkn)
O(k)
```java
class Solution {
    public ListNode mergeKLists(ListNode[] lists) {
        PriorityQueue<ListNode> pq = new PriorityQueue<>((a, b) -> a.val - b.val);
        for (int i = 0; i < lists.length; ++i) {
            if (lists[i] != null)
                pq.offer(lists[i]);
        }
        ListNode dummy = new ListNode(), p = dummy;
        while (!pq.isEmpty()) {
            p.next = pq.poll();
            p = p.next;
            if(p != null && p.next != null) pq.offer(p.next);
        }
        return dummy.next;
    }
}
```

## [面试题 17.11. 单词距离](https://leetcode.cn/problems/find-closest-lcci/)

> 数组，字符串
执行用时：10 ms, 在所有 Java 提交中击败了95.69%的用户
内存消耗：49.3 MB, 在所有 Java 提交中击败了22.59%的用户
```java
class Solution {
    public int findClosest(String[] words, String word1, String word2) {
        int l1 = -1, l2 = -1, min = Integer.MAX_VALUE;
        for (int i = 0; i < words.length; ++i) {
            if (word1.equals(words[i])) {
                if (l2 != -1) min = Math.min(min, i - l2);
                l1 = i;
            } else if (word2.equals(words[i])) {
                if (l1 != -1) min = Math.min(min, i - l1);
                l2 = i;
            }
        }
        return min;
    }
}
```

## [1021. 删除最外层的括号](https://leetcode.cn/problems/remove-outermost-parentheses/)

> 栈，字符串

```java
class Solution {
    public String removeOuterParentheses(String s) {
        StringBuilder sb = new StringBuilder();
        Deque<Character> dq = new ArrayDeque<>();
        for (int i = 0; i < s.length(); ++i) {
            char c = s.charAt(i);
            if (c == ')') {
                if (dq.size() > 1) sb.append(c);
                dq.pop();
            } else if (c == '(') {
                if (!dq.isEmpty()) sb.append(c);
                dq.push(c);
            }
        }
        return sb.toString();
    }
}
```
通过子数组和
```java
class Solution {
    public String removeOuterParentheses(String s) {
        int ans = 0;
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < s.length(); ++i) {
            char c = s.charAt(i);
            // 注意顺序
            if (c == ')') ans--;
            if (ans > 0) sb.append(c);
            if (c == '(') ans++;  
        }
        return sb.toString();
    }
}
```

## [1022. 从根到叶的二进制数之和](https://leetcode.cn/problems/sum-of-root-to-leaf-binary-numbers/)

> 树，深度优先搜索，二叉树

```java
class Solution {
    private int sum = 0;

    public int sumRootToLeaf(TreeNode root) {
        dfs(root, 0);
        return sum;
    }

    private void dfs(TreeNode root, int cur) {
        if (root == null) return;
        if (root.left == null && root.right == null) {
            sum += (cur << 1) + root.val;
            return;
        }
        cur = (cur << 1) + root.val;
        dfs(root.left, cur);
        dfs(root.right, cur);
    }
}
```
TODO 迭代

## [450. 删除二叉搜索树中的节点](https://leetcode.cn/problems/delete-node-in-a-bst/)

> 树，二叉搜索树，二叉树，递归

```java
class Solution {
    public TreeNode deleteNode(TreeNode root, int key) {
        if (root == null) {
            return null;
        }
        if (root.val > key) {
            root.left = deleteNode(root.left, key);
            return root;
        } else if (root.val < key) {
            root.right = deleteNode(root.right, key);
            return root;
        } else {
            if (root.left == null && root.right == null) return null;
            if (root.left == null) return root.right;
            if (root.right == null) return root.left;
            TreeNode successor = root.right;
            // 得到右子树的最小节点
            while (successor.left != null) successor = successor.left;
            root.right = deleteNode(root.right, successor.val);
            successor.left = root.left;
            successor.right = root.right;
            return successor;
        }
    }
}
```

## [929. 独特的电子邮件地址](https://leetcode.cn/problems/unique-email-addresses/)

> 数组，哈希表，字符串

```java
class Solution {
    public int numUniqueEmails(String[] emails) {
        Set<String> set = new HashSet<>();
        for (String email: emails) {
            String[] tmp = email.split("@");
            tmp[0] = tmp[0].replaceAll("\\.", "");
            tmp[0] = tmp[0].replaceAll("\\+.+", "");
            StringBuilder sb = new StringBuilder();
            sb.append(tmp[0]).append("@").append(tmp[1]);
            set.add(sb.toString());
        }
        return set.size();
    }
}
```
效率更高
```java
class Solution {
    public int numUniqueEmails(String[] emails) {
        Set<String> set = new HashSet<>();
        for (String email: emails) {
            StringBuilder sb = new StringBuilder();
            boolean isLocal = true;
            for (int i = 0; i < email.length(); ++i) {
                char c = email.charAt(i);
                if (isLocal && c == '.') continue;
                else if (isLocal && c == '+') {
                    while (email.charAt(i + 1) != '@') i++;
                } else if (c == '@') {
                    isLocal = false;
                    sb.append(c);
                } else {
                    sb.append(c);
                }
            }
            set.add(sb.toString());
        }
        return set.size();
    }
}
```

## [226. 翻转二叉树](https://leetcode.cn/problems/invert-binary-tree/)

> 树，深度优先搜索，广度优先搜索，二叉树

```java
class Solution {
    public TreeNode invertTree(TreeNode root) {
        if (root == null) return null;
        TreeNode tn = root.left;
        root.left = invertTree(root.right);
        root.right = invertTree(tn);
        return root;
    }
}
```

## [617. 合并二叉树](https://leetcode.cn/problems/merge-two-binary-trees/)

> 树，深度优先搜索，广度优先搜索，二叉树

```java
class Solution {
    public TreeNode mergeTrees(TreeNode root1, TreeNode root2) {
        if (root1 == null) return root2;
        if (root2 == null) return root1;
        int sum = root1.val + root2.val;
        return new TreeNode(sum, mergeTrees(root1.left, root2.left), mergeTrees(root1.right, root2.right));
    }
}
```

## [739. 每日温度](https://leetcode.cn/problems/daily-temperatures/)

> 栈，单调栈，数组

```java
class Solution {
    public int[] dailyTemperatures(int[] temperatures) {
        Deque<int[]> dq = new ArrayDeque<>();
        int n = temperatures.length;
        int[] ret = new int[n];
        for (int i = 0; i < n; ++i) {
            if (!dq.isEmpty() && dq.peek()[0] < temperatures[i]) {
                while (!dq.isEmpty() && dq.peek()[0] < temperatures[i]) {
                    int idx = dq.pop()[1];
                    ret[idx] = i - idx;
                }
            }
            dq.push(new int[]{temperatures[i], i});
        }
        while (!dq.isEmpty()) {
            int idx = dq.pop()[1];
            ret[idx] = 0;
        }
        return ret;
    }
}
```

## [5. 最长回文子串](https://leetcode.cn/problems/longest-palindromic-substring/)

> 字符串，动态规划，Manacher

```java
class Solution {
    public String longestPalindrome(String s) {
        // dp[i][j]表示以[i,j]是否为回文串
        int n = s.length();
        boolean[][] dp = new boolean[n][n];
        int l = 0, r = 0, maxLen = 1;
        for (int gap = 1; gap <= n; ++gap) {
            for (int start = 0; start + gap - 1 < n; start++) {
                int end = start + gap - 1;
                if (gap == 1) {
                    dp[start][end] = true;
                } else if (gap == 2) {
                    if (s.charAt(start) == s.charAt(end)) dp[start][end] = true;
                    else dp[start][end] = false;
                } else {
                    dp[start][end] = dp[start + 1][end - 1] & (s.charAt(start) == s.charAt(end));
                }
                if (dp[start][end] && gap > maxLen) {
                    maxLen = gap;
                    l = start;
                    r = end;
                }
            }
        }
        return s.substring(l, r + 1);
    }
}
```
TODO 中心扩展法，Manacher算法

## [142. 环形链表 II](https://leetcode.cn/problems/linked-list-cycle-ii/)

```java
public class Solution {
    public ListNode detectCycle(ListNode head) {
        if (head == null) return null;
        ListNode fast = head, slow = head;
        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;
            if (fast == slow) {
                // 注意通过公式推导出相遇点离相交点差从起点到相交点的距离
                fast = head;
                while (fast != slow) {
                    fast = fast.next;
                    slow = slow.next;
                }
                return slow;
            }
        }
        return null;
    }
}
```

## [78. 子集](https://leetcode.cn/problems/subsets/)

> 位运算，数组，回溯，递归

```java
class Solution {
    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> ret = new ArrayList<>();
        int n = nums.length;
        int upper = 1 << n;
        for (int i = 0; i < upper; ++i) {
            List<Integer> tmp = new ArrayList<>();
            for (int j = 0; j < 10; ++j) {
                if (((i >> j) & 1) == 1) {
                    tmp.add(nums[j]);
                }
            }
            ret.add(tmp);
        }
        return ret;
    }
}
```
TODO 递归

## [46. 全排列](https://leetcode.cn/problems/permutations/)

> 数组，回溯

```java
class Solution {
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> ret = new ArrayList<>();
        List<Integer> cur = new ArrayList<>();
        boolean[] vis = new boolean[nums.length];
        backtrace(nums, 0, vis, cur, ret);
        return ret;
    }

    private void backtrace(int[] nums, int pos, boolean[] vis, List<Integer> cur, List<List<Integer>> ret) {
        int n = nums.length;
        if (pos >= n) {
            // 一定要new ArrayList<>()取当前快照，不然后续cur会回溯为空
            ret.add(new ArrayList<>(cur));
            return;
        }
        for (int i = 0; i < n; ++i) {
            if (!vis[i]) {
                vis[i] = true;
                cur.add(nums[i]);
                backtrace(nums, pos + 1, vis, cur, ret);
                cur.remove(cur.size() - 1);
                vis[i] = false;
            }
        }
    }
}
```

## [114. 二叉树展开为链表](https://leetcode.cn/problems/flatten-binary-tree-to-linked-list/)

> 栈，树，深度优先搜索，链表，二叉树，递归

执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户
内存消耗：40.7 MB, 在所有 Java 提交中击败了87.48%的用户
```java
class Solution {
    public void flatten(TreeNode root) {
        // 空或者叶子节点退出
        if (root == null || root.left == null && root.right == null) return;
        // 右分支捋直
        flatten(root.right);
        // 左分支捋直
        flatten(root.left);
        // 左分支不为空，找到左分支最后一个节点后连接右分支，然后原地将根重置，即根的右分支置为左分支，左分支置为null
        if (root.left != null) {
            TreeNode p = root.left;
            while (p.right != null) {
                p = p.right;
            }
            p.right = root.right;
            root.right = root.left;
            root.left = null;
        }
    }
}
```
反向前序遍历更简单
```java
class Solution {
    TreeNode post = null;

    public void flatten(TreeNode root) {
        if (root == null) return;
        flatten(root.right);
        flatten(root.left);
        root.right = post;
        root.left = null;
        post = root;
    }
}
```

## [1038. 从二叉搜索树到更大和树](https://leetcode.cn/problems/binary-search-tree-to-greater-sum-tree/)
## [538. 把二叉搜索树转换为累加树](https://leetcode.cn/problems/convert-bst-to-greater-tree/)

> 树，深度优先搜索，二叉搜索树，二叉树，反序中序遍历

```java
class Solution {
    private int ans = 0;

    public TreeNode bstToGst(TreeNode root) {
        // 反序中序遍历并累加和
        if (root == null) return null;
        bstToGst(root.right);
        ans += root.val;
        root.val = ans;
        bstToGst(root.left);
        return root;
    }
}
```

## [剑指 Offer 21. 调整数组顺序使奇数位于偶数前面](https://leetcode.cn/problems/diao-zheng-shu-zu-shun-xu-shi-qi-shu-wei-yu-ou-shu-qian-mian-lcof/)

> 数组，双指针，排序

```java
class Solution {
    public int[] exchange(int[] nums) {
        int n = nums.length;
        int i = 0, j = nums.length - 1;
        while (i < j) {
            while (i < n && (nums[i] & 1) == 1) i++;
            while (j >= 0 && (nums[j] & 1) == 0) j--;
            if (i < j) {
                swap(nums, i, j);
                i++;
                j--;
            }
        }
        return nums;
    }

    private void swap(int[] nums, int i, int j) {
        int tmp = nums[i];
        nums[i] = nums[j];
        nums[j] = tmp;
    }
}
```

## [113. 路径总和 II](https://leetcode.cn/problems/path-sum-ii/)

> 树，深度优先搜索，回溯，二叉树

```java
class Solution {
    public List<List<Integer>> pathSum(TreeNode root, int targetSum) {
        List<List<Integer>> ret = new ArrayList<>();
        List<Integer> cur = new ArrayList<>();
        backtrace(root, targetSum, ret, cur, 0);
        return ret;
    }

    private void backtrace(TreeNode root, int targetSum, List<List<Integer>> ret, List<Integer> cur, int curSum) {
        if (root == null) return;
        if (root.left == null && root.right == null) {
            if (curSum + root.val == targetSum) {
                cur.add(root.val);
                ret.add(new ArrayList(cur));
                cur.remove(cur.size() - 1); // 重要
            }
            return;
        }
        curSum += root.val;
        cur.add(root.val);
        backtrace(root.left, targetSum, ret, cur, curSum);
        backtrace(root.right, targetSum, ret, cur, curSum);
        cur.remove(cur.size() - 1);
        curSum -= root.val;
    }
}
```

## [39. 组合总和](https://leetcode.cn/problems/combination-sum/)

> 数组，回溯

```java
class Solution {
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> ret = new ArrayList<>();
        List<Integer> cur = new ArrayList<>();
        backtrace(candidates, target, cur, ret, 0);
        return ret;
    }

    private void backtrace(int[] candidates, int target, List<Integer> cur, List<List<Integer>> ret, int idx) {
        if (idx == candidates.length) return;
        if (target == 0) {
            ret.add(new ArrayList<>(cur));
            return;
        }
        // 跳过当前数
        backtrace(candidates, target, cur, ret, idx + 1);
        // 选择当前数
        if (target >= candidates[idx]) {
            cur.add(candidates[idx]);
            backtrace(candidates, target - candidates[idx], cur, ret, idx);
            cur.remove(cur.size() - 1);
        }
    }
}
```

## [1037. 有效的回旋镖](https://leetcode.cn/problems/valid-boomerang/)

> 几何，数组，数学

```java
class Solution {
    public boolean isBoomerang(int[][] points) {
        return (points[1][1] - points[0][1]) * (points[2][0] - points[1][0]) != (points[2][1] - points[1][1]) * (points[1][0] - points[0][0]);
    }
}
```

## [497. 非重叠矩形中的随机点](https://leetcode.cn/problems/random-point-in-non-overlapping-rectangles/)

> 二分查找，随机化，前缀和，水塘抽样，别名抽样

```java
class Solution {
    private int[] preSum;
    private Random r;
    private int[][] rects;

    public Solution(int[][] rects) {
        this.rects = rects;
        int n = rects.length;
        preSum = new int[n + 1];
        preSum[0] = 0;
        for (int i = 1; i <= n; ++i) {
            int tt = (rects[i - 1][2] - rects[i - 1][0] + 1) * (rects[i - 1][3] - rects[i - 1][1] + 1);
            preSum[i] = preSum[i - 1] + tt;
        }
        r = new Random();
    }
    
    public int[] pick() {
        int selectId = r.nextInt(preSum[preSum.length - 1]); // [0..sum)
        int rectId = bs(selectId + 1) - 1;
        int[] rect = rects[rectId];
        int w = rect[2] - rect[0] + 1;
        int rest = selectId - preSum[rectId];
        int nh = rest / w;
        int x = rect[0] + rest - w * nh;
        int y = rect[1] + nh;
        return new int[]{x, y};
    }

    private int bs(int target) {
        int l = 0, r = preSum.length - 1;
        while (l <= r) {
            int mid = l + (r - l) / 2;
            if (preSum[mid] == target) {
                return mid;
            } else if (preSum[mid] < target) {
                l = mid + 1;
            } else {
                r = mid - 1;
            }
        }
        return l;
    }
}
```
TODO 水塘抽样，别名抽样

## [730. 统计不同回文子序列](https://leetcode.cn/problems/count-different-palindromic-subsequences/)

> 字符串，动态规划

```java
class Solution {
    public int countPalindromicSubsequences(String s) {
        // 表示以x为左右端点的不同非空回文子串
        // dp[x][i][j]=\sum(dp[y][i+1][j-1])+2/dp[x][i][j-1]/dp[x][i+1][j]/dp[x][i+1][j-1]
        // 初始化：dp[x][i][i]=0(s[i]!=x)/1(s[i]==x)
        // dp[x][i][j]=0(i<j)
        // \sum_x dp[x][0][n-1]
        int MOD = 1000000007;
        int n = s.length();
        int[][][] dp = new int[4][n][n];
        for (int gap = 0; gap < n; ++gap) {
            for (int i = 0; i + gap < n; ++i) {
                if (gap == 0) {
                    for (int x = 0; x < 4; ++x) {
                        dp[x][i][i] = s.charAt(i) == (char)(x + 'a') ? 1 : 0;
                    }
                    continue;
                }
                int j = i + gap;
                for (int x = 0; x < 4; ++x) {
                    char c = (char)(x + 'a');
                    if (s.charAt(i) == s.charAt(j) && s.charAt(i) == c) {
                        for (int z = 0; z < 4; ++z) {
                            dp[x][i][j] = (dp[x][i][j] + dp[z][i + 1][j - 1]) % MOD; 
                        }
                        dp[x][i][j] += 2;
                    } else if (s.charAt(i) == c) {
                        dp[x][i][j] = dp[x][i][j - 1];
                    } else if (s.charAt(j) == c) {
                        dp[x][i][j] = dp[x][i + 1][j];
                    } else {
                        dp[x][i][j] = dp[x][i + 1][j - 1];
                    }
                }
            }
        }
        int sum = 0;
        for (int z = 0; z < 4; ++z) {
            sum = (sum + dp[z][0][n - 1]) % MOD;
        }
        return sum;
    }
}
```
TODO 二维优化

## [926. 将字符串翻转到单调递增](https://leetcode.cn/problems/flip-string-to-monotone-increasing/)

> 字符串，动态规划

```java
class Solution {
    public int minFlipsMonoIncr(String s) {
        // dp[i][0]表示s[0..i]且翻转后s[i]结果为0的所求，dp[i][1]表示s[0..i]且翻转后s[i]结果为1的所求
        // dp[i][0]=dp[i-1][0]+II(s[i]==1)
        // dp[i][1]=min(dp[i-1][0], dp[i-1][1])+II(s[i]==0)
        // dp[0][0]=II(s[0]==1)/dp[0][1]==II(s[1]==0)
        // min(dp[n-1][0],dp[n-1][1])
        // 可以用滚动优化
        int dp_i_0 = 0, dp_i_1 = 0;
        for (int i = 0; i < s.length(); ++i) {
            dp_i_1 = Math.min(dp_i_0, dp_i_1) + '1' - s.charAt(i);
            dp_i_0 += s.charAt(i) - '0';
        }
        return Math.min(dp_i_0, dp_i_1);
    }
}
```

## [890. 查找和替换模式](https://leetcode.cn/problems/find-and-replace-pattern/)

> 数组，哈希表，字符串，归一化

执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户
内存消耗：39.9 MB, 在所有 Java 提交中击败了94.87%的用户
```java
class Solution {
    public List<String> findAndReplacePattern(String[] words, String pattern) {
        int[] parr = getParr(pattern);
        int n = pattern.length();
        List<String> ret = new ArrayList<>();
        for (String word: words) {
            if (word.length() == n) {
                int[] warr = getParr(word);
                boolean isSame = true;
                for (int i = 0; i < n; ++i) {
                    if (parr[i] != warr[i]) {
                        isSame = false;
                        break;
                    }
                }
                if (isSame) ret.add(word);
            }
        }
        return ret;
    }

    private int[] getParr(String p) {
        int[] poses = new int[26];
        int n = p.length();
        int[] ret = new int[n];
        for (int i = 0; i < n; ++i) {
            if (poses[p.charAt(i) - 'a'] == 0) {
                poses[p.charAt(i) - 'a'] = i + 1;
                ret[i] = i + 1;
            } else {
                ret[i] = poses[p.charAt(i) - 'a'];
            }
        }
        return ret;
    }
}
```

## [1051. 高度检查器](https://leetcode.cn/problems/height-checker/)

> 数组，计数排序，排序

暴力
```java
class Solution {
    public int heightChecker(int[] heights) {
        int n = heights.length;
        int[] newHeights = new int[n];
        System.arraycopy(heights, 0, newHeights, 0, n);
        Arrays.sort(newHeights);
        int cnt = 0;
        for (int i = 0; i < n; ++i) {
            cnt += newHeights[i] != heights[i] ? 1 : 0;
        }
        return cnt;
    }
}
```
计数排序
执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户
内存消耗：39.1 MB, 在所有 Java 提交中击败了68.13%的用户
```java
class Solution {
    public int heightChecker(int[] heights) {
        int[] hc = new int[101];
        for (int height: heights) {
            hc[height]++;
        }
        int cnt = 0, pos = 0, i = 0;
        while (i < heights.length) {
            while (hc[pos] == 0) pos++;
            while (hc[pos] > 0) {
                if (pos != heights[i]) {
                    cnt++;
                }
                ++i;
                hc[pos]--;
            }
        }
        return cnt;
    }
}
```

## [498. 对角线遍历](https://leetcode.cn/problems/diagonal-traverse/)

> 数组，矩阵，模拟

模拟
执行用时：2 ms, 在所有 Java 提交中击败了78.95%的用户
内存消耗：43.2 MB, 在所有 Java 提交中击败了46.15%的用户
```java
class Solution {
    public int[] findDiagonalOrder(int[][] mat) {
        int m = mat.length, n = mat[0].length, cnt = m * n;
        int x = 0, y = 0, nx = 0, ny = 0, pos = 1;
        int[][] dirs = new int[][]{{-1, 1}, {1, -1}};
        int next = 0;
        int[] ret = new int[cnt];
        ret[0] = mat[0][0];
        while (pos < cnt) {
            nx = x + dirs[next][0];
            ny = y + dirs[next][1];
            if (nx < 0 && ny < n) {
                // 上三角越界，右移
                y++;
                ret[pos++] = mat[x][y];
                next = 1;
            } else if (ny < 0 && nx < m) {
                // 左三角越界，下移
                x++;
                ret[pos++] = mat[x][y];
                next = 0;
            } else if (ny >= n) {
                // 右三角越界，下移
                x++;
                ret[pos++] = mat[x][y];
                next = 1;
            } else if (nx >= m) {
                // 下三角越界，右移
                y++;
                ret[pos++] = mat[x][y];
                next = 0;
            } else {
                // 不越界，方向保持不变
                x = nx;
                y = ny;
                ret[pos++] = mat[x][y];
            }
        }
        return ret;
    }
}
```
TODO 根据奇偶优化

## [719. 找出第 K 小的数对距离](https://leetcode.cn/problems/find-k-th-smallest-pair-distance/)

> 数组，双指针，二分查找，排序

```java
class Solution {
    public int smallestDistancePair(int[] nums, int k) {
        Arrays.sort(nums);
        int n = nums.length;
        int l = 0, r = nums[n - 1] - nums[0]; // 差值上下限
        while (l <= r) {
            int mid = l + (r - l) / 2;
            int cnt = getCnt(nums, mid); // 和为mid时有多少个数对
            if (cnt < k) {
                l = mid + 1;
            } else if (cnt >= k) {
                r = mid - 1;
            }
        }
        return l;
    }

    private int getCnt(int[] nums, int target) {
        int n = nums.length, ans = 0;
        // 固定右边，收缩左边
        for (int i = 0, j = 0; j < n; ++j) {
            while (nums[j] - nums[i] > target) i++;
            ans += j - i;
        }
        return ans;
    }
}
```

## [532. 数组中的 k-diff 数对](https://leetcode.cn/problems/k-diff-pairs-in-an-array/)

> 数组，哈希表，双指针，二分查找，排序

```java
class Solution {
    private Set<String> set = new HashSet<>();

    public int findPairs(int[] nums, int k) {
        Map<Integer, TreeSet<Integer>> map = new HashMap<>();
        int n = nums.length;
        for (int i = 0; i < n; ++i) {
            TreeSet<Integer> ts = map.containsKey(nums[i]) ? map.get(nums[i]) : new TreeSet<>();
            ts.add(i);
            map.put(nums[i], ts);
        }
        for (int i = 0; i < n; ++i) {
            getNewPairCnt(nums, map, nums[i] + k, i);
            getNewPairCnt(nums, map, nums[i] - k, i);
        }
        return set.size();
    }

    private void getNewPairCnt(int[] nums, Map<Integer, TreeSet<Integer>> map, int x, int pos) {
        if (!map.containsKey(x)) {
            return;
        }
        TreeSet<Integer> otherPoses = map.get(x);
        Set<Integer> headSet = otherPoses.headSet(pos);
        for (Integer i: headSet) {
            String str = Math.min(nums[i], nums[pos]) + "," + Math.max(nums[i], nums[pos]);
            if (!set.contains(str)) {
                set.add(str);
            }
        }
        Set<Integer> tailSet = otherPoses.tailSet(pos);
        for (Integer i: tailSet) {
            if (i != pos) {
                String str = Math.min(nums[i], nums[pos]) + "," + Math.max(nums[i], nums[pos]);
                if (!set.contains(str)) {
                    set.add(str);
                }
            }
        }
    }
}
```
简化
```java
class Solution {
    public int findPairs(int[] nums, int k) {
        Set<Integer> set = new HashSet<>(); // 找到后放入小的
        Set<Integer> vis = new HashSet<>(); // 访问过的数
        for (int num: nums) {
            if (vis.contains(num - k)) {
                set.add(num - k);
            }
            if (vis.contains(num + k)) {
                set.add(num);
            }
            vis.add(num);
        }
        return set.size();
    }
}
```
TODO 排序

## [958. 二叉树的完全性检验](https://leetcode.cn/problems/check-completeness-of-a-binary-tree/)

> 树，广度优先搜索，二叉树

```java
class Solution {
    public boolean isCompleteTree(TreeNode root) {
        // 层序遍历，如果遇到null后遇到的都是null，则是完全二叉树，否则不是
        Deque<TreeNode> dq = new ArrayDeque<>();
        dq.add(root);
        boolean isEnd = false;
        while (!dq.isEmpty()) {
            int size = dq.size();
            for (int i = 0; i < size; ++i) {
                TreeNode p = dq.poll();
                if (p.left != null) {
                    if (isEnd) return false;
                    dq.offer(p.left);
                } else {
                    isEnd = true;
                }
                if (p.right != null) {
                    if (isEnd) return false;
                    dq.offer(p.right);
                } else {
                    isEnd = true;
                }
            }
        }
        return true;
    }
}
```
TODO 采用编号

## [63. 不同路径 II](https://leetcode.cn/problems/unique-paths-ii/)

> 数组，动态规划，矩阵

DFS
```java
class Solution {
    public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        int m = obstacleGrid.length, n = obstacleGrid[0].length;
        int[][] mem = new int[m][n];
        for (int i = 0; i < m; ++i) {
            Arrays.fill(mem[i], -1);
        }
        return dfs(obstacleGrid, m, n, 0, 0, mem);
    }

    private int dfs(int[][] grid, int m, int n, int cx, int cy, int[][] mem) {
        if (cx < 0 || cy < 0 || cx >= m || cy >= n || grid[cx][cy] == 1) {
            return 0;
        }
        if (cx == m - 1 && cy == n - 1) {
            return 1;
        }
        if (mem[cx][cy] != -1) return mem[cx][cy];
        int sum = dfs(grid, m, n, cx + 1, cy, mem);
        sum += dfs(grid, m, n, cx, cy + 1, mem);
        mem[cx][cy] = sum;
        return sum;
    }
}
```
DP
```java
class Solution {
    public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        // dp[i][j]=og[i][j]==1?0:dp[i-1][j]+dp[i][j-1]
        // dp[0][0]=og[i][j]==1?0:1;
        // dp[m-1][n-1]
        int m = obstacleGrid.length, n = obstacleGrid[0].length;
        int[][] dp = new int[m][n];
        if (obstacleGrid[0][0] == 1) return 0;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i == 0 && j == 0) {
                    dp[i][j] = 1;
                    continue;
                }
                if (obstacleGrid[i][j] == 1) {
                    dp[i][j] = 0;
                    continue;
                }
                int sum = 0;
                if (i > 0) sum += dp[i - 1][j];
                if (j > 0) sum += dp[i][j - 1];
                dp[i][j] = sum;
            }
        }
        return dp[m - 1][n - 1];
    }
}
```
TODO 降为一维数组优化

## [124. 二叉树中的最大路径和](https://leetcode.cn/problems/binary-tree-maximum-path-sum/)

> 树，深度优先搜索，动态规划，二叉树

```java
class Solution {
    private int max = -1001;
    
    public int maxPathSum(TreeNode root) {
        dfs(root);
        return max;
    }

    private int dfs(TreeNode root) {
        if (root == null) return -10001;
        int a = dfs(root.left);
        int b = dfs(root.right);
        int c = root.val;
        // 最大贡献只能自己/自己+左节点最大贡献/自己+右节点最大贡献
        int ret = Math.max(c, Math.max(a + c, b + c));
        // 最大值可以再包括左+自己+右
        max = Math.max(max, Math.max(a + b + c, ret));
        return ret;
    }
}
```

## [1089. 复写零](https://leetcode.cn/problems/duplicate-zeros/)

> 数组，双指针

```java
class Solution {
    public void duplicateZeros(int[] arr) {
        Deque<Integer> dq = new ArrayDeque<>();
        for (int i = 0; i < arr.length; ++i) {
            if (!dq.isEmpty()) {
                int x = dq.pollFirst();
                if (arr[i] == 0) {
                    dq.offerLast(0);
                    dq.offerLast(0);
                } else {
                    dq.offerLast(arr[i]);
                }
                arr[i] = x;
            } else {
                if (arr[i] == 0) {
                    dq.offerLast(0);
                }
            }
        }
        return;
    }
}
```
TODO 双指针

## [剑指 Offer II 029. 排序的循环链表](https://leetcode.cn/problems/4ueAj6/)

> 链表

```java
/*
// Definition for a Node.
class Node {
    public int val;
    public Node next;

    public Node() {}

    public Node(int _val) {
        val = _val;
    }

    public Node(int _val, Node _next) {
        val = _val;
        next = _next;
    }
};
*/

class Solution {
    public Node insert(Node head, int insertVal) {
        if (head == null) {
            head = new Node(insertVal);
            head.next = head;
            return head;
        } else {
            Node h = head;
            while (true) {
                Node next = head.next;
                boolean flag = false;
                if (next.val != head.val) {
                    flag = true;
                }
                if (!flag && next == h ||
                    next.val < head.val && (insertVal <= next.val || insertVal >= head.val) ||
                    insertVal <= next.val && head.val <= insertVal) {
                    // 三种情况：全一样，直接插入/在节点处，插入节点/满足序列顺序插入两者之间
                    head.next = new Node(insertVal);
                    head.next.next = next;
                    return h;
                }
                head = head.next;
            }
        }
    }
}
```

## [508. 出现次数最多的子树元素和](https://leetcode.cn/problems/most-frequent-subtree-sum/)

> 树，深度优先搜索，哈希表，二叉树

```java
class Solution {
    private Map<Integer, Integer> map = new HashMap<>();
    private int max = 0;

    public int[] findFrequentTreeSum(TreeNode root) {
        dfs(root);
        List<Integer> list = new ArrayList<>();
        for (int key: map.keySet()) {
            if (map.get(key) == max) {
                list.add(key);
            }
        }
        int[] ret = new int[list.size()];
        for (int i = 0; i < list.size(); ++i) {
            ret[i] = list.get(i);
        }
        return ret;
    }

    private int dfs(TreeNode root) {
        if (root == null) return 0;
        int sum = root.val + dfs(root.left) + dfs(root.right);
        map.put(sum, map.getOrDefault(sum, 0) + 1);
        max = Math.max(max, map.get(sum));
        return sum;
    }
}
```

## [337. 打家劫舍 III](https://leetcode.cn/problems/house-robber-iii/)

> 树，深度优先搜索，动态规划，二叉树

```java
class Solution {
    public int rob(TreeNode root) {
        int[] ret = dfs(root);
        return Math.max(ret[0], ret[1]);
    }

    private int[] dfs(TreeNode root) {
        if (root == null) {
            return new int[]{0, 0};
        }
        int[] l = dfs(root.left);
        int[] r = dfs(root.right);
        int s = root.val + l[1] + r[1];
        int ns = Math.max(l[0], l[1]) + Math.max(r[0], r[1]);
        return new int[]{s, ns};
    }
}
```

## [96. 不同的二叉搜索树](https://leetcode.cn/problems/unique-binary-search-trees/)

> 树，二叉搜索树，数学，动态规划，二叉树，Catalan数

```java
class Solution {
    public int numTrees(int n) {
        long ans = 1;
        for (int i = 0; i < n; ++i) {
            ans = ans * (4 * i + 2) / (i + 2);
        }
        return (int)ans;
    }
}
```

## [513. 找树左下角的值](https://leetcode.cn/problems/find-bottom-left-tree-value/)

> 树，深度优先搜索，广度优先搜索，二叉树

层次遍历
```java
class Solution {
    public int findBottomLeftValue(TreeNode root) {
        List<Integer> cur = new ArrayList<>();
        Deque<TreeNode> dq = new ArrayDeque<>();
        dq.add(root);
        cur.add(root.val);
        while (!dq.isEmpty()) {
            int size = dq.size();
            boolean needClear = true;
            while (size-- > 0) {
                TreeNode tn = dq.pollFirst();
                if (tn.left != null) {
                    if (needClear) {
                        needClear = false;
                        cur.clear();
                    }
                    cur.add(tn.left.val);
                    dq.offerLast(tn.left);
                }
                if (tn.right != null) {
                    if (needClear) {
                        needClear = false;
                        cur.clear();
                    }
                    cur.add(tn.right.val);
                    dq.offerLast(tn.right);
                }
            }
        }
        return cur.get(0);
    }
}
```
DFS
```java
class Solution {
    private int maxLevel = 0;
    private int ret = 0;

    public int findBottomLeftValue(TreeNode root) {
        ret = root.val;
        dfs(root, 0);
        return ret;
    }

    private void dfs(TreeNode root, int level) {
        if (root == null) {
            return;
        }
        if (level > maxLevel) {
            maxLevel = level;
            ret = root.val;
        }
        dfs(root.left, level + 1);
        dfs(root.right, level + 1);
    }
}
```

## [515. 在每个树行中找最大值](https://leetcode.cn/problems/find-largest-value-in-each-tree-row/)

> 树，深度优先搜索，广度优先搜索，二叉树

```java
class Solution {
    public List<Integer> largestValues(TreeNode root) {
        List<Integer> ret = new ArrayList<>();
        if (root == null) return ret;
        Deque<TreeNode> dq = new ArrayDeque<>();
        dq.offerLast(root);
        while (!dq.isEmpty()) {
            int size = dq.size();
            int max = Integer.MIN_VALUE;
            while (size-- > 0) {
                TreeNode tn = dq.pollFirst();
                max = Math.max(max, tn.val);
                if (tn.left != null) dq.offerLast(tn.left);
                if (tn.right != null) dq.offerLast(tn.right);
            }
            ret.add(max);
        }
        return ret;
    }
}
```
TODO DFS

## [剑指 Offer II 091. 粉刷房子](https://leetcode.cn/problems/JEj789/)

> 数组，动态规划

```java
class Solution {
    public int minCost(int[][] costs) {
        // dp[i][j]表示第i个房子粉刷第j种颜色的最小成本
        // dp[i][j]=min(dp[i-1][0],dp[i-2][1],dp[i-3][2])+costs[i][j]
        // dp[0][j]=costs[0][j]
        // min(dp[n-1][j])
        int n = costs.length;
        int[][] dp = new int[n][3];
        dp[0][0] = costs[0][0];
        dp[0][1] = costs[0][1];
        dp[0][2] = costs[0][2];
        for (int i = 1; i < n; ++i) {
            dp[i][0] = Math.min(dp[i - 1][1], dp[i - 1][2]) + costs[i][0];
            dp[i][1] = Math.min(dp[i - 1][0], dp[i - 1][2]) + costs[i][1];
            dp[i][2] = Math.min(dp[i - 1][0], dp[i - 1][1]) + costs[i][2];
        }
        return Math.min(dp[n - 1][0], Math.min(dp[n - 1][1], dp[n - 1][2]));
    }
}
```
空间优化
```java
class Solution {
    public int minCost(int[][] costs) {
        int n = costs.length;
        int[] dp = new int[3];
        for (int i = 0; i < n; ++i) {
            if (i == 0) {
                dp[0] = costs[0][0];
                dp[1] = costs[0][1];
                dp[2] = costs[0][2];
            } else {
                int a = dp[0], b = dp[1], c = dp[2];
                dp[0] = Math.min(b, c) + costs[i][0];
                dp[1] = Math.min(a, c) + costs[i][1];
                dp[2] = Math.min(a, b) + costs[i][2];
            }
        }
        return Math.min(dp[0], Math.min(dp[1], dp[2]));
    }
}
```

## [710. 黑名单中的随机数](https://leetcode.cn/problems/random-pick-with-blacklist/)

> 哈希表，数学，随机化，映射

将黑名单映射到后面的白名单
```java
class Solution {
    private int ww = 0;
    private Map<Integer, Integer> map;
    private Random r;

    public Solution(int n, int[] blacklist) {
        ww = n - blacklist.length;
        r = new Random();
        map = new HashMap<>();
        int start = ww;
        Set<Integer> rbs = new HashSet<>();
        for (int i: blacklist) {
            if (i >= ww) rbs.add(i);
        }
        for (int i: blacklist) {
            if (i < ww) {
                while (rbs.contains(start)) start++;
                map.put(i, start);
                start++;
            }
        }
    }
    
    public int pick() {
        int ni = r.nextInt(ww);
        if (map.containsKey(ni)) return map.get(ni);
        else return ni;
    }
}
```
TODO 白名单线段计数列表->前缀和+二分

## [522. 最长特殊序列 II](https://leetcode.cn/problems/longest-uncommon-subsequence-ii/)

> 数组，哈希表，双指针，字符串，排序，LCS

```java
class Solution {
    public int findLUSlength(String[] strs) {
        int n = strs.length, maxLen = -1;
        for (int i = 0; i < n; ++i) { // 枚举候选独有子序列
            if (strs[i].length() < maxLen) continue;
            boolean canUpdate = true;
            for (int j = 0; j < n; ++j) {
                if (i != j && strs[i].length() <= strs[j].length() && isSub(strs[i], strs[j])) { // 是其他某个字符串的子序列，不符合题意
                    canUpdate = false;
                }
            }
            if (canUpdate) {
                maxLen = strs[i].length();
            }
        }
        return maxLen;
    }

    private boolean isSub(String s0, String s1) {
        int p0 = 0, p1 = 0;
        while (p0 < s0.length() && p1 < s1.length()) {
            if (s0.charAt(p0) != s1.charAt(p1)) p1++;
            else {
                p0++;
                p1++;
            }
        }
        return p0 >= s0.length();
    }
}
```

## [535. TinyURL 的加密与解密](https://leetcode.cn/problems/encode-and-decode-tinyurl/submissions/)

> 设计，哈希表，字符串，哈希函数

```java
public class Codec {
    Map<Integer, String> map = new HashMap<>();
    Random r = new Random();

    // Encodes a URL to a shortened URL.
    public String encode(String longUrl) {
        int key = 0;
        while (map.containsKey(key)) {
            key = r.nextInt(100000);
        }
        map.put(key, longUrl);
        return "http://tinyurl.com/" + key;
    }

    // Decodes a shortened URL to its original URL.
    public String decode(String shortUrl) {
        int idx = shortUrl.lastIndexOf('/') + 1;
        int key = Integer.parseInt(shortUrl.substring(idx));
        return map.getOrDefault(key, null);
    }
}
```

## [1175. 质数排列](https://leetcode.cn/problems/prime-arrangements/)

> 数学，欧拉筛，排列组合

```java
class Solution {
    private static int MOD = (int)1e9 + 7;

    public int numPrimeArrangements(int n) {
        // A合数XA质数
        // 欧拉筛
        boolean[] check = new boolean[n + 1];
        for (int i = 2; i <= n; ++i) {
            if (!check[i]) {
                for (int j = 2 * i; j <= n; j += i) {
                    check[j] = true;
                }
            }
        }
        int zCnt = 0;
        for (int i = 2; i <= n; ++i) {
            if (!check[i]) zCnt++;
        }
        long ret = aa(zCnt) * aa(n - zCnt) % MOD;
        return (int)ret;
    }

    private long aa(int cnt) {
        long ret = 1L;
        for (int i = cnt; i >= 2; --i) {
            ret *= i;
            ret %= MOD;
        }
        return ret;
    }
}
```

## [241. 为运算表达式设计优先级](https://leetcode.cn/problems/different-ways-to-add-parentheses/)

> DFS，记忆化搜索，递归，字符串，数学，动态规划，表达式

```java
class Solution {
    private char[] cs;

    public List<Integer> diffWaysToCompute(String expression) {
        cs = expression.toCharArray();
        return dfs(0, cs.length - 1);
    }

    private List<Integer> dfs(int l, int r) {
        List<Integer> ret = new ArrayList<>();
        for (int i = l; i <= r; ++i) {
            if (cs[i] >= '0' && cs[i] <= '9') continue;
            List<Integer> l1 = dfs(l, i - 1), l2 = dfs(i + 1, r);
            for (int a: l1) {
                for (int b: l2) {
                    switch(cs[i]) {
                        case '+':
                            ret.add(a + b);
                            break;
                        case '-':
                            ret.add(a - b);
                            break;
                        default:
                            ret.add(a * b);
                    }
                }
            }
        }
        if (ret.isEmpty()) {
            // 只有数字
            int number = 0;
            for (int j = l; j <= r; ++j) number = number * 10 + (cs[j] - '0');
            ret.add(number);
        }
        return ret;
    }
}
```
TODO 记忆化搜索，动态规划

## [556. 下一个更大元素 III](https://leetcode.cn/problems/next-greater-element-iii/)

> 数学，双指针，字符串，模拟

执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户
内存消耗：38.2 MB, 在所有 Java 提交中击败了61.82%的用户
```java
class Solution {
    public int nextGreaterElement(int n) {
        // 从后往前找到s[i]<s[j]的第一个，然后从后往前找到第一个比s[i]大的，调换i和j，然后s[i+1,n-1]逆序
        char[] nums = Integer.toString(n).toCharArray();
        int len = nums.length;
        int i = len - 1;
        while (i >= 1) {
            if (nums[i - 1] < nums[i]) break;
            i--;
        }
        if (i == 0) return -1;
        char x = nums[i - 1];
        int j = len - 1;
        while (nums[j] <= x) {
            j--;
        }
        // swap
        nums[i - 1] = nums[j];
        nums[j] = x;
        // reverse
        for (int l = i, r = len - 1; l < r; ++l, --r) {
            char t = nums[l];
            nums[l] = nums[r];
            nums[r] = t;
        }
        String num = new String(nums);
        long nn = Long.parseLong(num);
        if (nn > Integer.MAX_VALUE) return -1;
        else return (int)nn;
    }
}
```
TODO 空间优化

## [1200. 最小绝对差](https://leetcode.cn/problems/minimum-absolute-difference/submissions/)

> 数组，排序

```java
class Solution {
    public List<List<Integer>> minimumAbsDifference(int[] arr) {
        Arrays.sort(arr);
        int min = Integer.MAX_VALUE;
        for (int i = 1; i < arr.length; ++i) {
            int delta = arr[i] - arr[i - 1];
            if (delta < min) min = delta;
        }
        List<List<Integer>> ret = new ArrayList<>();
        for (int i = 1; i < arr.length; ++i) {
            int delta = arr[i] - arr[i - 1];
            if (delta == min) {
                List<Integer> tmp = new ArrayList<>();
                tmp.add(arr[i - 1]);
                tmp.add(arr[i]);
                ret.add(tmp);
            }
        }
        return ret;
    }
}
```

## [736. Lisp 语法解析](https://leetcode.cn/problems/parse-lisp-expression/)

> 栈，递归，哈希表，字符串

```java
class Solution {
    private Map<String, Deque<Integer>> map = new HashMap<>();

    public int evaluate(String expression) {
        if (expression.startsWith("(")) {
            // 表达式
            if (expression.startsWith("(let")) {
                int lastSpaceIdx = getLastSpaceIdx(expression);
                String params = expression.substring(5, lastSpaceIdx);
                int startIdx = 0;
                Set<String> keys = new HashSet<>();
                while (startIdx < params.length()) {
                    int idx1 = getFirstSpaceIdx(params, startIdx);
                    String key = params.substring(startIdx, idx1);
                    keys.add(key);
                    int idx2 = getFirstSpaceIdx(params, idx1 + 1);
                    int value = evaluate(params.substring(idx1 + 1, idx2));
                    if (map.containsKey(key)) {
                        map.get(key).push(value);
                    } else {
                        Deque<Integer> dq = new ArrayDeque<>();
                        dq.push(value);
                        map.put(key, dq);
                    }
                    startIdx = idx2 + 1;
                }
                int ret = evaluate(expression.substring(lastSpaceIdx + 1, expression.length() - 1));
                for (String usedKey: keys) {
                    map.get(usedKey).pop();
                }
                return ret;
            } else if (expression.startsWith("(add")) {
                int idx = getFirstSpaceIdx(expression, 5);
                return evaluate(expression.substring(5, idx)) 
                + evaluate(expression.substring(idx + 1, expression.length() - 1));
            } else {
                int idx = getFirstSpaceIdx(expression, 6);
                return evaluate(expression.substring(6, idx)) 
                * evaluate(expression.substring(idx + 1, expression.length() - 1));
            }
        } else if (expression.charAt(0) >= '0' && expression.charAt(0) <= '9' || expression.charAt(0) == '-') {
            // 数字
            return Integer.parseInt(expression);
        } else {
            // 变量
            return map.get(expression).peek();
        }
    }

    private int getFirstSpaceIdx(String str, int start) {
        int cnt = 0;
        int i = start;
        for (; i < str.length(); ++i) {
            char c = str.charAt(i);
            if (c == ' ' && cnt == 0) return i;
            else if (c == '(') cnt++;
            else if (c == ')') cnt--;
        }
        return i;
    }

    private int getLastSpaceIdx(String str) {
        int cnt = 0;
        for (int i = str.length() - 2; i >= 0; --i) {
            char c = str.charAt(i);
            if (c == ' ' && cnt == 0) return i;
            else if (c == '(') cnt--;
            else if (c == ')') cnt++;
        }
        return 0;
    }
}
```

## [648. 单词替换](https://leetcode.cn/problems/replace-words/)

> 字典树，数组，哈希，字符串

暴力
```java
class Solution {
    public String replaceWords(List<String> dictionary, String sentence) {
        Collections.sort(dictionary, (x, y) -> {
            return x.length() - y.length();
        });
        String[] sens = sentence.split(" ");
        for (int i = 0; i < sens.length; ++i) {
            for (int j = 0; j < dictionary.size(); ++j) {
                if (sens[i].startsWith(dictionary.get(j))) {
                    sens[i] = dictionary.get(j);
                }
            }
        }
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < sens.length; ++i) {
            sb.append(sens[i]);
            if (i < sens.length - 1) sb.append(" ");
        }
        return sb.toString();
    }
}
```

## [1217. 玩筹码](https://leetcode.cn/problems/minimum-cost-to-move-chips-to-the-same-position/)

> 贪心，数组，数学

执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户
内存消耗：39.4 MB, 在所有 Java 提交中击败了16.34%的用户
```java
class Solution {
    public int minCostToMoveChips(int[] position) {
        // 移动到同奇偶的位置上的时候无损，所以计算奇数位上数量和偶数位上数量的最小值即可
        int sum1 = 0, sum2 = 0;
        for (int item: position) {
            if ((item & 1) == 0) sum1++;
            else sum2++;
        }
        return Math.min(sum1, sum2);
    }
}
```

## [873. 最长的斐波那契子序列的长度](https://leetcode.cn/problems/length-of-longest-fibonacci-subsequence/)

> 数组，哈希表，动态规划

暴力
```java
class Solution {
    public int lenLongestFibSubseq(int[] arr) {
        Set<Integer> set = new HashSet<>();
        for (int i: arr) set.add(i);
        int n = arr.length, max = 0;
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                int next = arr[i] + arr[j];
                if (set.contains(next)) {
                    int pre = arr[j];
                    int cnt = 3;
                    while (true) {
                        int nnext = next + pre;
                        if (!set.contains(nnext)) break;
                        pre = next;
                        next = nnext;
                        cnt++;
                    }
                    max = Math.max(max, cnt);
                }
            }
        }
        return max;
    }
}
```
dp
```java
class Solution {
    public int lenLongestFibSubseq(int[] arr) {
        // 利用序列最后两个数来构建dp
        // dp[j][i]表示以arr[i],arr[j]作为最后两个子序列的最大长度
        // dp[j][i]=max(dp[k][j]+1,3)存在k使得arr[k]=arr[i]-arr[j]；0，不存在k
        // dp[j][i]=0
        // max(dp[j][i])
        int ans = 0, n = arr.length;
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < n; ++i) {
            map.put(arr[i], i);
        }
        int[][] dp = new int[n][n];
        for (int i = 0; i < n; ++i) {
            for (int j = i - 1; j >= 0 && arr[j] * 2 > arr[i]; j--) {
                int k = map.getOrDefault(arr[i] - arr[j], -1);
                if (k >= 0) {
                    // 存在
                    dp[j][i] = Math.max(dp[k][j] + 1, 3);
                } else {
                    // 不存在
                    dp[j][i] = 0;
                }
                ans = Math.max(ans, dp[j][i]);
            }
        }
        return ans;
    }
}
```

## [1252. 奇数值单元格的数目](https://leetcode.cn/problems/cells-with-odd-values-in-a-matrix/)

> 数组，数学，模拟

```java
class Solution {
    public int oddCells(int m, int n, int[][] indices) {
        Map<Integer, Integer> rmap = new HashMap<>();
        Map<Integer, Integer> cmap = new HashMap<>();
        for (int[] indice: indices) {
            rmap.put(indice[0], rmap.getOrDefault(indice[0], 0) + 1);
            cmap.put(indice[1], cmap.getOrDefault(indice[1], 0) + 1);
        }
        int cnt = 0;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                cnt += ((rmap.getOrDefault(i, 0) + cmap.getOrDefault(j, 0)) & 1) == 1 ? 1 : 0;
            }
        }
        return cnt;
    }
}
```
优化
```java
class Solution {
    public int oddCells(int m, int n, int[][] indices) {
        int[] row = new int[m], col = new int[n];
        for (int[] indice: indices) {
            row[indice[0]]++;
            col[indice[1]]++;
        }
        // res = oddx * (n - oddy) + oddy * (m - oddx)
        // 奇=奇+偶
        int oddx = 0, oddy = 0;
        for (int i = 0; i < m; ++i) {
            oddx += (row[i] & 1) == 1 ? 1 : 0;
        }
        for (int j = 0; j < n; ++j) {
            oddy += (col[j] & 1) == 1 ? 1 : 0;
        }
        return oddx * (n - oddy) + oddy * (m - oddx);
    }
}
```

## [735. 行星碰撞](https://leetcode.cn/problems/asteroid-collision/)

> 栈，数组

```java
class Solution {
    public int[] asteroidCollision(int[] asteroids) {
        Deque<Integer> dq = new ArrayDeque<>();
        for (int ast: asteroids) {
            boolean isAlive = true;
            while (isAlive && ast < 0 && !dq.isEmpty() && dq.peek() > 0) { // 正数直接入栈，负数则判断栈顶各自是否爆炸
                isAlive = dq.peek() < -ast; // 当前元素是否存活
                if (dq.peek() <= -ast) { // 栈顶元素是否存活
                    dq.pop();
                }
            }
            if (isAlive) dq.push(ast);
        }
        int size = dq.size();
        int[] ret = new int[size];
        for (int i = size - 1; i >= 0; --i) {
            ret[i] = dq.pop();
        }
        return ret;
    }
}
```

## [剑指 Offer II 041. 滑动窗口的平均值](https://leetcode.cn/problems/qIsx9U/)

> 设计，队列，数组，数据流，滑动窗口

```java
class MovingAverage {
    private Deque<Integer> q;
    private double sum = 0d;
    private int size;

    /** Initialize your data structure here. */
    public MovingAverage(int size) {
        q = new ArrayDeque<>();
        this.size = size;
    }
    
    public double next(int val) {
        sum += val;
        if (q.size() == this.size) {
            sum -= q.pollFirst();
        }
        q.offerLast(val);
        return sum / q.size();
    }
}
```

## [565. 数组嵌套](https://leetcode.cn/problems/array-nesting/)

> 深度优先搜索，数组，图

```java
class Solution {
    public int arrayNesting(int[] nums) {
        int n = nums.length, ans = 0;
        for (int i = 0; i < n; ++i) {
            int len = 0;
            while (nums[i] < n) {
                int num = nums[i];
                nums[i] = n;
                i = num;
                len++;
            }
            ans = Math.max(ans, len);
        }
        return ans;
    }
}
```

## [1260. 二维网格迁移](https://leetcode.cn/problems/shift-2d-grid/)

> 数组，矩阵，模拟

```java
class Solution {
    public List<List<Integer>> shiftGrid(int[][] grid, int k) {
        int m = grid.length, n = grid[0].length;
        int total = m * n;
        k %= total;
        List<List<Integer>> ret = new ArrayList<>();
        List<Integer> cur = new ArrayList<>();;
        int p = total - k;
        if (p >= total) p = 0;
        for (int i = 0; i < total; ++i) {
            int x = p / n, y = p % n;
            if (i == 0) {
                cur.add(grid[x][y]);
                p++;
                if (p == total) p = 0;
                continue;
            }
            if (i % n == 0) {
                ret.add(cur);
                cur = new ArrayList<>();
            }
            cur.add(grid[x][y]);
            p++;
            if (p == total) p = 0;
        }
        ret.add(cur);
        return ret;
    }
}
```

## [1184. 公交站间的距离](https://leetcode.cn/problems/distance-between-bus-stops/)

> 数组

```java
class Solution {
    public int distanceBetweenBusStops(int[] distance, int start, int destination) {
        int total = 0;
        for (int i: distance) {
            total += i;
        }
        int min = Math.min(start, destination), max = Math.max(start, destination);
        int sum = 0;
        for (int i = min; i < max; ++i) {
            sum += distance[i];
        }
        return Math.min(sum, total - sum);
    }
}
```

## [919. 完全二叉树插入器](https://leetcode.cn/problems/complete-binary-tree-inserter/)

> 树，广度优先搜索，设计，二叉树

```java
class CBTInserter {
    TreeNode root;
    Deque<TreeNode> dq;
    TreeNode parent;

    public CBTInserter(TreeNode root) {
        this.root = root;
        dq = new ArrayDeque<>();
    }
    
    public int insert(int val) {
        if (parent != null && parent.right == null) {
            parent.right = new TreeNode(val);
            return parent.val;
        }
        parent = root;
        dq.push(root);
        while (!dq.isEmpty()) {
            int size = dq.size();
            while (size-- > 0) {
                parent = dq.poll();
                if (parent.left == null) {
                    parent.left = new TreeNode(val);
                    return parent.val;
                } else if (parent.right == null) {
                    parent.right = new TreeNode(val);
                    return parent.val;
                }
                dq.offer(parent.left);
                dq.offer(parent.right);
            }
        }
        return parent.val;
    }
    
    public TreeNode get_root() {
        return root;
    }
}
```
用另一个队列记录倒数两排没排满的节点
```java
class CBTInserter {
    TreeNode root;
    Deque<TreeNode> dq;

    public CBTInserter(TreeNode root) {
        this.root = root;
        dq = new ArrayDeque<>();
        Queue<TreeNode> queue = new ArrayDeque<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            TreeNode node = queue.poll();
            if (node.left != null) {
                queue.offer(node.left);
            }
            if (node.right != null) {
                queue.offer(node.right);
            }
            if (!(node.left != null && node.right != null)) {
                dq.offer(node);
            }
        }
    }
    
    public int insert(int val) {
        TreeNode node = dq.peek();
        if (node.left == null) {
            node.left = new TreeNode(val);
            dq.offer(node.left);
            return node.val;
        }
        if (node.right == null) {
            node.right = new TreeNode(val);
            dq.offer(node.right);
            dq.poll();
            return node.val;
        }
        return node.val;
    }
    
    public TreeNode get_root() {
        return root;
    }
}
```
TODO 二进制

## [592. 分数加减运算](https://leetcode.cn/problems/fraction-addition-and-subtraction/)

> 数学，字符串，模拟

```java
class Solution {
    public String fractionAddition(String expression) {
        if (!expression.startsWith("-")) expression = '+' + expression;
        int n = expression.length();
        int i = 0;
        long up = 0, down = 1L;
        while (i < n) {
            boolean isNeg = expression.charAt(i) == '-';
            int cup = 0;
            i++;
            while (expression.charAt(i) != '/') {
                cup = cup * 10 + (expression.charAt(i) - '0');
                i++;
            }
            i++;
            int cdown = 0;
            while (i < n && expression.charAt(i) != '-' && expression.charAt(i) != '+') {
                cdown = cdown * 10 + (expression.charAt(i) - '0');
                i++;
            }
            up = isNeg ? cdown * up - cup * down : cdown * up + cup * down;
            down *= cdown;
        }
        if (up == 0) return "0/1";
        long g = gcd(Math.abs(up), down);
        return Long.toString(up / g) + "/" + Long.toString(down / g);
    }

    private long gcd(long a, long b) {
        return b > 0 ? gcd(b, a % b) : a;
    }
}
```

## [1331. 数组序号转换](https://leetcode.cn/problems/rank-transform-of-an-array/)

> 数组，哈希表，排序

```java
class Solution {
    public int[] arrayRankTransform(int[] arr) {
        Map<Integer, List<Integer>> map = new HashMap<>();
        int n = arr.length;
        for (int i = 0; i < n; ++i) {
            if (map.containsKey(arr[i])) {
                map.get(arr[i]).add(i);
            } else {
                List<Integer> list = new ArrayList<>();
                list.add(i);
                map.put(arr[i], list);
            }
        }
        int[] ret = new int[n];
        List<Integer> list = new ArrayList<>(map.keySet());
        Collections.sort(list);
        int cnt = 1;
        for (int i = 0; i < list.size(); ++i) {
            for (int k: map.get(list.get(i))) {
                ret[k] = cnt;
            }
            cnt++;
        }
        return ret;
    }
}
```
简化
```java
class Solution {
    public int[] arrayRankTransform(int[] arr) {
        int[] clone = arr.clone();
        Arrays.sort(clone);
        int n = arr.length;
        Map<Integer, Integer> map = new HashMap<>();
        int idx = 1;
        for (int item: clone) {
            if (!map.containsKey(item)) map.put(item, idx++);
        }
        int[] ret = new int[n];
        for (int i = 0; i < n; ++i) ret[i] = map.get(arr[i]);
        return ret;
    }
}
```

## [593. 有效的正方形](https://leetcode.cn/problems/valid-square/)

> 几何，数学

```java
class Solution {
    public boolean validSquare(int[] p1, int[] p2, int[] p3, int[] p4) {
        int[] arr = new int[]{
            distance(p1, p2),
            distance(p2, p3),
            distance(p3, p4),
            distance(p4, p1),
            distance(p1, p3),
            distance(p2, p4)
        };
        Arrays.sort(arr);
        return arr[0] != 0 && arr[0] == arr[3] && arr[4] == arr[5] && arr[0] * 2 == arr[4];
    }

    private int distance(int[] a, int[] b) {
        return (b[0] - a[0]) * (b[0] - a[0]) + (b[1] - a[1]) * (b[1] - a[1]);
    }
}
```

## [1161. 最大层内元素和](https://leetcode.cn/problems/maximum-level-sum-of-a-binary-tree/)

> 树，深度优先搜索，广度优先搜索，二叉树

```java
class Solution {
    public int maxLevelSum(TreeNode root) {
        if (root == null) return 0;
        int level = 1, max = root.val, maxLevel = 1;
        Deque<TreeNode> dq = new ArrayDeque<>();
        dq.offer(root);
        while (!dq.isEmpty()) {
            int size = dq.size();
            int curSum = 0;
            boolean hasLevel = false;
            while (size-- > 0) {
                TreeNode tn = dq.poll();
                if (tn.left != null) {
                    hasLevel = true;
                    curSum += tn.left.val;
                    dq.offer(tn.left);
                }
                if (tn.right != null) {
                    hasLevel = true;
                    curSum += tn.right.val;
                    dq.offer(tn.right);
                }
            }
            level++;
            if (hasLevel && curSum > max) {
                max = curSum;
                maxLevel = level;
            }
        }
        return maxLevel;
    }
}
```

## [1374. 生成每种字符都是奇数个的字符串](https://leetcode.cn/problems/generate-a-string-with-characters-that-have-odd-counts/)

> 字符串

```java
class Solution {
    public String generateTheString(int n) {
        StringBuilder sb = new StringBuilder();
        if ((n & 1) == 0) {
            // 偶数
            sb.append("b");
            n--;
        }
        while (n-- > 0) {
            sb.append("a");
        }
        return sb.toString();
    }
}
```
简化写法
```java
class Solution {
    public String generateTheString(int n) {
        // 或者一句话：
        // return (n & 1) == 1 ? "a".repeat(n) : "a".repeat(n - 1) + "b";
        StringBuilder sb = new StringBuilder();
        if ((n & 1) == 1) {
            return sb.append("a".repeat(n)).toString();
        }
        return sb.append("a".repeat(n - 1)).append("b").toString();
    }
}
```

## [1403. 非递增顺序的最小子序列](https://leetcode.cn/problems/minimum-subsequence-in-non-increasing-order/)

> 贪心，数组，排序

```java
class Solution {
    public List<Integer> minSubsequence(int[] nums) {
        Arrays.sort(nums);
        int sum = 0, n = nums.length;
        for (int i = 0; i < n; ++i) {
            sum += nums[i];
        }
        List<Integer> ret = new ArrayList<>();
        int s = 0, half = sum / 2;
        for (int i = n - 1; i >= 0; i--) {
            s += nums[i];
            ret.add(nums[i]);
            if (s > half) {
                break;
            }
        }
        return ret;
    }
}
```

## [1408. 数组中的字符串匹配](https://leetcode.cn/problems/string-matching-in-an-array/)

> 字符串，字符串匹配，KMP

```java
class Solution {
    public List<String> stringMatching(String[] words) {
        Set<String> ret = new HashSet<>();
        int n = words.length;
        for (int i = 0; i < n - 1; ++i) {
            for (int j = i + 1; j < n; ++j) {
                if (!words[i].equals(words[j])) {
                    if (words[j].indexOf(words[i]) != -1) {
                        ret.add(words[i]);
                        break;
                    }
                    if (words[i].indexOf(words[j]) != -1) {
                        ret.add(words[j]);
                    }
                }
            }
        }
        return new ArrayList<>(ret);
    }
}
```
TODO KMP

### [640. 求解方程](https://leetcode.cn/problems/solve-the-equation/)

> 数学，字符串，模拟

```java
class Solution {
    public String solveEquation(String equation) {
        boolean isLeft = true;
        int xs = 0, v = 0, n = equation.length(), i = 0, cn = 0;
        while (i < n) {
            char c = equation.charAt(i);
            if (c == '=') {
                if (cn != 0) {
                    v -= cn;
                    cn = 0;
                }
                isLeft = false;
                i++;
                continue;
            }
            if (c == 'x') {
                if (i == 0) {
                    xs = 1;
                } else {
                    char cb = equation.charAt(i - 1);
                    if (cb == '+' || cb == '=') {
                        xs += isLeft ? 1 : -1;
                    } else if (cb == '-') {
                        xs += isLeft ? -1 : 1;
                    } else {
                        xs = isLeft ? (xs + cn) : (xs - cn);
                        cn = 0;
                    }
                }
                i++;
                continue;
            } else {
                if (cn != 0) {
                    v = isLeft ? (v - cn) : (v + cn);
                    cn = 0;
                }
                boolean isNeg = false;
                if (c == '-') isNeg = true;
                if (c == '-' || c == '+') i++;
                while (i < n && Character.isDigit(equation.charAt(i))) {
                    cn *= 10;
                    cn += equation.charAt(i) - '0';
                    i++;
                }
                if (isNeg) cn = -cn;
                System.out.println("cn,v=" + cn + "," + v);
            }
        }
        if (cn != 0) {
            v += cn;
        }
        if (xs == 0 && v != 0) return "No solution";
        if (xs == 0 && v == 0) return "Infinite solutions";
        return "x=" + v / xs;
    }
}
```
TODO 可以优化下算法复杂度

## [1413. 逐步求和得到正数的最小值](https://leetcode.cn/problems/minimum-value-to-get-positive-step-by-step-sum/)

> 数组，前缀和

```java
class Solution {
    public int minStartValue(int[] nums) {
        int n = nums.length;
        int[] preSum = new int[n];
        preSum[0] = nums[0];
        int min = nums[0];
        for (int i = 1; i < n; ++i) {
            preSum[i] = preSum[i - 1] + nums[i];
            min = Math.min(min, preSum[i]);
        }
        return min >= 0 ? 1 : (-min + 1);
    }
}
```
TODO 不用前缀和也行，还能用二分

## [1417. 重新格式化字符串](https://leetcode.cn/problems/reformat-the-string/)

> 字符串

```java
class Solution {
    public String reformat(String s) {
        StringBuilder dsb = new StringBuilder(), csb = new StringBuilder();
        for (char c: s.toCharArray()) {
            if (Character.isDigit(c)) dsb.append(c);
            else csb.append(c);
        }
        if (Math.abs(dsb.length() - csb.length()) > 1) return "";
        StringBuilder ret = new StringBuilder();
        int i = 0, j = 0;
        while (i < dsb.length() && j < csb.length()) {
            ret.append(dsb.charAt(i)).append(csb.charAt(j));
            i++;
            j++;
        }
        if (i < dsb.length()) {
            ret.append(dsb.charAt(i));
        } else if (j < csb.length()) {
            ret.insert(0, csb.charAt(j));
        }
        return ret.toString();
    }
}
```

## [1422. 分割字符串的最大得分](https://leetcode.cn/problems/maximum-score-after-splitting-a-string/)

> 字符串

```java
class Solution {
    public int maxScore(String s) {
        int n = s.length();
        int[] la = new int[n], ra = new int[n];
        la[0] = s.charAt(0) == '0' ? 1 : 0;
        ra[n - 1] = s.charAt(n - 1) == '1' ? 1 : 0;
        for (int i = 1; i < n - 1; ++i) {
            la[i] = la[i - 1] + (s.charAt(i) == '0' ? 1 : 0);
        }
        for (int i = n - 2; i >= 1; --i) {
            ra[i] = ra[i + 1] + (s.charAt(i) == '1' ? 1 : 0);
        }
        int ret = 0;
        for (int i = 0; i < n - 1; ++i) {
            ret = Math.max(ret, la[i] + ra[i + 1]);
        }
        return ret;
    }
}
```
TODO 从左到右遍历即可

## [641. 设计循环双端队列](https://leetcode.cn/problems/design-circular-deque/)

> 设计，队列，数组，链表，循环队列

```java
class MyCircularDeque {
    private int[] elements;
    private int rear, front;
    private int capacity;

    public MyCircularDeque(int k) {
        capacity = k + 1;
        rear = front = 0;
        elements = new int[capacity];
    }
    
    public boolean insertFront(int value) {
        if (isFull()) return false;
        front = (front - 1 + capacity) % capacity;
        elements[front] = value;
        return true; 
    }
    
    public boolean insertLast(int value) {
        if (isFull()) return false;
        elements[rear] = value;
        rear = (rear + 1) % capacity;
        return true;
    }
    
    public boolean deleteFront() {
        if (isEmpty()) return false;
        front = (front + 1) % capacity;
        return true;
    }
    
    public boolean deleteLast() {
        if (isEmpty()) return false;
        rear = (rear - 1 + capacity) % capacity;
        return true;
    }
    
    public int getFront() {
        if (isEmpty()) return -1;
        return elements[front];
    }
    
    public int getRear() {
        if (isEmpty()) return -1;
        return elements[(rear - 1 + capacity) % capacity];
    }
    
    public boolean isEmpty() {
        return front == rear;
    }
    
    public boolean isFull() {
        return (rear + 1) % capacity == front;
    }
}
```
TODO 链表

## [1656. 设计有序流](https://leetcode.cn/problems/design-an-ordered-stream/)

> 设计，数组，哈希表，数据流

```java
class OrderedStream {
    private String[] arr;
    private int ptr;

    public OrderedStream(int n) {
        ptr = 1;
        arr = new String[n + 1];
    }
    
    public List<String> insert(int idKey, String value) {
        List<String> ret = new ArrayList<>();
        arr[idKey] = value;
        while (ptr < arr.length && arr[ptr] != null) {
            ret.add(arr[ptr++]);
        }
        return ret;
    }
}
```

## [1302. 层数最深叶子节点的和](https://leetcode.cn/problems/deepest-leaves-sum/)

> 树，深度优先搜索，广度优先搜索，二叉树，层次遍历

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
    public int deepestLeavesSum(TreeNode root) {
        if (root == null) return 0;
        Queue<TreeNode> dq = new ArrayDeque<>();
        dq.offer(root);
        int ret = root.val;
        while (!dq.isEmpty()) {
            int size = dq.size();
            boolean hasNode = false;
            int cur = 0;
            while (size-- > 0) {
                TreeNode tn = dq.poll();
                cur += tn.val;
                if (tn.left != null) {
                    hasNode = true;
                    dq.offer(tn.left);
                }
                if (tn.right != null) {
                    hasNode = true;
                    dq.offer(tn.right);
                }
            }
            ret = cur;
            if (!hasNode) break;
        }
        return ret;
    }
}
```

## [1757. 可回收且低脂的产品](https://leetcode.cn/problems/recyclable-and-low-fat-products/)

> 数据库

```SQL
SELECT product_id FROM Products WHERE low_fats = 'Y' AND recyclable = 'Y'
```

## [1455. 检查单词是否为句中其他单词的前缀](https://leetcode.cn/problems/check-if-a-word-occurs-as-a-prefix-of-any-word-in-a-sentence/)

> 字符串，字符串匹配

```java
class Solution {
    public int isPrefixOfWord(String sentence, String searchWord) {
        int wordCnt = 1;
        for (int i = 0; i < sentence.length(); ++i) {
            int id = check(sentence, i, searchWord);
            if (id == i) return wordCnt;
            i = id;
            wordCnt++;
        }
        return -1;
    }

    private int check(String sentence, int id, String searchWord) {
        int i = id;
        for (int j = 0; j < searchWord.length(); ++j) {
            if (sentence.charAt(i) != searchWord.charAt(j)) {
                while (i < sentence.length() && sentence.charAt(i) != ' ') i++;
                return i;
            }
            i++;
        }
        return id;
    }
}
```

## [655. 输出二叉树](https://leetcode.cn/problems/print-binary-tree/)

> 树，深度优先搜索，广度优先搜索，二叉树

```java
class Solution {
    public List<List<String>> printTree(TreeNode root) {
        int h = dfs(root, 0);
        h--;
        List<List<String>> ret = new ArrayList<>();
        if (root == null) return ret;
        int m = h + 1, n = (int)Math.pow(2, h + 1) - 1;
        String[][] mn = new String[m][n];
        Deque<TreeNode> dq = new ArrayDeque<>();
        dq.push(root);
        mn[0][(n - 1) / 2] = root.val + "";
        Map<TreeNode, String> cache = new HashMap<>();
        cache.put(root, 0 + "," + (n - 1) / 2);
        while (!dq.isEmpty()) {
            TreeNode tn = dq.poll();
            String[] pos = cache.get(tn).split(",");
            int x = Integer.parseInt(pos[0]), y = Integer.parseInt(pos[1]);
            if (tn.left != null) {
                dq.offer(tn.left);
                int xn = x + 1, yn = y - (int)Math.pow(2, h - x - 1);
                cache.put(tn.left, xn + "," + yn);
                mn[xn][yn] = tn.left.val + "";
            }
            if (tn.right != null) {
                dq.offer(tn.right);
                int xn = x + 1, yn = y + (int)Math.pow(2, h - x - 1);
                cache.put(tn.right, xn + "," + yn);
                mn[xn][yn] = tn.right.val + "";
            }
        }
        for (int i = 0; i < m; ++i) {
            List<String> rr = new ArrayList<>();
            for (int j = 0; j < n; ++j) {
                if (mn[i][j] == null) rr.add("");
                else rr.add(mn[i][j]);
            }
            ret.add(rr);
        }
        return ret;
    }

    private int dfs(TreeNode root, int h) {
        if (root == null) {
            return h;
        }
        return Math.max(dfs(root.left, h + 1), dfs(root.right, h + 1));
    }
}
```
TODO 都用DFS或BFS

## [1460. 通过翻转子数组使两个数组相等](https://leetcode.cn/problems/make-two-arrays-equal-by-reversing-sub-arrays/)

> 数组，哈希表，排序

```java
class Solution {
    public boolean canBeEqual(int[] target, int[] arr) {
        Arrays.sort(target);
        Arrays.sort(arr);
        return Arrays.equals(target, arr);
    }
}
```

## [1464. 数组中两元素的最大乘积](https://leetcode.cn/problems/maximum-product-of-two-elements-in-an-array/)

> 数组，排序，堆

```java
class Solution {
    public int maxProduct(int[] nums) {
        int a = 0, b = 0;
        for (int i = 0; i < nums.length; ++i) {
            if (a == 0) a = nums[i];
            else if (b == 0) {
                if (nums[i] > a) {
                    b = a;
                    a = nums[i];
                } else {
                    b = nums[i];
                }
            } else if (nums[i] > b) {
                if (nums[i] > a) {
                    b = a;
                    a = nums[i];
                } else {
                    b = nums[i];
                }
            }
        }
        return (a - 1) * (b - 1);
    }
}
```

## [2236. 判断根结点是否等于子结点之和](https://leetcode.cn/problems/root-equals-sum-of-children/)

> 树，二叉树

```java
class Solution {
    public boolean checkTree(TreeNode root) {
        return root.val == root.left.val + root.right.val;
    }
}
```

## [946. 验证栈序列](https://leetcode.cn/problems/validate-stack-sequences/)

> 栈，数组，模拟

```java
class Solution {
    public boolean validateStackSequences(int[] pushed, int[] popped) {
        Deque<Integer> dq = new ArrayDeque<>();
        for (int i = 0, j = 0; i < pushed.length; ++i) {
            dq.push(pushed[i]);
            while (!dq.isEmpty() && popped[j] == dq.peek()) {
                dq.pop();
                j++;
            }
        }
        return dq.isEmpty();
    }
}
```

## [1475. 商品折扣后的最终价格](https://leetcode.cn/problems/final-prices-with-a-special-discount-in-a-shop/)

> 栈，数组，单调栈

遍历
```java
class Solution {
    public int[] finalPrices(int[] prices) {
        int n = prices.length;
        int[] ret = new int[n];
        for (int i = 0; i < n; ++i) {
            int discount = 0;
            for (int j = i + 1; j < n; ++j) {
                if (prices[j] <= prices[i]) {
                    discount = prices[j];
                    break;
                }
            }
            ret[i] = prices[i] - discount;
        }
        return ret;
    }
}
```
单调栈
```java
class Solution {
    public int[] finalPrices(int[] prices) {
        int n = prices.length;
        int[] ret = new int[n];
        Deque<Integer> dq = new ArrayDeque<>();
        for (int i = n - 1; i >= 0; --i) {
            while (!dq.isEmpty() && dq.peek() > prices[i]) dq.pop();
            ret[i] = dq.isEmpty() ? prices[i] : prices[i] - dq.peek();
            dq.push(prices[i]);
        }
        return ret;
    }
}
```

## [687. 最长同值路径](https://leetcode.cn/problems/longest-univalue-path/)

> 树，深度优先搜索，二叉树

```java
class Solution {
    private int ans = 0;

    public int longestUnivaluePath(TreeNode root) {
        dfs(root);
        return ans;
    }

    // 以root作为起点往下的最长路径（仅一边）
    private int dfs(TreeNode root) {
        if (root == null) return 0;
        int left = dfs(root.left), right = dfs(root.right);
        int left1 = 0, right1 = 0;
        if (root.left != null && root.left.val == root.val) {
            left1 = left + 1;
        }
        if (root.right != null && root.right.val == root.val) {
            right1 = right + 1;
        }
        ans = Math.max(ans, left1 + right1);
        return Math.max(left1, right1);
    }

```

## [646. 最长数对链](https://leetcode.cn/problems/maximum-length-of-pair-chain/)

> 贪心，数组，动态规划，排序

```java
class Solution {
    public int findLongestChain(int[][] pairs) {
        Arrays.sort(pairs, (x, y) -> x[0] - y[0]);
        int ans = 1;
        int n = pairs.length;
        int[] dp = new int[n];
        Arrays.fill(dp, 1);
        for (int i = 1; i < n; ++i) {
            for (int j = 0; j < i; ++j) {
                if (pairs[j][1] < pairs[i][0]) {
                    dp[i] = Math.max(dp[i], dp[j] + 1);
                }
            }
        }
        return dp[n - 1];
    }
}
```
TODO dp+二分，贪心

## [1592. 重新排列单词间的空格](https://leetcode.cn/problems/rearrange-spaces-between-words/)

> 字符串

```java
class Solution {
    public String reorderSpaces(String text) {
        String[] strs = text.trim().split("\s+");
        int strCnt = 0;
        for (String str: strs) strCnt += str.length();
        StringBuilder sb = new StringBuilder();
        if (strs.length == 1) {
            sb.append(strs[0]);
            int rest = text.length() - sb.length();
            while (rest-- > 0) sb.append(' ');
            return sb.toString();
        }
        int spaceGap = (text.length() - strCnt) / (strs.length - 1);
        for (int i = 0; i < strs.length - 1; ++i) {
            sb.append(strs[i]);
            int x = spaceGap;
            while (x-- > 0) sb.append(' ');
        }
        sb.append(strs[strs.length - 1]);
        int len = text.length() - sb.length();
        while (len-- > 0) sb.append(" ");
        return sb.toString();
    }
}
```
TODO 双指针

## [1598. 文件夹操作日志搜集器](https://leetcode.cn/problems/crawler-log-folder/)

> 栈，数组，字符串

```java
class Solution {
    public int minOperations(String[] logs) {
        int deep = 0;
        for (String str: logs) {
            if (str.equals("./")) {
                continue;
            } else if (str.equals("../")) {
                deep -= 1;
                if (deep < 0) deep = 0;
            } else {
                deep += 1;
            }
        }
        return deep;
    }
}
```

## [669. 修剪二叉搜索树](https://leetcode.cn/problems/trim-a-binary-search-tree/)

> 树，深度优先搜索，二叉搜索树，二叉树

```java
class Solution {
    public TreeNode trimBST(TreeNode root, int low, int high) {
        if (root == null) return null;
        if (root.val < low) {
            return trimBST(root.right, low, high);
        }
        if (root.val > high) {
            return trimBST(root.left, low, high);
        }
        root.left = trimBST(root.left, low, high);
        root.right = trimBST(root.right, low, high);
        return root;
    }
}
```

## [1608. 特殊数组的特征值](https://leetcode.cn/problems/special-array-with-x-elements-greater-than-or-equal-x/)

> 数组，二分查找，排序，计数排序

```java
class Solution {
    public int specialArray(int[] nums) {
        Arrays.sort(nums);
        if (nums[nums.length - 1] == 0) return -1;
        int ans = 0;
        for (int i = nums.length - 1; i > 0; --i) {
            ans++;
            if (nums[i - 1] < ans && ans <= nums[i]) return ans;
            else if (nums[i] < ans) return -1;
        }
        return ans + 1;
    }
}
```
TODO 二分查找，计数排序

## [2235. 两整数相加](https://leetcode.cn/problems/add-two-integers/)

> 数学

```java
class Solution {
    public int sum(int num1, int num2) {
        return num1 + num2;
    }
}
```

## [1619. 删除某些元素后的数组均值](https://leetcode.cn/problems/mean-of-array-after-removing-some-elements/)

> 数组，排序

```java
class Solution {
    public double trimMean(int[] arr) {
        int n = arr.length;
        int start = (int)(n * 0.05);
        int rest = n - start * 2;
        Arrays.sort(arr);
        int sum = 0;
        int end = n - start;
        for (int i = start; i < end; i++) {
            sum += arr[i];
        }
        return sum / (double)rest;
    }
}
```
TODO 用通分后的19/20

## [1624. 两个相同字符之间的最长子字符串](https://leetcode.cn/problems/largest-substring-between-two-equal-characters/)

> 哈希表，字符串

```java
class Solution {
    public int maxLengthBetweenEqualCharacters(String s) {
        int max = -1;
        int[] ids = new int[26];
        Arrays.fill(ids, -1);
        for (int i = 0; i < s.length(); ++i) {
            int id = s.charAt(i) - 'a';
            if (ids[id] == -1) ids[id] = i;
            else max = Math.max(max, i - ids[id] - 1);
        }
        return max;
    }
}
```

## [1636. 按照频率将数组升序排序](https://leetcode.cn/problems/sort-array-by-increasing-frequency/)

> 数组，哈希表，排序

```java
class Solution {
    public int[] frequencySort(int[] nums) {
        int n = nums.length;
        int[] cnt = new int[201];
        for (int i = 0; i < n; ++i) {
            cnt[nums[i] + 100]++;
        }
        ArrayList[] lists = new ArrayList[101];
        for (int i = 200; i >= 0; --i) {
            if (cnt[i] != 0) {
                if (lists[cnt[i]] == null) {
                    lists[cnt[i]] = new ArrayList<>();
                }
                lists[cnt[i]].add(i - 100);
            }
        }
        int[] ret = new int[n];
        int j = 0;
        for (int i = 0; i < 101; ++i) {
            if (lists[i] != null) {
                List<Integer> list = lists[i];
                for (int item: list) {
                    int size = i;
                    while (size-- > 0) {
                        ret[j++] = item;
                    }
                }
            }
        }
        return ret;
    }
}
```
TODO Collection

## [1640. 能否连接形成数组](https://leetcode.cn/problems/check-array-formation-through-concatenation/)

> 数组，哈希表

```java
class Solution {
    public boolean canFormArray(int[] arr, int[][] pieces) {
        Map<Integer, Integer> map = new HashMap<>();
        int n = arr.length, m = pieces.length;
        for (int i = 0; i < m; ++i) {
            map.put(pieces[i][0], i);
        }
        for (int i = 0; i < n;) {
            if (!map.containsKey(arr[i])) return false;
            int idx = map.get(arr[i]);
            int len = pieces[idx].length;
            for (int j = 0; j < len; ++j) {
                if (arr[i + j] != pieces[idx][j]) return false;
            }
            i += len;
        }
        return true;
    }
}
```

## [1652. 拆炸弹](https://leetcode.cn/problems/defuse-the-bomb/)

> 数组，循环数组，滑动窗口

执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户
内存消耗：41.7 MB, 在所有 Java 提交中击败了27.50%的用户
```java
class Solution {
    public int[] decrypt(int[] code, int k) {
        int n = code.length;
        int[] res = new int[n];
        int[] tmp = new int[2 * n];
        for (int i = 0; i < n; ++i) {
            tmp[i] = code[i];
            tmp[n + i] = code[i];
        }
        int l = 0, r = 0;
        if (k > 0) {
            l = 1;
            r = k;
        } else if (k < 0) {
            l = n + k;
            r = n - 1;
        }
        if (k != 0) {
            for (int j = l; j <= r; ++j) {
                res[0] += tmp[j];
            }
        }
        for (int i = 1; i < n; ++i) {
            if (k != 0) {
                res[i] = res[i - 1] + tmp[++r] - tmp[l++];
            } else {
                res[i] = 0;
            }
        }
        return res;
    }
}
```

## [788. 旋转数字](https://leetcode.cn/problems/rotated-digits/)

> 数学，动态规划

```java
class Solution {
    public int rotatedDigits(int n) {
        // 0|1|8(0...n-1) & 至少 2|5|6|9中的一个
        int ans = 0;
        for (int i = 1; i <= n; ++i) {
            String str = Integer.toString(i);
            boolean flag = true, hasOne = false;
            for (char c: str.toCharArray()) {
                int x = c - '0';
                if (x == 2 || x == 5 || x == 6 || x == 9) {
                    hasOne = true;
                } else if (x != 0 && x != 1 && x != 8) {
                    flag = false;
                    break;
                }
            }
            ans += flag & hasOne ? 1 : 0;
        }
        return ans;
    }
}
```
TODO 数位DP

## [面试题 17.19. 消失的两个数字](https://leetcode.cn/problems/missing-two-lcci/)

> 位运算，数组，哈希

```java
class Solution {
    public int[] missingTwo(int[] nums) {
        int n = nums.length;
        int sum1 = (n + 3) * (n + 2) / 2, sum2 = 0, first = 0;
        for (int i = 0; i < nums.length; ++i) sum2 += nums[i];
        int twoSum = sum1 - sum2;
        int l = twoSum / 2;
        int sum3 = l * (l + 1) / 2, sum4 = 0;
        for (int i = 0; i < nums.length; ++i) {
            if (nums[i] <= l) sum4 += nums[i];
        }
        first = sum3 - sum4;
        return new int[]{first, twoSum - first};
    }
}
```
TODO 位运算

## [面试题 01.02. 判定是否互为字符重排](https://leetcode.cn/problems/check-permutation-lcci/)

> 哈希表，字符串，排序

```java
class Solution {
    public boolean CheckPermutation(String s1, String s2) {
        if (s1.length() != s2.length()) return false;
        int[] cnt = new int[26];
        for (int i = 0; i < s1.length(); ++i) {
            cnt[s1.charAt(i) - 'a']++;
            cnt[s2.charAt(i) - 'a']--;
        }
        for (int i = 0; i < 26; ++i) {
            if (cnt[i] != 0) return false;
        }
        return true;
    }
}
```
TODO 排序

## [面试题 01.09. 字符串轮转](https://leetcode.cn/problems/string-rotation-lcci/)

> 字符串，字符串匹配

```java
class Solution {
    public boolean isFlipedString(String s1, String s2) {
        if (s1.length() != s2.length()) return false;
        return (s1 + s1).contains(s2);
    }
}
```

## [1694. 重新格式化电话号码](https://leetcode.cn/problems/reformat-phone-number/)

> 字符串

```java
class Solution {
    public String reformatNumber(String number) {
        StringBuilder sb = new StringBuilder();
        int cnt = 0;
        for (int i = 0; i < number.length(); ++i) {
            if (Character.isDigit(number.charAt(i))) {
                if (cnt == 3) {
                    cnt = 1;
                    sb.append('-').append(number.charAt(i) - '0');
                } else {
                    cnt++;
                    sb.append(number.charAt(i) - '0');
                }
            }
        }
        int n = sb.length();
        if (sb.charAt(n - 2) == '-') {
            char c = sb.charAt(n - 3);
            sb.setCharAt(n - 3, '-');
            sb.setCharAt(n - 2, c);
        }
        return sb.toString();
    }
}
```

## [1784. 检查二进制字符串字段](https://leetcode.cn/problems/check-if-binary-string-has-at-most-one-segment-of-ones/)

> 字符串

执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户
内存消耗：39.5 MB, 在所有 Java 提交中击败了57.34%的用户
```java
class Solution {
    public boolean checkOnesSegment(String s) {
        return !s.contains("01");
    }
}
```

## [921. 使括号有效的最少添加](https://leetcode.cn/problems/minimum-add-to-make-parentheses-valid/)

> 栈，贪心，字符串

执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户
内存消耗：39.5 MB, 在所有 Java 提交中击败了51.53%的用户
```java
class Solution {
    public int minAddToMakeValid(String s) {
        int cnt = 0, cntL = 0;
        for (char c: s.toCharArray()) {
            if (c == '(') {
                cntL++;
            } else {
                if (cntL > 0) {
                    cntL--;
                } else {
                    cnt++;
                }
            }
        }
        return cnt + cntL;
    }
}
```

## [811. 子域名访问计数](https://leetcode.cn/problems/subdomain-visit-count/)

> 数组，哈希表，字符串，计数

```java
class Solution {
    public List<String> subdomainVisits(String[] cpdomains) {
        Map<String, Integer> counts = new HashMap<>();
        for (String cpdomain: cpdomains) {
            String[] sp = cpdomain.split(" ");
            int count = Integer.parseInt(sp[0]);
            String domain = sp[1];
            counts.put(domain, counts.getOrDefault(domain, 0) + count);
            for (int i = 0; i < domain.length(); ++i) {
                if (domain.charAt(i) == '.') {
                    String subDomain = domain.substring(i + 1);
                    counts.put(subDomain, counts.getOrDefault(subDomain, 0) + count);
                }
            }
        }
        List<String> ret = new ArrayList<>();
        for (String key: counts.keySet()) {
            ret.add(counts.get(key) + " " + key);
        }
        return ret;
    }
}
```

## [1800. 最大升序子数组和](https://leetcode.cn/problems/maximum-ascending-subarray-sum/)

> 数组

```java
class Solution {
    public int maxAscendingSum(int[] nums) {
        int max = nums[0], sum = nums[0];
        for (int i = 1; i < nums.length; ++i) {
            if (nums[i] > nums[i - 1]) {
                sum += nums[i];
                max = Math.max(max, sum);
            } else {
                sum = nums[i];
            }
        }
        return max;
    }
}
```

## [856. 括号的分数](https://leetcode.cn/problems/score-of-parentheses/)

> 栈，字符串

```java
class Solution {
    public int scoreOfParentheses(String s) {
        // 只需要计算()深度2^depth
        int depth = 0, res = 0;
        for (int i = 0; i < s.length(); ++i) {
            depth += (s.charAt(i) == '(' ? 1 : -1);
            if (s.charAt(i) == ')' && s.charAt(i - 1) == '(') res += 1 << depth;
        }
        return res;
    }
}
```
TODO 分治，栈

## [1790. 仅执行一次字符串交换能否使两个字符串相等](https://leetcode.cn/problems/check-if-one-string-swap-can-make-strings-equal/)

> 哈希表，字符串，计数

执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户
内存消耗：39.9 MB, 在所有 Java 提交中击败了30.28%的用户
```java
class Solution {
    public boolean areAlmostEqual(String s1, String s2) {
        if (s1.length() != s2.length()) return false;
        if (s1.equals(s2)) return true;
        int cnt = 0;
        char a = 0, b = 0;
        for (int i = 0; i < s1.length(); ++i) {
            if (s1.charAt(i) != s2.charAt(i)) {
                cnt++;
                if (cnt > 2) return false;
                if (cnt == 1) {
                    a = s1.charAt(i);
                    b = s2.charAt(i);
                } else {
                    if (!(s1.charAt(i) == b && s2.charAt(i) == a))
                        return false;
                }
            }
        }
        return cnt == 2;
    }
}
```

## [769. 最多能完成排序的块](https://leetcode.cn/problems/max-chunks-to-make-sorted/)

> 贪心，数组

```java
class Solution {
    public int maxChunksToSorted(int[] arr) {
        int curMax = arr[0], res = 1;
        for (int i = 1; i < arr.length; ++i) {
            if (arr[i] >= curMax) {
                curMax = arr[i];
                res++;
            }
        }
        return res;
    }
}
```

## [1441. 用栈操作构建数组](https://leetcode.cn/problems/build-an-array-with-stack-operations/)

> 栈，数组，模拟

```java
class Solution {
    public List<String> buildArray(int[] target, int n) {
        int x = 1;
        List<String> ret = new ArrayList<>();
        for (int i: target) {
            if (i == x) {
                ret.add("Push");
                x++;
            } else if (i > x) {
                int delta = i - x;
                for (int j = 0; j < delta; ++j) {
                    ret.add("Push");
                    ret.add("Pop");
                }
                ret.add("Push");
                x = i + 1;
            }
        }
        return ret;
    }
}
```

## [2413. 最小偶倍数](https://leetcode.cn/problems/smallest-even-multiple/)

> 数学，数论

```java
class Solution {
    public int smallestEvenMultiple(int n) {
        return ((n & 1) == 1) ? 2 * n : n;
    }
}
```

## [8. 字符串转换整数 (atoi)](https://leetcode.cn/problems/string-to-integer-atoi/)

> 字符串

```java
class Solution {
    public int myAtoi(String s) {
        int MAX = Integer.MAX_VALUE, MIN = Integer.MIN_VALUE;
        long MAX2 = (long)(Integer.MAX_VALUE) + 1L;
        long res = 0;
        int start = 0;
        s = s.trim();
        if (s.startsWith("-")) {
            start = 1;
        }
        if (s.startsWith("+")) {
            s = s.substring(1, s.length());
        }
        for (int i = start; i < s.length(); ++i) {
            if (s.charAt(i) > '9' || s.charAt(i) < '0') break;
            res *= 10;
            res += (s.charAt(i) - '0');
            if (res > MAX && start == 0) return MAX;
            if (res > MAX2 && start == 1) return MIN;
        }
        res = start == 0 ? res : -res;
        if (res > MAX) return MAX;
        if (res < MIN) return MIN;
        return (int)res;
    }
}
```
TODO 有限状态机

## [1700. 无法吃午餐的学生数量](https://leetcode.cn/problems/number-of-students-unable-to-eat-lunch/)

> 栈，队列，数组，模拟

执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户
内存消耗：39.3 MB, 在所有 Java 提交中击败了64.06%的用户
```java
class Solution {
    public int countStudents(int[] students, int[] sandwiches) {
        int one = 0, zero = 0;
        for (int i: students) {
            if (i == 0) zero++;
            else one++;
        }
        for (int i = 0; i < sandwiches.length; ++i) {
            if (sandwiches[i] == 0) {
                if (zero > 0) zero--;
                else return one;
            } else {
                if (one > 0) one--;
                else return zero;
            }
        }
        return 0;
    }
}
```

## [2351. 第一个出现两次的字母](https://leetcode.cn/problems/first-letter-to-appear-twice/)

> 哈希表，字符串，计数

```java
class Solution {
    public char repeatedCharacter(String s) {
        Set<Character> set = new HashSet<>();
        for (char c: s.toCharArray()) {
            if (set.contains(c)) return c;
            else set.add(c);
        }
        return '0';
    }
}
```

## [2373. 矩阵中的局部最大值](https://leetcode.cn/problems/largest-local-values-in-a-matrix/)

> 数组，矩阵

```java
class Solution {
    public int[][] largestLocal(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        int[][] ret = new int[m - 2][n - 2];
        for (int i = 1; i < m - 1; ++i) {
            for (int j = 1; j < n - 1; ++j) {
                ret[i - 1][j - 1] = min(grid, i, j);
            }
        }
        return ret;
    }

    private int min(int[][] grid, int i, int j) {
        int max = grid[i][j];
        for (int x = i - 1; x <= i + 1; ++x) {
            for (int y = j - 1; y <= j + 1; ++y) {
                if (x == i && y == j) continue;
                max = Math.max(max, grid[x][y]);
            }
        }
        return max;
    }
}
```

## [2315. 统计星号](https://leetcode.cn/problems/count-asterisks/)

> 字符串

执行用时：1 ms, 在所有 Java 提交中击败了100.00%的用户
内存消耗：39.4 MB, 在所有 Java 提交中击败了77.23%的用户
```java
class Solution {
    public int countAsterisks(String s) {
        boolean open = false;
        int ret = 0;
        for (char c: s.toCharArray()) {
            if (c == '|') open = !open;
            else if (c == '*' && !open) ret++;
        }
        return ret;
    }
}
```

## [915. 分割数组](https://leetcode.cn/problems/partition-array-into-disjoint-intervals/)

> 数组

```java
class Solution {
    public int partitionDisjoint(int[] nums) {
        int n = nums.length;
        int[] la = new int[n], ra = new int[n];
        la[0] = nums[0];
        ra[n - 1] = nums[n - 1];
        for (int i = 1; i < n; ++i) {
            la[i] = Math.max(la[i - 1], nums[i]);
        }
        for (int i = n - 2; i >= 0; --i) {
            ra[i] = Math.min(ra[i + 1], nums[i]);
        }
        for (int i = 0; i < n - 1; ++i) {
            if (la[i] <= ra[i + 1]) return i + 1;
        }
        return 0;
    }
}
```
一次遍历
```java
class Solution {
    public int partitionDisjoint(int[] nums) {
        int n = nums.length;
        int maxLeft = nums[0], curMax = nums[0], pos = 0;
        for (int i = 1; i < nums.length; ++i) {
            curMax = Math.max(curMax, nums[i]);
            if (maxLeft > nums[i]) {
                maxLeft = curMax;
                pos = i;
            }
        }
        return pos + 1;
    }
}
```

## [344. 反转字符串](https://leetcode.cn/problems/reverse-string/)

> 双指针，字符串

```java
class Solution {
    public void reverseString(char[] s) {
        char tmp = '0';
        int n = s.length, hn = n / 2;
        for (int i = 0; i < hn; ++i) {
            tmp = s[i];
            s[i] = s[n - i - 1];
            s[n - i - 1] = tmp;
        }
        return;
    }
}
```

## [2367. 算术三元组的数目](https://leetcode.cn/problems/number-of-arithmetic-triplets/)

> 数组，哈希表，双指针，枚举

```java
class Solution {
    public int arithmeticTriplets(int[] nums, int diff) {
        int n = nums.length;
        Set<Integer> set = new HashSet<>();
        for (int i: nums) set.add(i);
        int ret = 0;
        for (int i = 0; i < n; ++i) {
            if (set.contains(nums[i] + diff) && set.contains(nums[i] + 2 * diff)) ret++;
        }
        return ret;
    }
}
```
TODO 三指针

## [2220. 转换数字的最少位翻转次数](https://leetcode.cn/problems/minimum-bit-flips-to-convert-number/)

> 位运算
执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户
内存消耗：39 MB, 在所有 Java 提交中击败了5.10%的用户
```java
class Solution {
    public int minBitFlips(int start, int goal) {
        return Integer.bitCount(start ^ goal);
    }
}
```

## [1678. 设计 Goal 解析器](https://leetcode.cn/problems/goal-parser-interpretation/)

> 字符串

```java
class Solution {
    public String interpret(String command) {
        int n = command.length();
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < n; ++i) {
            if (command.charAt(i) == 'G') sb.append('G');
            else if (command.charAt(i) == '(') {
                if (command.charAt(i + 1) == ')') {
                    sb.append('o');
                    i++;
                } else {
                    sb.append("al");
                    i += 2;
                }
            }
        }
        return sb.toString();
    }
}
```

## [2037. 使每位学生都有座位的最少移动次数](https://leetcode.cn/problems/minimum-number-of-moves-to-seat-everyone/)

> 数组，排序

```java
class Solution {
    public int minMovesToSeat(int[] seats, int[] students) {
        Arrays.sort(seats);
        Arrays.sort(students);
        int sum = 0;
        for (int i = 0; i < seats.length; ++i) {
            sum += Math.abs(seats[i] - students[i]);
        }
        return sum;
    }
}
```

## [1704. 判断字符串的两半是否相似](https://leetcode.cn/problems/determine-if-string-halves-are-alike/)

> 字符串

```java
class Solution {
    public boolean halvesAreAlike(String s) {
        Set<Character> set = new HashSet<>();
        set.add('a');
        set.add('e');
        set.add('i');
        set.add('o');
        set.add('u');
        set.add('A');
        set.add('E');
        set.add('I');
        set.add('O');
        set.add('U');
        int half = s.length() / 2;
        int ret = 0;
        for (int i = 0; i < s.length(); ++i) {
            if (set.contains(s.charAt(i))) {
                if (i < half) {
                    ret++;
                } else {
                    ret--;
                }
            }
        }
        return ret == 0;
    }
}
```
TODO 简写

## [791. 自定义字符串排序](https://leetcode.cn/problems/custom-sort-string/)

> 哈希表，字符串，排序

```java
class Solution {
    public String customSortString(String order, String s) {
        int[] co = new int[26];
        int idx = 1;
        for (char c: order.toCharArray()) {
            co[c - 'a'] = idx;
            idx++;
        }
        char[] ss = s.toCharArray();
        Character[] cs = new Character[ss.length];
        for (int i = 0; i < cs.length; ++i) {
            cs[i] = Character.valueOf(ss[i]);
        }
        Arrays.sort(cs, (a, b) -> {
            return co[a.charValue() - 'a'] - co[b.charValue() - 'a'];
        });
        for (int i = 0; i < cs.length; ++i) {
            ss[i] = cs[i].charValue();
        }
        return new String(ss);
    }
}
```
TODO 计数排序

## [1710. 卡车上的最大单元数](https://leetcode.cn/problems/maximum-units-on-a-truck/)

> 贪心，数组，排序

```java
class Solution {
    public int maximumUnits(int[][] boxTypes, int truckSize) {
        // 贪心，挑大的先上
        Arrays.sort(boxTypes, (x, y) -> {
            return y[1] - x[1];
        });
        int sum = 0;
        for (int i = 0; i < boxTypes.length; i++) {
            if (truckSize > boxTypes[i][0]) {
                truckSize -= boxTypes[i][0];
                sum += boxTypes[i][0] * boxTypes[i][1];
            } else {
                sum += truckSize * boxTypes[i][1];
                break;
            }
        }
        return sum;
    }
}
```

## [775. 全局倒置与局部倒置](https://leetcode.cn/problems/global-and-local-inversions/)

> 数组，数学

```java
class Solution {
    public boolean isIdealPermutation(int[] nums) {
        // 是否有间隔1个以上的还倒置的，有返回false
        int n = nums.length;
        for (int i = 0; i < n; ++i) {
            if (nums[i] > i + 1 || nums[i] < i - 1) return false;
        }
        return true;
    }
}
```

## [2469. 温度转换](https://leetcode.cn/problems/convert-the-temperature/)

> 数学

```java
class Solution {
    public double[] convertTemperature(double celsius) {
        return new double[]{celsius + 273.15d, celsius * 1.8d + 32d};
    }
}
```

## [1742. 盒子中小球的最大数量](https://leetcode.cn/problems/maximum-number-of-balls-in-a-box/)

> 哈希表，数学，计数

```java
class Solution {
    public int countBalls(int lowLimit, int highLimit) {
        Map<Integer, Integer> map = new HashMap<>();
        int max = 0;
        for (int i = lowLimit; i <= highLimit; ++i) {
            int x = 0, y = i;
            while (y > 0) {
                x += y % 10;
                y /= 10;
            }
            map.put(x, map.getOrDefault(x, 0) + 1);
            max = Math.max(max, map.get(x));
        }
        return max;
    }
}
```

## [1752. 检查数组是否经排序和轮转得到](https://leetcode.cn/problems/check-if-array-is-sorted-and-rotated/)

> 数组

```java
class Solution {
    public boolean check(int[] nums) {
        int start = nums[0];
        boolean changed = false;
        for (int i = 1; i < nums.length; ++i) {
            if (!changed && nums[i] < nums[i - 1]) {
                changed = true;
                if (nums[i] > start) return false;
            } else if (changed && (nums[i] < nums[i - 1] || nums[i] > start)) {
                return false;
            }
        }
        return true;
    }
}
```

## [1758. 生成交替二进制字符串的最少操作数](https://leetcode.cn/problems/minimum-changes-to-make-alternating-binary-string/)

> 字符串

```java
class Solution {
    public int minOperations(String s) {
        int sum1 = 0, sum2 = 0;
        for (int i = 0; i < s.length(); ++i) {
            if ((i & 1) == 1) { // 奇数：sum1计1的数量，sum2计0的数量
                sum1 += s.charAt(i) - '0';
                sum2 += '1' - s.charAt(i);
            } else { // 偶数：sum1计0的数量，sum2计1的数量
                sum1 += '1' - s.charAt(i);
                sum2 += s.charAt(i) - '0';
            }
        }
        return Math.min(sum1, sum2);
    }
}
```
优化
```java
class Solution {
    public int minOperations(String s) {
        int sum = 0, n = s.length();
        for (int i = 0; i < n; ++i) {
            sum += s.charAt(i) - '0' ^ (i & 1);
        }
        return Math.min(sum, n - sum);
    }
}
```

## [1779. 找到最近的有相同 X 或 Y 坐标的点](https://leetcode.cn/problems/find-nearest-point-that-has-the-same-x-or-y-coordinate/)

> 数组

```java
class Solution {
    public int nearestValidPoint(int x, int y, int[][] points) {
        int min = Integer.MAX_VALUE, ret = -1;
        for (int i = 0; i < points.length; ++i) {
            if (points[i][0] == x) {
                int delta = Math.abs(points[i][1] - y);
                if (delta == 0) return i;
                if (delta < min) {
                    min = delta;
                    ret = i;
                }
            } else if (points[i][1] == y) {
                int delta = Math.abs(points[i][0] - x);
                if (delta == 0) return i;
                if (delta < min) {
                    min = delta;
                    ret = i;
                }
            }
        }
        return ret;
    }
}
```

## [2485. 找出中枢整数](https://leetcode.cn/problems/find-the-pivot-integer/)

> 数学

执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户
内存消耗：38.9 MB, 在所有 Java 提交中击败了21.66%的用户
```java
class Solution {
    public int pivotInteger(int n) {
        // (1+x)x=(x+n)(n-x+1)=>x^2=-x^2+n^2+n=>x^2=√(n^2+n)/2
        int a = (n * n + n) / 2;
        int b = (int)Math.sqrt(a);
        return b * b == a ? b : -1;
    }
}
```

## [1796. 字符串中第二大的数字](https://leetcode.cn/problems/second-largest-digit-in-a-string/)

> 哈希表，字符串

```java
class Solution {
    public int secondHighest(String s) {
        int max1 = -1, max2 = -1;
        for (char c: s.toCharArray()) {
            if (Character.isDigit(c)) {
                int x = c - '0';
                if (x > max1) {
                    max2 = max1;
                    max1 = x;
                } else if (x < max1 && x > max2) {
                    max2 = x;
                }
            }
        }
        return max2;
    }
}
```

## [1805. 字符串中不同整数的数目](https://leetcode.cn/problems/number-of-different-integers-in-a-string/)

> 哈希表，字符串

```java
class Solution {
    public int numDifferentIntegers(String word) {
        Set<String> set = new HashSet<>();
        StringBuilder sb = new StringBuilder();
        boolean digStart = true;
        char[] cs = word.toCharArray();
        for (int i = 0; i < cs.length; ++i) {
            if (!Character.isDigit(cs[i])) {
                digStart = true;
                if (sb.length() > 0) set.add(sb.toString());
                sb = new StringBuilder();
            } else {
                if (digStart && cs[i] == '0') {
                    if (i + 1 < cs.length && !Character.isDigit(cs[i + 1]) || i + 1 == cs.length) {
                        set.add("0");
                        digStart = false;
                    }
                    continue;
                }
                digStart = false;
                sb.append(cs[i]);
            }
        }
        if (sb.length() > 0) set.add(sb.toString());
        return set.size();
    }
}
```
TODO 双指针

## [1812. 判断国际象棋棋盘中一个格子的颜色](https://leetcode.cn/problems/determine-color-of-a-chessboard-square/)

> 数学，字符串
执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户
内存消耗：39.6 MB, 在所有 Java 提交中击败了5.21%的用户
```java
class Solution {
    public boolean squareIsWhite(String coordinates) {
        return (((coordinates.charAt(0) - 'a') & 1) == 1) ^ (((coordinates.charAt(1) - '0') & 1) == 0);
    }
}
```

## [1780. 判断一个数字是否可以表示成三的幂的和](https://leetcode.cn/problems/check-if-number-is-a-sum-of-powers-of-three/)

> 数学

```java
class Solution {
    public boolean checkPowersOfThree(int n) {
        while (n > 0) {
            if ((n % 3) == 2) return false;
            n /= 3; 
        }
        return true;
    }
}
```

## [1827. 最少操作使数组递增](https://leetcode.cn/problems/minimum-operations-to-make-the-array-increasing/)

> 贪心，数组

```java
class Solution {
    public int minOperations(int[] nums) {
        int next = nums[0] + 1, res = 0;
        for (int i = 1; i < nums.length; ++i) {
            if (next >= nums[i]) {
                res += next - nums[i];
                next++;
            } else {
                next = nums[i] + 1;
            }
        }
        return res;
    }
}
```

## [1945. 字符串转化后的各位数字之和](https://leetcode.cn/problems/sum-of-digits-of-string-after-convert/)

> 字符串，模拟

```java
class Solution {
    public int getLucky(String s, int k) {
        StringBuilder sb = new StringBuilder();
        for (char c: s.toCharArray()) {
            sb.append(c - 'a' + 1);
        }
        String tmp = sb.toString();
        int res = 0;
        while (k-- > 0) {
            res = 0;
            for (char c: tmp.toCharArray()) {
                res += c - '0';
            }
            tmp = String.valueOf(res);
        }
        return res;
    }
}
```

## [1753. 移除石子的最大得分](https://leetcode.cn/problems/maximum-score-from-removing-stones/)

> 贪心，数学，堆

```java
class Solution {
    public int maximumScore(int a, int b, int c) {
        int[] arr = new int[]{a, b, c};
        Arrays.sort(arr);
        int delta = arr[2] - arr[1];
        if (delta >= arr[0]) return arr[0] + arr[1];
        else {
            return arr[0] + arr[1] - (arr[0] - delta + 1) / 2;
        }
    }
}
```
简化
```java
class Solution {
    public int maximumScore(int a, int b, int c) {
        int max = Math.max(a, Math.max(b, c));
        int sum = a + b + c;
        return Math.min(sum - max, sum / 2);
    }
}
```

## [2500. 删除每行中的最大值](https://leetcode.cn/problems/delete-greatest-value-in-each-row/)

> 数组，矩阵，排序

```java
class Solution {
    public int deleteGreatestValue(int[][] grid) {
        for (int[] arr: grid) {
            Arrays.sort(arr);
        }
        int ret = 0;
        for (int i = 0; i < grid[0].length; ++i) {
            int max = 0;
            for (int j = 0; j < grid.length; ++j) {
                max = Math.max(max, grid[j][i]);
            }
            ret += max;
        }
        return ret;
    }
}
```

## [2027. 转换字符串的最少操作次数](https://leetcode.cn/problems/minimum-moves-to-convert-string/)

> 贪心，字符串

```java
class Solution {
    public int minimumMoves(String s) {
        int ret = 0;
        for (int i = 0; i < s.length(); ++i) {
            if (s.charAt(i) == 'X') {
                ret++;
                i += 2;
            }
        }
        return ret;     
    }
}
```

## [2278. 字母在字符串中的百分比](https://leetcode.cn/problems/percentage-of-letter-in-string/)

> 字符串

```java
class Solution {
    public int percentageLetter(String s, char letter) {
        int cnt = 0;
        for (char c: s.toCharArray()) {
            if (c == letter) {
                cnt++;
            }
        }
        return (int)cnt * 100 / s.length();
    }
}
```

## [2103. 环和杆](https://leetcode.cn/problems/rings-and-rods/)

> 哈希表，字符串

执行用时：0 ms, 在所有 Java 提交中击败了100.00%的用户
内存消耗：39.3 MB, 在所有 Java 提交中击败了70.28%的用户
```java
class Solution {
    public int countPoints(String rings) {
        boolean[][] arr = new boolean[10][4];
        int cnt = 0;
        for (int i = 0; i < rings.length() - 1; i += 2) {
            int idx = rings.charAt(i + 1) - '0';
            if (arr[idx][3]) continue; 
            int colorOffset = 0;
            if (rings.charAt(i) == 'G') {
                colorOffset = 1;
            } else if (rings.charAt(i) == 'B') {
                colorOffset = 2;
            }
            if (!arr[idx][colorOffset]) arr[idx][colorOffset] = true;
            if (arr[idx][0] && arr[idx][1] && arr[idx][2]) {
                arr[idx][3] = true;
                cnt++;
            }
        }
        return cnt;
    }
}
```

## [2042. 检查句子中的数字是否递增](https://leetcode.cn/problems/check-if-numbers-are-ascending-in-a-sentence/)

> 字符串

```java
class Solution {
    public boolean areNumbersAscending(String s) {
        int last = 0, temp = 0;
        for (char c: s.toCharArray()) {
            if (Character.isDigit(c)) {
                temp = temp * 10 + (c - '0');
            } else if (c == ' ') {
                if (temp != 0 && temp <= last) return false;
                if (temp != 0) last = temp;
                temp = 0;
            }
        }
        return temp == 0 || temp > last;
    }
}
```

## [2185. 统计包含给定前缀的字符串](https://leetcode.cn/problems/counting-words-with-a-given-prefix/)

> 数组，字符串

```java
class Solution {
    public int prefixCount(String[] words, String pref) {
        int ans = 0;
        for (String word: words) {
            if (word.startsWith(pref)) ans++;
        }
        return ans;
    }
}
```

## [2283. 判断一个数的数字计数是否等于数位的值](https://leetcode.cn/problems/check-if-number-has-equal-digit-count-and-digit-value/)

> 哈希表，字符串，计数

```java
class Solution {
    public boolean digitCount(String num) {
        int[] cnt = new int[10];
        int[] cnt2 = new int[10];
        for (int i = 0; i < num.length(); ++i) {
            cnt[i] = num.charAt(i) - '0';
            cnt2[num.charAt(i) - '0']++;
        }
        for (int i = 0; i < 10; ++i) {
            if (cnt[i] != cnt2[i]) return false;
        }
        return true;
    }
}
```

## [2293. 极大极小游戏](https://leetcode.cn/problems/min-max-game/)

> 数组，模拟

原地修改
```java
class Solution {
    public int minMaxGame(int[] nums) {
        int len = nums.length;
        while (len > 1) {
            boolean flag = true;
            int idx = 0;
            for (int i = 0; i < len; i += 2) {
                nums[idx++] = flag ? Math.min(nums[i], nums[i + 1]) : Math.max(nums[i], nums[i + 1]);
                flag = !flag;
            }
            len /= 2;
        }
        return nums[0];
    }
}
```
TODO 递归

## [2299. 强密码检验器 II](https://leetcode.cn/problems/strong-password-checker-ii/)

> 字符串

```java
class Solution {
    public boolean strongPasswordCheckerII(String password) {
        if (password.length() < 8) return false;
        boolean[] check = new boolean[4];
        for (int i = 0; i < password.length(); ++i) {
            if (!check[0] && Character.isLowerCase(password.charAt(i))) check[0] = true;
            if (!check[1] && Character.isUpperCase(password.charAt(i))) check[1] = true;
            if (!check[2] && Character.isDigit(password.charAt(i))) check[2] = true;
            if (!check[3] && "!@#$%^&*()-+".contains(String.valueOf(password.charAt(i)))) check[3] = true;
            if (i > 0 && password.charAt(i) == password.charAt(i - 1)) return false;
        }
        return check[0] && check[1] && check[2] && check[3];
    }
}
```
TODO HashSet

## [1817. 查找用户活跃分钟数](https://leetcode.cn/problems/finding-the-users-active-minutes/)

> 数组，哈希表

```java
class Solution {
    public int[] findingUsersActiveMinutes(int[][] logs, int k) {
        HashMap<Integer, Set<Integer>> map = new HashMap<>();
        for (int[] log: logs) {
            Set<Integer> set = map.getOrDefault(log[0], new HashSet<>());
            set.add(log[1]);
            map.put(log[0], set);
            //// 或者简单写法
            // map.putIfAbsent(log[0], new HashSet<>());
            // map.get(log[0]).add(log[1]);
        }
        int[] ans = new int[k];
        for (Integer key: map.keySet()) {
            int active = map.get(key).size();
            ans[active - 1]++;
        }
        return ans;
    }
}
```

## [2303. 计算应缴税款总额](https://leetcode.cn/problems/calculate-amount-paid-in-taxes/)

> 数组，模拟

```java
class Solution {
    public double calculateTax(int[][] brackets, int income) {
        double ans = 0, income2 = (double)income;
        for (int i = 0; i < brackets.length; ++i) {
            double last = i != 0 ? (double)brackets[i - 1][0] : 0;
            double delta = brackets[i][0] - last;
            if (income2 <= delta) {
                ans += income2 * brackets[i][1] * 0.01;
                break;
            } else {
                ans += delta * brackets[i][1] * 0.01; 
            }
            income2 -= delta;
        }
        return ans;
    }
}
```

## [2309. 兼具大小写的最好英文字母](https://leetcode.cn/problems/greatest-english-letter-in-upper-and-lower-case/)

> 哈希表，字符串，枚举，位运算

```java
class Solution {
    public String greatestLetter(String s) {
        int[] cnt = new int[26];
        for (int i = 0; i < s.length(); ++i) {
            char c = s.charAt(i);
            if (Character.isLowerCase(c)) {
                if (cnt[c - 'a'] == 0 || cnt[c - 'a'] == 2) cnt[c - 'a']++;
            } else {
                if (cnt[c - 'A'] == 0 || cnt[c - 'A'] == 1) cnt[c - 'A'] += 2;
            }
        }
        for (int i = 25; i >= 0; i--) {
            if (cnt[i] == 3) {
                return String.valueOf((char)('A' + i));
            }
        }
        return "";
    }
}
```
TODO 哈希表，位运算

## [1669. 合并两个链表](https://leetcode.cn/problems/merge-in-between-linked-lists/)

> 链表

```java
class Solution {
    public ListNode mergeInBetween(ListNode list1, int a, int b, ListNode list2) {
        ListNode p1 = null, p2 = null, p3 = null, p = list1;
        int cnt = 0;
        while (p != null) {
            if (cnt == a - 1) {
                p1 = p;
            } else if (cnt == b) {
                p2 = p.next;
                break;
            }
            p = p.next;
            cnt++;
        }
        p1.next = list2;
        p = list2;
        while (p.next != null) {
            p = p.next;
        }
        p.next = p2;
        return list1;
    }
}
```

## [2535. 数组元素和与数字和的绝对差](https://leetcode.cn/problems/difference-between-element-sum-and-digit-sum-of-an-array/)

> 数组，数学

```java
class Solution {
    public int differenceOfSum(int[] nums) {
        int sum = 0;
        for (int i = 0; i < nums.length; ++i) {
            int s = 0, tmp = nums[i];
            while (tmp > 0) {
                s += tmp % 10;
                tmp /= 10;
            }
            sum += nums[i] - s;
        }
        return Math.abs(sum);
    }
}
```

## [2319. 判断矩阵是否是一个 X 矩阵](https://leetcode.cn/problems/check-if-matrix-is-x-matrix/)

> 数组，矩阵

```java
class Solution {
    public boolean checkXMatrix(int[][] grid) {
        int m = grid.length;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < m; ++j) {
                if (i == j || m - i - 1 == j) {
                    if (grid[i][j] == 0) return false;
                } else {
                    if (grid[i][j] != 0) return false;
                }
            }
        }
        return true;
    }
}
```

## [2331. 计算布尔二叉树的值](https://leetcode.cn/problems/evaluate-boolean-binary-tree/)

> 树，深度优先搜索，二叉树

```java
class Solution {
    public boolean evaluateTree(TreeNode root) {
        if (root.left == null && root.right == null) return root.val == 1;
        boolean l = evaluateTree(root.left);
        boolean r = evaluateTree(root.right);
        return root.val == 2 ? (l | r) : (l & r);
    }
}
```

## [2335. 装满杯子需要的最短总时长](https://leetcode.cn/problems/minimum-amount-of-time-to-fill-cups/)

> 贪心，数组，排序，堆，模拟

模拟
```java
class Solution {
    public int fillCups(int[] amount) {
        int ans = 0;
        while (true) {
            if (amount[0] == 0 && amount[1] == 0 && amount[2] == 0) return ans;
            Arrays.sort(amount);
            int delta = amount[1] - amount[0];
            if (delta == 0) {
                if (amount[0] == 0) {
                    ans += amount[2];
                    break;
                } else {
                    amount[0]--;
                    amount[2]--;
                    ans++;
                }
            } else {
                amount[1] -= delta;
                amount[2] -= delta;
                ans += delta;
            }
        }
        return ans;
    }
}
```
TODO 贪心+分类讨论
```java
class Solution {
    public int fillCups(int[] amount) {
        Arrays.sort(amount);
        if (amount[0] + amount[1] <= amount[2]) return amount[2];
        return (amount[0] + amount[1] + amount[2] + 1) / 2;
    }
}
```

## [2341. 数组能形成多少数对](https://leetcode.cn/problems/maximum-number-of-pairs-in-array/)

> 数组，哈希表，计数

```java
class Solution {
    public int[] numberOfPairs(int[] nums) {
        int[] cnt = new int[101];
        int[] ans = new int[2];
        for (int num: nums) {
            cnt[num]++;
        }
        for (int i = 0; i < 101; ++i) {
            if (cnt[i] != 0) {
                if ((cnt[i] & 1) == 1) ans[1]++;
                ans[0] += cnt[i] / 2;
            }
        }
        return ans;
    }
}
```
TODO 用计数的奇偶来构造map

## [2347. 最好的扑克手牌](https://leetcode.cn/problems/best-poker-hand/)

> 数组，哈希表，计数

```java
class Solution {
    public String bestHand(int[] ranks, char[] suits) {
        if (suits[0] == suits[1] && suits[1] == suits[2] && suits[2] == suits[3] && suits[3] == suits[4]) return "Flush";
        int[] cnt = new int[13];
        for (int c: ranks) {
            cnt[c - 1]++;
        }
        int max = 0;
        for (int i: cnt) {
            max = Math.max(max, i);
        }
        if (max == 2) return "Pair";
        else if (max == 1) return "High Card";
        return "Three of a Kind";
    }
}
```

## [2357. 使数组中所有元素都等于零](https://leetcode.cn/problems/make-array-zero-by-subtracting-equal-amounts/)

> 贪心，数组，哈希表，排序，模拟，堆

排序+模拟
```java
class Solution {
    public int minimumOperations(int[] nums) {
        Arrays.sort(nums);
        int n = nums.length, tmp = 0, ans = 0;
        for (int i = 0; i < n; ++i) {
            if (nums[i] == 0 || nums[i] <= tmp) continue;
            ans++;
            tmp = nums[i];
        }
        return ans;
    }
}
```
哈希表
```java
class Solution {
    public int minimumOperations(int[] nums) {
        Set<Integer> set = new HashSet<>();
        for (int num: nums) {
            if (num != 0) set.add(num);
        }
        return set.size();
    }
}
```

## [2363. 合并相似的物品](https://leetcode.cn/problems/merge-similar-items/)

> 数组，哈希表，有序集合，排序

```java
class Solution {
    public List<List<Integer>> mergeSimilarItems(int[][] items1, int[][] items2) {
        Arrays.sort(items1, (x, y) -> x[0] - y[0]);
        Arrays.sort(items2, (x, y) -> x[0] - y[0]);
        List<List<Integer>> ret = new ArrayList<>();
        int l1 = items1.length, l2 = items2.length, p1 = 0, p2 = 0;
        while (p1 < l1 && p2 < l2) {
            List<Integer> it = new ArrayList<>();
            if (items1[p1][0] < items2[p2][0]) {
                it.add(items1[p1][0]);
                it.add(items1[p1][1]);
                ret.add(it);
                p1++;
            } else if (items1[p1][0] > items2[p2][0]) {
                it.add(items2[p2][0]);
                it.add(items2[p2][1]);
                ret.add(it);
                p2++;
            } else {
                it.add(items1[p1][0]);
                it.add(items1[p1][1] + items2[p2][1]);
                ret.add(it);
                p1++;
                p2++;
            }
        }
        if (p1 < l1) {
            while (p1 < l1) {
                List<Integer> it = new ArrayList<>();
                it.add(items1[p1][0]);
                it.add(items1[p1][1]);
                ret.add(it);
                p1++;
            }
        } else if (p2 < l2) {
            while (p2 < l2) {
                List<Integer> it = new ArrayList<>();
                it.add(items2[p2][0]);
                it.add(items2[p2][1]);
                ret.add(it);
                p2++;
            }
        }
        return ret;
    }
}
```
TODO 哈希表

## [2574. 左右元素和的差值](https://leetcode.cn/problems/left-and-right-sum-differences/)

> 数组，前缀和

```java
class Solution {
    public int[] leftRigthDifference(int[] nums) {
        int n = nums.length;
        int[] ls = new int[n], rs = new int[n];
        if (n > 1) {
            for (int i = 1; i < n; ++i) {
                ls[i] = ls[i - 1] + nums[i - 1];
            }
            for (int i = n - 2; i >= 0; --i) {
                rs[i] = rs[i + 1] + nums[i + 1];
            }
        }
        int[] res = new int[n];
        for (int i = 0; i < n; ++i) {
            res[i] = Math.abs(ls[i] - rs[i]);
        }
        return res;
    }
}
```
TODO 先求总和

## [2379. 得到 K 个黑块的最少涂色次数](https://leetcode.cn/problems/minimum-recolors-to-get-k-consecutive-black-blocks/)

> 字符串，滑动窗口

```java
class Solution {
    public int minimumRecolors(String blocks, int k) {
        int cur = 0, min = 0;
        for (int i = 0; i < k; ++i) {
            if (blocks.charAt(i) == 'W') cur++;
        }
        min = cur;
        for (int i = k; i < blocks.length(); ++i) {
            if (blocks.charAt(i - k) == 'W') cur--;
            if (blocks.charAt(i) == 'W') cur++;
            min = Math.min(min, cur);
        }
        return min;
    }
}
```

## [2383. 赢得比赛需要的最少训练时长](https://leetcode.cn/problems/minimum-hours-of-training-to-win-a-competition/)

> 贪心，数组

```java
class Solution {
    public int minNumberOfHours(int initialEnergy, int initialExperience, int[] energy, int[] experience) {
        int esum = 0;
        for (int ex: energy) esum += ex;
        esum = initialExperience > esum ? 0 : esum - initialEnergy + 1;
        for (int e : experience) {
            if (initialExperience <= e) {
                esum += e - initialExperience + 1;
                initialExperience = e + 1 + e;
            } else {
                initialExperience += e;
            }
        }
        return esum;
    }
}
```

## [2389. 和有限的最长子序列](https://leetcode.cn/problems/longest-subsequence-with-limited-sum/)

> 贪心，数组，二分查找，前缀和，排序

```java
class Solution {
    public int[] answerQueries(int[] nums, int[] queries) {
        Arrays.sort(nums);
        int m = nums.length, n = queries.length;
        int[] ans = new int[n];
        int[] pre = new int[m + 1];
        for (int i = 0; i < m; i++) {
            pre[i + 1] = pre[i] + nums[i];
        }
        for (int i = 0; i < n; i++) {
            ans[i] = binarySearch(pre, queries[i]) - 1;
        }
        return ans;
    }

    private int binarySearch(int[] pre, int q) {
        int l = 1, r = pre.length;
        while (l < r) {
            int mid = l + (r - l) / 2;
            if (pre[mid] > q) r = mid;
            else l = mid + 1;
        }
        return l;
    }
}
```

## [2395. 和相等的子数组](https://leetcode.cn/problems/find-subarrays-with-equal-sum/)

> 数组，哈希表

```java
class Solution {
    public boolean findSubarrays(int[] nums) {
        Set<Integer> set = new HashSet<>();
        int n = nums.length;
        for (int i = 0; i < n - 1; ++i) {
            int sum = nums[i] + nums[i + 1];
            if (set.contains(sum)) return true;
            set.add(sum);
        }
        return false;
    }
}
```

## [1683. 无效的推文](https://leetcode.cn/problems/invalid-tweets/)

> 数据库

```java
SELECT tweet_id FROM Tweets WHERE length(content) > 15
```

## [2399. 检查相同字母间的距离](https://leetcode.cn/problems/check-distances-between-same-letters/)

> 数组，哈希表，字符串

```java
class Solution {
    public boolean checkDistances(String s, int[] distance) {
        Map<Character, Integer> map = new HashMap<>();
        for (int i = 0; i < s.length(); ++i) {
            char c = s.charAt(i);
            if (map.containsKey(c)) {
                if (distance[c - 'a'] != i - map.get(c) - 1) return false;
            } else {
                map.put(c, i);
            }
        }
        return true;
    }
}
```

## [1041. 困于环中的机器人](https://leetcode.cn/problems/robot-bounded-in-circle/)

> 数学，字符串，模拟

```java
class Solution {
    public boolean isRobotBounded(String instructions) {
        // 执行完之后不在原点且朝北
        int x = 0, y = 0, dir = 0;
        int[][] direction = new int[][]{{0, 1}, {-1, 0}, {0, -1}, {1, 0}};
        for (int i = 0; i < instructions.length(); ++i) {
            char c = instructions.charAt(i);
            if (c == 'G') {
                x += direction[dir][0];
                y += direction[dir][1];
            } else if (c == 'L') {
                dir++;
                if (dir == 4) dir = 0;
            } else {
                dir--;
                if (dir == -1) dir = 3;
            }
        }
        return !(!(x == 0 && y == 0) && dir == 0);
    }
}
```

## [2404. 出现最频繁的偶数元素](https://leetcode.cn/problems/most-frequent-even-element/)

> 数组，哈希表，计数

```java
class Solution {
    public int mostFrequentEven(int[] nums) {
        Arrays.sort(nums);
        int cnt = 0, max = 0, ret = -1;
        for (int i = 0; i < nums.length; ++i) {
            if ((nums[i] & 1) == 0) {
                if (i != 0 && nums[i] == nums[i - 1]) cnt++;
                else cnt = 1;
                if (cnt > max) {
                    max = cnt;
                    ret = nums[i];
                }
            } else {
                cnt = 0;
            }
        }
        return ret;
    }
}
```
TODO 哈希表计数

## [2620. 计数器](https://leetcode.cn/problems/counter/)

> 闭包

```java
var createCounter = function(n) {
    return function() {
        return n++;
    };
};
```

## [2418. 按身高排序](https://leetcode.cn/problems/sort-the-people/)

> 数组，哈希表，字符串，排序

```java
class Solution {
    public String[] sortPeople(String[] names, int[] heights) {
        int n = names.length;
        Integer[] indices = new Integer[n];
        for (int i = 0; i < n; ++i) {
            indices[i] = i;
        }
        Arrays.sort(indices, (x, y) -> heights[y] - heights[x]);
        String[] res = new String[n];
        for (int i = 0; i < n; ++i) {
            res[i] = names[indices[i]];
        }
        return res;
    }
}
```

## [2432. 处理用时最长的那个任务的员工](https://leetcode.cn/problems/the-employee-that-worked-on-the-longest-task/)

> 数组

```java
class Solution {
    public int hardestWorker(int n, int[][] logs) {
        int max = logs[0][1], ret = logs[0][0];
        for (int i = 1; i < logs.length; ++i) {
            int delta = logs[i][1] - logs[i - 1][1];
            if (delta > max) {
                max = delta;
                ret = logs[i][0];
            } else if (delta == max) {
                if (logs[i][0] < ret) {
                    ret = logs[i][0];
                }
            }
        }
        return ret;
    }
}
```

## [2441. 与对应负数同时存在的最大正整数](https://leetcode.cn/problems/largest-positive-integer-that-exists-with-its-negative/)

> 哈希表，数组，双指针，排序

```java
class Solution {
    public int findMaxK(int[] nums) {
        Set<Integer> set = new HashSet<>();
        int max = -1;
        for (int i: nums) set.add(i);
        for (int i: nums) {
            if (i > max && set.contains(-i)) {
                max = i;
            }
        }
        return max;
    }
}
```
TODO 排序+双指针

## [2667. 创建 Hello World 函数](https://leetcode.cn/problems/create-hello-world-function/)

```java
var createHelloWorld = function() {
    return function(...args) {
        return "Hello World"
    }
};
```

## [2446. 判断两个事件是否存在冲突](https://leetcode.cn/problems/determine-if-two-events-have-conflict/)

> 数组，字符串

```java
class Solution {
    public boolean haveConflict(String[] event1, String[] event2) {
        int[] e1 = eval(event1), e2 = eval(event2);
        return !(e1[1] < e2[0] || e1[0] > e2[1]);
    }

    private int[] eval(String[] event) {
        int a1 = trans(event[0]), a2 = trans(event[1]);
        return new int[]{a1, a2};
    }

    private int trans(String time) {
        String[] times = time.split(":");
        return Integer.parseInt(times[0]) * 60 + Integer.parseInt(times[1]);
    }
}
```
时间能直接比较!
```java
class Solution {
    public boolean haveConflict(String[] event1, String[] event2) {
        return !(event1[1].compareTo(event2[0]) < 0 || event1[0].compareTo(event2[1]) > 0);
    }
}
```

## [2455. 可被三整除的偶数的平均值](https://leetcode.cn/problems/average-value-of-even-numbers-that-are-divisible-by-three/)

> 数组，数学

```java
class Solution {
    public int averageValue(int[] nums) {
        int sum = 0, n = 0;
        for (int i: nums) {
            if (i % 6 == 0) {
                sum += i;
                n++;
            }
        }
        if (n == 0) return 0;
        return sum / n;
    }
}
```

## [2465. 不同的平均值数目](https://leetcode.cn/problems/number-of-distinct-averages/)

```java
class Solution {
    public int distinctAverages(int[] nums) {
        Set<Integer> set = new HashSet<>();
        Arrays.sort(nums);
        int i = 0, j = nums.length - 1;
        while (i < j) {
            set.add(nums[i] + nums[j]);
            i++;
            j--;
        }
        return set.size();
    }
}
```

## [2460. 对数组执行操作](https://leetcode.cn/problems/apply-operations-to-an-array/)

```java
class Solution {
    public int[] applyOperations(int[] nums) {
        int n = nums.length;
        int[] ret = new int[n];
        int j = 0;
        for (int i = 0; i < n - 1; ++i) {
            if (nums[i] == nums[i + 1] && nums[i] != 0) {
                ret[j++] = nums[i] * 2;
                nums[i + 1] = 0;
            } else {
                if (nums[i] != 0) ret[j++] = nums[i];
            }
        }
        if (nums[n - 1] != 0) ret[j++] = nums[n - 1];
        return ret;
    }
}
```
TODO swap => 空间O(1)

## [2352. 相等行列对](https://leetcode.cn/problems/equal-row-and-column-pairs/)

```java
class Solution {
    public int equalPairs(int[][] grid) {
        int n = grid.length, ret = 0;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (eq(grid, i, j, n)) {
                    ret++;
                }
            }
        }
        return ret;
    }

    private boolean eq(int[][] grid, int i, int j, int n) {
        boolean flag = true;
        for (int x = 0; x < n; ++x) {
            if (grid[i][x] != grid[x][j]) flag = false;
        }
        return flag;
    }
}
```
TODO 哈希表

## [2475. 数组中不等三元组的数目](https://leetcode.cn/problems/number-of-unequal-triplets-in-array/)

> 数组，哈希表

```java
class Solution {
    public int unequalTriplets(int[] nums) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int i: nums) {
            map.put(i, map.getOrDefault(i, 0) + 1);
        }
        int cur = 0, ret = 0, n = nums.length;
        for (int key: map.keySet()) {
            ret += cur * map.get(key) * (n - cur - map.get(key));
            cur += map.get(key);
        }
        return ret;
    }
}
```

## [2481. 分割圆的最少切割次数](https://leetcode.cn/problems/minimum-cuts-to-divide-a-circle/)

> 几何，数学

```java
class Solution {
    public int numberOfCuts(int n) {
        return n == 1 ? 0 : ((n & 1) == 1 ? n : n / 2);
    }
}
```

## [2496. 数组中字符串的最大值](https://leetcode.cn/problems/maximum-value-of-a-string-in-an-array/)

> 数组，字符串

```java
class Solution {
    public int maximumValue(String[] strs) {
        int max = 0;
        for (String str: strs) {
            max = Math.max(max, countStrNum(str));
        }
        return max;
    }

    private int countStrNum(String str) {
        int sum = 0;
        for (int i = 0; i < str.length(); ++i) {
            if (Character.isDigit(str.charAt(i))) {
                sum = sum * 10 + (str.charAt(i) - '0');
            } else {
                return str.length();
            }
        }
        return sum;
    }
}
```
内存消耗优化
```java
class Solution {
    public int maximumValue(String[] strs) {
        int max = 0;
        for (String str: strs) {
            boolean isDigit = true;
            for (int i = 0; i < str.length(); ++i) {
                isDigit &= Character.isDigit(str.charAt(i));
            }
            max = Math.max(max, isDigit ? Integer.parseInt(str) : str.length());
        }
        return max;
    }
}
```

## [2744. 最大字符串配对数目](https://leetcode.cn/problems/find-maximum-number-of-string-pairs/submissions/)

> 字符串

```java
class Solution {
    public int maximumNumberOfStringPairs(String[] words) {
        int num = 0, n = words.length;
        for (int i = 0; i < n - 1; ++i) {
            for (int j = i + 1; j < n; ++j) {
                if (words[i].equals(reverse(words[j]))) {
                    num++;
                }
            }
        }
        return num;
    }

    private String reverse(String word) {
        StringBuilder sb = new StringBuilder();
        for (int i = word.length() - 1; i >= 0; i--) {
            sb.append(word.charAt(i));
        }
        return sb.toString();
    }
}
```
HashSet
```java
class Solution {
    public int maximumNumberOfStringPairs(String[] words) {
        int num = 0;
        Set<String> set = new HashSet<>();
        for (String str: words) {
            set.add(str);
        }
        for (String word: set) {
            String rword = reverse(word);
            if (word.equals(rword)) continue;
            if (set.contains(rword)) {
                num++;
            }
        }
        return num / 2;
    }

    private String reverse(String word) {
        StringBuilder sb = new StringBuilder();
        for (int i = word.length() - 1; i >= 0; i--) {
            sb.append(word.charAt(i));
        }
        return sb.toString();
    }
}
```

## [2490. 回环句](https://leetcode.cn/problems/circular-sentence/)

> 字符串

```java
class Solution {
    public boolean isCircularSentence(String sentence) {
        String[] sens = sentence.split(" ");
        boolean ret = true;
        for (int i = 1; i < sens.length; ++i) {
            if (sens[i].charAt(0) != sens[i - 1].charAt(sens[i - 1].length() - 1)) return false;
        }
        return sens[sens.length - 1].charAt(sens[sens.length - 1].length() - 1) == sens[0].charAt(0);
    }
}
```
TODO 用空格前后判断更简单

## [2656. K 个元素的最大和](https://leetcode.cn/problems/maximum-sum-with-exactly-k-elements/description/)

> 贪心，数组

```java
class Solution {
    public int maximizeSum(int[] nums, int k) {
        int max = 0;
        for (int num: nums) {
            max = Math.max(max, num);
        }
        return (max * 2 + k - 1) * k / 2;
    }
}
```
TODO 取最大值可以用：Arrays.stream(nums).max().getAsInt();

## [2600. K 件物品的最大和](https://leetcode.cn/problems/k-items-with-the-maximum-sum/description/)

> 贪心，数学

```java
class Solution {
    public int kItemsWithMaximumSum(int numOnes, int numZeros, int numNegOnes, int k) {
        if (k <= numOnes) return k;
        else if (numOnes + numZeros > k && k > numOnes) return numOnes;
        else return numOnes - (k - numOnes - numZeros);
    }
}
```
TODO return min(numOnes, k) - max(k - numOnes - numZeros, 0)

## [2544. 交替数字和](https://leetcode.cn/problems/alternating-digit-sum/description/)

```java
class Solution {
    public int alternateDigitSum(int n) {
        String ns = Integer.toString(n);
        boolean flag = true;
        int sum = 0;
        for (char c: ns.toCharArray()) {
            int x = c - '0';
            sum = flag ? sum + x : sum - x;
            flag = !flag;
        }
        return sum;
    }
}
```

## [415. 字符串相加](https://leetcode.cn/problems/add-strings/description/)

> 数学，字符串，模拟

```java
class Solution {
    public String addStrings(String num1, String num2) {
        int n1 = num1.length(), n2 = num2.length(), carry = 0, mn = Math.max(n1, n2);
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < mn; ++i) {
            int x = 0, y = 0;
            if (n1 > i) x = num1.charAt(n1 - i - 1) - '0';
            if (n2 > i) y = num2.charAt(n2 - i - 1) - '0';
            int s = x + y + carry;
            if (s > 9) {
                carry = 1;
                s -= 10;
            } else {
                carry = 0;
            }
            sb.append(String.valueOf(s));
        }
        if (carry > 0) sb.append("1");
        return sb.reverse().toString();
    }
}
```

## [860. 柠檬水找零](https://leetcode.cn/problems/lemonade-change/description/)

> 贪心，数组

```java
class Solution {
    public boolean lemonadeChange(int[] bills) {
        if (bills[0] > 5) return false;
        int a = 1, b = 0;
        for (int i = 1; i < bills.length; ++i) {
            if (bills[i] == 5) {
                a++;
            } else if (bills[i] == 10) {
                b++;
                if (a == 0) return false;
                else a--;
            } else {
                if (b >= 1) {
                    b--;
                    if (a >= 1) {
                        a--;
                    } else {
                        return false;
                    }
                } else {
                    if (a >= 3) {
                        a -= 3;
                    } else {
                        return false;
                    }
                }
            }
        }
        return true;
    }
}
```

## [1837. K 进制表示下的各位数字总和](https://leetcode.cn/problems/sum-of-digits-in-base-k/description/)

> 数学

```java
class Solution {
    public int sumBase(int n, int k) {
        int sum = 0;
        while (n > 0) {
            sum += n % k;
            n /= k;
        }
        return sum;
    }
}
```

## [771. 宝石与石头](https://leetcode.cn/problems/jewels-and-stones/description/)

> 哈希表，字符串

```java
class Solution {
    public int numJewelsInStones(String jewels, String stones) {
        int sum = 0;
        boolean[] status = new boolean[52];
        for (char c: jewels.toCharArray()) {
            if (Character.isUpperCase(c)) {
                status[c - 'A'] = true;
            } else {
                status[26 + c - 'a'] = true;
            }
        }
        for (char c: stones.toCharArray()) {
            if (Character.isUpperCase(c)) {
                sum += status[c - 'A'] ? 1 : 0;
            } else {
                sum += status[26 + c - 'a'] ? 1 : 0;
            }
        }
        return sum;
    }
}
```

## [2651. 计算列车到站时间](https://leetcode.cn/problems/calculate-delayed-arrival-time/description/)

> 数学

```java
class Solution {
    public int findDelayedArrivalTime(int arrivalTime, int delayedTime) {
        return (arrivalTime + delayedTime) % 24;
    }
}
```

## [2769. 找出最大的可达成数字](https://leetcode.cn/problems/find-the-maximum-achievable-number/description/)

> 数学

```java
class Solution {
    public int theMaximumAchievableX(int num, int t) {
        return num + t * 2;
    }
}
```

## [2703. 返回传递的参数的长度](https://leetcode.cn/problems/return-length-of-arguments-passed/description/)

> JS

```js
/**
 * @return {number}
 */
var argumentsLength = function(...args) {
    return args.length
};

/**
 * argumentsLength(1, 2, 3); // 3
 */
```

## [1572. 矩阵对角线元素的和](https://leetcode.cn/problems/matrix-diagonal-sum/description/)

> 数组，矩阵

```java
class Solution {
    public int diagonalSum(int[][] mat) {
        int m = mat.length, sum = 0;
        for (int i = 0; i < m; ++i) {
            sum += mat[i][i];
        }
        for (int i = m - 1; i >= 0; --i) {
            sum += mat[i][m - i - 1];
        }
        if ((m & 1) == 1) {
            int n = m / 2;
            sum -= mat[n][n];
        }
        return sum;
    }
}
```

## [2396. 严格回文的数字](https://leetcode.cn/problems/strictly-palindromic-number/description/)

> 脑筋急转弯

```java
class Solution {
    public boolean isStrictlyPalindromic(int n) {
        return false;
    }
}
```

## [2807. 在链表中插入最大公约数](https://leetcode.cn/problems/insert-greatest-common-divisors-in-linked-list/description/)

> 数字，链表，数学

```java
class Solution {
    public ListNode insertGreatestCommonDivisors(ListNode head) {
        ListNode p = head;
        while (p != null && p.next != null) {
            ListNode nn = new ListNode(gcd(p.val, p.next.val));
            nn.next = p.next;
            p.next = nn;
            p = p.next.next;
        }
        return head;
    }

    private int gcd(int a, int b) {
        if (b == 0) return a;
        return gcd(b, a % b);
    }
}
```

## [2682. 找出转圈游戏输家](https://leetcode.cn/problems/find-the-losers-of-the-circular-game/description/)

> 数组，哈希表，模拟

```java
class Solution {
    public int[] circularGameLosers(int n, int k) {
        boolean[] status = new boolean[n];
        int p = 0;
        status[0] = true;
        for (int i = 1; i < n; i++) {
            p = (p + i * k) % n;
            if (status[p]) break;
            else status[p] = true;
        }
        List<Integer> list = new ArrayList<>();
        for (int i = 0; i < n; ++i) {
            if (!status[i]) list.add(i + 1);
        }
        int len = list.size();
        int[] ret = new int[len];
        for (int i = 0; i < len; ++i) ret[i] = list.get(i);
        return ret;
    }
}
```

## [2723. 添加两个 Promise 对象](https://leetcode.cn/problems/add-two-promises/description/)

> JS

```javascript
var addTwoPromises = async function(promise1, promise2) {
    return await promise1 + await promise2;
};
```

## [849. 到最近的人的最大距离](https://leetcode.cn/problems/maximize-distance-to-closest-person/description/?lang=pythondata)

> 数组

```java
class Solution {
    public int maxDistToClosest(int[] seats) {
        int res = 0;
        int l = 0;
        while (l < seats.length && seats[l] == 0) {
            ++l;
        }
        res = Math.max(res, l);
        while (l < seats.length) {
            int r = l + 1;
            while (r < seats.length && seats[r] == 0) {
                ++r;
            }
            if (r == seats.length) {
                res = Math.max(res, r - l - 1);
            } else {
                res = Math.max(res, (r - l) / 2);
            }
            l = r;
        }
        return res;
    }
}
```

## [2511. 最多可以摧毁的敌人城堡数目](https://leetcode.cn/problems/maximum-enemy-forts-that-can-be-captured)

> 数组，双指针

```java
class Solution {
    public int captureForts(int[] forts) {
        // 1和-1之间的0的个数的最大值
        int pre = -1, i = 0, n = forts.length, max = 0;
        while (i < n) {
            if (forts[i] == -1 || forts[i] == 1) {
                if (pre != -1) {
                    if (forts[pre] != forts[i]) {
                        max = Math.max(max, i - pre - 1);
                    }
                }
                pre = i;
            }
            i++;
        }
        return max;
    }
}
```

## [2605. 从两个数字数组里生成最小数字](https://leetcode.cn/problems/form-smallest-number-from-two-digit-arrays)

> 数组，哈希表，枚举

```java
class Solution {
    public int minNumber(int[] nums1, int[] nums2) {
        Arrays.sort(nums1);
        Arrays.sort(nums2);
        Set<Integer> set = new HashSet<>();
        for (int i: nums1) {
            set.add(i);
        }
        for (int i: nums2) {
            if (set.contains(i)) return i;
        }
        int a = Math.min(nums1[0], nums2[0]);
        int b = Math.max(nums1[0], nums2[0]);
        return a * 10 + b;
    }
}
```

## [2798. 满足目标工作时长的员工数目](https://leetcode.cn/problems/number-of-employees-who-met-the-target)

> 数组，枚举

```java
class Solution {
    public int numberOfEmployeesWhoMetTarget(int[] hours, int target) {
        int cnt = 0;
        for (int hour: hours) {
            if (hour >= target) cnt++;
        }
        return cnt;
    }
}
```

## [LCP50.宝石补给](https://leetcode.cn/problems/WHnhjV)

> 模拟

```java
class Solution {
    public int giveGem(int[] gem, int[][] operations) {
        for (int[] op: operations) {
            int n = gem[op[0]] / 2;
            gem[op[1]] += n;
            gem[op[0]] -= n;
        }
        int max = gem[0], min = gem[0];
        for (int i = 1; i < gem.length; ++i) {
            max = Math.max(max, gem[i]);
            min = Math.min(min, gem[i]);
        }
        return max - min;
    }
}
```

## [198.打家劫舍](https://leetcode.cn/problems/house-robber)

> 数组，动态规划

```java
class Solution {
    public int rob(int[] nums) {
        int n = nums.length;
        int[][] max = new int[n][2];
        max[0][0] = 0;
        max[0][1] = nums[0];
        for (int i = 1; i < n; ++i) {
            max[i][0] = Math.max(max[i - 1][1], max[i - 1][0]);
            max[i][1] = max[i - 1][0] + nums[i];
        }
        return Math.max(max[n - 1][0], max[n - 1][1]);
    }
}
```
一维即可
```java
class Solution {
    public int rob(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int length = nums.length;
        if (length == 1) {
            return nums[0];
        }
        int[] dp = new int[length];
        dp[0] = nums[0];
        dp[1] = Math.max(nums[0], nums[1]);
        for (int i = 2; i < length; i++) {
            dp[i] = Math.max(dp[i - 2] + nums[i], dp[i - 1]);
        }
        return dp[length - 1];
    }
}
```

## [2828.判别首字母缩略词](https://leetcode.cn/problems/check-if-a-string-is-an-acronym-of-words/)

> 数组，字符串

```java
class Solution {
    public boolean isAcronym(List<String> words, String s) {
        StringBuilder sb = new StringBuilder();
        for (String word: words) {
            sb.append(word.charAt(0));
        }
        return sb.toString().equals(s);
    }
}
```
优化
```java
class Solution {
    public boolean isAcronym(List<String> words, String s) {
        if (words.size() != s.length()) return false;
        for (int i = 0; i < words.size(); ++i) {
            if (words.get(i).charAt(0) != s.charAt(i)) return false;
        }
        return true;
    }
}
```

## [2591.将钱分给最多的儿童](https://leetcode.cn/problems/distribute-money-to-maximum-children)

> 贪心，数学

```java
class Solution {
    public int distMoney(int money, int children) {
        if (money < children) return -1;
        if (money > children * 8) return children - 1;
        int rest = money - children;
        int res = rest / 7;
        return res == 0 ? 0 : ((rest % 7) == 3 ? (children - res > 1 ? res : res - 1) : res);
    }
}
```

## [2582. 递枕头](https://leetcode.cn/problems/pass-the-pillow)

> 数学，模拟

```java
class Solution {
    public int passThePillow(int n, int time) {
        int x = time / (n - 1);
        int y = time % (n - 1);
        return (x & 1) == 1 ? n - y : y + 1;
    }
}
```

## [145. 二叉树的后序遍历](https://leetcode.cn/problems/binary-tree-postorder-traversal)

> 栈，树，DFS，二叉树

```java
class Solution {
    public List<Integer> postorderTraversal(TreeNode root) {
        if (root == null) return new ArrayList<>();
        List<Integer> list1 = postorderTraversal(root.left);
        List<Integer> list2 = postorderTraversal(root.right);
        list1.addAll(list2);
        list1.add(root.val);
        return list1;
    }
}
```

## [2578. 最小和分割](https://leetcode.cn/problems/split-with-minimum-sum)

> 贪心，数学，排序

```java
class Solution {
    public int splitNum(int num) {
        String str = String.valueOf(num);
        char[] chars = str.toCharArray();
        // char[] chars = Integer.toString(num).toCharArray();
        Arrays.sort(chars);
        int a = 0, b = 0, n = chars.length;
        for (int i = 0; i < n; i += 2) {
            a = a * 10 + chars[i] - '0';
            if (i + 1 < n) b = b * 10 + chars[i + 1] - '0';
        }
        return a + b;
    }
}
```

## [2562. 找出数组的串联值](https://leetcode.cn/problems/find-the-array-concatenation-value)

> 数组，双指针，模拟

```java
class Solution {
    public long findTheArrayConcVal(int[] nums) {
        int n = nums.length;
        long res = 0l;
        for (int i = 0; i < n; ++i) {
            if (i > n - i - 1) break;
            if (i == n - i - 1) {
                res += nums[i];
            } else {
                String tmp = String.valueOf(nums[n - i - 1]);
                res += nums[i] * (long)Math.pow(10, tmp.length()) + (long)nums[n - i - 1];
            }
        }
        return res;
    }
}
```
TODO 双指针

## [2652. 倍数求和](https://leetcode.cn/problems/sum-multiples)

> 数学

```java
class Solution {
    public int sumOfMultiples(int n) {
        if (n == 1) return 0;
        return sumOfMultiples(n - 1) + ((n % 3 == 0 || n % 5 == 0 || n % 7 == 0) ? n : 0);
    }
}
```
容斥原理
```java
class Solution {
    public int sumOfMultiples(int n) {
        return f(n, 3) + f(n, 5) + f(n, 7) - f(n, 15) - f(n, 21) - f(n, 35) + f(n, 105);
    }

    private int f(int n, int m) {
        return (m + n / m * m) * (n / m) / 2;
    }
}
```

## [2525. 根据规则将箱子分类](https://leetcode.cn/problems/categorize-box-according-to-criteria)

> 数学

```java
class Solution {
    public String categorizeBox(int length, int width, int height, int mass) {
        boolean isBulky = false, isHeavy = mass >= 100;
        long volume = (long)length * (long)width * (long)height;
        if (length >= 1e4 || width >= 1e4 || height >= 1e4 || volume >= 1e9)
            isBulky = true;
        if (isBulky && isHeavy) return "Both";
        if (!isBulky && !isHeavy) return "Neither";
        if (isBulky && !isHeavy) return "Bulky";
        return "Heavy";
    }
}
```

## [2678. 老人的数目](https://leetcode.cn/problems/number-of-senior-citizens)

> 数组，字符串

```java
class Solution {
    public int countSeniors(String[] details) {
        int ans = 0;
        for (String detail: details) {
            if ((detail.charAt(11) - '6') >= 0 && !(detail.charAt(11) == '6' && detail.charAt(12) == '0'))
                ans++;
        }
        return ans;
    }
}
```
用substring
```java
class Solution {
    public int countSeniors(String[] details) {
        int ans = 0;
        for (String detail: details) {
            if (Integer.parseInt(detail.substring(11, 13)) > 60) {
                ans++;
            }
        }
        return ans;
    }
}
```

## [2520. 统计能整除数字的位数](https://leetcode.cn/problems/count-the-digits-that-divide-a-number)

> 数学

```java
class Solution {
    public int countDigits(int num) {
        int ans = 0, y = num;
        while (y > 0) {
            if (num % (y % 10) == 0) ans++;
            y /= 10;
        }
        return ans;
    }
}
```

## [2558. 从数量最多的堆取走礼物](https://leetcode.cn/problems/take-gifts-from-the-richest-pile)

> 数组，模拟，堆

```java
class Solution {
    public long pickGifts(int[] gifts, int k) {
        PriorityQueue<Integer> pq = new PriorityQueue<>((a, b) -> b - a);
        for (int i: gifts) pq.offer(i);
        for (int i = 0; i < k; ++i) pq.offer((int)Math.sqrt(pq.poll()));
        long ans = 0;
        while(!pq.isEmpty()) ans += (long)pq.poll();
        return ans;
    }
}
```

## [统计范围内的元音字符串数](https://leetcode.cn/problems/count-the-number-of-vowel-strings-in-range)

> 数组，字符串

```java
class Solution {
    public int vowelStrings(String[] words, int left, int right) {
        Set<Character> set = new HashSet<>();
        set.add('a');
        set.add('e');
        set.add('i');
        set.add('o');
        set.add('u');
        int ans = 0;
        for (int i = left; i <= right; ++i) {
            int n = words[i].length();
            if (set.contains(words[i].charAt(0)) && set.contains(words[i].charAt(n - 1)))
                ans++;
        }
        return ans;
    }
}
```

## [最长平衡子字符串](https://leetcode.cn/problems/find-the-longest-balanced-substring-of-a-binary-string)

> 字符串

```java
class Solution {
    public int findTheLongestBalancedSubstring(String s) {
        int max = 0, zn = 0, on = 0;
        char[] cs = s.toCharArray();
        zn = cs[0] == '0' ? 1 : 0;
        for (int i = 1; i < cs.length; ++i) {
            if (cs[i] == '0' && cs[i - 1] == '0') {
                zn++;
            } else if (cs[i] == '0' && cs[i - 1] == '1') {
                max = Math.max(max, Math.min(zn, on) * 2);
                on = 0;
                zn = 1;
            } else {
                on++;
            }
        }
        return Math.max(max, Math.min(zn, on) * 2);
    }
}
```

## [2760. 最长奇偶子数组](https://leetcode.cn/problems/longest-even-odd-subarray-with-threshold)

> 数组，滑动窗口

```java
class Solution {
    public int longestAlternatingSubarray(int[] nums, int threshold) {
        int max = 0;
        for (int i = 0; i < nums.length; ++i) {
            if (nums[i] % 2 == 0 && nums[i] <= threshold) {
                int j = i + 1;
                while (j < nums.length) {
                    if ((nums[j] % 2 != nums[j - 1] % 2) && nums[j] <= threshold) {
                        j++;
                    } else {
                        break;
                    }
                }
                max = Math.max(max, j - i);
                i = j - 1;
            }
        }
        return max;
    }
}
```

## [2824. 统计和小于目标的下标对数目](https://leetcode.cn/problems/count-pairs-whose-sum-is-less-than-target)

> 数组，双指针，排序

```java
class Solution {
    public int countPairs(List<Integer> nums, int target) {
        int n = nums.size(), ans = 0;
        for (int i = 0; i < n - 1; ++i) {
            for (int j = i + 1; j < n; ++j) {
                if (nums.get(i) + nums.get(j) < target)
                    ans++;
            }
        }
        return ans;
    }
}
```
TODO 二分/双指针

# Java算法模板

## BFS

如果不需要确定当前遍历到了哪一层

```java
queue.push(root)
while queue 不空：
    cur = queue.pop()
    for 节点 in cur的所有相邻节点：
        if 该节点有效且未访问过：
            queue.push(该节点)
```

如果要确定当前遍历到了哪一层

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

双向BFS

```java
// 创建「两个队列」分别用于两个方向的搜索；
// 创建「两个哈希表」用于「解决相同节点重复搜索」和「记录转换次数」；
// 为了尽可能让两个搜索方向“平均”，每次从队列中取值进行扩展时，先判断哪个队列容量较少；
// 如果在搜索过程中「搜索到对方搜索过的节点」，说明找到了最短路径。

// d1、d2 为两个方向的队列
// m1、m2 为两个方向的哈希表，记录每个节点距离起点的 
// 只有两个队列都不空，才有必要继续往下搜索
// 如果其中一个队列空了，说明从某个方向搜到底都搜不到该方向的目标节点
while(!d1.isEmpty() && !d2.isEmpty()) {
    if (d1.size() < d2.size()) {
        update(d1, m1, m2);
    } else {
        update(d2, m2, m1);
    }
}

// update 为从队列 d 中取出一个元素进行「一次完整扩展」的逻辑
void update(Deque d, Map cur, Map other) {}
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

直接用函数

```java
Arrays.binarySearch(arr, fromIndex, toIndex, key)
```

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
        int left = 0, right = nums.length - 1; // 定义target在左闭右闭的区间里，[left, right]
        while (left <= right) { // 当left==right，区间[left, right]依然有效，所以用 <=（平衡查找+1-1，不平衡用<）
            int mid = left + (right - left) / 2; // 防止溢出 等同于(left + right)/2
            if (nums[mid] < target) {
                // 更新left
                left = mid + 1; // target 在右区间，所以[mid + 1, right]
            } else if (nums[mid] > target) {
                // 更新right
                right = mid - 1; // target 在左区间，所以[left, mid - 1]
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
        while(i < j && arr[j] >= pivot) j--;
        arr[i] = arr[j];
        while(i < j && arr[i] <= pivot) i++;
        arr[j] = arr[i];
    }
    arr[i] = pivot;
    QuickSort(arr, left, i - 1);
    QuickSort(arr, i + 1, right);
}
```

快速选择算法（求第k大元素）

```java
public void nthElement(List<Integer> results, int left, int kth, int right) {
    if (left == right)
        return;
    int pivot = (int) (left + Math.random() * (right - left + 1));
    swap(results, pivot, right);
    // 三路划分（three-way partition）
    int sepl = left - 1, sepr = left - 1;
    for (int i = left; i <= right; i++) {
        if (results.get(i) > results.get(right)) {
            swap(results, ++sepr, i);
            swap(results, ++sepl, sepr);
        } else if (results.get(i) == results.get(right)) {
            swap(results, ++sepr, i);
        }
    }
    if (sepl < left + kth && left + kth <= sepr) {
        return;
    } else if (left + kth <= sepl) {
        nthElement(results, left, kth, sepl);
    } else {
        nthElement(results, sepr + 1, kth - (sepr - left + 1), right);
    }
}
public void swap(List<Integer> results, int index1, int index2) {
    int temp = results.get(index1);
    results.set(index1, results.get(index2));
    results.set(index2, temp);
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
> 
> https://mp.weixin.qq.com/s?__biz=MzU4NDE3MTEyMA==&mid=2247488490&idx=1&sn=db2998cb0e5f08684ee1b6009b974089&chksm=fd9cb8f5caeb31e3f7f67dba981d8d01a24e26c93ead5491edb521c988adc0798d8acb6f9e9d&token=1232059512&lang=zh_CN#rd

```java
class Trie {
    TrieNode root;

    class TrieNode {
        String word = ""; // 在结尾处记录单词（可选）
        boolean end = false; // 是否单词结尾（可选）
        int frequency = 0; // 频数统计（可选）
        TrieNode[] child = new TrieNode[26];
    }

    public Trie() {
        root = new TrieNode();
    }

    public void insert(String s) {
        TrieNode p = root;
        for(int i = 0; i < s.length(); i++) {
            int u = s.charAt(i) - 'a';
            if (p.child[u] == null)
                p.child[u] = new TrieNode();
            p = p.child[u]; 
        }
        p.word = s;
        p.end = true;
        ++p.frequency;
    }

    public boolean search(String s) {
        TrieNode p = root;
        for(int i = 0; i < s.length(); i++) {
            int u = s.charAt(i) - 'a';
            if (p.child[u] == null)
                return false;
            p = p.child[u]; 
        }
        return p.end;
    }

    public boolean startsWith(String s) {
        TrieNode p = root;
        for(int i = 0; i < s.length(); i++) {
            int u = s.charAt(i) - 'a';
            if (p.child[u] == null)
                return false;
            p = p.child[u]; 
        }
        return true;
    }
}
```
字典树
执行用时：5 ms, 在所有 Java 提交中击败了100.00%的用户
内存消耗：52.4 MB, 在所有 Java 提交中击败了37.98%的用户
```java
class Solution {
    class Trie {
        private TrieNode root;

        class TrieNode {
            TrieNode[] children = new TrieNode[26];
            String word;
        }

        Trie () {
            root = new TrieNode();
        }

        public void insert(String word) {
            TrieNode p = root;
            for (int i = 0; i < word.length(); ++i) {
                int idx = word.charAt(i) - 'a';
                if (p.children[idx] == null) p.children[idx] = new TrieNode();
                p = p.children[idx];
            }
            p.word = word;
        }

        public String getRoot(String word) {
            TrieNode p = root;
            for (int i = 0; i < word.length(); ++i) {
                int idx = word.charAt(i) - 'a';
                if (p.children[idx] != null) {
                    if (p.children[idx].word != null) return p.children[idx].word;
                } else {
                    return null;
                }
                p = p.children[idx];
            }
            return null;
        }
    }

    public String replaceWords(List<String> dictionary, String sentence) {
        Trie trie = new Trie();
        for (String root: dictionary) {
            trie.insert(root);
        }
        String[] sens = sentence.split(" ");
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < sens.length; ++i) {
            String x = trie.getRoot(sens[i]);
            if (x != null) {
                sb.append(x);
            } else {
                sb.append(sens[i]);
            }
            if (i < sens.length - 1) sb.append(" ");
        }
        return sb.toString();
    }
}
```

### 线段树（Segment Tree）

简化版

```java
public Node build(int[] a, int l, int r) {
    if (l == r)
        return new Node(a[l]);
    int mid = l + (r - l) / 2;
    Node ln = build(a, l, mid);
    Node rn = build(a, mid + 1, r);
    return pushUp(ln, rn);
}
```

### 树状数组

## 图

### 建图
#### 邻接矩阵

#### 邻接表

#### 边数组

#### 链式前向星
```java
// 无向图双向边，所以*2
static int N = 100010, M = N * 2;
static int[] head = new int[N], to = new int[M], next = new int[M];
int idx = 0;
void add(int f, int t) {
    to[idx] = t; // 本边的终止节点
    next[idx] = head[f]; // 本边指向之前的头边，形成新的头边（类似链表头插）
    head[f] = idx++; // 指针指向新头边
}
// 从编号0的点开始BFS，求从0到所有点的路径长度
static INF = 0x3f3f3f3f;
static int[] dist = new int[N];
void bfs() {
    Arrays.fill(head, -1);
    Arrays.fill(dist, INF);
    for (int[] e : edges) {
        // 无向图记得add两次
        add(e[0], e[1]);
        add(e[1], e[0]);
    }
    Deque<Integer> d = new ArrayDeque<>();
    d.addLast(0);
    dist[0] = 0;
    while (!d.isEmpty()) {
        int t = d.pollFirst();
        for (int i = head[t]; i != -1; i = next[i]) {
            int j = to[i];
            if (dist[j] != INF) continue;
            dist[j] = dist[t] + 1;
            d.addLast(j);
        }
    }
}
```

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

### 思考模板

- 状态定义：一维还是二维？
- 转移方程：有几种状态？状态之间依赖关系？
- 初始条件
- 结果表示
- 是否可以采用优化：滚动数组/滚动变量

### 子串遍历

检查回文串

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

### 状态搜索

### 贪心

## 单调栈

## 字符串
### KMP
```java
public class Kmp {
    /**
     * KMP 匹配
     */
    public static int kmp(String str, String pattern){
        // 计算前缀长度表
        int[] prefixLens = calPrefixLen(pattern);
        // 查找匹配位置
        for(int i = 0, j = 0; i < str.length(); i++){
            while(j > 0 && str.charAt(i) != pattern.charAt(j))
                j = prefixLens[j - 1];
            if(str.charAt(i) == pattern.charAt(j))
                j++;
            if(j == pattern.length())
                return i - j + 1;
        }
        return -1;
    }

    /**
     * 计算前缀长度表
     */
    public static int[] calPrefixLen(String pattern){
        int[] prefixLens = new int[pattern.length()];
        prefixLens[0] = 0;
        for(int i = 1, j = 0; i < pattern.length(); i++){
            while(j > 0 && pattern.charAt(j) != pattern.charAt(i))
                j = next[j - 1];
            if(pattern.charAt(i) == pattern.charAt(j))
                j++;
            prefixLens[i] = j;
        }
        return prefixLens;
    }
}
```

## 数学

### 最大公约数gcd

```java
public int gcd(int a, int b) { // 口诀bbaba
    return b > 0 ? gcd(b, a % b) : a;
}
```

### 最小公倍数lcm

### 快速幂

```java
public int pow(int x, int n) {
    int ans = 1;
    while (n > 0) {
        if ((n & 1) == 1) {
            ans *= x;
        }
        x *= x;
        n >>= 1;
    }
    return ans;
}
```

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

矩阵快速幂

![快速幂](https://i.loli.net/2021/09/04/XCav6lpUJgWyIe2.png)

```java
public int[][] pow(int[][] a, int n) {
    int[][] ret = {{1, 0}, {0, 1}};
    while (n > 0) {
        if ((n & 1) == 1) { // 幂二进制为1的时候才乘
            ret = multiply(ret, a);
        }
        n >>= 1;
        a = multiply(a, a); // 下一个2^x是当前的2次方
    }
    return ret;
}

public int[][] multiply(int[][] a, int[][] b) {
    int w = a.length;
    int[][] c = new int[w][w];
    for (int i = 0; i < w; i++) {
        for (int j = 0; j < w; j++) {
            c[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j];
        }
    }
    return c;
}
```

## 随机化

### 水塘抽样

https://zhuanlan.zhihu.com/p/107793995

# Java常用数据结构

**Queue VS Deque**

| `Queue` Method                                                                           | Equivalent `Deque` Method                                                                        |
| ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| [`add(e)`](https://docs.oracle.com/javase/10/docs/api/java/util/Deque.html#add(E))       | [`addLast(e)`](https://docs.oracle.com/javase/10/docs/api/java/util/Deque.html#addLast(E))       |
| [`offer(e)`](https://docs.oracle.com/javase/10/docs/api/java/util/Deque.html#offer(E))   | [`offerLast(e)`](https://docs.oracle.com/javase/10/docs/api/java/util/Deque.html#offerLast(E))   |
| [`remove()`](https://docs.oracle.com/javase/10/docs/api/java/util/Deque.html#remove())   | [`removeFirst()`](https://docs.oracle.com/javase/10/docs/api/java/util/Deque.html#removeFirst()) |
| [`poll()`](https://docs.oracle.com/javase/10/docs/api/java/util/Deque.html#poll())       | [`pollFirst()`](https://docs.oracle.com/javase/10/docs/api/java/util/Deque.html#pollFirst())     |
| [`element()`](https://docs.oracle.com/javase/10/docs/api/java/util/Deque.html#element()) | [`getFirst()`](https://docs.oracle.com/javase/10/docs/api/java/util/Deque.html#getFirst())       |
| [`peek()`](https://docs.oracle.com/javase/10/docs/api/java/util/Deque.html#peek())       | [`peekFirst()`](https://docs.oracle.com/javase/10/docs/api/java/util/Deque.html#peekFirst())     |

**Stack VS Deque**

| Stack Method                                                                         | Equivalent `Deque` Method                                                                        |
| ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------ |
| [`push(e)`](https://docs.oracle.com/javase/10/docs/api/java/util/Deque.html#push(E)) | [`addFirst(e)`](https://docs.oracle.com/javase/10/docs/api/java/util/Deque.html#addFirst(E))     |
| [`pop()`](https://docs.oracle.com/javase/10/docs/api/java/util/Deque.html#pop())     | [`removeFirst()`](https://docs.oracle.com/javase/10/docs/api/java/util/Deque.html#removeFirst()) |
| [`peek()`](https://docs.oracle.com/javase/10/docs/api/java/util/Deque.html#peek())   | [`peekFirst()`](https://docs.oracle.com/javase/10/docs/api/java/util/Deque.html#peekFirst())     |

## Queue 队列

- offer/add 添加尾部
- poll/remove 删除头部
- peek/element 查询头部
- size/isEmpty 容量

## Deque 双端队列

> Deque继承Queue，实现了ArrayDeque和LinkedList

![image-20210221013330334](https://i.loli.net/2021/02/21/kEis72yF9G3Q8HM.png)

- add/addAll/addFirst/addLast/offer/offerFirst/offerLast/push 添加
- remove/removeFirst/removeLast/poll/pollFirst/pollLast/pop 删除
- element/getFirst/getLast/peek/peekFirst/peekLast 获取查询
- contains 存在查询
- iterator/descendingIterator 迭代查询
- size 容量

ArrayDeque 双端队列数组实现（几乎没有容量限制，线程不安全，禁止null）

```java
Queue<Integer> queue = new ArrayDeque<>(); // 队列常用 直接offer和poll即可
Deque<Integer> deque = new ArrayDeque<>(); // 作为栈：push开头添加，pop开头删除，peek开头返回；作为双端队列：用offerXXX和pollXXX
```

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

哈希表的遍历

```java
for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
    Integer val = entry.getValue();
    // ...
}
```

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

# Java常见技巧

## 数据转化类

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

- `Set<String>`转`String[]`

```java
String[] string = set.toArray(new String[set.size()]);
```

- `String[]` 转 `Set<String>`

```java
Set<String> set = new HashSet<String>(Arrays.asList(strings));
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

- `Map.values`转`List`

```java
ArrayList<Object> list = new ArrayList(resultMap.values());
```

- `Map`转`Set`

```java
map.keySet();
map.entrySet();
```

- `Set`转`Map`
遍历添加暂无简洁方法

- `List`转`Set`

```java
HashSet<String> set = new HashSet<String>(list);
```

- `Set`转`List`

```java
List<T> list = new ArrayList<>();
list.addAll(set);
```

## 字符串类

- 判断字符是否为字母或者数字

```java
Character.isLetterOrDigit(c) 两者
Character.isLetter(c) 字母
Character.isDigit(c) 数字
```

- 判断字符是否为空白符
```java
Character.isWhitespace(c)
```

- 字符转化为大小写

```java
Character.toLowerCase(c) / Character.toUpperCase(c)
```

- 字符判断大小写

```java
Character.isLowerCase(c) / Character.isUpperCase(c)
```

- 修改字符串某一位

```java
StringBuilder sb = new StringBuilder(str);
sb.setCharAt(1, 'a');
```

- 比较数组是否相等
```java
Arrays.equals(arr1, arr2);
```

## 数组类

+ 数组初始化：Arrays.fill(arr, Integer.MAX_VALUE)

+ 数组排序
  
  ```java
  Arrays.sort(A); // 如果是字符数组，则是先大写后小写，用的是双轴快排
  Arrays.sort(A, Collections.reverseOrder()); // 逆序，必须是Integer[]，或者从后往前处理
  Arrays.sort(A, String.CASE_INSENSITIVE_ORDER); // 字符串排序，忽略大小写
  // 对于基本类型的数组如int[]/double[]/char[]，Arrays类只提供了默认的升序排列，没有降序
  int[] newA = Arrays.stream(A).boxed().sorted((a, b) -> b - a).mapToInt(p -> p).toArray();
  // Integer[]逆序
  Arrays.sort(A, new Comparator<Integer>(){
      public int compare(Integer a, Integer b){
          return b - a;
      }
  });
  Arrays.sort(A, (a, b) -> b - a);
  // 给某区间按c比较器（可选）排序
  Arrays.sort(A, fromIndex, toIndex, c);
  // 按字符串长度排序
  Arrays.sort(A, (a, b) -> Integer.signum(a.length() - b.length()));
  Arrays.sort(A, Comparator.comparingInt(String::length));
  Arrays.sort(A, (a, b) -> a.length() - b.length());
  Arrays.sort(A, (String a, String b) -> { return Integer.signum(a.length() - b.length())});
  // 二维数组排序
  Arrays.sort(AA, new Comparator<?>() {
      @Override
      private int Compare(int[] a, int[] b) {
          return a[0] > b[0] ? -1: 1;
      }
  })
  ```
- 数组相等
  
  ```java
  Arrays.equals(arr1, arr2)
  ```

- 数组复制
  
  ```java
  Arrays.copyOfRange(nums, 0, k)
  System.arraycopy(Object src【原数组】, int srcPos【原数组开始位置】, Object dest【目标数组】, int destPos【目标数组开始位置】, int length【拷贝长度】);
  ```

- 数组求和、找最大、找最小（效率不高）
  
  ```java
  int max = Arrays.stream(arr).max().getAsInt();
  int min = Arrays.stream(arr).min().getAsInt();
  int total = Arrays.stream(arr).sum()
  ```

- 数组打印
  
  ```java
  Object[] arr;
  for (int i = 0; i < arr.length; i++)
      System.out.print(arr[i] + ", "); 
  for(Object item: arr) 
      System.out.println(item + ", ");
  System.out.println(Arrays.toString(arr));
  System.out.println(Arrays.asList(arr));
  Arrays.asList(arr).stream().forEach(item -> System.out.println(item));
  ```

## 集合类

+ ArrayList简单构造：Arrays.asList(a,b)

+ 创建哑节点 `dummyHead`，令 `dummyHead.next = head`。引入哑节点是为了便于在 `head` 节点之前插入节点。

+ 计数哈希表构建：map.put(val, map.getOrDefault(val, 0) + 1);

+ 哈希表（复杂value）获取key并操作，eg：
  
  ```java
  Map<Integer, PriorityQueue<Character>> map = new HashMap<>();
  map.computeIfAbsent(key, key -> new PriorityQueue<>()).offer(ch);
  ```

+ 大顶堆：new PriorityQueue<>(Collections.reverseOrder());或者new PriorityQueue<>((x,y)->y-x);
- 复杂Map遍历
  
  ```java
  // 例如Map<Integer, int[]>
  for (Map.Entry<Integer, int[]> entry : map.entrySet()) {
      int[] arr = entry.getValue();
      //...
  }
  ```

- List求和（流）
  
  ```java
  long sum = list.stream().mapToLong(User::getAge).sum(); // List<User>
  long sum = list.stream().reduce(Integer::sum).orElse(0); // List<Integer>
  ```

- 栈和队列
  
  ```java
  // 栈（DFS非递归常用）
  Deque<T> stack = new LinkedList<>(); // push/pop/peek
  // 队列（BFS常用）
  Queue<T> q = new LinkedList<>(); // offer/poll
  // 通用
  LinkedList<T> l = new LinkedList<>(); // +first/+last
  ```

## 位运算类

- 判断奇偶：(x & 1) == 1, (x & 1) == 0

- 求模：x & (y - 1) <=> x % y

- 位移代替除法：x >> 1 <=> x / 2

- 最低位的1变成0：x &= (x - 1)

- 得到最低位的1的值(lowbit)，形如100...0：**x & -x**（常用）; x & (x ^ (x - 1))【常应用于树状数组，状压DP，二进制或位运算题中】

- 0和1翻转：n ^= 1

- 判断是否全1：(n & (n + 1)) == 0

- x & ~x <=> 0

- 指定位置的位运算
    将X最右边的n位清零：x & (~0 << n)
    获取x的第n位值：(x >> n) & 1
    获取x的第n位的幂值：x & (1 << n)
    仅将第n位置为1：x | (1 << n)
    仅将第n位置为0：x & (~(1 << n))
    将x最高位至第n位（含）清零：x & ((1 << n) - 1)
    将第n位至第0位（含）清零：x & (~((1 << (n + 1)) - 1))

- 异或结合律
  
    (类似点乘法结合律)
  
    x ^ 0 = x, x ^ x = 0
    x ^ (~0) = ~x, x ^ (~x) = ~0
    a ^ b = c, a ^ c = b, b ^ c = a
  
    字母表示：(a ^ b) ^ c = a ^ (b ^ c)
    图形表示：(☆ ^ ◇) ^ △ = ☆ ^ (◇ ^ △)

- 大小字母位运算技巧
  
    大写变小写、小写变大写：字符 ^= 32 （大写 ^= 32 相当于 +32，小写 ^= 32 相当于 -32）
    大写变小写、小写变小写：字符 |= 32 （大写 |= 32 就相当于+32，小写 |= 32 不变）
    大写变大写、小写变大写：字符 &= -33 （大写 ^= -33 不变，小写 ^= -33 相当于 -32）

- int整数表示字符串字母集合（不考虑char出现的频数）：对每个ch计算mask |= (1 << (ch - 'a'))

- int的bit数
  
  ```java
  public int countOnes(int x) {
      int ones = 0;
      while (x > 0) {
          x &= (x - 1); // 法1：最低位的1变成0
          // x -= x & -x; // 法2：减去得到最低位的1的值
          ones++;
      }
      return ones;
  }
  // 或者
  Integer.bitCount(x)
  ```

- [位运算的奇技淫巧（二） - RioTian - 博客园](https://www.cnblogs.com/RioTian/p/13598747.html)

- [Integer常用函数和位运算技巧](https://blog.csdn.net/youyou1543724847/article/details/52385775)
  
  ```java
  前导0计数:Integer.numberOfLeadingZeros
  后缀0计数:Integer.numberOfTailingZeros（最后一个1的位置）
  ```

- 二进制子集遍历
  
  ```java
  for (int x = mask; x != 0; x = (x - 1) & mask) {
      System.out.println(x);
  }
  ```

## 流类

- 求ArrayList的和

```java
// List<Integer>类型
list.stream().reduce(Integer::sum).orElse(0);
// List<User>类型
list.stream().mapToInt(User::getScore).sum();
```

## 其他类
- 获取随机数
  
  ```java
  int x = (int)(Math.random() * total) + 1;
  ```

- Math计算判断整数
  
  ```java
  Math.abs(x - Math.round(x)) < Math.pow(10, -14);
  ```

- 求模防止溢出（X是一个可能溢出的计算式）
  
  ```java
  (int)((long)X % MOD)
  ```

# 未完成

## [493. 翻转对](https://leetcode-cn.com/problems/reverse-pairs/)

## [147. 对链表进行插入排序](https://leetcode-cn.com/problems/insertion-sort-list/)

## [135. 分发糖果](https://leetcode-cn.com/problems/candy/)

## [803. 打砖块](https://leetcode.cn/problems/bricks-falling-when-hit/)

## [1489. 找到最小生成树里的关键边和伪关键边](https://leetcode-cn.com/problems/bricks-falling-when-hit/)

## [480. 滑动窗口中位数](https://leetcode-cn.com/problems/sliding-window-median/)【重点看看】

## [354. 俄罗斯套娃信封问题](https://leetcode-cn.com/problems/russian-doll-envelopes/)【重点看看】

## [1006. 笨阶乘](https://leetcode-cn.com/problems/clumsy-factorial/)

## [87. 扰乱字符串](https://leetcode-cn.com/problems/scramble-string/)

## [28. 实现 strStr()](https://leetcode-cn.com/problems/implement-strstr/)【重点看看】

## [1473. 粉刷房子 III](https://leetcode-cn.com/problems/paint-house-iii/)

## [1707. 与数组中元素的最大异或值](https://leetcode-cn.com/problems/maximum-xor-with-an-element-from-array/)

## [879. 盈利计划](https://leetcode-cn.com/problems/profitable-schemes/)

## [218. 天际线问题](https://leetcode-cn.com/problems/the-skyline-problem/)

## [1818. 绝对差值和](https://leetcode-cn.com/problems/minimum-absolute-sum-difference/)

## [1846. 减小和重新排列数组后的最大元素](https://leetcode-cn.com/problems/maximum-element-after-decreasing-and-rearranging/)

## [1713. 得到子序列的最少操作次数](https://leetcode-cn.com/problems/minimum-operations-to-make-a-subsequence/)

## [987. 二叉树的垂序遍历](https://leetcode-cn.com/problems/vertical-order-traversal-of-a-binary-tree/)

## [743. 网络延迟时间](https://leetcode-cn.com/problems/network-delay-time/)（重要）

## [611. 有效三角形的个数](https://leetcode-cn.com/problems/valid-triangle-number/)

## [802. 找到最终的安全状态](https://leetcode-cn.com/problems/find-eventual-safe-states/)

## [847. 访问所有节点的最短路径](https://leetcode-cn.com/problems/shortest-path-visiting-all-nodes/)

## [457. 环形数组是否存在循环](https://leetcode-cn.com/problems/circular-array-loop/)

## [313. 超级丑数](https://leetcode-cn.com/problems/super-ugly-number/)

## [233. 数字 1 的个数](https://leetcode-cn.com/problems/number-of-digit-one/)

## [552. 学生出勤记录 II](https://leetcode-cn.com/problems/student-attendance-record-ii/)

## [502. IPO](https://leetcode-cn.com/problems/ipo/)

## [352. 将数据流变为多个不相交区间](https://leetcode-cn.com/problems/data-stream-as-disjoint-intervals/)

## [282. 给表达式添加运算符](https://leetcode-cn.com/problems/expression-add-operators/)

## [638. 大礼包](https://leetcode-cn.com/problems/shopping-offers/)

## [240. 搜索二维矩阵 II](https://leetcode-cn.com/problems/search-a-2d-matrix-ii/)

## [488. 祖玛游戏](https://leetcode-cn.com/problems/zuma-game/)

## [629. K个逆序对数组](https://leetcode-cn.com/problems/k-inverse-pairs-array/)

## [677. 键值映射](https://leetcode-cn.com/problems/map-sum-pairs/)

## [391. 完美矩形](https://leetcode-cn.com/problems/perfect-rectangle/)

## [630. 课程表 III](https://leetcode-cn.com/problems/course-schedule-iii/)

## [851. 喧闹和富有](https://leetcode-cn.com/problems/loud-and-rich/)

## [475. 供暖器](https://leetcode-cn.com/problems/heaters/)

## [1044. 最长重复子串](https://leetcode-cn.com/problems/longest-duplicate-substring/)

## [913. 猫和老鼠](https://leetcode-cn.com/problems/cat-and-mouse/)

## [1036. 逃离大迷宫](https://leetcode-cn.com/problems/escape-a-large-maze/)

## [373. 查找和最小的 K 对数字](https://leetcode-cn.com/problems/find-k-pairs-with-smallest-sums/)

## [1220. 统计元音字母序列的数目](https://leetcode-cn.com/problems/count-vowels-permutation/)

## [2029. 石子游戏 IX](https://leetcode-cn.com/problems/stone-game-ix/)

## [跳跃游戏 IV](https://leetcode-cn.com/problems/jump-game-iv/solution/tiao-yue-you-xi-iv-by-leetcode-solution-zsix/)

## [2045. 到达目的地的第二短时间](https://leetcode-cn.com/problems/second-minimum-time-to-reach-destination/)

## [1996. 游戏中弱角色的数量](https://leetcode-cn.com/problems/the-number-of-weak-characters-in-the-game/)

## [1765. 地图中的最高点](https://leetcode-cn.com/problems/map-of-highest-peak/)

## [1414. 和为 K 的最少斐波那契数字数目](https://leetcode-cn.com/problems/find-the-minimum-number-of-fibonacci-numbers-whose-sum-is-k/)

## [1219. 黄金矿工](https://leetcode-cn.com/problems/path-with-maximum-gold/)

## [1001. 网格照明](https://leetcode-cn.com/problems/grid-illumination/)

## [1719. 重构一棵树的方案数](https://leetcode-cn.com/problems/number-of-ways-to-reconstruct-a-tree/)

## [838. 推多米诺](https://leetcode-cn.com/problems/push-dominoes/)

## [1994. 好子集的数目](https://leetcode-cn.com/problems/the-number-of-good-subsets/)

## [1601. 最多可达成的换楼请求数目](https://leetcode-cn.com/problems/maximum-number-of-achievable-transfer-requests/)

## [6. Z 字形变换](https://leetcode-cn.com/problems/zigzag-conversion/)

## [564. 寻找最近的回文数](https://leetcode-cn.com/problems/find-the-closest-palindrome/)

## [2104. 子数组范围和](https://leetcode-cn.com/problems/sum-of-subarray-ranges/)

## [2049. 统计最高分的节点数目](https://leetcode-cn.com/problems/count-nodes-with-the-highest-score/)

## [432. 全 O(1) 的数据结构](https://leetcode-cn.com/problems/all-oone-data-structure/)

## [1606. 找到处理最多请求的服务器](https://leetcode-cn.com/problems/find-servers-that-handled-most-number-of-requests/)

## [420. 强密码检验器](https://leetcode-cn.com/problems/strong-password-checker/)

## [307. 区域和检索 - 数组可修改](https://leetcode-cn.com/problems/range-sum-query-mutable/)

## [310. 最小高度树](https://leetcode-cn.com/problems/minimum-height-trees/)

## [388. 文件的最长绝对路径](https://leetcode-cn.com/problems/longest-absolute-file-path/)

## [396. 旋转函数](https://leetcode-cn.com/problems/rotate-function/)

## [587. 安装栅栏](https://leetcode-cn.com/problems/erect-the-fence/)

## [691. 贴纸拼词](https://leetcode.cn/problems/stickers-to-spell-word/)

## [675. 为高尔夫比赛砍树](https://leetcode.cn/problems/cut-off-trees-for-golf-event/)

## [剑指 Offer II 114. 外星文字典](https://leetcode.cn/problems/Jf1JuT/)

## [473. 火柴拼正方形](https://leetcode.cn/problems/matchsticks-to-square/)

## [732. 我的日程安排表 III](https://leetcode.cn/problems/my-calendar-iii/)

## [715. Range 模块](https://leetcode.cn/problems/range-module/)

## [30. 串联所有单词的子串](https://leetcode.cn/problems/substring-with-concatenation-of-all-words/)

## [324. 摆动排序 II](https://leetcode.cn/problems/wiggle-sort-ii/)

## [871. 最低加油次数](https://leetcode.cn/problems/minimum-number-of-refueling-stops/)

## [729. 我的日程安排表 I](https://leetcode.cn/problems/my-calendar-i/) - 线段树

## [741. 摘樱桃](https://leetcode.cn/problems/cherry-pickup/)

## [676. 实现一个魔法字典](https://leetcode.cn/problems/implement-magic-dictionary/)

## [745. 前缀和后缀搜索](https://leetcode.cn/problems/prefix-and-suffix-search/)

## [558. 四叉树交集](https://leetcode.cn/problems/logical-or-of-two-binary-grids-represented-as-quad-trees/)

## [749. 隔离病毒](https://leetcode.cn/problems/contain-virus/)

## [731. 我的日程安排表 II](https://leetcode.cn/problems/my-calendar-ii/)

## [757. 设置交集大小至少为2](https://leetcode.cn/problems/set-intersection-size-at-least-two/)

## [814. 二叉树剪枝](https://leetcode.cn/problems/binary-tree-pruning/)

## [剑指 Offer II 115. 重建序列](https://leetcode.cn/problems/ur2n8P/)

## [1206. 设计跳表](https://leetcode.cn/problems/design-skiplist/)

## [952. 按公因数计算最大组件大小](https://leetcode.cn/problems/largest-component-size-by-common-factor/)

## [622. 设计循环队列](https://leetcode.cn/problems/design-circular-queue/)

## [899. 有序队列](https://leetcode.cn/problems/orderly-queue/)

## [623. 在二叉树中增加一行](https://leetcode.cn/problems/add-one-row-to-tree/)

## [636. 函数的独占时间](https://leetcode.cn/problems/exclusive-time-of-functions/)

## [761. 特殊的二进制序列](https://leetcode.cn/problems/special-binary-string/)

## [640. 求解方程](https://leetcode.cn/problems/solve-the-equation/)

## [1282. 用户分组](https://leetcode.cn/problems/group-the-people-given-the-group-size-they-belong-to/)

## [768. 最多能完成排序的块 II](https://leetcode.cn/problems/max-chunks-to-make-sorted-ii/)

## [1224. 最大相等频率](https://leetcode.cn/problems/maximum-equal-frequency/)

## [654. 最大二叉树](https://leetcode.cn/problems/maximum-binary-tree/)

## [782. 变为棋盘](https://leetcode.cn/problems/transform-to-chessboard/)

## [658. 找到 K 个最接近的元素](https://leetcode.cn/problems/find-k-closest-elements/)

## [662. 二叉树最大宽度](https://leetcode.cn/problems/maximum-width-of-binary-tree/)

## [793. 阶乘函数后 K 个零](https://leetcode.cn/problems/preimage-size-of-factorial-zeroes-function/)

## [998. 最大二叉树 II](https://leetcode.cn/problems/maximum-binary-tree-ii/)

## [1582. 二进制矩阵中的特殊位置](https://leetcode.cn/problems/special-positions-in-a-binary-matrix/)

## [652. 寻找重复的子树](https://leetcode.cn/problems/find-duplicate-subtrees/)

## [828. 统计子串中的唯一字符](https://leetcode.cn/problems/count-unique-characters-of-all-substrings-of-a-given-string/)

## [857. 雇佣 K 名工人的最低成本](https://leetcode.cn/problems/minimum-cost-to-hire-k-workers/)

## [672. 灯泡开关 Ⅱ](https://leetcode.cn/problems/bulb-switcher-ii/)

## [850. 矩形面积 II](https://leetcode.cn/problems/rectangle-area-ii/)

## [827. 最大人工岛](https://leetcode.cn/problems/making-a-large-island/)

## [698. 划分为k个相等的子集](https://leetcode.cn/problems/partition-to-k-equal-sum-subsets/)

## [854. 相似度为 K 的字符串](https://leetcode.cn/problems/k-similar-strings/)

## [707. 设计链表](https://leetcode.cn/problems/design-linked-list/)

## [面试题 17.09. 第 k 个数](https://leetcode.cn/problems/get-kth-magic-number-lcci/)

## [面试题 01.08. 零矩阵](https://leetcode.cn/problems/zero-matrix-lcci/)

## [777. 在LR字符串中交换相邻字符](https://leetcode.cn/problems/swap-adjacent-in-lr-string/)

## [927. 三等分](https://leetcode.cn/problems/three-equal-parts/)

## [870. 优势洗牌](https://leetcode.cn/problems/advantage-shuffle/)

## [801. 使序列递增的最小交换次数](https://leetcode.cn/problems/minimum-swaps-to-make-sequences-increasing/)

## [817. 链表组件](https://leetcode.cn/problems/linked-list-components/)

## [940. 不同的子序列 II](https://leetcode.cn/problems/distinct-subsequences-ii/)

## [886. 可能的二分法](https://leetcode.cn/problems/possible-bipartition/)

## [904. 水果成篮](https://leetcode.cn/problems/fruit-into-baskets/)

## [902. 最大为 N 的数字组合](https://leetcode.cn/problems/numbers-at-most-n-given-digit-set/)

## [779. 第K个语法符号](https://leetcode.cn/problems/k-th-symbol-in-grammar/)

## [901. 股票价格跨度](https://leetcode.cn/problems/online-stock-span/)

## [1235. 规划兼职工作](https://leetcode.cn/problems/maximum-profit-in-job-scheduling/)

## [934. 最短的桥](https://leetcode.cn/problems/shortest-bridge/)

## [862. 和至少为 K 的最短子数组](https://leetcode.cn/problems/shortest-subarray-with-sum-at-least-k/)

## [907. 子数组的最小值之和](https://leetcode.cn/problems/sum-of-subarray-minimums/)

## [784. 字母大小写全排列](https://leetcode.cn/problems/letter-case-permutation/)

## [481. 神奇字符串](https://leetcode.cn/problems/magical-string/)

## [1620. 网络信号最好的坐标](https://leetcode.cn/problems/coordinate-with-maximum-network-quality/submissions/)

## [1668. 最大重复子字符串](https://leetcode.cn/problems/maximum-repeating-substring/)

## [754. 到达终点数字](https://leetcode.cn/problems/reach-a-number/)

## [1106. 解析布尔表达式](https://leetcode.cn/problems/parsing-a-boolean-expression/)

## [816. 模糊坐标](https://leetcode.cn/problems/ambiguous-coordinates/)

## [764. 最大加号标志](https://leetcode.cn/problems/largest-plus-sign/)

## [864. 获取所有钥匙的最短路径](https://leetcode.cn/problems/shortest-path-to-get-all-keys/)

## [790. 多米诺和托米诺平铺](https://leetcode.cn/problems/domino-and-tromino-tiling/)

## [805. 数组的均值分割](https://leetcode.cn/problems/split-array-with-same-average/)

## [792. 匹配子序列的单词数](https://leetcode.cn/problems/number-of-matching-subsequences/)

## [891. 子序列宽度之和](https://leetcode.cn/problems/sum-of-subsequence-widths/)

## [799. 香槟塔](https://leetcode.cn/problems/champagne-tower/)

## [808. 分汤](https://leetcode.cn/problems/soup-servings/)

## [878. 第 N 个神奇数字](https://leetcode.cn/problems/nth-magical-number/)

## [795. 区间子数组个数](https://leetcode.cn/problems/number-of-subarrays-with-bounded-maximum/)

## [809. 情感丰富的文字](https://leetcode.cn/problems/expressive-words/)

## [882. 细分图中的可到达节点](https://leetcode.cn/problems/reachable-nodes-in-subdivided-graph/)

## [813. 最大平均值和的分组](https://leetcode.cn/problems/largest-sum-of-averages/)

## [895. 最大频率栈](https://leetcode.cn/problems/maximum-frequency-stack/)

## [1774. 最接近目标价格的甜点成本](https://leetcode.cn/problems/closest-dessert-cost/)

## [1687. 从仓库到码头运输箱子](https://leetcode.cn/problems/delivering-boxes-from-storage-to-ports/)

## [1775. 通过最少操作次数使数组的和相等](https://leetcode.cn/problems/equal-sum-arrays-with-minimum-number-of-operations/)

## [1691. 堆叠长方体的最大高度](https://leetcode.cn/problems/maximum-height-by-stacking-cuboids/)

## [1781. 所有子字符串美丽值之和](https://leetcode.cn/problems/sum-of-beauty-of-all-substrings/)

## [1697. 检查边长度限制的路径是否存在](https://leetcode.cn/problems/checking-existence-of-edge-length-limited-paths/)

## [1785. 构成特定和需要添加的最少元素](https://leetcode.cn/problems/minimum-elements-to-add-to-form-a-given-sum/)

## [1764. 通过连接另一个数组的子数组得到一个数组](https://leetcode.cn/problems/form-array-by-concatenating-subarrays-of-another-array/)

## [1703. 得到连续 K 个 1 的最少相邻交换次数](https://leetcode.cn/problems/minimum-adjacent-swaps-for-k-consecutive-ones/)

## [1971. 寻找图中是否存在路径](https://leetcode.cn/problems/find-if-path-exists-in-graph/)

## [1760. 袋子里最少数目的球](https://leetcode.cn/problems/minimum-limit-of-balls-in-a-bag/)

## [1799. N 次操作后的最大分数和](https://leetcode.cn/problems/maximize-score-after-n-operations/)

## [1754. 构造字典序最大的合并字符串](https://leetcode.cn/problems/largest-merge-of-two-strings/submissions/)

## [1739. 放置盒子](https://leetcode.cn/problems/building-boxes/)

## [1759. 统计同构子字符串的数目](https://leetcode.cn/problems/count-number-of-homogenous-substrings/)

## [1750. 删除字符串两端相同字符后的最短长度](https://leetcode.cn/problems/minimum-length-of-string-after-deleting-similar-ends/)

## [2032. 至少在两个数组中出现的值](https://leetcode.cn/problems/two-out-of-three/)

## [855. 考场就座](https://leetcode.cn/problems/exam-room/)

## [1801. 积压订单中的订单总数](https://leetcode.cn/problems/number-of-orders-in-the-backlog/)

## [1802. 有界数组中指定下标处的最大值](https://leetcode.cn/problems/maximum-value-at-a-given-index-in-a-bounded-array/)

## [1803. 统计异或值在范围内的数对有多少](https://leetcode.cn/problems/count-pairs-with-xor-in-a-range/)

## [2180. 统计各位数字之和为偶数的整数个数](https://leetcode.cn/problems/count-integers-with-even-digit-sum/)

## [1658. 将 x 减到 0 的最小操作数](https://leetcode.cn/problems/minimum-operations-to-reduce-x-to-zero/)

## [1806. 还原排列的最少操作步数](https://leetcode.cn/problems/minimum-number-of-operations-to-reinitialize-a-permutation/)

## [753. 破解保险箱](https://leetcode.cn/problems/cracking-the-safe/)

## [1807. 替换字符串中的括号内容](https://leetcode.cn/problems/evaluate-the-bracket-pairs-of-a-string/)

## [2287. 重排字符形成目标字符串](https://leetcode.cn/problems/rearrange-characters-to-make-target-string/)

## [1819. 序列中不同最大公约数的数目](https://leetcode.cn/problems/number-of-different-subsequences-gcds/)

## [1813. 句子相似性 III](https://leetcode.cn/problems/sentence-similarity-iii/)

## [1814. 统计一个数组中好对子的数目](https://leetcode.cn/problems/count-nice-pairs-in-an-array/)

## [1825. 求出 MK 平均值](https://leetcode.cn/problems/finding-mk-average/)

## [1824. 最少侧跳次数](https://leetcode.cn/problems/minimum-sideway-jumps/)

## [1815. 得到新鲜甜甜圈的最多组数](https://leetcode.cn/problems/maximum-number-of-groups-getting-fresh-donuts/)

## [1828. 统计一个圆中点的数目](https://leetcode.cn/problems/queries-on-number-of-points-inside-a-circle/)

## [1632. 矩阵转换后的秩](https://leetcode.cn/problems/rank-transform-of-a-matrix/)

## [1663. 具有给定数值的最小字符串](https://leetcode.cn/problems/smallest-string-with-a-given-numeric-value/)

## [1664. 生成平衡数组的方案数](https://leetcode.cn/problems/ways-to-make-a-fair-array/)

## [2325. 解密消息](https://leetcode.cn/problems/decode-the-message/)

## [1129. 颜色交替的最短路径](https://leetcode.cn/problems/shortest-path-with-alternating-colors/)

## [1145. 二叉树着色游戏](https://leetcode.cn/problems/binary-tree-coloring-game/)

## [1798. 你能构造出连续值的最大数目](https://leetcode.cn/problems/maximum-number-of-consecutive-values-you-can-make/)

## [1210. 穿过迷宫的最少移动次数](https://leetcode.cn/problems/minimum-moves-to-reach-target-with-rotations/)

## [1604. 警告一小时内使用相同员工卡大于等于三次的人](https://leetcode.cn/problems/alert-using-same-key-card-three-or-more-times-in-a-one-hour-period/)

## [1233. 删除子文件夹](https://leetcode.cn/problems/remove-sub-folders-from-the-filesystem/)

## [1797. 设计一个验证系统](https://leetcode.cn/problems/design-authentication-manager/)

## [1223. 掷骰子模拟](https://leetcode.cn/problems/dice-roll-simulation/)

## [1138. 字母板上的路径](https://leetcode.cn/problems/alphabet-board-path/)

## [1234. 替换子串得到平衡字符串](https://leetcode.cn/problems/replace-the-substring-for-balanced-string/)

## [1124. 表现良好的最长时间段](https://leetcode.cn/problems/longest-well-performing-interval/)

## [1250. 检查「好数组」](https://leetcode.cn/problems/check-if-it-is-a-good-array/)

## [1139. 最大的以 1 为边界的正方形](https://leetcode.cn/problems/largest-1-bordered-square/)

## [1237. 找出给定方程的正整数解](https://leetcode.cn/problems/find-positive-integer-solution-for-a-given-equation/)

## [1792. 最大平均通过率](https://leetcode.cn/problems/maximum-average-pass-ratio/)

## [1326. 灌溉花园的最少水龙头数目](https://leetcode.cn/problems/minimum-number-of-taps-to-open-to-water-a-garden/)

## [1140. 石子游戏 II](https://leetcode.cn/problems/stone-game-ii/)

## [1238. 循环码排列](https://leetcode.cn/problems/circular-permutation-in-binary-representation/)

## [1247. 交换字符使得字符串相同](https://leetcode.cn/problems/minimum-swaps-to-make-strings-equal/)

## [1255. 得分最高的单词集合](https://leetcode.cn/problems/maximum-score-words-formed-by-letters/)

## [1144. 递减元素使数组呈锯齿状](https://leetcode.cn/problems/decrease-elements-to-make-array-zigzag/)

## [面试题 05.02. 二进制数转字符串](https://leetcode.cn/problems/bianry-number-to-string-lcci/)

## [1487. 保证文件名唯一](https://leetcode.cn/problems/making-file-names-unique/)

## [982. 按位与为零的三元组](https://leetcode.cn/problems/triples-with-bitwise-and-equal-to-zero/)

## [1599. 经营摩天轮的最大利润](https://leetcode.cn/problems/maximum-profit-of-operating-a-centennial-wheel/)

## [1653. 使字符串平衡的最少删除次数](https://leetcode.cn/problems/minimum-deletions-to-make-string-balanced/)

## [1096. 花括号展开 II](https://leetcode.cn/problems/brace-expansion-ii/)

## [剑指 Offer 47. 礼物的最大价值](https://leetcode.cn/problems/li-wu-de-zui-da-jie-zhi-lcof/)

## [1590. 使数组和能被 P 整除](https://leetcode.cn/problems/make-sum-divisible-by-p/)

## [面试题 17.05.  字母与数字](https://leetcode.cn/problems/find-longest-subarray-lcci/)

## [1617. 统计子树中城市之间最大距离](https://leetcode.cn/problems/count-subtrees-with-max-distance-between-cities/)

## [1605. 给定行和列的和求可行矩阵](https://leetcode.cn/problems/find-valid-matrix-given-row-and-column-sums/)

## [1615. 最大网络秩](https://leetcode.cn/problems/maximal-network-rank/)

## [2488. 统计中位数为 K 的子数组](https://leetcode.cn/problems/count-subarrays-with-median-k/)

## [1616. 分割两个字符串得到回文串](https://leetcode.cn/problems/split-two-strings-to-make-palindrome/)

## [1625. 执行操作后字典序最小的字符串](https://leetcode.cn/problems/lexicographically-smallest-string-after-applying-operations/)

## [1012. 至少有 1 位重复的数字](https://leetcode.cn/problems/numbers-with-repeated-digits/)

## [2469. 温度转换](https://leetcode.cn/problems/convert-the-temperature/)

## [1626. 无矛盾的最佳球队](https://leetcode.cn/problems/best-team-with-no-conflicts/)

## [1630. 等差子数组](https://leetcode.cn/problems/arithmetic-subarrays/)

## [1032. 字符流](https://leetcode.cn/problems/stream-of-characters/)

## [1574. 删除最短的子数组使剩余数组有序](https://leetcode.cn/problems/shortest-subarray-to-be-removed-to-make-array-sorted/)

## [1638. 统计只差一个字符的子串数目](https://leetcode.cn/problems/count-substrings-that-differ-by-one-character/)

## [1092. 最短公共超序列](https://leetcode.cn/problems/shortest-common-supersequence/)

## [1641. 统计字典序元音字符串的数目](https://leetcode.cn/problems/count-sorted-vowel-strings/)

## [1637. 两点之间不包含任何点的最宽垂直区域](https://leetcode.cn/problems/widest-vertical-area-between-two-points-containing-no-points/)

## [831. 隐藏个人信息](https://leetcode.cn/problems/masking-personal-information/)

## [1039. 多边形三角剖分的最低得分](https://leetcode.cn/problems/minimum-score-triangulation-of-polygon/)

## [1053. 交换一次的先前排列](https://leetcode.cn/problems/previous-permutation-with-one-swap/)

## [1000. 合并石头的最低成本](https://leetcode.cn/problems/minimum-cost-to-merge-stones/)

## [2427. 公因子的数目](https://leetcode.cn/problems/number-of-common-factors/)

## [1017. 负二进制转换](https://leetcode.cn/problems/convert-to-base-2/submissions/)

## [1125. 最小的必要团队](https://leetcode.cn/problems/smallest-sufficient-team/)

## [1019. 链表中的下一个更大节点](https://leetcode.cn/problems/next-greater-node-in-linked-list/)

## [1147. 段式回文](https://leetcode.cn/problems/longest-chunked-palindrome-decomposition/)

## [1023. 驼峰式匹配](https://leetcode.cn/problems/camelcase-matching/)

## [1042. 不邻接植花](https://leetcode.cn/problems/flower-planting-with-no-adjacent/)

## [1157. 子数组中占绝大多数的元素](https://leetcode.cn/problems/online-majority-element-in-subarray/)

## [2409. 统计共同度过的日子数](https://leetcode.cn/problems/count-days-spent-together/)

## [1026. 节点与其祖先之间的最大差值](https://leetcode.cn/problems/maximum-difference-between-node-and-ancestor/)

## [1043. 分隔数组以得到最大和](https://leetcode.cn/problems/partition-array-for-maximum-sum/)

## [1187. 使数组严格递增](https://leetcode.cn/problems/make-array-strictly-increasing/)

## [1027. 最长等差数列](https://leetcode.cn/problems/longest-arithmetic-subsequence/)

## [1105. 填充书架](https://leetcode.cn/problems/filling-bookcase-shelves/)

## [1163. 按字典序排在最后的子串](https://leetcode.cn/problems/last-substring-in-lexicographical-order/)

## [1031. 两个非重叠子数组的最大和](https://leetcode.cn/problems/maximum-sum-of-two-non-overlapping-subarrays/)

## [1048. 最长字符串链](https://leetcode.cn/problems/longest-string-chain/)

## [1172. 餐盘栈](https://leetcode.cn/problems/dinner-plate-stacks/)

## [2423. 删除字符使频率相同](https://leetcode.cn/problems/remove-letter-to-equalize-frequency/)

## [1033. 移动石子直到连续](https://leetcode.cn/problems/moving-stones-until-consecutive/)

## [1376. 通知所有员工所需的时间](https://leetcode.cn/problems/time-needed-to-inform-all-employees/)

## [970. 强整数](https://leetcode.cn/problems/powerful-integers/)

## [1003. 检查替换后的词是否有效](https://leetcode.cn/problems/check-if-word-is-valid-after-substitutions/)

## [2106. 摘水果](https://leetcode.cn/problems/maximum-fruits-harvested-after-at-most-k-steps/)

## [1419. 数青蛙](https://leetcode.cn/problems/minimum-number-of-frogs-croaking/)

## [1010. 总持续时间可被 60 整除的歌曲](https://leetcode.cn/problems/pairs-of-songs-with-total-durations-divisible-by-60/)

## [1263. 推箱子](https://leetcode.cn/problems/minimum-moves-to-move-a-box-to-their-target-location/)

## [2437. 有效时间的数目](https://leetcode.cn/problems/number-of-valid-clock-times/)

## [1015. 可被 K 整除的最小整数](https://leetcode.cn/problems/smallest-integer-divisible-by-k/)

## [1016. 子串能表示从 1 到 N 数字的二进制串](https://leetcode.cn/problems/binary-string-with-substrings-representing-1-to-n/)

## [1330. 翻转子数组得到最大的数组值](https://leetcode.cn/problems/reverse-subarray-to-maximize-array-value/)

## [1054. 距离相等的条形码](https://leetcode.cn/problems/distant-barcodes/)

## [1072. 按列翻转得到最大值等行数](https://leetcode.cn/problems/flip-columns-for-maximum-number-of-equal-rows/)

## [1335. 工作计划的最低难度](https://leetcode.cn/problems/minimum-difficulty-of-a-job-schedule/)

## [1073. 负二进制数相加](https://leetcode.cn/problems/adding-two-negabinary-numbers/)

## [1079. 活字印刷](https://leetcode.cn/problems/letter-tile-possibilities/)

## [1373. 二叉搜索子树的最大键值和](https://leetcode.cn/problems/maximum-sum-bst-in-binary-tree/)

## [LCP 33. 蓄水](https://leetcode.cn/problems/o8SXZn/)

## [1080. 根到叶路径上的不足节点](https://leetcode.cn/problems/insufficient-nodes-in-root-to-leaf-paths/)

## [1090. 受标签影响的最大值](https://leetcode.cn/problems/largest-values-from-labels/)

## [1377. T 秒后青蛙的位置](https://leetcode.cn/problems/frog-position-after-t-seconds/)

## [2451. 差值数组不同的字符串](https://leetcode.cn/problems/odd-string-difference/)

## [1091. 二进制矩阵中的最短路径](https://leetcode.cn/problems/shortest-path-in-binary-matrix/)

## [1093. 大样本统计](https://leetcode.cn/problems/statistics-from-a-large-sample/)

## [1439. 有序矩阵中的第 k 个最小数组和](https://leetcode.cn/problems/find-the-kth-smallest-sum-of-a-matrix-with-sorted-rows/)

## [1110. 删点成林](https://leetcode.cn/problems/delete-nodes-and-return-forest/)

## [1130. 叶值的最小代价生成树](https://leetcode.cn/problems/minimum-cost-tree-from-leaf-values/)

## [2517. 礼盒的最大甜蜜度](https://leetcode.cn/problems/maximum-tastiness-of-candy-basket/)

## [2559. 统计范围内的元音字符串数](https://leetcode.cn/problems/count-vowel-strings-in-ranges/)

## [1156. 单字符重复子串的最大长度](https://leetcode.cn/problems/swap-for-longest-repeated-character-substring/)

## [2611. 老鼠和奶酪](https://leetcode.cn/problems/mice-and-cheese/)

## [1240. 铺瓷砖](https://leetcode.cn/problems/tiling-a-rectangle-with-the-fewest-squares/)

## [2699. 修改图中的边权](https://leetcode.cn/problems/modify-graph-edge-weights/)

## [1170. 比较字符串最小字母出现频次](https://leetcode.cn/problems/compare-strings-by-frequency-of-the-smallest-character/)

## [1171. 从链表中删去总和值为零的连续节点](https://leetcode.cn/problems/remove-zero-sum-consecutive-nodes-from-linked-list/)

## [1483. 树节点的第 K 个祖先](https://leetcode.cn/problems/kth-ancestor-of-a-tree-node/)

## [1375. 二进制字符串前缀一致的次数](https://leetcode.cn/problems/number-of-times-binary-string-is-prefix-aligned/)

## [1177. 构建回文串检测](https://leetcode.cn/problems/can-make-palindrome-from-substring/)

## [1494. 并行课程 II](https://leetcode.cn/problems/parallel-courses-ii/)

## [1254. 统计封闭岛屿的数目](https://leetcode.cn/problems/number-of-closed-islands/)

## [1262. 可被三整除的最大和](https://leetcode.cn/problems/greatest-sum-divisible-by-three/)

## [1595. 连通两组点的最小成本](https://leetcode.cn/problems/minimum-cost-to-connect-two-groups-of-points/)

## [LCP 41. 黑白翻转棋](https://leetcode.cn/problems/fHi6rV/)

## [面试题 16.19. 水域大小](https://leetcode.cn/problems/pond-sizes-lcci/)

## [1659. 最大化网格幸福感](https://leetcode.cn/problems/maximize-grid-happiness/)

## [1401. 圆和矩形是否有重叠](https://leetcode.cn/problems/circle-and-rectangle-overlapping/)

## [1186. 删除一次得到子数组最大和](https://leetcode.cn/problems/maximum-subarray-sum-with-one-deletion/)

## [1681. 最小不兼容性](https://leetcode.cn/problems/minimum-incompatibility/)

## [1253. 重构 2 行二进制矩阵](https://leetcode.cn/problems/reconstruct-a-2-row-binary-matrix/)

## [2. 两数相加](https://leetcode.cn/problems/add-two-numbers/description/)

## [445. 两数相加 II](https://leetcode.cn/problems/add-two-numbers-ii/description/)

## [2679. 矩阵中的和](https://leetcode.cn/problems/sum-in-a-matrix/description/)

## [2178. 拆分成最多数目的正偶数之和](https://leetcode.cn/problems/maximum-split-of-positive-even-integers/)

## [2532. 过桥的时间](https://leetcode.cn/problems/time-to-cross-a-bridge/description/)

## [167. 两数之和 II - 输入有序数组](https://leetcode.cn/problems/two-sum-ii-input-array-is-sorted/)

## [15. 三数之和](https://leetcode.cn/problems/3sum/description/)

## [16. 最接近的三数之和](https://leetcode.cn/problems/3sum-closest/description/)

## [1911. 最大子序列交替和](https://leetcode.cn/problems/maximum-alternating-subsequence-sum/description/)

## [931. 下降路径最小和](https://leetcode.cn/problems/minimum-falling-path-sum/description/)

## [18. 四数之和](https://leetcode.cn/problems/4sum/description/)

## [979. 在二叉树中分配硬币](https://leetcode.cn/problems/distribute-coins-in-binary-tree/description/)

## [834. 树中距离之和](https://leetcode.cn/problems/sum-of-distances-in-tree/description/)

## [1851. 包含每个查询的最小区间](https://leetcode.cn/problems/minimum-interval-to-include-each-query/description/)

## [874. 模拟行走机器人](https://leetcode.cn/problems/walking-robot-simulation/description/)

## [918. 环形子数组的最大和](https://leetcode.cn/problems/maximum-sum-circular-subarray/description/)

## [1499. 满足不等式的最大值](https://leetcode.cn/problems/max-value-of-equation/description/)

## [2208. 将数组和减半的最少操作次数](https://leetcode.cn/problems/minimum-operations-to-halve-array-sum/description/)

## [2569. 更新数组后处理求和查询](https://leetcode.cn/problems/handling-sum-queries-after-update/description/)

## [2050. 并行课程 III](https://leetcode.cn/problems/parallel-courses-iii/description/)

## [2681. 英雄的力量](https://leetcode.cn/problems/power-of-heroes/description/)

## [822. 翻转卡片游戏](https://leetcode.cn/problems/card-flipping-game/)

## [722. 删除注释](https://leetcode.cn/problems/remove-comments/)

## [980. 不同路径 III](https://leetcode.cn/problems/unique-paths-iii/)

## [24. 两两交换链表中的节点](https://leetcode.cn/problems/swap-nodes-in-pairs/)

## [1749. 任意子数组和的绝对值的最大值](https://leetcode.cn/problems/maximum-absolute-sum-of-any-subarray/)

## [1289. 下降路径最小和 II](https://leetcode.cn/problems/minimum-falling-path-sum-ii/description/)

## [1749. 任意子数组和的绝对值的最大值](https://leetcode.cn/problems/maximum-absolute-sum-of-any-subarray/description/)

## [833. 字符串中的查找与替换](https://leetcode.cn/problems/find-and-replace-in-string/description/)

## [1444. 切披萨的方案数](https://leetcode.cn/problems/number-of-ways-of-cutting-a-pizza/description/)

## [1388. 3n 块披萨](https://leetcode.cn/problems/pizza-with-3n-slices/description/)

## [2337. 移动片段得到字符串](https://leetcode.cn/problems/move-pieces-to-obtain-a-string/description/)

## [1782. 统计点对的数目](https://leetcode.cn/problems/count-pairs-of-nodes/description/)

## [1267. 统计参与通信的服务器](https://leetcode.cn/problems/count-servers-that-communicate/description/)

## [1448. 统计二叉树中好节点的数目](https://leetcode.cn/problems/count-good-nodes-in-binary-tree/description/)

## [56. 合并区间](https://leetcode.cn/problems/merge-intervals/description/)

## [823. 带因子的二叉树](https://leetcode.cn/problems/binary-trees-with-factors/description/)

## [1654. 到家的最少跳跃次数](https://leetcode.cn/problems/minimum-jumps-to-reach-home/description/)

## [1761. 一个图中连通三元组的最小度数](https://leetcode.cn/problems/minimum-degree-of-a-connected-trio-in-a-graph/description/)

## [2240. 买钢笔和铅笔的方案数](https://leetcode.cn/problems/number-of-ways-to-buy-pens-and-pencils/)

## [1921. 消灭怪物的最大数量](https://leetcode.cn/problems/eliminate-maximum-number-of-monsters/)

## [449. 序列化和反序列化二叉搜索树](https://leetcode.cn/problems/serialize-and-deserialize-bst)

## [1123. 最深叶节点的最近公共祖先](https://leetcode.cn/problems/lowest-common-ancestor-of-deepest-leaves/)

## [2594. 修车的最少时间](https://leetcode.cn/problems/minimum-time-to-repair-cars)

## [207. 课程表](https://leetcode.cn/problems/course-schedule)

## [210. 课程表 II](https://leetcode.cn/problems/course-schedule-ii)

## [630. 课程表 III](https://leetcode.cn/problems/course-schedule-iii)

## [1462. 课程表 IV](https://leetcode.cn/problems/course-schedule-iv)

## [2596. 检查骑士巡视方案](https://leetcode.cn/problems/check-knight-tour-configuration)

## [可以攻击国王的皇后](https://leetcode.cn/problems/queens-that-can-attack-the-king)

## [打家劫舍 III](https://leetcode.cn/problems/house-robber-iii)

## [打家劫舍 IV](https://leetcode.cn/problems/house-robber-iv)

## [2603. 收集树中金币](https://leetcode.cn/problems/collect-coins-in-a-tree)

## [1993.树上的操作](https://leetcode.cn/problems/operations-on-tree)

## [146. LRU 缓存](https://leetcode.cn/problems/lru-cache)

## [1333. 餐厅过滤器](https://leetcode.cn/problems/filter-restaurants-by-vegan-friendly-price-and-distance)

## [2251. 花期内花的数目](https://leetcode.cn/problems/number-of-flowers-in-full-bloom)

## [2136. 全部开花的最早一天](https://leetcode.cn/problems/earliest-possible-day-of-full-bloom)

## [309. 买卖股票的最佳时机含冷冻期](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-with-cooldown)

## [2731. 移动机器人](https://leetcode.cn/problems/movement-of-robots)

## [2512. 奖励最顶尖的 K 名学生](https://leetcode.cn/problems/reward-top-k-students)

## [1488. 避免洪水泛滥](https://leetcode.cn/problems/avoid-flood-in-the-city)

## [137. 只出现一次的数字 II](https://leetcode.cn/problems/single-number-ii)

## [2530. 执行 K 次操作后的最大分数](https://leetcode.cn/problems/maximal-score-after-applying-k-operations)

## [1726. 同积元组](https://leetcode.cn/problems/tuple-with-same-product)

## [2316. 统计无向图中无法互相到达点对数](https://leetcode.cn/problems/count-unreachable-pairs-of-nodes-in-an-undirected-graph)

## [1402. 做菜顺序](https://leetcode.cn/problems/reducing-dishes)

## [1155. 掷骰子等于目标和的方法数](https://leetcode.cn/problems/number-of-dice-rolls-with-target-sum)

## [2698. 求一个整数的惩罚数](https://leetcode.cn/problems/find-the-punishment-number-of-an-integer)

## [割后面积最大的蛋糕](https://leetcode.cn/problems/maximum-area-of-a-piece-of-cake-after-horizontal-and-vertical-cuts)

## [2003. 每棵子树内缺失的最小基因值](https://leetcode.cn/problems/smallest-missing-genetic-value-in-each-subtree)

## [2127. 参加会议的最多员工数](https://leetcode.cn/problems/maximum-employees-to-be-invited-to-a-meeting)

## [117. 填充每个节点的下一个右侧节点指针 II](https://leetcode.cn/problems/populating-next-right-pointers-in-each-node-ii)

## [2258. 逃离火灾](https://leetcode.cn/problems/escape-the-spreading-fire)

## [2300. 咒语和药水的成功对数](https://leetcode.cn/problems/successful-pairs-of-spells-and-potions)

## [1334. 阈值距离内邻居最少的城市](https://leetcode.cn/problems/find-the-city-with-the-smallest-number-of-neighbors-at-a-threshold-distance)

## [2736. 最大和查询](https://leetcode.cn/problems/maximum-sum-queries)

## [2342. 数位和相等数对的最大和](https://leetcode.cn/problems/max-sum-of-a-pair-with-equal-sum-of-digits)

## [2216. 美化数组的最少删除数](https://leetcode.cn/problems/minimum-deletions-to-make-array-beautiful)

## [2304. 网格中的最小路径代价](https://leetcode.cn/problems/minimum-path-cost-in-a-grid)

## [1410. HTML 实体解析器](https://leetcode.cn/problems/html-entity-parser)
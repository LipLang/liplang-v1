# LIP
`Lisp`语言(LISt Processor)等价于在`树状列表`(而非`线性纸带`)上的一种`图灵机`.
引入整数后, 整个语言可以相应简化, 操作可以更加直接, 思想也可以更加简明.
将其名为`LIP`: (List of Integers) Processor.
进一步引入浮点数后, 可做更多的计算, 足够日常把玩.

```lip
# 选择排序(示例代码):
(let sortlist (lambda (lst): # 定义函数, 并给其赋名(逗号,分号;冒号:均作为空格处理)
  (case (isnull lst): # 条件(判断是否空表)
    (block  # 若非空表, 排序(这个分支在条件 **不成立时** 执行):
      (let temp-min (min lst))                         # 找到最小元素的值
      (let temp-pos (find temp-min lst))               # 找到最小元素所在位置
      (cons temp-min (sortlist (delete temp-pos lst))) # 将最小元素移到最前面, 并递归排序后面的子表
    );
    _;    # 若为空表, 返回() (这个分支在条件成立时执行)
    None  # 其他情况, 出错时才会用到
))) # end case, lambda, let

(sortlist {9 4 7 3 8 6}) # (3 4 6 7 8 9)
```

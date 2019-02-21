# This is the repository for Entity alignment by rule mining

## Todo

1. 依照目前的形式，我们仍需要先训练TransE，使用韩旭的代码会是一个很好地决定，所以先转化格式至韩旭的代码，有趣的是，我不需要做transe的test，所以只用分两个组就好。

The idea for this project is developed as the following

## Frame

![entity alignment](https://raw.githubusercontent.com/acharkq/pictures/master/rule%20based%20entity%20alignment.jpg?token=Ag6_nBY5bk9J6eMtTqUJoPMdYSTGisbdks5caiVxwA%3D%3D)

1. Rule based graph completion
   1. TransE:
      1. 根据两个pattern从数据中抽取rule candidates
      2. 在图上使用TransE分别学习embedding
      3. 使用embedding计算candidate的truth value
      4. 依据truth value选出top k rules （truth value是针对单个rule实例的，怎么从rule实例的truth value演化到rule的truth value）（策略：top k取多少，精确重要还是数量重要？）
      5. 使用rule完成graph completion，**这一步很关键**：需要借用seed alignment做到rule transfer，规则较复杂。

      * 已有论文：**top k rule需要手工筛选 -- 是否仿照？**

      * 已有论文：论文的第5步采用另一种方法不直接得将rule inferred triple加入训练集当中，而是**持续隐式地优化真值**。但这样的结果GCN无法显式利用，且与我们的故事不完全相同。

      * Issue：第5步之后：需不需要返回持续优化embedding从而得到更多的rule？（rule的总量不确定，估计是并不很大，**不如等到entity和relation seeds扩张之后，综合该部分结果**，一起迭代，从而获得更高的效率）

   2. AMIE+：获得更多的rule，完成graph completion

2. relation weighting + GCN

   1. 依据图中公式计算relation权重，拍的照片中有对各个参数的解释，记得看到过relation权重的另一种计算方法，在知乎介绍GCN的博客上，参考一下。
   2. 曹的计划是GCN share parameter，和已有论文一致。
   3. 能不能这里的graph embedding和TransE得到的结果share，**这样做能不能在以后的迭代中提高TransE训练结果**（实验验证，暂时觉得不可行，由于TransE训练时两张图分别进行）
   4. 优化函数如图中，最好借鉴GCN的文章
   5. 预测更多的entity alignment和relation alignment
   6. 选择：
      1. 直接将new alignment代入1.1.5，即曹老师提出的扩大$E_S, R_s$从而扩大$\hat R$
      2. 回到1.1.2，这次优化真值
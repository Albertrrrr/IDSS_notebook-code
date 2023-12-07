## Vector space
**vector** : 我们将把向量视为实数的有序元组 $[x_1, x_2, \dots x_n], x_i \in \mathbb{R}$  一个向量有固定的维数 $n$，即元组的长度。我们可以把向量的每个元素想象成代表与所有其他元素**正交的方向**上的距离。

We consider vectors to be ordered tuples of real numbers.A vector has a fixed dimension $n$, which is the length of the tuple. We can imagine each element of the vector as representing a distance in an **direction orthogonal** to all the other elements.

* 考虑 3D 向量 [5, 7, 3]。这是 $\real^3$ 中的一个点，它由 
* Consider the 3D vector [5, 7, 3]. This is a point in $\real^3$, which is formed of:

            5 * [1,0,0] +
            7 * [0,1,0] +
            3 * [0,0,1]

每个向量 [1,0,0]、[0,1,0]、[0,0,1] 都指向一个独立的方向（正交方向），长度为 1。向量 [5,7,3] 可以看作是这些正交单位向量（称为**"基向量 "**）的加权和。向量空间有三个独立的基，因此是三维的。

Each of these vectors [1,0,0], [0,1,0], [0,0,1] is pointing in a independent direction (orthogonal direction) and has length one. The vector [5,7,3] can be thought of a weighted sum of these orthogonal unit vectors (called **"basis vectors"**). The vector space has three independent bases, and so is three dimensional.

* **标量乘法 scalar multiplication** so that $a{\bf x}$  is defined for any scalar $a$. For real vectors, $a{\bf x} = [a x_1, a x_2, \dots a x_n]$, elementwise scaling.
    * $(\real, \real^n) \rightarrow \real^n$
* **向量加法 vector addition** so that ${\bf x} + {\bf y}$ vectors ${\bf x, y}$ of equal dimension. For real vectors, ${\bf x} + {\bf y} = [x_1 + y_1, x_2 + y_2, \dots x_d + y_d]$ the elementwise sum
    * $(\real^n, \real^n) \rightarrow \real^n$
* **norm** $||{\bf x}||$ which allows the length of vectors to be measured 主要用于强调在二维或三维空间中的直观几何含义.
    * $\real_n \rightarrow \real_{\geq 0} ~~ ~~~ \| \mathbf{x} \|_2 = \sqrt{x_1^2 + x_2^2 + \dots + x_n^2}$ 
    * `numpy.linalg.norm`
* **vector addition** so that ${\bf x} + {\bf y}$ vectors ${\bf x, y}$ of equal dimension. For real vectors, ${\bf x} + {\bf y} = [x_1 + y_1, x_2 + y_2, \dots x_d + y_d]$ the elementwise sum
    * $(\real^n, \real^n) \rightarrow \real^n$
* 内积和norm求夹角 $\theta = \arccos\left(\frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\|\|\mathbf{b}\|}\right)$  内积基本就反应了他们的夹角，因为norm可以被看作为常量 
* `np.degrees(np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))`
* `np.inner(x,y)`

* **线性插值 linear interpolation** between two vectors. Linear interpolation of two values is governed by a parameter $\alpha$, and is just: 线性插值就是构造x和y中所有的向量
$$\text{lerp}(\vec{x}, \vec{y}, \alpha) = (1-\alpha) \vec{x} + (\alpha) \vec{y}$$

## Different Norm
* L0 norm ：通常被定义为向量中非零元素的数量 $||{\bf x}||_0 = 非零元素的个数$
* L1 norm ： 是向量中各个元素绝对值之和 $||{\bf x}||_1 = \sum_{i=1}^n|x_i|$
* L2 norm : 通常被称为欧几里得范数，是向量元素平方的和再开平方根。
* Infinity norm: 又称最大范数，是向量的各个分量绝对值的最大值 $||{\bf x}||_\infty = \max(|x_1|,||x_2,...,|x_n|)$
* Lp-norm : 是欧几里得范数的一种泛化形式，用于测量向量在p维空间中的长度:$\| \mathbf{x} \|_p = \left( \sum_{i=1}^{n} |x_i|^p \right)^{\frac{1}{p}}$

<img src="imgs/pnorms.png">

## Matrice
### 线性变换 linear transform 
* 旋转（Rotation）:旋转变换改变向量的方向而保持其长度不变。在二维空间中，一个围绕原点旋转角度 $\theta$ 的旋转变换可以通过以下矩阵实现：
$$\begin{pmatrix}
\cos(\theta) & -\sin(\theta) \\
\sin(\theta) & \cos(\theta)
\end{pmatrix}$$
* 缩放（Scaling）:缩放变换改变向量的长度。在二维空间中，一个沿x轴和y轴分别缩放 a 和 
b 倍的缩放变换可以通过以下矩阵实现：
$$\begin{pmatrix}
a & 0 \\
0 & b
\end{pmatrix} $$
* 剪切（Shearing）:剪切变换会使对象倾斜，改变其形状而保持面积（或体积）不变。在二维空间中，一个沿x轴的剪切可以通过以下矩阵实现：
$$\begin{pmatrix}
1 & k \\
0 & 1
\end{pmatrix}
$$
* 反射（Reflection）：反射变换会使向量在某个轴上翻转。例如，在二维空间中，一个关于x轴的反射可以通过以下矩阵实现：
$$
\begin{pmatrix}
1 & 0 \\
0 & -1
\end{pmatrix}
$$
### 矩阵乘法 Multiplying a Vector by Matrix 
`numpy.dot()` 矩阵乘法的定义是：如果 $A$ 表示线性变换 $f(\vec{x})$，而
$B$ 表示线性变换 $g(\vec{x})$，那么 $BA\vec{x} = g(f(\vec{x}))$。
* $A$ is $p \times q$ and
* $B$ is $q \times r$.
* 
If $C=AB$ then $C_{ij}=\sum_k a_{ik} b_{kj}$


This is the **outer product** of two vectors, every possible combination of their elements:

$$\vec{x} \otimes \vec{y} = \vec{x}^T \vec{y}$$

and the product of a 1xN with an Nx1 vector is a 1x1 matrix; a scalar. This is exactly the **inner product** of two vectors: 内积用于计算相似性，如余弦相似度。

$$\vec{x} \bullet \vec{y} = \vec{x}\vec{y^T} ,$$
and is only defined for vectors $\vec{x}, \vec{y}$ of the same length.

### vranice 方差 
数据大小的密度
$$\sigma^2 =  \frac{1}{N-1} \sum_{i=0}^{N-1} (x_i - \mu_i)^2$$
### covranice 协方差 
$$\Sigma_{ij} = \frac{1}{N-1} \sum_{k=1}^{N} (X_{ki}-\mu_i)(X_{kj}-\mu_j) $$
`np.cov()`
两个变量的协方差：

矩阵中的非对角线元素表示这两个变量之间的协方差。如果我们将这两个变量标记为 X 和 Y，则这两个元素分别是 Cov(X,Y）和 Cov(Y,X)。在协方差的情况下，Cov(X,Y) 等于 Cov(Y,X)。
$$\begin{pmatrix}
\text{Var}(X) & \text{Cov}(X, Y) \\
\text{Cov}(Y, X) & \text{Var}(Y)
\end{pmatrix}
$$ 
五维情况下：
$$\begin{pmatrix}
\text{Var}(X_1) & \text{Cov}(X_1, X_2) & \text{Cov}(X_1, X_3) & \text{Cov}(X_1, X_4) & \text{Cov}(X_1, X_5) \\
\text{Cov}(X_2, X_1) & \text{Var}(X_2) & \text{Cov}(X_2, X_3) & \text{Cov}(X_2, X_4) & \text{Cov}(X_2, X_5) \\
\text{Cov}(X_3, X_1) & \text{Cov}(X_3, X_2) & \text{Var}(X_3) & \text{Cov}(X_3, X_4) & \text{Cov}(X_3, X_5) \\
\text{Cov}(X_4, X_1) & \text{Cov}(X_4, X_2) & \text{Cov}(X_4, X_3) & \text{Var}(X_4) & \text{Cov}(X_4, X_5) \\
\text{Cov}(X_5, X_1) & \text{Cov}(X_5, X_2) & \text{Cov}(X_5, X_3) & \text{Cov}(X_5, X_4) & \text{Var}(X_5)
\end{pmatrix}
$$
#### 作用
* 衡量变量间的线性关系：协方差矩阵提供了数据集中各个变量之间线性关系的度量。矩阵中的每个元素代表了一对变量之间的协方差，表明它们是否同时增减。正协方差表示两个变量正相关（一个增加时另一个也增加），负协方差表示它们负相关（一个增加时另一个减少）。
* Measuring linear relationships between variables: the covariance matrix provides a measure of the linear relationship between the variables in the data set. Each element of the matrix represents the covariance between a pair of variables, indicating whether they increase or decrease simultaneously.

* 数据特征的提取：通过对协方差矩阵进行特征分解，可以提取数据的主要成分或方向，这是主成分分析（PCA）的基础。PCA通过协方差矩阵识别数据中的主要变化方向，帮助减少数据的维度，同时尽可能保留重要的信息。 Extraction of data features: The main components or directions of the data can be extracted by feature decomposition of the covariance matrix, which is the basis of Principal Component Analysis (PCA).
  
* 数据的多维度分布理解：协方差矩阵反映了多维数据集中各维度的联合变异性。通过分析协方差矩阵，可以理解数据各维度间的相互作用，这对于多变量分析非常重要。Understanding the multidimensional distribution of data: The covariance matrix reflects the joint variability of dimensions in a multidimensional dataset. 


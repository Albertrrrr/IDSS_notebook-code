## 1. Computational linear algebra and optimisation
### (a)
- (i)
  * Reviewing the data in Tabel 1, since the magnitude and distribution of each dimension varies widely(e.g., the mean value of dimension 3 is 78.1, while the mean value of dimension 8 is -159.3)
  * In this case, the aim is computing the similarity bewteen the tracks, so it's better to choose L2 norm to solve this problem.
  * For example, when we compute the distance of two feature vectorsl in non-normalisation situation, some features may contribute too much thee overall distance due to their large magnitude, which may mask the effect of other features.
  * L2 norm can be defined as $x_{norm} = \frac{x_i}{\sqrt{\sum_i^n x_i^2}}$
  
  - (ii)
    ```
    x_upload_norm = x_upload / np.linalg.norm(x_upload)
    X_dataset_norm = X_dataset / np.linalg.norm(X_dataset, axis=1, keepdims=True)
    
    distances = np.linalg.norm(X_dataset_norm - x_upload_norm, axis=1)
    closest_match_index = np.argmin(distances)

    ```

### (b)
- (i)
  * the dimension of X is 101750 x 15
  * the dimension of w is 1 x 15
  * the dimension of y is 101750 x 1 

- (ii)
  * A common approach is to use linear least squares. The goal of the least squares method is to minimise the sum of squares of the prediction errors, which can find the w.
  * First step is performing a singular value decomposition of the matrix X. $X = U\Sigma V^T$
  * Second step is computing $X^+ = V \Sigma^+ U^T$  其中，$\Sigma^+$是$\Sigma$的伪逆，即将$\Sigma$的非零元素替换为它们的倒数，零元素保持为零。
  * Finally, we can directly get w by $w=X^+ y$

### (c)
- (i)
  * We can use the PCA to finish this 3D transformer.
  ```
  # 计算每个特征的均值和标准差
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    # 标准化数据
    X_normalized = (X - mean) / std

    # Step 2: 计算协方差矩阵
    cov_matrix = np.cov(X_normalized.T)

    # Step 3: 计算协方差矩阵的特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Step 4: 对特征向量进行排序，并选择主成分
    # 对特征值从大到小排序，获取排序后的特征值的索引
    sorted_indices = np.argsort(eigenvalues)[::-1]

    # 选择前k个特征向量，k是你想要的主成分数量
    k = 3  
    principal_components = eigenvectors[:, sorted_indices[:k]]

    # Step 5: 将原始数据转换到新的特征空间
    X_pca = X_normalized.dot(principal_components)
    
  ```
- (ii)
  * Figure 1 shows the eigenspectrum of the covariance of the matrix X. From this eigen-spectrum, it can be seen that the two largest eigenvalues are much larger than the other eigenvalues, which indicates that the variance in the dataset is mainly concentrated on these two eigenvectors.
  * To more efficiently, we can use the PCA method to decrease the dimension of data, to remove the not main features.
  * In this case, 2D will be more better, becuase two largest eigenvalues are much larger than the other eigenvalues.
  
  ### (d)
  * SGD can solve this issue
  * For each training sample i, the predicted output of the model can be expressed as $\hat{g}_i =x_i w^T + b$。
  * For each training instance i, the squared loss function is denoted as
    $$L_i(w, b) = ||\hat{g}_i - g_i||_2^2$$ 
  *  **w gradient**：
  $$ \nabla_{w} L_i(w, b) = (\hat{g}_i - g_i) x_i $$

  * **b gradient**：
  $$ \nabla_{b} L_i(w, b) = \hat{g}_i - g_i $$

  * - Update w：$w \leftarrow w - \alpha \nabla_{w} L_i(w, b)$
    - Update b：$b \leftarrow b - \alpha \nabla_{b} L_i(w, b)$
    - $\alpha$ is the learning rate

## 2 Probabilities & Bayes rule
### (a)
- (i_1)
  * $P(D|T) = \frac{P(T|D)P(D)}{P(T)}$ $P(T|D)= \frac{28}{28+3}$ $P(D) = \rho$
  * $P(T) = P(T|D) P(D) + P(T|D^-) P(D^-) = \frac{28}{28+3} \rho +  \frac{12}{12+89} (1-\rho)$ 
  * $P(D|T) = \frac{\frac{28}{28+3} \rho}{\frac{28}{28+3} \rho +  \frac{12}{12+89} (1-\rho)}$
- (i_2) 
  * $P(D|T^-) = \frac{P(T^-|D)P(D)}{P(T^-)}$ $P(T^-|D)= \frac{3}{28+3}$ $P(D) = \rho$
  * $P(T^-) = P(T^-|D) P(D) + P(T^-|D^-) P(D^-) = \frac{3}{28+3} \rho +  \frac{89}{12+89} (1-\rho)$ 
  * $P(D|T^-) = \frac{\frac{3}{28+3} \rho}{\frac{3}{28+3} \rho +  \frac{89}{12+89} (1-\rho)}$
  
- (ii)
  * $S_p = \frac{TN}{TN+FP} = \frac{89}{89+28} = 76\%$
  * Because of the high sensitivity $S_p$, it is appropriate for regular testing of people working with vulnerabel population.
  * the rate of true diseased people is 24%，This is not the case when the treatment has serious side effects, and only 24 per cent of people actually need to be treated.
  * As for applying to the whole population to find all diseasd people is not suitable, due to high rate of healthy negative, and it's diffcult for goverment to treat these individuals.

- (iii)
  * Now we can easily know $P(T) = 0.02 ~~ P(T^-) = 0.98$
  * $P(T) = P(T|D) P(D) + (1 - P(T|D)) P(D^-) = \frac{28}{28+3} \rho +  \frac{3}{28+3} (1-\rho) = 0.98$ -> 
### (b)
   * 联立方程组 得$\rho=$% 
   * $$\begin{align}
        0.7 & = \frac{P(T|D)\rho}{P(T)} \\
        0.01 & = \frac{P(T^-|D)\rho}{P(T^-)} \\
        P(T) &= P(T|D)\rho + (1-P(T|D))(1-\rho) 
      \end{align}$$

#### 3.(a)
- a(i)
    * L(x) is the loss funciton 
    * x is a matrix of parameters representing the linear relation between each content and the rating. The vector of dimension would be 1x6.
    * A = $[p_m,p_n,p_b,p_f,p_c,p_a]$ is the number of hours for each content type, a metrix of dimenstion would be 120x6. Each row correspond to one month of historical data.
    * y is a vector representing each month's score from audience  during 10 years, and vector of dimension would be 120x1.
    * The outcome will be the coefficient for the linear relation $\hat x$, which is the best fit of this dataset $

- a(ii)
  * The biggest issule of this case is not a linear relation. linear regression cannot solve this problem. 
  * Residual analysis. we can examine the difference between the model's predicted and actual values. If the residuals show a non-random pattern, this may indicate that the model is not capturing some key features in the data. Also, if the model is good, it should be very small.

- b(i)
- b(ii)
  * Beacause the model is complexible, gradient descent is likely to be the most efficient optimisation in rhis case.
  * The set of parameters $\theta = [b,\alpha_m,\beta_m,\mu_m,..\alpha_a,\beta_a,\mu_a]$
  * the issue can be represented as :
    $$\argmin_{\theta} L(\theta) = \sum_p ||\hat r(p) - r(p)||^2_2$$


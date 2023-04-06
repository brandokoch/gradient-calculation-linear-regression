# Linear regression gradient calculation

This post takes you through the gradient calculation for the linear regression cost function. A deeper dive with more context is available at the [Accompanying blog post](https://blog.brandokoch.com/posts/2023_linear_regression/linear_regression.html)

## Linear regression equation 
Below is the linear regression equation:
```math
$$ y = \sum_{j=0}^{n} \theta_j x_j $$
```

Below is the linear regression cost function
```math
\begin{aligned}
& J = \frac{1}{2m} \sum_{i=1}^{m} (\sum_{j=0}^{n} \theta_j x_j^{(i)} - y^{(i)})^2 
\end{aligned}
```

## Non-vectorized derivative calculation  
A non-vectorized calculation of derivatives is performed in this section. First, a list of derivation rules we may need to use are listed below:

```math
\begin{aligned}
\frac{d}{dx}(f(x)+g(x)) & = \frac{d}{dx}f(x) + \frac{d}{dx}g(x)  \quad\quad && \text{(1.1) Sum rule }\\
f(x) & =ax^b \Rightarrow \frac{d}{dx}f(x) = bax^{b-1}   \quad\quad && \text{(1.2) Power rule }\\
h(x) & = f(x)g(x) \Rightarrow \frac{d}{dx}h(x) = f(x)\frac{d}{dx}g(x) + \frac{d}{dx}f(x)g(x) \quad\quad && \text{(1.3) Product rule}\\
h(x) & = g(f(x)) \Rightarrow \frac{d}{dx}h(x) = \frac{d}{dx}g(f(x)) \frac{d}{dx}f(x)  \quad\quad && \text{(1.4) Chain rule} \\
\frac{d}{dx} \left( \frac{f(x)}{g(x)} \right) & = \frac{g(x)\frac{d}{dx}f(x) - f(x)\frac{d}{dx}g(x)}{(g(x))^2} \quad\quad && \text{(1.5) Quotient rule}
\end{aligned}
```
We proceed with the step-by-step derivation of the cost function. Since the cost function is multivariable, our derivatives turn into partial derivatives that need to be calculated for each parameter $\theta_{k}$.

```math
\begin{aligned} 
\frac{\partial}{\partial \theta_k }J 
& = \frac{\partial}{\partial \theta_k} \left( \frac{1}{2m} \sum_{i=1}^{m} \left( \sum_{j=0}^{n} \theta_j x_j^{(i)} - y^{(i)} \right)^2 \right)  \quad\quad &&\text{apply (1.4), (1.2)}\\
& =   \frac{1}{2m} \sum_{i=1}^{m} 2  \left( \sum_{j=0}^{n} \theta_j x_j^{(i)} - y^{(i)} \right)  \frac{\partial}{\partial \theta_k} \left( \sum_{j=0}^{n} \theta_j x_j^{(i)} - y^{(i)} \right)  \quad\quad &&\text{apply (1.2)} \\
& =   \frac{1}{m} \sum_{i=1}^{m}  \left( \sum_{j=0}^{n} \theta_j x_j^{(i)} - y^{(i)} \right)  x_k^{(i)} 
\end{aligned}
```

## Vectorizing the derivation result
Vectorization of our derivation result is performed here. The inner sum can be rewritten as a dot product or a trivial matrix multiplication.

```math
\begin{aligned} 
& \frac{\partial}{\partial \theta_k }J = \frac{1}{m} \sum_{i=1}^{m}  \left( \sum_{j=0}^{n} \theta_j x_j^{(i)} - y^{(i)} \right)  x_k^{(i)} \\
& \frac{\partial}{\partial \theta_k }J =  \left( \frac{1}{m} \sum_{i=1}^{m}  \left( \boldsymbol{\theta}^T \textbf{x}^{(i)} - y^{(i)} \right)  x_k^{(i)}  \right) 
\end{aligned}
```

Calculating all the partial derivaties at once can be expressed as calculating the vector of partial derivatives, i.e. the gradient. 

```math
\begin{aligned}
\nabla_{\boldsymbol{\theta}} J & = \left( \frac{\partial}{\partial \theta_0} J, \frac{\partial}{\partial \theta_1} J, \dots, \frac{\partial}{\partial \theta_n} J \right) \\
& = \left( \frac{1}{m} \sum_{i=1}^{m}  \left( \boldsymbol{\theta}^T \textbf{x}^{(i)} - y^{(i)} \right)  x_0^{(i)} , \dots, \frac{1}{m} \sum_{i=1}^{m}  \left( \boldsymbol{\theta}^T \textbf{x}^{(i)} - y^{(i)} \right)  x_n^{(i)} \right) \\
& = \frac{1}{m} \sum_{i=1}^{m}  \left( \boldsymbol{\theta}^T \textbf{x}^{(i)} - y^{(i)} \right)  \textbf{x}^{(i)} \\
\end{aligned}
```

Last sum to vectorize is the summation over the dataset. To vectorize we first introduce a matrix $\textbf{X}$ where each row is $(\textbf{x}^{(i)})^T$. Likewise, we introduce $\textbf{y}$ as a vector of outputs. Predictions are now of the form $\textbf{X}\boldsymbol{\theta}$ and the vector of residuals is $\textbf{X}\boldsymbol{\theta} - \textbf{y}$. We are left to take care of our $\textbf{x}^{(i)}$ in the previous equation. Remember that the gradient must be a vector.

Let us first understand what is done in the previous equation with $\textbf{x}^{(i)}$. Our parentheses, when solved, produced a single number, and $\textbf{x}^{(i)}$ was a vector. The summation, therefore, performs a summation over individual scalar vector products. Returning back to our new equation we have a vector of residuals. Each value of that vector needs to multiply the corresponding vector $\textbf{x}^{(i)}$ and all of the results need to be summed. If we transpose $\textbf{X}$ and put it on the left side of the equation we get exactly that. 
```math
$$ \nabla_{\boldsymbol{\theta}} J  = \frac{1}{m} \textbf{X}^T(\textbf{X}\boldsymbol{\theta} - \textbf{y}) $$
```

## Vectorized derivative calculation

Vectorization is usually done before any derivation rules are used. In this section we vectorize immediately and apply new derivation rules for the vectorized form. 

```math
\begin{aligned}
& \frac{1}{2m} \sum_{i=1}^{m} \left( \sum_{j=0}^{n} \theta_j x_j^{(i)} - y^{(i)} \right)^2  \\
& \frac{1}{2m} \sum_{i=1}^{m} \left( \boldsymbol{\theta}^T \textbf{x}^{(i)} - y^{(i)} \right)^2 \\
& \frac{1}{2m}  ( \textbf{X}\boldsymbol{\theta} - \textbf{y} )^2 \\
\end{aligned}
```

We will focus on the term without normalization and account for it at the end. Square operation will be split as follows.

```math
\begin{aligned}
& ( \textbf{X}\boldsymbol{\theta} - \textbf{y} )^2 = ( \textbf{X}\boldsymbol{\theta} - \textbf{y} )^T ( \textbf{X}\boldsymbol{\theta} - \textbf{y} ) 
\end{aligned}
```

Before calculating the gradient, we will list the calculus / linear algebra rules that will be useful.

```math
\begin{aligned}
\textbf{a}^T \textbf{b} & = \textbf{b}^T \textbf{a} \quad\quad && \text{(2.1)}\\
(\textbf{A} \textbf{B})^T & = \textbf{B}^T \textbf{A}^T \quad\quad && \text{(2.2)}\\
(\textbf{A} + \textbf{B})^T & = \textbf{A}^T + \textbf{B}^T \quad\quad && \text{(2.3)}\\
\nabla_{x} \textbf{b}^T \textbf{x} & = \textbf{b} \quad\quad && \text{(2.4)}\\
\nabla_{x} \textbf{x}^T \textbf{A} \textbf{x} & = 2 \textbf{A} \textbf{x} \quad\quad && \text{(2.5)}\\
\nabla_{\boldsymbol{x}} \textbf{W}\textbf{x} & = \textbf{W} \quad\quad && \text{(2.6)}\\
\nabla_{\boldsymbol{x}} \textbf{x}^T\textbf{W} & = \textbf{W}^T \quad\quad && \text{(2.7)}\\
a^T & =a   \quad\quad && \text{(2.8)}
\end{aligned}
```

Gradient calculation follows.

```math
\begin{aligned}
& = \nabla_{\boldsymbol{\theta}} ( \textbf{X}\boldsymbol{\theta} - \textbf{y} )^T ( \textbf{X}\boldsymbol{\theta} - \textbf{y} ) \quad\quad &&\text{apply (2.3), (2.2)}\\
& = \nabla_{\boldsymbol{\theta}} ( \boldsymbol{\theta}^T \textbf{X}^T - \textbf{y}^T ) ( \textbf{X}\boldsymbol{\theta} - \textbf{y} ) \\
& = \nabla_{\boldsymbol{\theta}} ( \boldsymbol{\theta}^T\textbf{X}^T\textbf{X}\boldsymbol{\theta} - \textbf{y}^T\textbf{X}\boldsymbol{\theta} - \boldsymbol{\theta}^T\textbf{X}^T\textbf{y} + \textbf{y}^T\textbf{y} ) \\
\end{aligned}

```

The term $\boldsymbol{\theta}^T\textbf{X}^T\textbf{y}$ is actually a scalar if you consider individual element dimensions. A transpose of a scalar is that same scalar, so a transpose applied to it wouldn't change anything but it would simplify our equation.

```math
\begin{aligned}
& = \nabla_{\boldsymbol{\theta}} ( \boldsymbol{\theta}^T\textbf{X}^T\textbf{X}\boldsymbol{\theta} - \textbf{y}^T\textbf{X}\boldsymbol{\theta} - \boldsymbol{\theta}^T\textbf{X}^T\textbf{y} + \textbf{y}^T\textbf{y} ) &&\text{apply (2.8)}\\
& = \nabla_{\boldsymbol{\theta}} ( \boldsymbol{\theta}^T\textbf{X}^T\textbf{X}\boldsymbol{\theta} - \textbf{y}^T\textbf{X}\boldsymbol{\theta} - ( \boldsymbol{\theta}^T\textbf{X}^T\textbf{y} )^T + \textbf{y}^T\textbf{y} ) &&\text{apply (2.2)}\\
& = \nabla_{\boldsymbol{\theta}} ( \boldsymbol{\theta}^T\textbf{X}^T\textbf{X}\boldsymbol{\theta} - \textbf{y}^T\textbf{X}\boldsymbol{\theta} - \textbf{y}^T\textbf{X}\boldsymbol{\theta}  + \textbf{y}^T\textbf{y} ) \\
& = \nabla_{\boldsymbol{\theta}} ( \boldsymbol{\theta}^T\textbf{X}^T\textbf{X}\boldsymbol{\theta} - 2\textbf{y}^T\textbf{X}\boldsymbol{\theta}  + \textbf{y}^T\textbf{y} ) \\
& = \nabla_{\boldsymbol{\theta}} ( \boldsymbol{\theta}^T\textbf{X}^T\textbf{X}\boldsymbol{\theta} ) - 2\nabla_{\boldsymbol{\theta}} (\textbf{y}^T\textbf{X}\boldsymbol{\theta} ) + \nabla_{\boldsymbol{\theta}} (\textbf{y}^T\textbf{y} ) \\
\end{aligned}
```

We take a look at each gradient individually:

```math
\begin{aligned}
& \nabla_{\boldsymbol{\theta}} (\boldsymbol{\theta}^T\textbf{X}^T\textbf{X}\boldsymbol{\theta} ) = 2 \textbf{X}^T\textbf{X}\boldsymbol{\theta} &&\text{apply (2.5)}\\
& \nabla_{\boldsymbol{\theta}} (\textbf{y}^T\textbf{X}\boldsymbol{\theta} ) = \nabla_{\boldsymbol{\theta}} ((\textbf{X}^T\textbf{y})^T\boldsymbol{\theta}) = \textbf{X}^T\textbf{y} &&\text{apply (2.2), (2.6)}\\
& \nabla_{\boldsymbol{\theta}} (\textbf{y}^T\textbf{y} ) = 0 
\end{aligned}
```

We obtain:  
```math
\begin{aligned}
& = 2 \textbf{X}^T\textbf{X}\boldsymbol{\theta} -  2 \textbf{X}^T\textbf{y} \\
& = 2 \textbf{X}^T(\textbf{X}\boldsymbol{\theta} - \textbf{y})
\end{aligned}
```

Accounting for the normalization we arrive at:
```math
\begin{aligned}
& \frac{1}{2m} 2 \textbf{X}^T(\textbf{X}\boldsymbol{\theta} - \textbf{y}) = \frac{1}{m} \textbf{X}^T(\textbf{X}\boldsymbol{\theta} - \textbf{y})
\end{aligned}
```
















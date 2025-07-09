# Neural Network Math

- A **Neural Network** is a **computational model** that is **inspired** by the structure and function of the **human brain**
    - Composed of **interconnected layers** of simple processing units called **neurons**
    - By adjusting the **parameters** of the **neurons**, **Neural networks** can **approximate complex functions**
- An **epoch** is **one full pass through the training dataset**
- A **batch** is a **subset** of **training examples** that are **processed together**
- The **learning rate $\eta$ is the** step size ****in **parameter updates**
- **Generalization**
    - The ability of a model to make accurate predictions on unseen data
        - The ultimate goal is to build a model that is able to **generalize** as accurately as possible
- **Overfitting**
    - When a model is too **complex**
        - Occurs when a model is over-trained on data
        - The model is unable to generalize the underlying patterns in the data
- **Underfitting**
    - When a model is too **simple**
        - Occurs when a model is under-trained on data
        - The model is unable to capture the underlying patterns in the data

# Architecture

- A **Neuron** is the **basic computing unit** of a **Neural Network**
    - **Neurons** **compute** a **weighted sum** of **inputs plus** a **bias**
    - The **weights $W$** are the **parameters** that are **multiplied** by the **input**
        - **Each feature** of the **input** is **multiplied** by a **weight**
    - The **bias $b$** is the parameter that is **added**
        - **Layer specific** offsets
- **Activation functions are nonlinear functions** applied to the **output** of a **neuron**
    - By introducing **nonlinearity, activation functions** allow a **network** to **stack layers** without collapsing into a **single linear transformation**
    - Enables **Neural networks** to **approximate complex, real-world functions**
- **Layers** are **collections of neurons** with the **same input**
    - The **input layer** receives **raw input features**
    - The **hidden layers** perform **intermediate computations**
    - The **output layer** produces the **final predictions**

# Forward Pass

- The **forward pass** is the **process** of taking an **input vector $x$** and **passing** it **through layers**
    
    $$
    x \in \R^{n_0 \times 1}
    $$
    
    - **$x$ shape**: $(n_0 \times 1)$
    - The **output** is **produced** by the **final layer**
        - The **final output vector of the last layer** is $a^{(L)}$, also known as $\hat{y}$
            - **$a^{(L)}$ shape**: $(n_L \times 1)$
- For a **layer** $l \in \{ 1, ..., L \}$ with $n_{l-1}$ **inputs** and $n_l$ **outputs**
    
    $$
    z^{(l)}=W^{(l)} \cdot a^{(l-1)}+b^{(l)}
    $$
    
    - Where:
        - $a^{(l - 1)}$ is the **output vector** of the **previous layer**
            
            $$
            a^{(l-1)} \in \R^{n_{l-1}\times 1}
            $$
            
            - **$a^{(l-1)}$ shape**: $(n_{l-1} \times 1)$
        - $W^{(l)}$ is the **weight matrix**
            
            $$
            W^{(l)} \in \R^{n_l \times n_{l - 1}}
            $$
            
            - **$W^{(l)}$ shape**: $(n_l \times n_{l-1})$
        - $b^{(l)}$ is the **bias vector**
            
            $$
            b^{(l)} \in \R^{n_l \times 1}
            $$
            
            - **$b^{(l)}$ shape**: $(n_l \times 1)$
        - $z^{(l)}$ is the **pre-activation output vector**
            
            $$
            z^{(l)} \in \R^{n_l \times 1} 
            $$
            
            - **$z^{(l)}$ shape**: $(n_l \times 1)$
    - Implement ****a **linear (fully-connected) layer** in **PyTorch with:**
        - **`nn.Linear(in_features, out_features)`**
- An **activation function** can then be applied to the **pre-activation output vector**
    - Outputs the **final** **output vector** for layer $l$
        
        $$
        a^{(l)}=\sigma^{(l)}(z^{(l)})
        $$
        
        - Where:
            - $\sigma^{(l)}$ is an **activation function**
                - **Sigmoid, Tanh, ReLU, Leaky ReLU, Softmax,** etc
            - $a^{(l)}$ is the **output vector** for layer $l$
                
                $$
                a^{(l)} \in \R^{n_l \times 1} 
                $$
                
                - **$a^{(l)}$ shape**: $(n_l \times 1)$

# Activation Functions

- **Activation functions are nonlinear functions** applied to the **output** of a **neuron**
    - By introducing **nonlinearity, activation functions** allow a **network** to **stack layers** without collapsing into a **single linear transformation**
    - Enables **Neural networks** to **approximate complex, real-world functions**

## Sigmoid

- The **sigmoid activation function** is used for **binary outputs** and **probability modeling**
    - Outputs values are between $[0, 1]$
- The **sigmoid activation function** is given as:
    
    $$
    \sigma(z)=\frac{1}{1+e^{-z}}
    $$
    
- The **derivative of the sigmoid activation function** is given as:
    
    $$
    \sigma^{\prime}(z)=\sigma(z)(1-\sigma(z))
    $$
    
- Implement the **sigmoid activation function** with **`nn.Sigmoid()(input)`** or **`torch.nn.functional.sigmoid(input)`**

### Strengths and Weaknesses

- **Strengths**
    - **Smooth**, bounded in $(0,1)$
        - **Useful** and **interpretable** as a **probability**
    - **Easy** to **implement**
- **Weaknesses**
    - **Vanish gradients problem**

## ReLU

- The **ReLU activation function** is one of the **most commonly used** activation functions because of its **simplicity**
    - Output values are between $[0, z]$
- The **ReLU activation function** is given as:
    
    $$
    relu(z)= \max(0,z)
    $$
    
- The **derivative of the ReLU activation function** is given as:
    
    $$
    relu^{\prime}(z)=
    \begin{cases}1,&z>0\\0,&z\le0\end{cases}
    
    $$
    
- Implement the **ReLU activation function** with **`nn.ReLU()(input)`** or **`torch.nn.functional.relu(input)`**

### Strengths and Weaknesses

- **Strengths**
    - **Simple** and **fast to compute**
    - **Mitigates** the **vanishing gradient problem** for $x > 0$
    - Usually **trains faster** in **deep networks**
- **Weaknesses**
    - **Dying ReLU problem**
        - **Neurons** get **stuck** **outputting** $0$ ****for $x \leq 0$
    - **Not zero-centered**

## Softmax

- The **softmax activation function** is used for **multi-class probabilities**
    - Output values are between  $[0, 1]$
    - All **output probabilities must add up** to $1$
- The **softmax activation function** is given as:
    
    $$
    softmax_i(z)= \frac{e^{z_i}}{\sum_je^{z_j}}
    $$
    
- The **derivative of the softmax activation function** is given as:
    
    $$
    \text{Jacobian matrix: }\partial_i\partial_j=softmaxi(\delta ij-softmax_j)
    $$
    
- Implement the **Softmax activation function** with **`nn.Softmax(dim=None)(input)`** or **`torch.nn.functional.relu(input, dim=None)`**

### Strengths and Weaknesses

- **Strengths**
    - Produces a **valid probability distribution over classes**
    - Gradients **naturally compare all outputs**
- **Weaknesses**
    - **Changing one logit affects all probabilities**

## Tanh

- The **Tanh** **activation function** is used to produce **zero-centered outputs**
    - Outputs are between $[-1, 1]$
- The **Tanh activation function** is given as:
    
    $$
    tanh(z)= \frac{e^z-e^{-z}}{e^z+e^{-z}}
    $$
    
- The **derivative of the Tanh activation function** is given as:
    
    $$
    tanh^{\prime}(z)=1-tanh^2(z)
    $$
    
- Implement the **Tanh activation function** with **`nn.Tanh()(input)`** or **`torch.nn.functional.tanh(input)`**

### Strengths and Weaknesses

- **Strengths**
    - **Outputs** are **centered around 0**
        - **Faster** **training** and **convergence**
- **Weaknesses**
    - Can **saturate** and **suffer** from **vanishing gradients** for **large inputs**

## Leaky ReLU

- The **Leaky ReLU activation function** is used to avoid the **dying ReLU problem** and **allow** **negative outputs**
    - $\alpha$ is a **hyperparameter**
- The **Leaky ReLU activation function** is given as:
    
    $$
    L \space ReLU(z)=max(\alpha z,z)
    $$
    
- The **derivative of the Leaky ReLU activation function** is given as:
    
    $$
    L \space ReLU^{\prime}(z)=
    \begin{cases}1,&z>0 \\ \alpha,&z\le0\end{cases}
    $$
    
- Implement the **ReLU activation function** with **`nn.LeakyReLU(negative_slope=0.01)(input)`** or **`torch.nn.functional.leaky_relu(input, negative_slope=0.01)`**

### Strengths and Weaknesses

- **Strengths**
    - Keeps a **small gradient** for $x < 0$
        - Avoids **dead neurons**
    - **Simple** and **efficient**
- **Weaknesses**
    - Requires **tuning** the **hyperparameter** $\alpha$

## ELU

- Smoothes at $x=0$ for a **continuous derivative**, giving **nonzero gradients when negative**
    - $\alpha$ is a **hyperparameter**
- The **ELU activation function** is given as:
    
    $$
    ELU(z)= 
    \begin{cases}
    x, & x \ge 0\\
    \alpha \bigl(e^x - 1\bigr), & x < 0
    \end{cases}
    
    $$
    
- The **derivative of the ELU activation function** is given as:
    
    $$
    ELU^{\prime}(z)= 
    \begin{cases}
    1, & x \ge 0\\
    \alpha e^x, & x < 0
    \end{cases}
    $$
    
- Implement the **ELU activation function** with **`nn.ELU(alpha=1.0)(input)`** or **`torch.nn.functional.elu(input, alpha=1.0)`**

### Strengths and Weaknesses

- **Strengths**
    - **Smooth transition** at $0$
        - Allows for a **continuous derivative**
    - **Faster convergence**
- **Weaknesses**
    - **Expensive exponential computation** for $x < 0$
    - Requires **tuning** the **hyperparameter** $\alpha$

# Loss Functions

- A **loss function** is a **mathematical function** that **measures** the **discrepancy (error) between** a **model’s predictions** and the **true target values**
    - Outputs a **non-negative error** for each prediction
- During training, the **loss function** is used to **compute gradients** and **update the model’s parameters**
    - The goal is to **minimize the overall loss**, which **improves model performance**
- Different **loss functions** are used for **different tasks**

## Regression

### Mean Absolute Error (MAE)

- **MAE** measures the **average absolute difference** between **predictions** and **targets**
    - **MAE weights all errors the same**
    - **MAE is not differentiable** at $0$

$$
L_{MAE}(\hat{y},y)=\frac{1}{m}\sum_{i=1}^m |\hat{y}_i-y_i|
$$

- Where:
    - $m$ is the **number of samples** in the **batch**
    - $y_i$ is the **true target value** for sample $i$
    - $\hat{y}_i$ is the **model’s predicted value** for sample $i$
- The **partial derivative** of the **MAE with respect to** the **model’s predictions  $\hat{y}_i$ is**
    
    $$
    \frac{\partial L}{\partial\hat{y}_i} = \frac{1}{m} \cdot sign(\hat{y}_i-y_i)\cdot 1 = \frac{1}{m} sign(\hat{y}_i-y_i)
    $$
    
- Implement the **MAE loss function** with **`nn.L1Loss()(input)`** or **`torch.nn.functional.l1_loss(input)`**

### Mean Squared Error (MSE)

- **MSE** measures the **average squared difference** between **predictions** and **targets**
    - **MSE punishes large errors stronger**

$$
L_{MSE}(\hat{y},y)=\frac{1}{m}\sum_{i=1}^m(\hat{y}_i-y_i)^2
$$

- Where:
    - $m$ is the **number of samples** in the **batch**
    - $y_i$ is the **true target value** for sample $i$
    - $\hat{y}_i$ is the **model’s predicted value** for sample $i$
- The **partial derivative** of the **MSE with respect to** the **model’s predictions  $\hat{y}_i$ is**
    
    $$
    \frac{\partial L}{\partial\hat{y}_i} = \frac{1}{m} \cdot 2(\hat{y}_i-y_i)\cdot 1 = \frac{2}{m}(\hat{y}_i-y_i)
    $$
    
- Implement the **MSE loss function** with **`nn.MSELoss()(input)`** or **`torch.nn.functional.mse_loss(input)`**

## Classification

### Binary Cross Entropy (BCE)

- Used for **binary classification**

$$
L_{BCE}(\hat{y},y)=-\frac{1}{m}\sum_{i=1}^m[y_i \ln (\hat{y}_i)+(1-y_i) \ln(1 - \hat{y}_i)]
$$

- Where:
    - $\hat{y}_i = \sigma(z_i)$ is the **predicted probability** for the **positive class**
        - The **sigmoid activation function** is used for **binary outputs**
    - $m$ is the **number of samples** in the **batch**
    - $y_i \in (0,1)$ is the **true binary target** for sample $i$
    - $\hat{y}_i \in (0,1)$ is the **model’s predicted probability** of the **positive class** for sample $i$
        - Comes from the output of the **sigmoid function**
- The **partial derivative** of the **BCE with respect to** the **model’s pre-activation output predictions $z_i$ is**
    
    $$
    \frac{\partial L}{\partial z_i} = \frac{\partial L}{\partial \hat{y}_i} \cdot \frac{\partial \hat{y}_i}{\partial z_i}
    $$
    
    - We need to use the **chain rule** and **multiply** $\frac{\partial L}{\partial \hat{y}_i}$ and $\frac{\partial \hat{y}_i}{\partial z_i}$
    - Find $\frac{\partial L}{\partial\hat{y}_i}$:
        
        $$
        \frac{\partial L}{\partial\hat{y}_i} = \frac{\partial L}{\partial\hat{y}_i}(-[y_i \ln (\hat{y}_i)+(1-y_i) \ln(1 - \hat{y}_i)])
        $$
        
        - Take the derivative with respect to $\hat{y}_i$:
            
            $$
            \frac{\partial L}{\partial\hat{y}_i} = \frac{1}{m} \cdot (- \frac{y_i}{\hat{y}_i}- \frac{1 - y_i}{1-\hat{y}_i} \cdot -1) = \\ \frac{1}{m} \cdot (- \frac{y_i}{\hat{y}_i}+\frac{1 - y_i}{1-\hat{y}_i})
            $$
            
        - Simplify to get:
            
            $$
            \frac{\partial L}{\partial\hat{y}_i} = \frac{1}{m} (\frac{\hat{y}_i-y_i\hat{y}_i-y_i+y_i\hat{y}_i}{\hat{y}_i(1-\hat{y}_i)}) = \frac{1}{m}(\frac{\hat{y}_i-y_i}{\hat{y}_i(1-\hat{y}_i)})
            $$
            
    - Find $\frac{\partial \hat{y}_i}{\partial z_i}$:
        
        $$
        \frac{\partial \hat{y}_i}{\partial z_i} = \frac{\partial \hat{y}_i}{\partial z_i}(\sigma(z_i))
        $$
        
        - Because it is **Binary Cross Entropy**, the **sigmoid activation function** is used **by default** for **binary probability outputs**
        - Take the derivative with respect to $z_i$
            
            $$
            \frac{\partial \hat{y}_i}{\partial z_i} = \sigma^{\prime}(z_i)= \sigma(z_i)(1-\sigma(z_i))
            $$
            
        - Since we know that $\hat{y}_i = \sigma(z_i)$, we can rewrite the above as:
            
            $$
            \frac{\partial \hat{y}_i}{\partial z_i} = \hat{y}_i(1- \hat{y}_i)
            $$
            
    - Now, we can **multiply** the two **partial derivatives** to get the final $\frac{\partial L}{\partial z_i}$
        
        $$
        \frac{\partial L}{\partial z_i} = \frac{\partial L}{\partial \hat{y}_i} \cdot \frac{\partial \hat{y}_i}{\partial z_i} = \frac{1}{m}\cdot \frac{\hat{y}_i-y_i}{\hat{y}_i(1-\hat{y}_i)} \cdot \hat{y}_i(1- \hat{y}_i)
        $$
        
        - **Simplify** and **cancel** the **denominators**
        - The **partial derivative** of the **BCE with respect to** the **model’s final pre-activation output $z_i$** when using a **sigmoid activation function** on $z_i$ as:
        
        $$
        \frac{\partial L}{\partial z_i} =  \frac{1}{m}(\hat{y}_i-y_i)
        $$
        
- Implement the **BCE loss function** with **`nn.BCELoss()(input)`** or **`torch.nn.functional.binary_cross_entropy(input)`**
- Implement the **BCE loss with logits function** with **`nn.BCEWithLogitsLoss()(input)`** or **`torch.nn.functional.binary_cross_entropy_with_logits(input)`**
    - **Combines** a **sigmoid activation function** and the **BCE Loss Computation**

### Cross Entropy (CE)

- Used for **multi-class classification**

$$
L_{CE}(\hat{y},y)=-\frac{1}{m}\sum_{i=1}^m\sum_{k=1}^K [y_{i,k} \ln(\hat{y}_{i,k})]
$$

- Where:
    - $\hat{y}_i=softmax(z_i)$ is the **$K$-way probability vector** for example $i$
    - $m$ is the **number of samples** in the **batch**
    - $K$ is the **number of classes**
    - $y_{i,k} \in (0,1)$ is the **true one-hot target** that sample $i$ belongs to class $k$
    - $\hat{y}_{i,k} \in (0,1)$ is the **model’s predicted probability for class $k$** for sample $i$
        - Comes from the output of the **softmax function**
- The **partial derivative** of the **CE with respect to** the **model’s model’s final pre-activation output $z_i$** of **class $k$** when using a **softmax activation function** on $z_i$ as:
    
    $$
    \frac{\partial L}{\partial z_{i,k}} =  \frac{1}{m} (\hat{y}_{i,k} - y_{i,k})
    $$
    
- Implement the **CE loss function** with **`nn.CrossEntropyLoss()(input)`** or **`torch.nn.functional.cross_entropy(input)`**
    - Computes the **Cross Entropy Loss** between the **input logits** and the **target**
    - **Combines** a **softmax activation function** and a **CE Loss Computation**

# Optimization Process

- Given a network with $L$ layers, inputs $x = a^{(0)}$, and parameters $W^{(l)}$ and $b^{(l)}$ for each layer:
    - Do a **forward pass**
        - For $l = 1$ to $L$
            
            $$
            z^{(l)}=W^{(l)} \cdot a^{(l-1)}+b^{(l)}
            $$
            
            $$
            a^{(l)}=\sigma^{(l)}(z^{(l)})
            $$
            
            - Where:
                - $z^{(l)}$ is the **pre-activation function output**
                - $\sigma^{(l)}$ is the **activation function** for the **layer**
                - $a^{(l)}$ is the **layer output**
    - **Compute** the **loss**
        
        $$
        L = L(a^{(L)}, \hat{y})
        $$
        
    - Perform **backpropagation** to get the **partial derivative** of the **Loss Function** with respect to the **pre-activation layer output**

## Backpropagation

- **Backpropagation** is the **algorithm** for **computing** the **gradient** of a **loss function** with respect to **all trainable parameters** in a **neural network**
    - Leverages the **chain rule of calculus** to **propagate error signals backwards through the network**
        - From ****the **output layer** to the **input layer**
        - Each of the **computed gradients** is used to **update** the **weight** or **bias** by **gradient descent**
- The **chain rule** is used to **calculate** the **partial derivative** of the **Loss Function** with respect to the **pre-activation layer output**
- For the **errors** of the **output layers**
    
    $$
    \frac{\partial L}{\partial z^{(L)}} = \frac{\partial L}{\partial a^{(L)}} \cdot \frac{\partial a^{(L)}}{\partial z^{(L)}} = \frac{\partial L}{\partial a^{(L)}} \cdot \sigma^{\prime (L)}(z^{(L)})
    $$
    
    - When a **final activation function** is not used, **ignore the term**
    - Where:
        - $\frac{\partial L}{\partial z^{(L)}}$ is the **error signal** from the **final layer**
- For the **errors of the hidden layers** (for $l = L - 1, \dots, 1$)
    
    $$
    \frac{\partial L}{\partial z^{(l)}} = \frac{\partial L}{\partial a^{(l)}} \cdot \frac{\partial a^{(l)}}{\partial z^{(l )}} = \frac{1}{m} \cdot \sigma^{\prime(l)}(z^{(l)}) \cdot (W^{(l + 1)\top} \frac{\partial L}{\partial z^{(l + 1)}} +b^{(l + 1)})
    $$
    
    - Where:
        - $\frac{\partial L}{\partial z^{(l + 1)}}$  is the **error signal** from the **next layer**
        - $W^{(l + 1)}$ is the **weight matrix connecting layer $l$** to **layer** $l + 1$
- **Compute** the **partial derivative** of the **loss function with respect to** the **layer’s parameters** using the **chain rule**
    
    $$
    \frac{\partial L}{\partial W^{(l)}}= \frac{\partial L}{\partial z^{(l)}} \cdot \frac{\partial z^{(l)}}{\partial W^{(l)}} = \frac{\partial L}{\partial z^{(l)}} \cdot a^{(l-1) \top}
    $$
    
    $$
    \frac{\partial L}{\partial b^{(l)}}= \frac{\partial L}{\partial z^{(l)}} \cdot \frac{\partial z^{(l)}}{\partial b^{(l)}} = \frac{\partial L}{\partial z^{(l)}} \cdot 1
    $$
    
- In **PyTorch**, **backpropagation** is performed with **`loss.backward()`**

## Backpropagation Example

- As an example to show the calculations of **backpropagation**, we will have a simple **Neural Network** with **3 layers**
    - The first 2 **layers** will use the **ReLU activation function**
    - The **final output layer** will have a **sigmoid activation function** for **binary classification**
    - The **loss function** will be the **Binary Cross Entropy (BCE) Loss** for b**inary classification**
        
        $$
        L_{BCE}(\hat{y},y)=-\frac{1}{m}\sum_{i=1}^m[y_i \ln (\hat{y}_i)+(1-y_i) \ln(1 - \hat{y}_i)]
        $$
        
- The **network** will have:
    
    $$
    z^{(1)}=W^{(1)} \cdot x +b^{(1)}
    $$
    
    $$
    a^{(1)}= relu(z^{(1)})
    $$
    
    $$
    z^{(2)}=W^{(2)} \cdot a^{(1)} +b^{(2)}
    $$
    
    $$
    a^{(2)}= relu(z^{(2)})
    $$
    
    $$
    z^{(3)}=W^{(3)} \cdot a^{(2)} +b^{(3)}
    $$
    
    $$
    a^{(3)}= \sigma(z^{(3)})
    $$
    
- Because the **final predictions** $a^{(3)}$ and the **targets** $y$ are **matrices**, the loss function becomes
    
    $$
    L_{BCE}(a^{(3)},y)=-\frac{1}{m}[y \ln (a^{(3)})+(1 - y) \ln(1 - a^{(3)})]
    $$
    
- The **process** of **backpropagation** involves taking the **partial derivatives** of the **loss function** with respect to the **model weights** and **biases** **moving backwards through** the **network**
    - We can get the **partial derivatives** of **previous layers** using the **chain rule**
- **Layer $3$ Partial Derivatives**
    - Using the **derivation from above**, we know that the **partial derivative** of the **BCE with respect to** the **model’s pre-activation output predictions $z^{(3)}$** using the **chain rule** is
        
        $$
        \frac{\partial L}{\partial z^{(3)}} = \frac{\partial L}{\partial a^{(3)}} \cdot \frac{\partial a^{(3)}}{\partial z^{(3)}} = \frac{1}{m}(a^{(3)} - y)
        $$
        
    - To get the **partial derivative** of the **BCE with respect** to the **last layer’s weights $W^{(3)}$** and **biases $b^{(3)}$** use the **chain rule**
        - In **layer $3$**, we have $z^{(3)}=W^{(3)} \cdot a^{(2)} +b^{(3)}$
            - The **partial derivative with respect to** $W^{(3)}$ is $a^{(2)}$ **(simple differentiation)** because they are being **multiplied**
            - The **partial derivative with respect to** $b^{(3)}$ is $1$ **(simple differentiation)** because it is a **constant**
        
        $$
        \frac{\partial L}{\partial W^{(3)}}= \frac{\partial L}{\partial z^{(3)}} \cdot \frac{\partial z^{(3)}}{\partial W^{(3)}} = \frac{1}{m}(a^{(3)} - y)\cdot a^{(2) \top}
        $$
        
        $$
        \frac{\partial L}{\partial b^{(3)}}= \frac{\partial L}{\partial z^{(3)}} \cdot \frac{\partial z^{(3)}}{\partial b^{(3)}} = \frac{1}{m} (a^{(3)} - y) \cdot 1
        $$
        
    - These **partial derivatives** are used in the **gradient vector** and are **updated** to **minimize** the **loss** using an **optimization algorithm**
- **Layer $2$ Partial Derivatives**
    - Moving **backwards** **through** the **network**, the **input** to **layer** $3$ is $a^{(2)}$
    - We need to find the **partial derivative** of the **loss function with respect to $a^{(2)}$**
        - In **layer $3$**, we have $z^{(3)}=W^{(3)} \cdot a^{(2)} +b^{(3)}$
            - The **partial derivative with respect to** $a^{(2)}$ is $W^{(3)}$ **(simple differentiation)** because they are being **multiplied**
            
            $$
            \frac{\partial L}{\partial a^{(2)}}= \frac{\partial L}{\partial z^{(3)}} \cdot \frac{\partial z^{(3)}}{\partial a^{(2)}} = \frac{1}{m}(a^{(3)} - y)\cdot W^{(3)}
            $$
            
    - Moving **backwards** through the **network**, the **ReLU** **activation function** is **applied** so that $a^{(2)}= relu(z^{(2)})$
    - We need to find the **partial derivative** of the **loss function with respect to** $z^{(2)}$
        - The **derivative** of **the ReLU activation function** will be denoted with $relu^{\prime}$
            
            $$
            \frac{\partial L}{\partial z^{(2)}}= \frac{\partial L}{\partial a^{(2)}} \cdot \frac{\partial a^{(2)}}{\partial z^{(2)}} = \frac{1}{m}(a^{(3)} - y)\cdot W^{(3)} \cdot relu^{\prime}(z^{(2)})
            $$
            
    - To get the **partial derivative** of the **BCE with respect** to the **layer’s weights $W^{(2)}$** and **biases $b^{(2)}$** use the **chain rule**
        - In **layer $2$**, we have $z^{(2)}=W^{(2)} \cdot a^{(1)} +b^{(2)}$
            - The **partial derivative with respect to** $W^{(2)}$ is $a^{(1)}$ **(simple differentiation)** because they are being **multiplied**
            - The **partial derivative with respect to** $b^{(2)}$ is $1$ **(simple differentiation)** because it is a **constant**
        
        $$
        \frac{\partial L}{\partial W^{(2)}}= \frac{\partial L}{\partial z^{(2)}} \cdot \frac{\partial z^{(2)}}{\partial W^{(2)}} = \frac{1}{m}(a^{(3)} - y)\cdot W^{(3)} \cdot relu^{\prime}(z^{(2)}) \cdot a^{(1) \top}
        $$
        
        $$
        \frac{\partial L}{\partial b^{(2)}}= \frac{\partial L}{\partial z^{(2)}} \cdot \frac{\partial z^{(2)}}{\partial b^{(2)}} = \frac{1}{m}(a^{(3)} - y)\cdot W^{(3)} \cdot relu^{\prime}(z^{(2)}) \cdot 1
        $$
        
        - These **partial derivatives** are used in the **gradient vector** and are **updated** to **minimize** the **loss** using an **optimization algorithm**
- **Layer $1$ Partial Derivatives**
    - Moving **backwards** **through** the **network**, the **input** to **layer** $2$ is $a^{(1)}$
    - We need to find the **partial derivative** of the **loss function with respect to $a^{(1)}$**
        - In **layer $2$**, we have $z^{(2)}=W^{(2)} \cdot a^{(1)} +b^{(2)}$
            - The **partial derivative with respect to** $a^{(1)}$ is $W^{(2)}$ **(simple differentiation)** because they are being **multiplied**
            
            $$
            \frac{\partial L}{\partial a^{(1)}}= \frac{\partial L}{\partial z^{(2)}} \cdot \frac{\partial z^{(2)}}{\partial a^{(1)}} = \frac{1}{m}(a^{(3)} - y)\cdot W^{(3)} \cdot relu^{\prime}(z^{(2)}) \cdot W^{(2)}
            $$
            
    - Moving **backwards** through the **network**, the **ReLU** **activation function** is **applied** so that $a^{(1)}= relu(z^{(1)})$
    - We need to find the **partial derivative** of the **loss function with respect to** $z^{(1)}$
        - The **derivative** of **the ReLU activation function** will be denoted with $relu^{\prime}$
            
            $$
            \frac{\partial L}{\partial z^{(1)}}= \frac{\partial L}{\partial a^{(1)}} \cdot \frac{\partial a^{(1)}}{\partial z^{(1)}} = \frac{1}{m}(a^{(3)} - y)\cdot W^{(3)} \cdot relu^{\prime}(z^{(2)}) \cdot W^{(2)} \cdot relu^{\prime}(z^{(1)})
            $$
            
    - To get the **partial derivative** of the **BCE with respect** to the **layer’s weights $W^{(1)}$** and **biases $b^{(1)}$** use the **chain rule**
        - In **layer $1$**, we have $z^{(1)}=W^{(1)} \cdot x +b^{(1)}$
            - The **partial derivative with respect to** $W^{(1)}$ is $x$ **(simple differentiation)** because they are being **multiplied**
            - The **partial derivative with respect to** $b^{(1)}$ is $1$ **(simple differentiation)** because it is a **constant**
        
        $$
        \frac{\partial L}{\partial W^{(1)}}= \frac{\partial L}{\partial z^{(1)}} \cdot \frac{\partial z^{(1)}}{\partial W^{(1)}} = \frac{1}{m}(a^{(3)} - y)\cdot W^{(3)} \cdot relu^{\prime}(z^{(2)}) \cdot W^{(2)} \cdot relu^{\prime}(z^{(1)}) \cdot x^\top
        $$
        
        $$
        \frac{\partial L}{\partial b^{(1)}}= \frac{\partial L}{\partial z^{(1)}} \cdot \frac{\partial z^{(1)}}{\partial b^{(1)}} = \frac{1}{m}(a^{(3)} - y)\cdot W^{(3)} \cdot relu^{\prime}(z^{(2)}) \cdot W^{(2)} \cdot relu^{\prime}(z^{(1)}) \cdot 1
        $$
        
        - These **partial derivatives** are used in the **gradient vector** and are **updated** to **minimize** the **loss** using an **optimization algorithm**

## Vanishing and Exploding Gradients

- **Vanishing** and **exploding gradients** are **phenomena** that occur when **training deep networks** via **backpropagation**
    - **Vanishing Gradients Problem**
        - Occurs when the **magnitude** of **gradients shrinks towards $0$**
        - **Early layers learn very slowly** (or not at all)
            - Results in **poor feature extraction** and **suboptimal performance**
    - **Exploding Gradients Problem**
        - Occurs when the **magnitude** of **gradients grows without bound**
        - **Updates** become **numerically unstable**
            - **Loss** **may** **diverge** and **training can fail**
- **Both** problems are **caused** by the **repeated multiplication** of **small** or **large** factors during **backpropagation**
    - In the example above, there are only $3$ layers, and there are already **several** **terms** **being** **multiplied**
        - As the **network** becomes **deeper (more layers are added)**, there are **more terms** being **multiplied**
            - The **gradient problems become** more **prominent**

### Solutions

- **Careful Weight Initialization**
- **Normalization Layers**
    - **Batch Normalization**, **Layer Normalization**, and **Group Normalization re-center** and **re-scale** the **features** of each **layer**
        - **Prevents** **distributions** from **drifting into saturation** where the features are **tiny** or **extremely** **large**
- **Skip/Residual Connections**
    - Allows **gradients** to **bypass layers** and **non-linearities**, mitigating the **vanishing gradient problem**
- **Gradient Clipping**
    - To solve the **exploding gradients problem, clip** the **norm**

## Gradient Descent

- **Gradient Descent** is an **iterative optimization algorithm** for finding the **local minima** of a **differentiable function**
    - Here, the **differentiable function** is the **Loss Function** $L$
- Using the **partial derivatives**, we have to get the **gradient**
    - The **gradient** is a **vector** that **combines all** of the **partial derivatives**
        - **Vector** that contains **all partial derivatives** of the **loss function** **with respect to each parameter**
        - The **gradient** points in the **direction** of **steepest increase** for the **loss function $L$**
    
    $$
    \nabla_\theta L(\theta)
    \begin{pmatrix}
    \frac{\partial L}{\partial \theta_1}\\[6pt]
    \vdots\\[3pt]
    \frac{\partial L}{\partial \theta_n}
    \end{pmatrix}
    $$
    
    - Where:
        - $\nabla_\theta L(\theta)$ is the **gradient**
        - $\theta$ denotes **all** of the **model parameters**
        - $\frac{\partial L}{\partial \theta_i}$ is the **partial derivative** of the **loss function** with respect to a **model parameter**
- At every iteration of **gradient descent**, the **update rule** for the **parameters $\theta$** is:
    
    $$
    \theta = \theta - (\eta \cdot \nabla_{\theta}L(\theta))
    $$
    
    - Where:
        - $\eta$ is the **learning rate**
            - $\eta > 0$
        - **Subtract** the **partial derivative** to take a small step in the **opposite direction** of the **maximum increase**
            - Doing this **reduces** the **loss** at each **iteration**
- **For example, given** a **layer** $l$, the **gradients** of the **layer** for **parameters** $W^{(l)}$ and $b^{(l)}$ are given as:
    
    $$
    \nabla_{W^{(l)}}L=\frac{\partial L}{\partial W^{(l)}}
    $$
    
    $$
    \nabla_{b^{(l)}}L=\frac{\partial L}{\partial b^{(l)}}
    $$
    
    - The **gradient descent update** **rule for $W^{(l)}$** and $b^{(l)}$ ****is:
        
        $$
        W^{(l)}=W^{(l)}- (\eta \cdot \nabla_{W^{(l)}}L)
        $$
        
        $$
        b^{(l)}=b^{(l)}- (\eta \cdot \nabla_{b^{(l)}}L)
        $$

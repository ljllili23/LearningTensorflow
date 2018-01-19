# Deep MNIST for Experts



## Setup

---

### Load MNIST Data

`mnist` is a lightweight class which stores the training, validation and testing sets as NumPy arrays.

it also provides a function for iterating through data minibatches, which we will use blow.

> 用什么方法把一般的数据处理成这样？

###### NOTICE

---

using `feed_dict` to replace the `placeholder` tensors `x` and `y_` with the training examples. Note that you can replace any tensor in your computation graph using `feed_dict` -- it's not restricted to just `placeholder`s.



### Weight Initialization

To create this model, we're going to need to create a lot of weights and biases. 

One should generally initialize weights with a small amount of noise for symmetry breaking, and ***to prevent 0 gradients.***

 Since we're using [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) neurons, it is also good practice to initialize them with a slightly positive initial bias to avoid "dead neurons". 

Instead of doing this repeatedly while we build the model, let's create two handy functions to do it for us.

```python
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)
#initial 初始值
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
```

> `tf.truncated_normal` outputs random values from a truncated normal distribution.(截断正态分布)
>
> The generated values follow a normal distribution with specified mean and standard deviation,
>
> - **except that values whose magnitude is more than 2 standard deviations from the mean are dropped and re-picked.**
>
> **stddev**: A 0-D Tensor or Python value of type `dtype`. The standard deviation of the truncated normal distribution.

## Convolution and Pooling

*convolution operation:*

- How do we handle the boundaries?
- What's our stride size?

In this example, we're going to choose the vanilla version.(香草版)

> what's **vanilla**?
>
> In information technology, vanilla (pronounced vah-NIHL-uh) is an adjective meaning **plain **or **basic**. 
>
> The unfeatured version of a product is sometimes referred to as the vanilla version.
>
> The term is based on the fact that vanilla is the most popular or at least the most commonly served flavor of ice cream. **the default ice cream.**

​	Our convolutions uses a stride of one and are zero padded so that the output is the same size as the input.

*pooling operation:*

Our pooling is plain old max pooling over 2x2 blocks.

```python
def conv2d(x, W):
  return tf.nn.conv2d(x,W,strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
  
```

### First Convolutional Layer

The convolution will consist of convolution, followed by max pooling.

The convolution will compute 32 features for each 5*5 patch.?

Its weight tensor will have a shape of `[5,5,1,32]`. 

- The fisrt two dimensions are the patch size,
- the next is the number of input channels?
- and the last is the number of output channels.

```python
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
```


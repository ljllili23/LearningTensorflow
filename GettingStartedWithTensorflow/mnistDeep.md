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

One should generally initialize weights with a small amount of noise for symmetry breaking, and to prevent 0 gradients.

 Since we're using [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) neurons, it is also good practice to initialize them with a slightly positive initial bias to avoid "dead neurons". 

Instead of doing this repeatedly while we build the model, let's create two handy functions to do it for us.

```python
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
```

> `tf.truncated_normal` outputs random values from a truncated normal distribution.(截断正态分布)
>
> The generated values follow a normal distribution with specified mean and standard deviation,
>
> **except that values whose magnitude is more than 2 standard deviations from the mean are dropped and re-picked.**
>
> **stddev**: A 0-D Tensor or Python value of type `dtype`. The standard deviation of the truncated normal distribution.




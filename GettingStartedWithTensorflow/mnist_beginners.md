# MNIST Tutorial

*author: LeeJiangLee*

*2018/01/16*

---

MNIST is a simple computer vision dataset. It consists of images of handwritten digits like these:

![img](https://www.tensorflow.org/images/MNIST.png)





In this tutorial, we're going to train a model to look at images and predict what digits they are.

[mnist_softmax.py](https://www.github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/examples/tutorials/mnist/mnist_softmax.py)



## The MNIST Data

---

[Yann LeCun's website](http://yann.lecun.com/exdb/mnist/)

```python
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
```

every MNIST data point has two parts: an image of a handwritten digit and a corresponding label. We'll call the images "x" and the labels "y".

Each image is 28 pixels by 28 pixels. We can interpret this as a big array of numbers:![img](https://www.tensorflow.org/images/MNIST-Matrix.png)

We can flatten this array into a vector of 28x28 = 784 numbers. 

The result is that `mnist.train.images` is a tensor (an n-dimensional array) with a shape of `[55000, 784]`. 

![img](https://www.tensorflow.org/images/mnist-train-xs.png)

### NOTICE

---

What's `one-hot vectors`?

For example, 3 would be [0,0,0,1,0,0,0,0,0,0]. 

Consequently, `mnist.train.labels` is a`[55000, 10]` array of floats.

![img](https://www.tensorflow.org/images/mnist-train-ys.png)





## Softmax Regressions

---

every image in MNIST is of a handwritten digit between 0 and 9. 

So there are ten possible things that a given image can be. We want to be able to look at an image and give the ***probabilities*** for it being each digit.

For example, our model might look at a picture of a nine and be 80% sure it's a nine, but give a 5% chance to it being an eight***(because of the top loop)***  ?

If you want to assign probabilities to an object being one of several different things, **softmax** is the thing to do.

==Even later on, when we train more sophisticated models, the final step will be a layer of softmax.==

A softmax regression has two steps:

1. first we add up the evidence of our input being in certain classes,
2. and then we convert that evidence into probabilities.

We do a weighted sum of the pixel intensities.

- the weight is negative if that pixel having a high intensity is evidence against the image being in that class.(类)
- kand positive if it is evidence in favor.

The following diagram shows the weights on model learned for each of these classes. ==Red== represents ==nagative== weights, while ==blue== represents ==positive== weights.

> 下面这几个图是同一个权值矩阵对0~9，10个不同的图片的激活图。（应该是用来识别‘3’的。)

![img](https://www.tensorflow.org/images/softmax-weights.png)



We also add some extra evidence called a bias. 

输入是$x$，其是第$i$类的证据为：
$$
\text{evidence}i = \sum_j W{i,~j}x_j + b_i
$$

> $W_i$ is the weights and $b_i$ is the bias for class $i$, and $j$ is an index for summing over the pixels in our image x.

$$
y=\text{softmax}(\text{evidence})
$$

Here softmax is serving as an "activation" or "link" function, shaping the output of our linear function into the form we want.

 You can think of it as converting tallies of evidence into probabilities of our input being in each class.
$$
\text{softmax}(evidence) = \text{normalize}(\exp(evidence))
$$
也就是归一化成概率分布。

可以把softmax regression看成下面的过程。

![img](https://www.tensorflow.org/images/softmax-regression-scalargraph.png)

![imag](https://www.tensorflow.org/images/softmax-regression-vectorequation.png)


$$
y = \text{softmax}(Wx+b)
$$

## Implementing the Regression

---

Tensorflow lets us describe a graph of interacting operations that run entirely outside Python.

[mnist_beginners.py](C:\Users\Shin\Documents\GitHub\LearningTensorflow\GettingStartedWithTensorflow\mnist_beginners.py)



To use TensorFlow, first we need to import it.

```
import tensorflow as tf

```

We describe these interacting operations by manipulating symbolic variables. Let's create one:

```
x = tf.placeholder(tf.float32, [None, 784])

```

`x` isn't a specific value. It's a `placeholder`, a value that we'll input when we ask TensorFlow to run a computation. We want to be able to input any number of MNIST images, each flattened into a 784-dimensional vector. We represent this as a 2-D tensor of floating-point numbers, with a shape `[None, 784]`. (Here `None` means that a dimension can be of any length.)

We also need the weights and biases for our model. We could imagine treating these like additional inputs, but TensorFlow has an even better way to handle it: `Variable`. A `Variable` is a modifiable tensor that lives in TensorFlow's graph of interacting operations. It can be used and even modified by the computation. For machine learning applications, one generally has the model parameters be `Variable`s.

```
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

```

We create these `Variable`s by giving `tf.Variable` the initial value of the `Variable`: in this case, we initialize both `W` and `b` as tensors full of zeros. Since we are going to learn `W`and `b`, it doesn't matter very much what they initially are.

Notice that `W` has a shape of [784, 10] because we want to multiply the 784-dimensional image vectors by it to produce 10-dimensional vectors of evidence for the difference classes. `b` has a shape of [10] so we can add it to the output.

We can now implement our model. It only takes one line to define it!

```
y = tf.nn.softmax(tf.matmul(x, W) + b)

```

First, we multiply `x` by `W` with the expression `tf.matmul(x, W)`. This is flipped from when we multiplied them in our equation, where we had $Wx$, as a small trick to deal with `x` being a 2D tensor with multiple inputs. We then add `b`, and finally apply `tf.nn.softmax`.

That's it. It only took us one line to define our model, after a couple short lines of setup. That isn't because TensorFlow is designed to make a softmax regression particularly easy: it's just a very flexible way to describe many kinds of numerical computations, from machine learning models to physics simulations. And once defined, our model can be run on different devices: your computer's CPU, GPUs, and even phones!



## Training

---

In order to train our model, we need to define what it means for the model to be good.

actually, in machine learning we typically define what it means for a model to be bad.

**We call this the cost, or the loss.**



One very common, very nice function to determine the loss of a model is called "**cross entropy**"交叉熵。
$$
H_{y'}(y) = -\sum_i y'_i \log(y_i)
$$

```python
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
```

functions used above:

1. tf.reduce_mean
2. tf.reduce_sum
3. tf.log

**comparation:**

- `tf.matmul` means matrix multiply.
- `*` means multiply each element.
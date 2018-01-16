# MNIST Tutorial

*authour: LeeJiangLee*

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

> 下面这几个图是同一个权值矩阵对0~9，10个不同的图片的激活图。（应该是用来识别‘3’的。

![img](https://www.tensorflow.org/images/softmax-weights.png)
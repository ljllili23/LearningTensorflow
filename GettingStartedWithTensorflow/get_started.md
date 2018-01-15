This is my first DOC about Tensorflow.千里之行始于足下！

# Getting Started With TensorFlow

---

TensorFlow Core programs consist of two discrete(constituting a separate entity or part) sections:

1. Buiding the computational graph.
1. Runing the computational graph.

 **A computational graph** is a series of TensorFlow operations arranged into a graph of nodes.

To actually evaluates the nodes, we must run the computational graph within a session.

A **session ** encapsulates the control and state of the TensorFlow runtime.

encapsulate压缩；将…装入胶囊；将…封进内部；概述

    node1 = tf.constant(3.0, dtype=tf.float32)
    node2 = tf.constant(4.0) #also tf.float32 implicitly
    print(node1, node2)
    sess=tf.Session()
    print(sess.run([node1,node2]))

notice:

Does the `[]` means the variables must be arrays?

combining Tensor nodes with operations(operations are also nodes)

    from __future__ import
    print_functionnode3 = tf.add(node1, node2)
    print("node3:", node3)
    print("sess.run(node3):", sess.run(node3))

![](https://www.tensorflow.org/images/getting_started_add.png)

## Three methods to create a Tensors

---

- `tf.constant`
- `tf.placeholder`
- `tf.Variable`

A **placeholder ** is a promise to provide a value later.

We can evaluate this graph with multiple inputs by using the **feed_dict ** argument to the  [**run method**](https://www.tensorflow.org/api_docs/python/tf/Session#run) ** ** to feed _concrete _ values to the **placeholders** :

    print(sess.run(adder_node, {a: 3, b: 4.5}))
    print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))

In machine learning we will typically want a model that take _arbitrary _ inputs, such as the one above. To make the model _trainable_ , we need to be able to _modify _ the graph to get new outputs with the same input. **Variables ** allow us to add trainable parameters to a graph.

    W = tf.Variable([.3], dtype=tf.float32)
    b = tf.Variable([-.3], dtype=tf.float32)
    x = tf.placeholder(tf.float32)
    linear_model = W*x + b

NOTICE:

---

the different between placeholder and Variable is that:

- **`[tf.placeholder](https://www.tensorflow.org/api_docs/python/tf/placeholder)`**

will be fed concrete values when we evaluate the graph.but can not be changed after the first run.

- **`[tf.Variable](https://www.tensorflow.org/api_docs/python/tf/Variable)`**

variables are not initialized when you call `tf.Variable` .

allow us to add trainable parameters to a graph, namely the parameters can be changed after the first run.

- **`[tf.constant](https://www.tensorflow.org/api_docs/python/tf/constant)`**

constants are initialized when you call `tf.constant` , and their value can never change.

To initialize all the variables in a TensorFlow program, you must explicity call a special operation as follow:

    init = tf.global_variables_initializer()
    sess.run(init)

---

## Warning!

- It is important to realize init is a **handle ** to the TensorFlow sub-graph that initializes all the global variables.

- Until we call sess.run, the variables are uninitialized.

# Loss function

---

A loss function measures how far apart the current model is from the provided data.

Three operation used below:

1.  [tf.square](https://www.tensorflow.org/api_docs/python/tf/square)
1.  [tf.reduce_sum](https://www.tensorflow.org/api_docs/python/tf/reduce_sum)
1.  [tf.assign](https://www.tensorflow.org/api_docs/python/tf/assign)

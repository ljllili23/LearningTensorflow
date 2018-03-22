### Tip1

As a rule of thumb, increasing the number of hidden layers and neurons typically creates a more powerful model, which requires more data to train effectively.

作为一个经验法则，**增加隐藏层和神经元的数量**通常会创建一个更强大的模型，这**需要更多的数据**进行有效训练。

---

### Tip2

 [**Neural networks**](https://developers.google.com/machine-learning/glossary/#neural_network) can find complex relationships between features and the label.

 a [**fully connected neural network**](https://developers.google.com/machine-learning/glossary/#fully_connected_layer), means that the neurons in one layer take inputs from *every* neuron in the previous layer.

---

### Tip3

```
classifier.train(
        input_fn=lambda:train_input_fn(train_feature, train_label, args.batch_size),
        steps=args.train_steps)

```

The `steps` argument tells `train` to stop training after the specified number of iterations. Increasing `steps` increases the amount of time the model will train. Counter-intuitively, training a model longer does not guarantee a better model. The default value of `args.train_steps` is 1000. The number of steps to train is a [**hyperparameter**](https://developers.google.com/machine-learning/glossary/#hyperparameter) you can tune. Choosing the right number of steps usually requires both experience and experimentation. (超参数)

---


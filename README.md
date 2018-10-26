# T-GANs

code for <a href="https://arxiv.org/abs/1810.10948">Training Generative Adversarial Networks Via Turing Test</a><br>
(There may be a delay of arxiv version, please get lastest version <a href="https://github.com/bojone/T-GANs/blob/master/paper%20-%20Training%20Generative%20Adversarial%20Networks%20Via%20Turing%20Test.pdf">here</a>)

Require: Keras 2.2.0+

<font color="red"><strong>before using the code, you must change the source code of keras !!</strong></font><br>
(you can run it without changing keras, but the effect will be worse.)

`keras/engine/base_layer.py`, replace

```
    def add_weight(self,
                   name,
                   shape,
                   dtype=None,
                   initializer=None,
                   regularizer=None,
                   trainable=True,
                   constraint=None):
        """Adds a weight variable to the layer.
        # Arguments
            name: String, the name for the weight variable.
            shape: The shape tuple of the weight.
            dtype: The dtype of the weight.
            initializer: An Initializer instance (callable).
            regularizer: An optional Regularizer instance.
            trainable: A boolean, whether the weight should
                be trained via backprop or not (assuming
                that the layer itself is also trainable).
            constraint: An optional Constraint instance.
        # Returns
            The created weight variable.
        """
        initializer = initializers.get(initializer)
        if dtype is None:
            dtype = K.floatx()
        weight = K.variable(initializer(shape),
                            dtype=dtype,
                            name=name,
                            constraint=constraint)
        if regularizer is not None:
            with K.name_scope('weight_regularizer'):
                self.add_loss(regularizer(weight))
        if trainable:
            self._trainable_weights.append(weight)
        else:
            self._non_trainable_weights.append(weight)
        return weight
```
with
```
    def add_weight(self,
                   name,
                   shape,
                   dtype=None,
                   initializer=None,
                   regularizer=None,
                   trainable=True,
                   constraint=None):
        """Adds a weight variable to the layer.
        # Arguments
            name: String, the name for the weight variable.
            shape: The shape tuple of the weight.
            dtype: The dtype of the weight.
            initializer: An Initializer instance (callable).
            regularizer: An optional Regularizer instance.
            trainable: A boolean, whether the weight should
                be trained via backprop or not (assuming
                that the layer itself is also trainable).
            constraint: An optional Constraint instance.
        # Returns
            The created weight variable.
        """
        initializer = initializers.get(initializer)
        if dtype is None:
            dtype = K.floatx()
        weight = K.variable(initializer(shape),
                            dtype=dtype,
                            name=name,
                            constraint=None)
        if regularizer is not None:
            with K.name_scope('weight_regularizer'):
                self.add_loss(regularizer(weight))
        if trainable:
            self._trainable_weights.append(weight)
        else:
            self._non_trainable_weights.append(weight)
        if constraint is not None:
            return constraint(weight)
        return weight
```  

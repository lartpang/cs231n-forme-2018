from builtins import object, range

import numpy as np

from cs231n.layer_utils import *
from cs231n.layers import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3 * 32 * 32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        self.params['W1'] = np.random.normal(scale=weight_scale,
                                             size=(input_dim, hidden_dim))
        self.params['b1'] = np.zeros(shape=(hidden_dim,))
        self.params['W2'] = np.random.normal(scale=weight_scale,
                                             size=(hidden_dim, num_classes))
        self.params['b2'] = np.zeros(shape=(num_classes,))

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # affine_1 W1 input*hidden
        #  affine_1_out = X.dot(self.params['W1']) + self.params['b1']
        # relu_1
        #  relu_1_out = np.maximum(0, affine_1_out)
        # affine_2 W2 hidden*classes
        #  affine_2_out = relu_1_out.dot(self.params['W2']) + self.params['b2']
        # 最后的softmax不计算，因为得分的计算计算到最后的分类之前即可
        #  scores = affine_2_out
        # 思路是上面的思路，但是实际计算的时候需要注意具体的数据形式，这里还是用已有
        # 的函数计算比较方便
        layer_1, cache_1 = affine_relu_forward(
            X, self.params['W1'], self.params['b1'])
        layer_2, cache_2 = affine_forward(
            layer_1, self.params['W2'], self.params['b2'])
        scores = layer_2
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # 计算softmax的损失函数
        loss, dscores = softmax_loss(scores, y)
        # 正则项
        loss += 0.5 * self.reg * (np.sum(np.square(self.params['W2'])) +
                                  np.sum(np.square(self.params['W1'])))
        # 对第二个affine求导数
        dlayer_1, grads['W2'], grads['b2'] = affine_backward(dscores, cache_2)
        # 对第一个affine求导数
        _, grads['W1'], grads['b1'] = affine_relu_backward(dlayer_1, cache_1)
        # 正则化
        grads['W2'] += self.reg * self.params['W2']
        grads['W1'] += self.reg * self.params['W1']

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3 * 32 * 32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        # 隐含层加输出层总数
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # 这里要留心，实际上这是一个针对任意层网络实现的一个计算，所以要针对可变量
        # hidden_dims 实现可变的层次安排
        for i in range(self.num_layers):
            # 计算第一层的参数，也就是i = 0
            if i == 0:
                self.params['W1'] = np.random.normal(scale=weight_scale,
                                                     size=(input_dim,
                                                           hidden_dims[i]))
                self.params['b1'] = np.zeros(hidden_dims[i])
            # 描述最后一层的映射
            elif i == self.num_layers - 1:
                self.params['W' + str(i + 1)] = \
                    np.random.normal(scale=weight_scale,
                                     size=(hidden_dims[i - 1],
                                           num_classes))
                self.params['b' + str(i + 1)] = np.zeros(num_classes)
            # 描述中间层
            else:
                self.params['W' + str(i + 1)] = \
                    np.random.normal(scale=weight_scale,
                                     size=(hidden_dims[i - 1],
                                           hidden_dims[i]))
                self.params['b' + str(i + 1)] = np.zeros(hidden_dims[i])
            # 描述normalization层
            if i != self.num_layers - 1:
                if normalization == 'batchnorm' or normalization == 'layernorm':
                    self.params['gamma' + str(i + 1)] = np.ones(hidden_dims[i])
                    self.params['beta' + str(i + 1)] = np.zeros(hidden_dims[i])
            # “布置”完各层映射参数
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == 'batchnorm':
            self.bn_params = [{'mode': 'train'}
                              for i in range(self.num_layers - 1)]
        if self.normalization == 'layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization == 'batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # 这里的实现和之前的相同，同样需要考虑处理任意层次的问题
        cache = {}
        # 防止影响原本的数据
        scores = X.copy()
        # 按层数循环
        for i in range(1, self.num_layers + 1):
            # 各层首先必有affine
            scores, cache['fc'+str(i)] = \
                affine_forward(scores,
                               self.params['W' + str(i)],
                               self.params['b' + str(i)])

            # 最后一层没有relu，norm，dropout运算，所以不考虑，这里非最后一层
            if i < self.num_layers:
                # 前面的各层里不一定有norm层，需要判断
                if self.normalization == "batchnorm":
                    # 特征数目
                    # D = scores.shape[1]
                    scores, cache['bn'+str(i)] = \
                        batchnorm_forward(scores,
                                          self.params['gamma' + str(i)],
                                          self.params['beta' + str(i)],
                                          self.bn_params[i-1])
                    # self.bn_params[i-1] since the provided code above initilizes bn_params
                    # for layers from index 0, here we index layer from 1.
                elif self.normalization == "layernorm":
                    scores, cache['bn'+str(i)] = \
                        layernorm_forward(scores,
                                          self.params['gamma' + str(i)],
                                          self.params['beta' + str(i)],
                                          self.bn_params[i-1])
                # 必有的ReLU层
                scores, cache['relu'+str(i)] = relu_forward(scores)

                # 可能有的dropout层
                if self.use_dropout:
                    scores, cache['dropout' + str(i)] = \
                        dropout_forward(scores, self.dropout_param)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # 计算损失函数，损失函数结果就是最后一层softmax的计算结果加上之前各层的正则项
        loss, dscores = softmax_loss(scores, y)
        # 计算损失函数
        for i in range(1, self.num_layers + 1):
            loss += 0.5 * self.reg * np.sum(self.params['W' + str(i)] ** 2)
        # 这里使用列表生成式的方式来避免显式的 for
        # loss += 0.5 * self.reg * sum([np.sum(self.params['W' + str(i)]**2) \
        #                                                  for i in range(1, self.num_layers + 1)])
        # 计算每一层的偏导数，各层组成情况：
        # {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

        # 计算方式1
        # 先计算最后的一层
        #  dscores, grads['W' + str(self.num_layers)], \
        #      grads['b' + str(self.num_layers)] \
        #      = affine_relu_backward(dscores, caches[self.num_layers - 1])
        #  grads['W' + str(str(i))] += self.reg * self.params['W' + str(i)]

        #  for i in range(self.num_layers - 1, 0, -1):
        #      if self.use_dropout:
        #          pass

        #      if self.normalization == 'batchnorm':
        #          dscores, grads['W' + str(i)], grads['b' + str(i)], \
        #              grads['gamma' + str(i)], grads['beta' + str(i)] = \
        #              affine_relu_bn_backward(dscores, caches[i - 1])
        #          grads['W' + str(i)] += self.reg * self.params['W' + str(i)]
        #      elif self.normalization == 'layernorm':
        #          dscores, grads['W' + str(i)], grads['b' + str(i)], \
        #              grads['gamma' + str(i)], grads['beta' + str(i)] = \
        #              affine_relu_ln_backward(dscores, caches[i - 1][-1])
        #          grads['W' + str(i)] += self.reg * self.params['W' + str(i)]
        #      else:
        #          dscores, grads['W' + str(i)], grads['b' + str(i)] = \
        #              affine_relu_backward(dscores, caches[i - 1])
        #          grads['W' + str(i)] += self.reg * self.params['W' + str(i)]

        # 计算方式2
        # 这里的实现，更加注重各个环节的考虑，与其像上面那样，连带考虑计算，到不如直接分布，有需要时判断运算即可
        for i in range(self.num_layers, 0, -1):
            # No ReLU, dropout, Batchnorm for the last layer
            if i < self.num_layers:
                # 使用随机失活时，求其反向传播
                if self.use_dropout:
                    dscores = dropout_backward(
                        dscores, cache['dropout' + str(i)])

                # 求出必有的ReLU反向传播
                dscores = relu_backward(dscores, cache['relu' + str(i)])

                # 再考虑是否有归一化
                if self.normalization == "batchnorm":
                    dscores, grads['gamma'+str(i)], grads['beta'+str(i)] = \
                        batchnorm_backward(dscores, cache['bn'+str(i)])
                elif self.normalization == "layernorm":
                    dscores, grads['gamma'+str(i)], grads['beta'+str(i)] = \
                        layernorm_backward(dscores, cache['bn'+str(i)])

            # 最后一层，只有affine
            dscores, grads['W' + str(i)], grads['b' + str(i)] = \
                affine_backward(dscores, cache['fc' + str(i)])

            grads['W' + str(i)] += self.reg * self.params['W' + str(i)]

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

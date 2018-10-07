from builtins import range

import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.
    N个样本，各个样本的维度有后面的尺寸决定，现在要将输入的数据转换成一个向量，
    长度是原本各个尺寸的乘积。

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    N = x.shape[0]
    # 将多维e的x归置到二维，各个样本的数据都被压缩到一个维度
    new_x = x.reshape([N, -1])
    out = np.dot(new_x, w) + b
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.
    核心：关键是理解针对不同类型的层的运算究竟是什么？
    仿射层实际上就是一个简单的加权映射，所以它的求导也就是按照那个线性公式计算即可

    Inputs:
    - dout: Upstream derivative, of shape (N,
    M)，dout实际上是从后面的层传回来的偏导数，它就是后面的部分对于out的偏导数，
    再乘以out对于输入的偏导数，就是想要的偏导数结果了。
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    N = x.shape[0]
    # dx N,d1,d2,...,dk = dout N,M × w.T D,M
    dx = dout.dot(w.T).reshape(x.shape)
    # dw D,M = x.T D,N ×dout N,M
    dw = x.reshape([N, -1]).T.dot(dout)
    # db M, = dout N,M 沿axis=0方向累加
    db = np.sum(dout, axis=0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    # 对于numpy数组对象而言，.copy()实现的是深拷贝，类似python内置函数deepcopy()
    out = x.copy()
    out[x <= 0] = 0
    # 若是不用这个.copy()，直接赋值，这是浅拷贝，两个变量仍是引用一个地址
    # also
    #  out = np.maximum(0, x)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # 这里要算的是最后的输出关于x的偏导数，所以不只要算ReLU的偏导数，还要乘以dout
    dx = dout * (x >= 0)
    # also
    #  dx = dout
    #  dx[x < 0] = 0

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    # 到了batchnorm的数据已经是二维的了
    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        # # You should store the output in the variable out. Any intermediates
        # # that you need for the backward pass should be stored in the cache
        # # variable.
        # #
        # # You should also use your computed sample mean and variance together
        # # with the momentum variable to update the running mean and running
        # # variance, storing your result in the running_mean and running_var
        # # variables.
        # #
        # # Note that though you should be keeping track of the running
        # # variance, you should normalize the data based on the standard
        # # deviation (square root of variance) instead!
        # # Referencing the original paper (https://arxiv.org/abs/1502.03167)
        # # might prove to be helpful.
        # #
        #######################################################################
        sample_mean = np.mean(x, axis=0)
        # 计算标准差（这之中使用了参数epsilon)
        sample_var = np.var(x, axis=0)
        # 动量方式滑动更新策略
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
        # 计算归一化值
        x_normed = (x - sample_mean) / np.sqrt(sample_var + eps)
        # 变换重构
        out = gamma * x_normed + beta
        # 保存参数
        cache = (x, gamma, beta, x_normed, sample_mean, sample_var, eps)

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # 测试，不需要在计算均值方差，而是使用已有的running值
        # 计算归一化值
        x_normed = (x - bn_param['running_mean']) / \
            np.sqrt(bn_param['running_var'] + eps)
        # 变换重构
        out = gamma * x_normed + beta

        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # - x: Data of shape (N, D)
    # - gamma: Scale parameter of shape (D,)
    # - beta: Shift paremeter of shape (D,)
    # - bn_param: Dictionary with the following keys:
    #   - mode: 'train' or 'test'; required
    #   - eps: Constant for numeric stability
    #   - momentum: Constant for running mean / variance.
    #   - running_mean: Array of shape (D,) giving running mean of features
    #   - running_var Array of shape (D,) giving running variance of features
    x, gamma, beta, x_normed, sample_mean, sample_var, eps = cache
    # 罗列计算
    # dout实际上就是dl/dy
    # 对于beta，gamma求导，实际上是一个特征有一个beta，gamma，这导致求导是多个分支相加
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_normed, axis=0)

    # 先计算出对归一化的x偏导
    dx_normed = dout * gamma.reshape([1, -1])

    # dy/dx有三条路径，一条是对归一化公式里的x，另外就是对均值里的和标准差里的x
    # dl/dvar
    dvar = -0.5 * np.sum(dx_normed * (x - sample_mean), axis=0) * \
        ((sample_var + eps)**(-1.5))
    # dl/dmean
    # 注意这里有两处均值，一处分子上的mu一处分母上的方差里的mu
    dmean = -np.sum(dx_normed / np.sqrt(sample_var + eps), axis=0) \
        - 2 * dvar / x.shape[0] * np.sum(x - sample_mean, axis=0)
    # dl/dx
    dx = dx_normed / np.sqrt(sample_var + eps) + 2 * dvar * (x - sample_mean) / \
        x.shape[0] + dmean / x.shape[0]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    x, gamma, beta, x_normed, sample_mean, sample_var, eps = cache
    # 对于beta，gamma求导，实际上是一个特征有一个beta，gamma，这导致求导是多个分支相加
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_normed, axis=0)

    # dy/dx有三条路径，一条是对归一化公式里的x，另外就是对均值里的和标准差里的x
    # dl/dx 这里合到了一起
    dx = \
        dout * gamma / np.sqrt(sample_var + eps) \
        - np.sum(dout * gamma * (x - sample_mean), axis=0) * \
        ((sample_var + eps) ** -1.5) \
        * (x - sample_mean) / x.shape[0] \
        - np.sum(dout * gamma / np.sqrt(sample_var + eps), axis=0) / \
        x.shape[0] \
        + np.sum(dout * gamma * (x - sample_mean), axis=0) * \
        ((sample_var + eps) ** -1.5) \
        / x.shape[0] * np.sum(x - sample_mean, axis=0) / x.shape[0]
    # dx = \
    #   dx_normed / np.sqrt(sample_var + eps) \
    #
    #
    #   2 * dvar * (x - sample_mean) / x.shape[0] \
    #
    #
    #   dmean / x.shape[0]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get('eps', 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # 因为layernorm是关于特征的处理，x到这里的时候，已经是二维的了。
    x_new = x.copy()
    # 这里的计算均值方差，是针对一个样本而言
    sample_mean = np.mean(x_new, axis=1)
    sample_var = np.var(x_new, axis=1)

    # 这里是按照不同的样本进行的均值方差计算
    x_new = x_new - sample_mean.reshape([-1, 1])
    x_new = x_new / np.sqrt(sample_var.reshape([-1, 1]) + eps)
    out = gamma * x_new + beta
    cache = (x, gamma, beta, x_new, sample_mean, sample_var, eps)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    x, gamma, beta, x_, sample_mean, sample_var, eps = cache
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_, axis=0)

    dx_ = dout * gamma
    dvar = -0.5 * np.sum(dx_.T * (x.T - sample_mean), axis=0) * \
        ((sample_var + eps) ** -1.5)
    dmean = -np.sum(dx_.T / np.sqrt(sample_var + eps), axis=0) \
        - 2 * dvar / x.shape[1] * np.sum(x.T - sample_mean, axis=0)
    dx = dx_.T / np.sqrt(sample_var + eps) + 2 * dvar * \
        (x.T - sample_mean) / x.shape[1] + dmean / x.shape[1]
    dx = dx.T

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # 对于dropout而言，每一个ReLU输出背后要跟着一个乘以掩膜的过程，这里只处理一层即可
        mask = (np.random.randn(*x.shape) < p) / p
        out = mask * x
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # 对于一层的dropout而言，测试、预测并不需要处理
        out = x
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        dx = dout * mask
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.


    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # 这里要使用函数 np.pad 来进行矩阵的填充，就是实现补零操作
    pad = conv_param['pad']
    stride = conv_param['stride']
    N, _, H, W = x.shape
    F, _, HH, WW = w.shape
    # 因为后面是用的需要整数类型
    H_ = 1 + (H + 2 * pad - HH) // stride
    W_ = 1 + (W + 2 * pad - WW) // stride
    # 这里只是计算出了对应的大小形状，想要计算具体的内容需要进一步计算卷积的过程
    # 在对x进行处理的时候，需要先计算x_pad，这样才可以实现后面的卷积操作
    # 这里的添加pad主要是在HW维度上进行的处理，对于pad_width实际上只写一侧的时候代表两侧
    # 添加同样的pad量
    pad_width = ((0, ), (0, ), (pad, ), (pad, ))
    x_pad = np.pad(x, pad_width, 'constant')
    # 初始化数据空间
    out = np.zeros((N, F, H_, W_))
    # 卷积计算
    # 对于不同的输入的样本进行循环
    for n in range(N):
        # 对于多个卷积核（层）进行循环
        for f in range(F):
            # 开始处理卷积操作，平移h方向
            for h_ in range(H_):
                # 平移w方向
                for w_ in range(W_):
                    # 对于前两维度，也就是样本数和通道数，样本数随n循环，而对于通道维度而言，
                    # 并不需要处理，因为卷积核和图像是有相同的通道维度
                    # 关键是在 H×W 的维度上该如何处理
                    # 每次和卷积核做乘法的是x_pad的对应的块
                    # 而卷积核的形状本就是与之对应的，只需要循环卷积核f即可
                    # 计算后求和并加上偏置项，偏置项与卷积核相对应，都是权重关系
                    out[n, f, h_, w_] = np.sum(
                        x_pad[n,
                              :,
                              h_ * stride: h_ * stride + HH,
                              w_ * stride: w_ * stride + WW] * w[f, :]
                    ) + b[f]
                    # out 表示第n个样本与第f个卷积核之间对应的卷积关系

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    x, w, b, conv_param = cache
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    _, _, dHH, dWW = dout.shape
    stride, pad = conv_param['stride'], conv_param['pad']

    pad_width = ((0, ), (0, ), (pad, ), (pad, ))
    x_pad = np.pad(x, pad_width, 'constant')

    # 预先设定空间
    dx_pad = np.zeros_like(x_pad)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    # 循环
    for n in range(N):
        for f in range(F):
            # 对于偏置项而言，它的对应关系就是第f个卷积核所对应的第f个偏置项
            # 这里以移动卷积核卷积的方式来计算，而笔记中讲解的是把同一个卷积核对不同位置的卷积，
            # 认为是多个相同的卷积核对于各自的一小块区域的乘积
            # 所以对于b的偏导数，需要对于同一个卷积核应对的多个输入样本的累和
            db[f] += np.sum(dout[n, f])
            for dhh in range(dHH):
                for dww in range(dWW):
                    # 从偏导数计算中，可以看出来对于dwf而言，结果是dout与对应的卷积核感受野的乘积
                    dw[f] += x_pad[n, :,
                                   dhh * stride: dhh * stride + HH,
                                   dww * stride: dww * stride + WW] * \
                        dout[n, f, dhh, dww]
                    dx_pad[n, :,
                           dhh * stride: dhh * stride + HH,
                           dww * stride: dww * stride + WW]\
                        += w[f] * dout[n, f, dhh, dww]
    dx = dx_pad[:, :, pad: pad + H, pad: pad + W]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']

    # 注意取整的除法操作
    H_ = 1 + (H - pool_height) // stride
    W_ = 1 + (W - pool_width) // stride
    out = np.zeros((N, C, H_, W_))

#    for n in range(N):
    for h in range(H_):
        for w in range(W_):
            # 使用的max pooling，返回第2、3维度的最大值
            # 这里np.max是矩阵内部的最大值，np.maximum是比较两个矩阵的逐位比较取其最大
            # 这里没有专门计算N,C维度，因为可以同时得出来，
            out[:, :, h, w] = np.max(x[:, :,
                                       h * stride: h * stride + pool_height,
                                       w * stride: w * stride + pool_width],
                                     axis=(-1, -2))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    x, pool_param = cache
    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    H_ = 1 + (H - pool_height) // stride
    W_ = 1 + (W - pool_width) // stride
    dx = np.zeros_like(x)

    for n in range(N):
        for c in range(C):
            for h_ in range(H_):
                for w_ in range(W_):
                    # 将平面索引或平面索引数组转换为坐标数组，而argmax实现的是总的平坦数组的索引
                    # 而np.unravel_index的dim项，表示指定的原本的数据形状，因为argmax得出来的是平坦的结果
                    # np.unravel_index需要借助制定形状来从而转换平坦索引到坐标元组
                    # 因为使用的是max pooling所以导数也是很直接，对于大于0的部分为1，小于0的部分为0
                    # 配合dout，则大于0的位置都是dout，小于0的位置为0
                    # pooling层输出对于输入的x的偏导数就是在各个pooling感受野上，最大值位置处取1，感受野其余位置为0
                    ind = np.unravel_index(np.argmax(x[n, c,
                                                       h_ * stride: h_ * stride + pool_height,
                                                       w_ * stride: w_ * stride + pool_width],
                                                     axis=None),
                                           (pool_height, pool_width))
                    dx[n, c,
                       h_ * stride: h_ * stride + pool_height,
                       w_ * stride: w_ * stride + pool_width][ind] = dout[n, c, h_, w_]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    N, C, H, W = x.shape
    # 调整x到 N×H×W × C 大小，四维压缩到二维，保留C维度
    # np.transpose可以调整各维度的顺序，对于二维而言也就是转置了
    x_new = np.reshape(np.transpose(x, (0, 2, 3, 1)), (-1, C))

    out, cache = batchnorm_forward(x_new, gamma, beta, bn_param)

    # 调整out从(N, H, W, C)到(N, C, H, W)
    out = np.transpose(np.reshape(out, (N, H, W, C)), (0, 3, 1, 2))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # 获取参数
    N, C, H, W = dout.shape
    # 调整压缩维度
    dout_new = np.reshape(np.transpose(dout, (0, 2, 3, 1)), (-1, C))
    dx, dgamma, dbeta = batchnorm_backward(dout_new, cache)
    # 调整
    dx = np.transpose(np.reshape(dx, (N, H, W, C)), (0, 3, 1, 2))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get('eps', 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    N, C, H, W = x.shape
    # GroupNorm：将channel方向分group，然后每个group内做归一化，算(C//G)*H*W的均值
    x = np.reshape(x, (N * G, C // G * H * W))

    # 注意这里的实现,可以看出来，均值方差使用的都是现算的结果，不同于BN的过程
    x = x.T
    mu = np.mean(x, axis=0)
    # 下面的sq计算价值不大
    xmu = x - mu
    sq = xmu ** 2
    var = np.var(x, axis=0)

    sqrtvar = np.sqrt(var + eps)
    ivar = 1. / sqrtvar
    xhat = xmu * ivar

    xhat = np.reshape(xhat.T, (N, C, H, W))
    out = gamma[np.newaxis, :, np.newaxis, np.newaxis] * \
        xhat + beta[np.newaxis, :, np.newaxis, np.newaxis]
    cache = (xhat, gamma, xmu, ivar, sqrtvar, var, eps, G)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    N, C, H, W = dout.shape
    xhat, gamma, xmu, ivar, sqrtvar, var, eps, G = cache
    dxhat = dout * gamma[np.newaxis, :, np.newaxis, np.newaxis]
    dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)
    dgamma = np.sum(dout * xhat, axis=(0, 2, 3), keepdims=True)
    dxhat = np.reshape(dxhat, (N * G, C // G * H * W)).T
    xhat = np.reshape(xhat, (N * G, C // G * H * W)).T
    Nprime, Dprime = dxhat.shape
    dx = 1.0 / Nprime * ivar * \
        (Nprime * dxhat - np.sum(dxhat, axis=0) -
         xhat * np.sum(dxhat * xhat, axis=0))
    dx = np.reshape(dx.T, (N, C, H, W))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    这里的x相当公式里的s，表示得分情况，由于不再是对于w求导（之前的实现是对于w的
    求导，所以还是需要留意下计算方法，和dW略有不同

    假定式子形式为d/dsk，式子中的可变参量为j，有sj以及syi
    当k=yi时，这个存在累和，为不满足边界值（计算损失）的情况的数目
    当k!=yi，仅为普通的j时，此时仅有一项偏导数为非零，结果为1
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    # 计算出来的margins表示的是所有得分损失的分散分布情况，它的第(i,j)个元素表示
    # 第j类对于第i个输入的计算的损失情况
    loss = np.sum(margins) / N
    # 将第i个的针对所有的类的损失求和，就是Li，将所有的输入加和，得到了总的损失

    # 下面的描述是关于wj求导的思路，而非本次需要的对于x的导数
    # 对应第i个样本的正确分类Wyi的梯度。它的计算方法是，有多少个Wj导致边界值不被
    # 满足，从而就对损失函数产生多少次贡献。从对应的函数导数上来看，由于对于任意
    # 的j!=yi，都存在倒数d/dWj，所以总的导数是一个加和的形式。
    # 这个次数乘以Xi并取负数就是Wyi行对应的梯度。
    # 对应第i个样本的不正确分类行Wi的梯度。注意在求导中，只有 i==j 的那一行，
    # 才有可能对梯度产生贡献，i<>j时整个式子均为常数，而常数求导为零。所以最终
    # 导数形式与式子1）不同，没有了求和符号。

    # 这里计算的就是各个样本中有多少次边界不被满足，要计算损失，要用来计算对于xyi
    # 的梯度
    num_pos = np.sum(margins > 0, axis=1)

    dx = np.zeros_like(x)

    # 计算示性函数的结果
    dx[margins > 0] = 1

    # 分类正确的位置，在margin中本身为0，减去num_pos，不用关心上一句的作用
    dx[np.arange(N), y] -= num_pos
    dx /= N

    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    同理，这里也是对于dx的求导，需注意与之前不同
    """
    # 先平移数据
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    # 获得对各个样本，将其自己关于各个分类的指数得分相加，获得分母部分，先把方
    # 便整体计算的部分求解出来
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)

    # 先算了log，再算了指数，和起来算了一个softmax函数
    # 这里可以用来计算损失函数
    log_probs = shifted_logits - np.log(Z)
    # 这里可以用来计算关于x的导数
    probs = np.exp(log_probs)

    N = x.shape[0]
    # 注意这里使用的是 log_probs ，也就是损失函数Li
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    # 这里使用深拷贝，不干扰原本的probs
    dx = probs.copy()
    # 只需要在 dLi/dsk k=yi
    # 的位置减去1即可，从而整体是各个分类器损失函数对于x的偏导数
    dx[np.arange(N), y] -= 1
    dx /= N

    return loss, dx

from random import shuffle

import numpy as np


def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    # 一个单独的矩阵乘法Wx_i就高效地并行评估10个不同的分类器（每个分类器针对一
    # 个分类），其中每个类的分类器就是W的一个行向量。
    # W 3703X10
    # X 500X3703
    scores = X[i].dot(W)
    # scores 表示的是得分情况，可是下面的scores[y[i]]又表示什么含义？
    # 可以参见“线性分类”部分的损失函数，这里构建的是syi
    correct_class_score = scores[y[i]]
    # 这里实现的是针对第i个数据的多类SVM损失函数
    for j in range(num_classes):
      # 这里实现的就是如同“线性分类”部分的损失函数也就是sj-syi+delta

      # 这里只是将所有不正确分类相加
      margin = scores[j] - correct_class_score + 1 # note delta = 1

      if margin > 0:
          if j == y[i]:
            # 这里为什么不能将dW[:, y[i]] += -X[i] 方在这里？
            # 仔细比较，可以发现，两个位置相加执行的次数不同
            # 也就是说，相加，也只是在 j!=y[i]时执行
            # 也就是说，SVM损失函数的求导项也只是对于 j!=y[i]时执行
            # ？？？
            continue
          else:
            loss += margin
            dW[:, j] += X[i]
            dW[:, y[i]] += (-X[i])
        # 只有超过一定的范围，才会认定损失
        # SVM损失函数想要正确分类类别yi的分数比不正确类别分数高，而且至少要高
        # delta，如果不满足，就开始计算损失值
        # 对于损失函数的偏导数，套用公式，使用的是上面margin

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  # 这里的reg就是公式里的参数lambda
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  # 计算得分函数
  scores = X.dot(W)
  # 得到一个500X10的矩阵
  # 正确分类的得分提取
  correct_class_score = scores[np.arange(num_train), y]
  # 参数的意义：axis=None，时候就会flatten当前矩阵，实际上就是变成了一个行向量
             #  axis=0,沿着y轴复制，实际上增加了行数
             #  axis=1,沿着x轴复制，实际上增加列数
  # 此时correct_class_score和scores大小不匹配，需要调整
  correct_class_score = np.reshape( \
                            np.repeat(correct_class_score, num_classes),
                            (num_train, num_classes))
  # 计算损失函数
  margin = np.maximum(0, scores - correct_class_score + 1)
  margin[np.arange(num_train), y] = 0
  # margin 存放的是那个折叶函数的结果

  loss = np.sum(margin)
  loss /= num_train
  loss += reg * np.sum(W * W)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  # gradient
  # 梯度里的损失函数直接使用示性函数的结果
  # 这是折叶函数的导数，1或0
  margin[margin > 0] = 1
  margin[margin <= 0] = 0

  row_sum = np.sum(margin, axis = 1)
  margin[np.arange(num_train), y] = -row_sum
  dW += np.dot(X.T, margin)
  dW /= num_train
  dW += 2 * reg * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW

from random import shuffle

import numpy as np


def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]

  for i in range(num_train):
      scores = X[i].dot(W)
      # 为了避免因为数值过大，而导致结果不稳定，则将数据平移
      shift_scores = scores - max(scores)
      loss_i = -shift_scores[y[i]] + np.log(np.sum(np.exp(shift_scores)))
      loss += loss_i

      for j in range(num_classes):
          softmax_output = np.exp(shift_scores[j]) / np.sum(np.exp(shift_scores))
          # 对第j个分类器偏导
          if j == y[i]:
              # 正好是对应的实际类别时
              dW[:, j] += (-1 + softmax_output) * X[i]
          else:
              # 非自己的类别
              dW[:, j] += softmax_output * X[i]


  loss /= num_train
  loss += reg * np.sum(W * W)
  dW = dW / num_train + 2 * reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  scores = X.dot(W)
  # 因为是直接使用了矩阵的运算，所以使得数据应该是多个的数据的集合——一个得分向量
  shift_scores = scores - np.max(scores, axis = 1).reshape(-1, 1)
  # 可以直接利用矩阵除以总和来得到对应的softmax函数的输出矩阵
  softmax_output = np.exp(shift_scores) / np.sum(np.exp(shift_scores), \
                                                 axis = 1).reshape(-1, 1)
  # 这里计算出来了各个样本对于各个分类的得分的softmax输出，而对于最终的损失函数
  # 的计算，是针对各个样本属于自己实际类别得分的一个求和平均
  # 因为这个矩阵是利用 np.exp(shift_scores) 为基础得到的
  # y是各元素表示对应的类别序号
  loss = -np.sum(np.log(softmax_output[range(num_train), list(y)]))
  loss /= num_train
  loss += reg * np.sum(W * W)

  # 拷贝一份输出
  dS = softmax_output.copy()
  # 对于类别为实际的分类列，相较于错误分类在softmax函数的基础上，多减了一个一
  dS[range(num_train), list(y)] += -1
  # 各个样本对新的“分类器”进行计算，结果就是偏导数
  dW = (X.T).dot(dS)
  dW = dW / num_train + 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

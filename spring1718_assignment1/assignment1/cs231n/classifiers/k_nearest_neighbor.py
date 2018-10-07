import numpy as np


class KNearestNeighbor(object):
  """ a kNN classifier with L2 distance """

  def __init__(self):
    pass

  def train(self, X, y):
    """
    Train the classifier. For k-nearest neighbors this is just
    memorizing the training data.

    Inputs:
    - X: A numpy array of shape (num_train, D) containing the training data
      consisting of num_train samples each of dimension D.
    - y: A numpy array of shape (N,) containing the training labels, where
         y[i] is the label for X[i].
    """
    self.X_train = X
    self.y_train = y

  def predict(self, X, k=1, num_loops=0):
    """
    Predict labels for test data using this classifier.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data consisting
         of num_test samples each of dimension D.
    - k: The number of nearest neighbors that vote for the predicted labels.
    - num_loops: Determines which implementation to use to compute distances
      between training points and testing points.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].
    """
    if num_loops == 0:
      dists = self.compute_distances_no_loops(X)
    elif num_loops == 1:
      dists = self.compute_distances_one_loop(X)
    elif num_loops == 2:
      dists = self.compute_distances_two_loops(X)
    else:
      raise ValueError('Invalid value %d for num_loops' % num_loops)

    return self.predict_labels(dists, k=k)

  def compute_distances_two_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a nested loop over both the training data and the
    test data.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data.

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the Euclidean distance between the ith test point and the jth training
      point.
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):
      for j in range(num_train):
        #####################################################################
        # TODO:                                                             #
        # Compute the l2 distance between the ith test point and the jth    #
        # training point, and store the result in dists[i, j]. You should   #
        # not use a loop over dimension.                                    #
        #####################################################################
        #  pass
        # 第i个测试样本与第j个训练样本各特征差值平方和又开根号，这里使用向量
        dists[i, j] = np.sqrt((X[i, :] - self.X_train[j, :]).dot( \
            (X[i, :] - self.X_train[j, :])))
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
    return dists

  def compute_distances_one_loop(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a single loop over the test data.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):
      #######################################################################
      # TODO:                                                               #
      # Compute the l2 distance between the ith test point and all training #
      # points, and store the result in dists[i, :].                        #
      #######################################################################
      #  pass
      matrix_temp = self.X_train - X[i, :]
      # 这里错误了，要注意实际求解距离的对应关系，而且这样会导致数据计算出现问题
      #  matrix_temp_one_loop = (matrix_temp[i, :]).dot(matrix_temp.T)
      #  matrix_temp_one_loop = np.sum(matrix_temp * matrix_temp, axis=1)
      matrix_temp_one_loop = np.sum(np.square(matrix_temp), axis = 1)
      dists[i, :] = np.sqrt(matrix_temp_one_loop)

      #######################################################################
      #                         END OF YOUR CODE                            #
      #######################################################################
    return dists

  def compute_distances_no_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    #########################################################################
    # TODO:                                                                 #
    # Compute the l2 distance between all test points and all training      #
    # points without using any explicit loops, and store the result in      #
    # dists.                                                                #
    #                                                                       #
    # You should implement this function using only basic array operations; #
    # in particular you should not use functions from scipy.                #
    #                                                                       #
    # HINT: Try to formulate the l2 distance using matrix multiplication    #
    #       and two broadcast sums.                                         #
    #########################################################################
    #  pass
    # 错误思路
    #  X_test_test = X.dot(X.T)
    #  X_train_train = self.X_train.dot(self.X_train.T)
    #  X_test_train = X.dot(self.X_train.T)
    #  X_train_test = self.X_train.dot(X.T)

    # 下面的实现主要是依靠将差值平方进行了拆分
    # 比对最后的结果dists矩阵和原本的 test * train 样本矩阵，实际上元素上的差异
    # 主要是在于前者的第 (i,j) 项是表示
    # (testi-trainj).^2，也就是testi.^2+trainj.^2-2*testi*trainj
    # 对于矩阵的形式就是如下了
    dists = np.sqrt(-2*np.dot(X, self.X_train.T) + \
                        np.sum(np.square(self.X_train), axis = 1) + \
                        np.transpose([np.sum(np.square(X), axis = 1)]))
    # 注意上面第三行的中括号，里面需要他来实现行向量到列向量的转置
    #########################################################################
    #                         END OF YOUR CODE                              #
    #########################################################################
    return dists

  def predict_labels(self, dists, k=1):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Inputs:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      gives the distance betwen the ith test point and the jth training point.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].
    """
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in range(num_test):
      # A list of length k storing the labels of the k nearest neighbors to
      # the ith test point.
      closest_y = []
      #########################################################################
      # TODO:                                                                 #
      # Use the distance matrix to find the k nearest neighbors of the ith    #
      # testing point, and use self.y_train to find the labels of these       #
      # neighbors. Store these labels in closest_y.                           #
      # Hint: Look up the function numpy.argsort.                             #
      #########################################################################
      #  pass
      # np.argsort()默认 返回来的结果含义：前后的顺序表示从小到大，但是具体的值
      # 表示对应位次的数据原本的序号
      # 对于矩阵，axis=1的时候是按照横着比较大小，=0时，按照列比较大小
      index_sort_i_test  = np.argsort(dists[i, :])
      k_nearest_i_test = index_sort_i_test[0:k, ]
      closest_y = self.y_train[k_nearest_i_test]

      #########################################################################
      # TODO:                                                                 #
      # Now that you have found the labels of the k nearest neighbors, you    #
      # need to find the most common label in the list closest_y of labels.   #
      # Store this label in y_pred[i]. Break ties by choosing the smaller     #
      # label.                                                                #
      #########################################################################
      #  pass
      count_i_test = np.bincount(closest_y)
      y_pred[i] = np.argmax(count_i_test)

      #########################################################################
      #                           END OF YOUR CODE                            #
      #########################################################################

    return y_pred

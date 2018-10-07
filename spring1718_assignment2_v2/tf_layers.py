import tensorflow as tf
import numpy as np


def load_cifar10(num_training=49000, num_validation=1000, num_test=10000):
    """
    Fetch the CIFAR-10 dataset from the web and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.
    """
    cifar10 = tf.keras.datasets.cifar10.load_data()
    (X_train, y_train), (X_test, y_test) = cifar10
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int32).flatten()
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.int32).flatten()

    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    mean_pixel = X_train.mean(axis=(0, 1, 2), keepdims=True)
    std_pixel = X_train.std(axis=(0, 1, 2), keepdims=True)
    X_train = (X_train - mean_pixel) / std_pixel
    X_val = (X_val - mean_pixel) / std_pixel
    X_test = (X_test - mean_pixel) / std_pixel

    return X_train, y_train, X_val, y_val, X_test, y_test


class Dataset(object):
    def __init__(self, X, y, batch_size, shuffle=False):
        """
        Construct a Dataset object to iterate over data X and labels y

        Inputs:
        - X: Numpy array of data, of any shape
        - y: Numpy array of labels, of any shape but with y.shape[0] == X.shape[0]
        - batch_size: Integer giving number of elements per minibatch
        - shuffle: (optional) Boolean, whether to shuffle the data on each epoch
        """
        assert X.shape[0] == y.shape[0], 'Got different numbers of data and labels'
        self.X, self.y = X, y
        self.batch_size, self.shuffle = batch_size, shuffle

    def __iter__(self):
        N, B = self.X.shape[0], self.batch_size
        idxs = np.arange(N)
        if self.shuffle:
            np.random.shuffle(idxs)
        return iter((self.X[i:i+B], self.y[i:i+B]) for i in range(0, N, B))


def check_accuracy(sess, dset, x, scores, is_training=None):
    """
    Check accuracy on a classification model.

    Inputs:
    - sess: A TensorFlow Session that will be used to run the graph
    - dset: A Dataset object on which to check accuracy
    - x: A TensorFlow placeholder Tensor where input images should be fed
    - scores: A TensorFlow Tensor representing the scores output from the
      model; this is the Tensor we will ask TensorFlow to evaluate.

    Returns: Nothing, but prints the accuracy of the model
    """
    num_correct, num_samples = 0, 0
    for x_batch, y_batch in dset:
        feed_dict = {x: x_batch, is_training: 0}
        scores_np = sess.run(scores, feed_dict=feed_dict)
        y_pred = scores_np.argmax(axis=1)
        num_samples += x_batch.shape[0]
        num_correct += (y_pred == y_batch).sum()
    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f%%)' %
          (num_correct, num_samples, 100 * acc))


def train_part34(model_init_fn, optimizer_init_fn, num_epochs=1):
    """
    Simple training loop for use with models defined using tf.keras. It trains
    a model for one epoch on the CIFAR-10 training set and periodically checks
    accuracy on the CIFAR-10 validation set.

    Inputs:
    - model_init_fn: A function that takes no parameters; when called it
      constructs the model we want to train: model = model_init_fn()
    - optimizer_init_fn: A function which takes no parameters; when called it
      constructs the Optimizer object we will use to optimize the model:
      optimizer = optimizer_init_fn()
    - num_epochs: The number of epochs to train for

    Returns: Nothing, but prints progress during trainingn
    """
    tf.reset_default_graph()
    with tf.device(device):
        x = tf.placeholder(tf.float32, [None, 32, 32, 3])
        y = tf.placeholder(tf.int32, [None])

        is_training = tf.placeholder(tf.bool, name='is_training')
        scores = model_init_fn(x, is_training)
        print(scores)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y, logits=scores)
        loss = tf.reduce_mean(loss)

        optimizer = optimizer_init_fn()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        t = 0
        for epoch in range(num_epochs):
            print('Starting epoch %d' % epoch)
            for x_np, y_np in train_dset:
                feed_dict = {x: x_np, y: y_np, is_training: 1}
                loss_np, _ = sess.run([loss, train_op], feed_dict=feed_dict)
                if t % print_every == 0:
                    print('Iteration %d, loss = %.4f' % (t, loss_np))
                    check_accuracy(sess, val_dset, x, scores,
                                   is_training=is_training)
                    print()
                t += 1


def model_init_fn(inputs, is_training):
    model = None

    def fire_module(inputs, s1, e1, e3):
        fire_squeeze = \
            tf.layers.conv2d(inputs, s1,
                             kernel_size=(1, 1), padding='SAME',
                             activation=tf.nn.relu, strides=(1, 1),
                             kernel_initializer=tf.glorot_uniform_initializer())
        fire_expand1 = \
            tf.layers.conv2d(fire_squeeze, e1,
                             kernel_size=(1, 1), padding='SAME',
                             activation=tf.nn.relu, strides=(1, 1),
                             kernel_initializer=tf.glorot_uniform_initializer())
        fire_expand2 = \
            tf.layers.conv2d(fire_squeeze, e3,
                             kernel_size=(2, 2), padding='SAME',
                             activation=tf.nn.relu, strides=(1, 1),
                             kernel_initializer=tf.glorot_uniform_initializer())
        fire_merge = tf.concat([fire_expand1, fire_expand2], 3)
        return fire_merge

    conv1 = tf.layers.conv2d(inputs, 96, kernel_size=(7, 7), activation=tf.nn.relu,
                             kernel_initializer=tf.glorot_uniform_initializer(),
                             strides=(1, 1), padding='SAME')
    pool1 = tf.layers.max_pooling2d(conv1, pool_size=(2, 2), strides=(2, 2))
    fire2 = fire_module(pool1, 16, 64, 64)
    fire3 = fire_module(fire2, 16, 64, 64)
    fire4 = fire_module(fire2, 32, 128, 128)

    dp4 = tf.layers.dropout(inputs=fire4, training=is_training)
    pool4 = tf.layers.max_pooling2d(dp4, pool_size=(2, 2), strides=(2, 2))
    flat5 = tf.layers.flatten(pool4)
    net = tf.layers.dense(flat5, units=10,
                          kernel_initializer=tf.glorot_uniform_initializer())
    return net


def optimizer_init_fn():
    optimizer = None
    optimizer = tf.train.AdamOptimizer(learning_rate=4e-4)
    return optimizer


NHW = (0, 1, 2)
# 载入数据
X_train, y_train, X_val, y_val, X_test, y_test = load_cifar10()
print("数据读取完毕")

# 读取数据
train_dset = Dataset(X_train, y_train, batch_size=64, shuffle=True)
val_dset = Dataset(X_val, y_val, batch_size=64, shuffle=False)
test_dset = Dataset(X_test, y_test, batch_size=64)
print("数据处理完毕")

device = '/gpu:0'
print_every = 700
num_epochs = 10
train_part34(model_init_fn, optimizer_init_fn, num_epochs)
print("模型训练结束")

import numpy as np
import matplotlib.pyplot as plt
import math
import os
import pickle
import tensorflow as tf
import sys

data_path = "/Users/Ben/data/CIFAR-10/"

def one_hot_encoded(class_numbers, num_classes=None):
    num_classes = np.max(class_numbers) + 1
    return np.eye(num_classes, dtype=float)[class_numbers]

def _get_file_path(filename=""):
    return os.path.join(data_path, "cifar-10-batches-py/", filename)

def _unpickle(filename):
    file_path = _get_file_path(filename)
    print("Loading data: " + file_path)

    with open(file_path, mode='rb') as file:
        data = pickle.load(file, encoding='bytes')

    return data

def _convert_images(raw):
    raw_float = np.array(raw, dtype=float) / 255.0

    images = raw_float.reshape([-1, 3, 32, 32])

    images = images.transpose([0, 2, 3, 1])

    return images

def _load_data(filename):
    data = _unpickle(filename)

    raw_images = data[b'data']

    cls = np.array(data[b'labels'])

    images = _convert_images(raw_images)

    return images, cls

#def load_class_names():
 #   raw = _unpickle(filename="batches.meta")[b'label_names']

  #  names = [x.decode('utf-8') for x in raw]

 #   return names

def load_training_data():
    images = np.zeros(shape=[50000, 32, 32, 3], dtype=float)
    cls = np.zeros(shape=[50000], dtype=int)

    begin = 0

    for i in range(5):
        images_batch, cls_batch = _load_data(filename="data_batch_" + str(i + 1))

        num_images = len(images_batch)

        end = begin + num_images

        images[begin:end, :] = images_batch

        cls[begin:end] = cls_batch

        begin = end

    return images, cls, one_hot_encoded(class_numbers=cls, num_classes=10)

def load_test_data():
    images, cls = _load_data(filename="test_batch")

    return images, cls, one_hot_encoded(class_numbers=cls, num_classes=10)

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = 32
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000
batch_size = 10

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.001       # Initial learning rate.

X = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='X')

kernel = tf.get_variable('weights', [5, 5, 3, 64],
                           initializer=tf.truncated_normal_initializer(stddev=.01, dtype=tf.float32), dtype=tf.float32)

biases = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0.0), dtype=tf.float32)

kernel2 = tf.get_variable('weights2', [5, 5, 64, 64],
                           initializer=tf.truncated_normal_initializer(stddev=.01, dtype=tf.float32), dtype=tf.float32)

biases2 = tf.get_variable('biases2', [64], initializer=tf.constant_initializer(0.1), dtype=tf.float32)

weights3 = tf.get_variable('weights3', [4096, 384],
                           initializer=tf.truncated_normal_initializer(stddev=.04, dtype=tf.float32), dtype=tf.float32)
biases3 = tf.get_variable('biases3', [384], initializer=tf.constant_initializer(0.1), dtype=tf.float32)

weights4 = tf.get_variable('weights4', [384, 192],
                           initializer=tf.truncated_normal_initializer(stddev=.04, dtype=tf.float32), dtype=tf.float32)
biases4 = tf.get_variable('biases4', [192], initializer=tf.constant_initializer(0.1), dtype=tf.float32)

weights5 = tf.get_variable('weights5', [192, NUM_CLASSES],
                           initializer=tf.truncated_normal_initializer(stddev=1/192.0, dtype=tf.float32), dtype=tf.float32)
biases5 = tf.get_variable('biases5', [NUM_CLASSES], initializer=tf.constant_initializer(0.0), dtype=tf.float32)

def inference(images):

  conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')

  pre_activation = tf.nn.bias_add(conv, biases)

  conv1 = tf.nn.relu(pre_activation, name='conv1')

  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')

  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

  conv2 = tf.nn.conv2d(norm1, kernel2, [1, 1, 1, 1], padding='SAME')

  pre_activation2 = tf.nn.bias_add(conv2, biases2)

  conv2f = tf.nn.relu(pre_activation2, name='conv2f')

  print(conv2f)


  norm2 = tf.nn.lrn(conv2f, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')

  print(norm2)

  pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')

  print(pool2)

  reshape = tf.reshape(pool2, [batch_size, -1])

  dim = reshape.get_shape()[1].value

  local3 = tf.nn.relu(tf.matmul(reshape, weights3) + biases3, name='local3')

  local4 = tf.nn.relu(tf.matmul(local3, weights4) + biases4, name='local4')

  softmax_linear = tf.add(tf.matmul(local4, weights5), biases5, name='softmax_linear')

  return softmax_linear

tpred = inference(X)

Y = tf.placeholder(tf.float32, shape=[None], name='Y')

def loss(logits, labels):
  """Add L2Loss to all the trainable variables.
  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]
  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')

tloss = loss(tpred, Y)

def train(total_loss, global_step):
  """Train CIFAR-10 model.
  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.
  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.summary.scalar('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_sum = tf.summary.scalar("loss", total_loss)

  # Compute gradients.
  opt = tf.train.GradientDescentOptimizer(lr)
  grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
      tf.summary.histogram(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
      if grad is not None:
          tf.summary.histogram(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
      train_op = tf.no_op(name='train')

  return train_op

global_step = tf.Variable(0, name='global_step', trainable=False)

mytrain_op = train(tloss, global_step)
#load data

xtrain = np.empty((50000, 32, 32, 3))

ytrain = np.empty(50000)

xtest = np.empty((10000, 32, 32, 3))

ytest = np.empty(10000)

xtrain, ytrain, _ = load_training_data()

xtest, ytest, _ = load_test_data()

xtrain = np.float32(xtrain / 255.0)

ytrain = np.float32(ytrain)

xtest = np.float32(xtest / 255.0)

ytest = np.float32(ytest)

batches = 40
batch = 0
test = 0
tests = 50
correct = 0.0
epoch = 0
epochs = 2
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
with sess.as_default():
    for epoch in range(epochs):
        np.random.shuffle(xtrain)
        np.random.shuffle(ytrain)
        for batch in range(batches):
            _, currentloss = sess.run([mytrain_op, tloss], feed_dict={X: xtrain[batches*batch_size:batches*batch_size + batch_size, :, :, :],
                                      Y: ytrain[batches*batch_size:batches*batch_size + batch_size]})
            batch += 1
        epoch += 1

    for test in range(tests):
        testresults = tpred.eval(feed_dict={X: xtest[test*batch_size:test*batch_size + batch_size, :, :, :]})
        matches = (np.argmax(testresults, axis=1) == ytest[test*batch_size:test*batch_size + batch_size])
        print(testresults)
        print(np.argmax(testresults, axis=1))
        print(ytest[test*batch_size:test*batch_size + batch_size])
        correct += int(np.sum(matches))
        test += 1
print(correct/(tests*batch_size*epochs))
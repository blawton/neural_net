import numpy as np
import matplotlib.pyplot as plt
import math
import os
import pickle
import sys
import time

print(sys.version)
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

#LayerCode

def fc_forward(inputs, weight, bias):
    """Compute forward propagation for a fully-connected layer.

    Args:
        inputs: The input image stack of the layer, of the shape (width, height,
            depth).
        weight: A numpy array of shape (n_in, n_out), where n_in = width *
            height * depth.
        bias: A numpy array of shape (n_out,).

    Returns:
        outputs: The output of the layer, of the shape (n_out,).
        cache: The tuple (inputs, weight, bias).
    """
    input_vec = np.reshape(inputs, -1)
    outputs = np.dot(weight.T, input_vec) + bias
    cache = (inputs, weight, bias)
    return outputs, cache


def fc_backward(dout, cache):
    """Compute backward propagation for a fully-connected layer.

    Args:
        dout: The gradient w.r.t. the output of the layer, of shape (n_out,).
        cache: The cache in the return of fc_forward.

    Returns:
        din: The gradient w.r.t. the input of the layer, of shape (width,
            height, depth).
        dweight: The gradient w.r.t. the weight, of shape (n_in, n_out).
        dbias: The gradient w.r.t. the bias, of shape (n_out,).
    """
    inputs, weight, _ = cache
    din_vec = np.dot(weight, dout)
    din = np.reshape(din_vec, inputs.shape)
    dweight = np.outer(inputs, dout)
    dbias = dout
    return din, dweight, dbias


def softmax(inputs, label):
    """Compute softmax function, loss function and its gradient.

    Args:
        inputs: The input scores, of shape (c,), where x[j] is the score for the
            j-th class.
        label: The true label of the input, taking from {0, 1, ..., c-1}.

    Returns:
        probs: The probabilities for each class.
        loss: The value of loss function.
        din: The gradient of loss function.
    """
    #print(inputs)
    input_shift = inputs - max(inputs)
    sum_exp = sum(np.exp(input_shift))
    log_probs = input_shift - math.log(sum_exp)
    probs = np.exp(log_probs)
    loss = -log_probs[label]
    din = probs.copy()
    din[label] -= 1
    #print(inputs)
    #print(probs)
    return probs, loss, din

xtrain = np.empty((50000, 32, 32, 3))

ytrain = np.empty((50000, 10))

#load_class_names()

xtrain,_,ytrain = load_training_data()

W0 = np.random.normal(0., .01, (16,3,3,3))
#b0 = np.zeros(16)

W1 = np.random.normal(0., .01, (3600,10))
b1 = np.random.normal(0., .01, 10)

channels = 16
def pool(input):
    cache=np.empty((channels, 30, 30))
    output=np.empty((channels, 15, 15))
    for chan in range(0, channels):
        for x2 in range(0, 15):
            for x1 in range(0, 15):
                output[chan, x1, x2] = np.amax(input[chan, (2*x1):(2*x1 + 2), (2*x2):(2*x2 + 2)])
    return output

def filter(filters, input):
    output = np.empty((16, 30, 30))
    for chan in range(16):
        for x1 in range(30):
            for x2 in range(30):
                output[chan,x1,x2] = np.sum(input[x1:x1+3, x2:x2+3,:]*filters[chan,:,:,:])
    return(output)
pool_tester = xtrain[1, 0:30, 0:30]
filter_tester = xtrain[0]
#print(filter(W0, filter_tester))

def relu_forward(input):
    #print(W0[1,:,:,:])
    return(np.maximum(input, 0))

relu_test= np.array([-3,1,-.2,0,1,4,1,-.05,-6,5])
dout_test = ([2,2,2,2,2,4,4,4,4,4])
def relu_backward(dout, cache):
    inputs = cache
    B = np.array([x>0. for x in inputs])
    #print(B)
    din = np.multiply(B.astype(int), dout)
    return(din)
def W0_backward(dout, inputs):
    dweight=np.zeros((16,3,3,3))
    for k in range(16):
        for p1 in range(30):
            for p2 in range(30):
                dweight[k,:,:,:] += (inputs[p1:p1+3,p2:p2+3,:]*dout[k, p1, p2])
    return(dweight)

def pool_backward(dout, inputs):
    din = np.zeros(inputs.shape)
    for chan in range(0, inputs.shape[0]):
        for x2 in range(0,int(inputs.shape[1]/2)):
            for x1 in range(0,int(inputs.shape[2]/2)):
                mindex = np.argmax(inputs[chan, (2*x1):(2*x1 + 2),(2*x2):(2*x2 + 2)])
                din[chan, 2*x1 + int(mindex/2),2*x2 + (mindex%2)]=dout[chan, x1, x2]
                #print(2*x1 + (mindex/2))
                #print(2*x2 + (mindex%2))
    return(din)
batches = 500
batch_size = 1
correct = 0.0
eta = .001
startt= time.time()
for i1 in range(batches):
    sumdW0 = 0.0
    sumdW1 = 0.0
    sumdb1 = 0.0
    for i2 in range(batch_size):
        filtered = filter(W0, xtrain[batch_size*i1 + i2, :, :, :])
        pooled = pool(relu_forward(filtered))
        probs, loss, dinsoft = softmax(fc_forward(pooled, W1, b1)[0], np.argmax(ytrain[batch_size*i1 + i2,:]))
        dinfc, dW1, db1 = fc_backward(dinsoft, (pooled, W1, b1))
        #for i in range(16):
            #print(relu_backward(pool_backward(dinfc, relu_forward(filtered)), filtered)[i,:,:])
        dW0 = W0_backward(relu_backward(pool_backward(dinfc, relu_forward(filtered)), filtered) , xtrain[batch_size*i1 + i2, :, :, :])
        #print(dW0)
        #print(dW1)
        #print(db1)
        if np.argmax(probs)==np.argmax(ytrain[batch_size*i1 + i2,:]):
            correct += 1.
        sumdW0 += dW0
        sumdW1 += dW1
        sumdb1 += db1
    dW0tot = sumdW0/batch_size
    dW1tot = sumdW1/batch_size
    db1tot = sumdb1/batch_size
    #print(dW0tot)
    #print(dW1tot)
    #print(db1tot)
    W0 -= (eta*dW0tot)
    W1 -= (eta*dW1tot)
    b1 -= (eta*db1tot)
endt=time.time()

print(endt-startt)
print(correct/float(batch_size*batches))
#print(relu_backward(dout_test, relu_test))
#print(pool_tester_b)
#print(pool_backward(np.ones((1,8,8)), pool_tester_b))
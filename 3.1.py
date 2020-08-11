import numpy as np
import matplotlib.pyplot as plt
import math
import os
import pickle
import sys
import time

print(sys.version)

#Important variables used to retrieve data and srtructure neural net
data_path = "/CIFAR-10/"
channels = 16

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

ylabels = np.empty(50000)
#load_class_names()

xtrain, ylabels ,ytrain = load_training_data()
#ylabels = np.ones(50000, dtype=int)*7
xtest, ytest, _ = load_test_data()

xtrain = (xtrain-np.mean(xtrain))/255.0

W0 = np.random.normal(0., .01, (channels,3,3,3))
b0 = np.random.normal(0, .01, channels)

W1 = np.random.normal(0., .01, (3600,10))
b1 = np.random.normal(0, .01, 10)


#A pooling layer that takes the maximum of (2,2) squares in the input image to reduce dimensionality
def pool(input):
    cache=np.empty((2, channels, 15, 15), dtype=np.int)
    output=np.empty((channels, 15, 15), dtype=np.int)
    for chan in range(0, channels):
        for x2 in range(0, 15):
            for x1 in range(0, 15):
                mindex = np.argmax(input[chan, (2*x1):(2*x1 + 2), (2*x2):(2*x2 + 2)])
                cache[0, chan, x1, x2] = mindex // 2
                cache[1, chan, x1, x2] = mindex % 2
                output[chan, x1, x2] = input[chan,(2*x1)+cache[0, chan, x1, x2], (2*x2)+cache[1, chan, x1, x2]]
    return output, cache

#Test pooling with random data
pool_tester = np.random.randint(8, high=None, size=(channels, 30, 30))
print(pool_tester)
poold=pool(pool_tester)
print(poold[0])
print(poold[1])

#Defines convolution of input with each filter in a (channel,3,3,3)-shaped array
def filter(filters, b, input):
    output = np.empty((channels, 30, 30))
    for chan in range(channels):
        for x1 in range(30):
            for x2 in range(30):
                output[chan,x1,x2] = np.sum(input[x1:x1+3, x2:x2+3,:]*filters[chan,:,:,:]) + b[chan]
    return(output)

#Test the convolution leayer with W0 as the weights, b0 as constants,
#and the first piece of training data as input
filter_tester = xtrain[0]
print(filter(W0, b0, filter_tester))


#rectified linear function used to introduce a nonlinearity to neural net
def relu_forward(input):
    #print(input)
    cache = np.array([x>0. for x in input])
    output = np.multiply(cache, input)
    #print(output)
    return(output, cache)

#Backwards propogation of the rectifier (ReLU) gradient
def relu_backward(dout, cache):
    din = np.multiply(dout, cache)
    return(din)

#Tests backwards prop. of rectifier gradient
relu_test= np.array([-3,1,-.2,0,1,4,1,-.05,-6,5])
dout_test = ([2,2,2,2,2,4,4,4,4,4])
print(relu_backward(dout_test, relu_test))

#Backward prop. of gradient from first layer (convolution w/ filters)
def W0_backward(dout, inputs):
    dweight=np.zeros((channels,3,3,3))
    db0=np.zeros(channels)
    for k in range(channels):
        for p1 in range(30):
            for p2 in range(30):
                dweight[k,:,:,:] += (inputs[p1:p1+3,p2:p2+3,:]*dout[k, p1, p2])
                db0[k] += dout[k, p1, p2]
    return(dweight, db0)

#Backwards prop. of the gradient from the pool layer
def pool_backward(dout, inputs, cache):
    din = np.zeros(inputs.shape)
    for chan in range(0, inputs.shape[0]):
        for x2 in range(0,int(inputs.shape[1]/2)):
            for x1 in range(0,int(inputs.shape[2]/2)):
                din[chan, (2*x1)+cache[0, chan, x1, x2],(2*x2)+cache[1, chan, x1, x2]]=dout[chan, x1, x2]
                #print(2*x1 + (mindex/2))
                #print(2*x2 + (mindex%2))
    return(din)

#test pool_backward with the test data from before, using the cache from poold
print(pool_backward(np.ones((channels,15,15)), pool_tester, poold[1]))

#Specification of training+testing parameters incl. batches, step size, batch size, tests
batches = 1000
batch_size = 1
correct = 0.0
eta = .001
tests = 300

#Train the neural net using the parameters specified above
startt = time.time()
for i1 in range(batches):
    filtered = filter(W0, b0, xtrain[i1, :, :, :])
    relud, cacher = relu_forward(filtered)
    pooled, cachep = pool(relud)
    probs, loss, dinsoft = softmax(fc_forward(pooled, W1, b1)[0], ylabels[i1])
    dinfc, dW1, db1 = fc_backward(dinsoft, (pooled, W1, b1))
    #for i in range(channels):
    #print(relu_backward(pool_backward(dinfc, relu_forward(filtered)), filtered)[i,:,:])
    dW0, db0 = W0_backward(relu_backward(pool_backward(dinfc, relud, cachep), filtered) , xtrain[i1, :, :, :])
    #print(dW0)
    #print(dW1)
    #print(db1)
    #print(dW0tot)
    #print(dW1tot)
    #print(db1tot)
    W0 -= (eta*dW0)
    b0 -= (eta*db0)
    W1 -= (eta*dW1)
    b1 -= (eta*db1)
endt = time.time()

#Random choice of cifar data to test neural net
testers = np.random.choice(5000, tests, replace=False)

#Run the test of the neural net trained above
for i3 in range(tests):
    filtered = filter(W0, b0, xtest[testers[i3], :, :, :])
    relud = relu_forward(filtered)[0]
    pooled = pool(relud)[0]
    probs, loss, dinsoft = softmax(fc_forward(pooled, W1, b1)[0], ytest[testers[i3]])
    #print(np.argmax(probs))
    #print("should be =" + str(ytest[i3]))
    if np.argmax(probs) == ytest[testers[i3]]:
        correct += 1.
        #print(correct)

#Print train timing and accuracy information
print(correct/tests)
print(endt-startt)
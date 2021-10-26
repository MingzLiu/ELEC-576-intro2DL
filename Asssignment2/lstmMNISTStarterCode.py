import tensorflow as tf
if(tf.__version__.split('.')[0]=='2'):
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import matplotlib.pyplot as plt

import plotFunc

import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)  # call mnist function

learningRate = 1e-3
trainingIters = 30000
batchSize = 100
displayStep = 10

nInput = 28  # we want the input to take the 28 pixels
nSteps = 28  # every 28
nHidden = 256  # number of neurons for the RNN
nClasses = 10  # this is MNIST so you know

x = tf.placeholder('float', [None, nSteps, nInput])
y = tf.placeholder('float', [None, nClasses])

weights = {
    'out': tf.Variable(tf.random_normal([nHidden, nClasses]))
}

biases = {
    'out': tf.Variable(tf.random_normal([nClasses]))
}


def RNN(x, weights, biases):
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, nInput])
    x = tf.split(x, nSteps, 0)  # configuring so you can get it as needed for the 28 pixels

    # find which lstm to use in the documentation
    # lstmCell = tf.compat.v1.nn.rnn_cell.LSTMCell(nHidden, forget_bias=1.0, reuse=tf.AUTO_REUSE)
    # lstmCell = rnn_cell.GRUCell(nHidden)
    lstmCell = rnn_cell.BasicRNNCell(nHidden, reuse=tf.AUTO_REUSE)

    outputs, states = rnn.static_rnn(lstmCell, x, dtype = tf.float32)  # for the rnn where to get the output and hidden state

    return tf.matmul(outputs[-1], weights['out']) + biases['out']


pred = RNN(x, weights, biases)

# optimization
# create the cost, optimization, evaluation, and accuracy
# for the cost softmax_cross_entropy_with_logits seems really good
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(cost)

correctPred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
init = tf.global_variables_initializer()
losses = []
accs = []
with tf.Session() as sess:
    sess.run(init)
    step = 1

    while step * batchSize < trainingIters:
        batchX, batchY = mnist.train.next_batch(batchSize)  # mnist has a way to get the next batch
        batchX = batchX.reshape((batchSize, nSteps, nInput))

        sess.run(optimizer, feed_dict={x: batchX, y: batchY})

        acc = sess.run(accuracy, feed_dict={x: batchX, y: batchY})
        loss = sess.run(cost, feed_dict={x: batchX, y: batchY})

        losses.append(loss)
        accs.append(acc)

        if step % displayStep == 0:
            print("Iter " + str(step * batchSize) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print('Optimization finished')

    testData = mnist.test.images.reshape((-1, nSteps, nInput))
    testLabel = mnist.test.labels
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: testData, y: testLabel}))

    sess.close()

# plotFunc.plotFunc(accs, losses, "BasicRNNCell")
# plotFunc.plotFunc(accs, losses, "GRUCell")
# plotFunc.plotFunc(accs, losses, "BasicLSTMCell")
plotFunc.plotFunc(accs, losses, "256 units")

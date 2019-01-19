import tensorflow as tf
import numpy as np

test_data_size = 2000
iterations = 50000
learn_rate = 0.005


def generate_test_values():
    train_x = []
    train_y = []

    for _ in xrange(test_data_size):
        x = np.random.rand()
        y_f = 8 / x
        train_y.append(y_f)
        train_x.append(x)

    return np.transpose([train_x]), np.transpose([train_y])


train_dataset,train_values = generate_test_values()

a = tf.Variable(tf.random_normal([1]), name="a")
x = tf.placeholder(tf.float32, [None, 1], name="idf")
y = tf.placeholder(tf.float32, [None, 1])
model = tf.divide(a, x)

cost = tf.reduce_mean(tf.square(y - model))
train = tf.train.GradientDescentOptimizer(learn_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)

    for _ in xrange(iterations):
        session.run(train, feed_dict={
            x: train_dataset,
            y: train_values
        })

    print "cost = {}".format(session.run(cost, feed_dict={
        x: train_dataset,
        y: train_values
    }))

    print "a = {}".format(session.run(a))

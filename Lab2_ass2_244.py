
# coding: utf-8

# In[26]:


import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
rng = numpy.random


# In[43]:


# Parameters
learning_rate_1 = 0.005
learning_rate_2 = 0.0005
learning_rate_3 = 0.001
training_epochs_1 = 100
training_epochs_2 = 500
training_epochs_3 = 100000
display_step_1 = 5
display_step_2 = 10
display_step_3 = 500


# In[28]:


train_X = numpy.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1,4.5,4.32,6.434,6.765,3.123,9,6,3.44])
train_Y = numpy.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3,7.042,10.791,5.313,7.997,5.654,9.27,3.1,4.5])
n_samples = train_X.shape[0]
n_samples_1 = train_Y.shape[0]
print(n_samples,n_samples_1)


# In[29]:


# tf Graph Input
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Set model weights
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")


# In[30]:


# Construct a linear model
pred = tf.add(tf.multiply(X, W), b)


# In[33]:


#learning rate = 0.005
#training epoch = 100
#J(θ)=1/2m * ∑ i=1 to m (hθ(x(i))−y(i))^2
# Mean squared error
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
# Gradient descent
optimizer = tf.train.AdagradOptimizer(learning_rate_1).minimize(cost)


# In[34]:


# Initializing the variables
init = tf.global_variables_initializer()


# In[47]:


# Launch the graph
#
with tf.Session() as sess:
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs_1):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        #Display logs per epoch step
        if (epoch+1) % display_step_1 == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
            print ("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), 
                "W=", sess.run(W), "b=", sess.run(b))

    print ("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print ("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

    #Graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()


# In[36]:


#learning rate = 0.0005
#training epoch = 500
#J(θ)=1/2m * ∑ i=1 to m (hθ(x(i))−y(i))^2
# Mean squared error
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
# Gradient descent
optimizer = tf.train.AdagradOptimizer(learning_rate_2).minimize(cost)


# In[37]:


# Initializing the variables
init = tf.global_variables_initializer()


# ## Launch the graph
# with tf.Session() as sess:
#     sess.run(init)
# 
#     # Fit all training data
#     for epoch in range(training_epochs_2):
#         for (x, y) in zip(train_X, train_Y):
#             sess.run(optimizer, feed_dict={X: x, Y: y})
# 
#         #Display logs per epoch step
#         if (epoch+1) % display_step_2 == 0:
#             c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
#             print ("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), 
#                 "W=", sess.run(W), "b=", sess.run(b))
# 
#     print ("Optimization Finished!")
#     training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
#     print ("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')
# 
#     #Graphic display
#     plt.plot(train_X, train_Y, 'ro', label='Original data')
#     plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
#     plt.legend()
#     plt.show()

# In[40]:


#learning_rate_3 = 0.001
#training_epochs_3 = 100000
#J(θ)=1/2m * ∑ i=1 to m (hθ(x(i))−y(i))^2
# Mean squared error
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
# Gradient descent
optimizer = tf.train.AdagradOptimizer(learning_rate_3).minimize(cost)


# In[41]:


# Initializing the variables
init = tf.global_variables_initializer()


# In[45]:


# Launch the graph
#
with tf.Session() as sess:
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs_3):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        #Display logs per epoch step
        if (epoch+1) % display_step_3 == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
            print ("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), 
                "W=", sess.run(W), "b=", sess.run(b))

    print ("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print ("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

    #Graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()


# In[ ]:





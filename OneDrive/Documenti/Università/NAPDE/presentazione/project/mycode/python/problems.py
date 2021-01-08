# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 11:02:31 2020

@author: beacr
"""

from pdebase import *

#2-dimensional smooth case, boundary data = 0 :
# delta(u(x,y)) = -2*pi^2 * sin(pi*x) * sin(pi*y)
class Problem1(NNPDE):
    # data to be modified
    def exactsol(self,x,y):
        return np.sin(np.pi * x) * np.sin(np.pi * y)

    def A(self, x):
        return 0

    def B(self, x):
        return x[:, 0] * (1 - x[:, 0]) * x[:, 1] * (1 - x[:, 1])

    def f(self, x):
        return -2 * np.pi ** 2 * tf.sin(np.pi * x[:, 0]) * tf.sin(np.pi * x[:, 1])

    def loss_function(self):
        deltah = compute_delta(self.u, self.x)
        delta = self.f(self.x)
        res = tf.reduce_sum((deltah - delta) ** 2)
        assert_shape(res, ())
        return res
        # end modification
				
				
#2-dimensional smooth case with (null) boundary data to be approximated
#  delta(u(x,y)) = -2*pi^2 * sin(pi*x) * sin(pi*y)
class Problem1_BD(NNPDE2):

    def exactsol(self,x,y):
        return np.sin(np.pi * x) * np.sin(np.pi * y)

    def tfexactsol(self,x):
        return tf.sin(np.pi * x[:,0]) * tf.sin(np.pi * x[:,1])


    def f(self, x):
        return -2 * np.pi ** 2 * tf.sin(np.pi * x[:, 0]) * tf.sin(np.pi * x[:, 1])

    def loss_function(self):
        deltah = compute_delta(self.u, self.x)
        delta = self.f(self.x)
        res = tf.reduce_sum((deltah - delta) ** 2)
        assert_shape(res, ())
        return res


#2-dimensional peak case with (null) boundary data to be approximated
#  delta(u(x,y)) = 2*alpha^2 * ((x-xc)^2 + (y-yc)^2) * (1 + 2*alpha^2 * (x^2 + y^2)) +
#                      + 4*alpha^2 * (x^2 + y^2) - pi^2 * sin(pi*x)
class ProblemPeak_BD(NNPDE2):
    def __init__(self, batch_size, N, refn):
        self.alpha = 1000
        self.xc = 0.5
        self.yc = 0.5
        NNPDE2.__init__(self,batch_size, N, refn)

    def bsubnetwork(self, x, reuse = False):
        with tf.variable_scope("boundary"):
            for i in range(self.N):
                x = tf.layers.dense(x, 256, activation=tf.nn.tanh, name="bdense{}".format(i), reuse=reuse)
            x = tf.layers.dense(x, 1, activation=None, name="blast", reuse=reuse)
            x = tf.squeeze(x, axis=1)
            assert_shape(x, (None,))
        return x

		##loss function as defined in the base class (SSE)
    # def loss_function(self):
    #     deltah = compute_delta(self.u, self.x)
    #     delta = self.f(self.x)
    #     delta = tf.clip_by_value(delta, -1e2, 1e2)
    #     deltah = tf.clip_by_value(deltah, -1e2, 1e2)
    #     res = tf.reduce_sum((deltah - delta) ** 2)
    #     assert_shape(res, ())
    #     return res

		#loss function defined as sum of (errors / laplacian(u))^2
    def loss_function(self):
        deltah = compute_delta(self.u, self.x)
        delta = self.f(self.x)
        # delta = tf.clip_by_value(delta, -1e2, 1e2)
        # deltah = tf.clip_by_value(deltah, -1e2, 1e2)
        # weight = tf.clip_by_norm(1/delta**2, 10)
        # weight = tf.reduce_sum(delta**2)/delta**2
        res = tf.reduce_sum( 1/deltah**2 * (deltah - delta) ** 2)
        assert_shape(res, ())
        return res

    def exactsol(self, x, y):
        return np.exp(-self.alpha*((x-self.xc)**2+(y-self.yc)**2)) + np.sin(np.pi * x)

    def tfexactsol(self, x):
        return tf.exp(-1000 * ((x[:,0] - self.xc) ** 2 + (x[:,1] - self.yc) ** 2))+ tf.sin(np.pi * x[:,0])

    def f(self, x):
        return -4*self.alpha*self.tfexactsol(self.x) + 4*self.alpha**2*self.tfexactsol(self.x)* \
                                                       ((x[:, 0] - self.xc) ** 2 + (x[:, 1] - self.yc) ** 2) - np.pi**2 * tf.sin(np.pi * x[:,0])

    def train(self, sess, i=-1):
        # self.X = rectspace(0,0.5,0.,0.5,self.n)
        bX = np.zeros((4*self.batch_size, 2))
        bX[:self.batch_size,0] = np.random.rand(self.batch_size)
        bX[:self.batch_size,1] = 0.0

        bX[self.batch_size:2*self.batch_size, 0] = np.random.rand(self.batch_size)
        bX[self.batch_size:2*self.batch_size, 1] = 1.0

        bX[2*self.batch_size:3*self.batch_size, 0] = 0.0
        bX[2*self.batch_size:3*self.batch_size, 1] = np.random.rand(self.batch_size)

        bX[3*self.batch_size:4*self.batch_size, 0] = 1.0
        bX[3 * self.batch_size:4 * self.batch_size, 1] = np.random.rand(self.batch_size)

        for _ in range(5):
            _, bloss = sess.run([self.opt1, self.bloss], feed_dict={self.x_b: bX})

        X = np.random.rand(self.batch_size, 2)
        # if i>50:
        X = np.concatenate([X,rectspace(0.4,0.5,0.4,0.5,5)], axis=0)
        _, loss = sess.run([self.opt2, self.loss], feed_dict={self.x: X})

        if i % 10 == 0:
            print("Iteration={}, bloss = {}, loss= {}".format(i, bloss, loss))
				
				########## record loss ############
        self.rbloss.append(bloss)
        self.rloss.append(loss)
        uh = sess.run(self.u, feed_dict={self.x: self.refX}) #approximate solutions at each iteration
        Z = uh.reshape((self.refn, self.refn))
        uhref = self.exactsol(self.X, self.Y)
        self.rl2.append( np.sqrt(np.mean((Z-uhref)**2)) )
        ########## record loss ############



#2-dimensional singularity case, boundary data
#delta(u(x,y)) = -0.24*y^(-1.4)
class ProblemBLSingularity_BD(NNPDE2):
    def __init__(self, batch_size, N, refn):
        self.alpha = 0.6
        NNPDE2.__init__(self,batch_size, N, refn)

    def exactsol(self, x, y):
        return y**0.6

    def tfexactsol(self, x):
        return tf.pow(x[:,1],0.6)

    def f(self, x):
        return self.alpha*(self.alpha-1)*x[:,1]**(self.alpha-2)

    def loss_function(self):
        deltah = compute_delta(self.u, self.x)
        delta = self.f(self.x)
        delta = tf.clip_by_value(delta, -1e2, 1e2)
        deltah = tf.clip_by_value(deltah, -1e2, 1e2)
        res = tf.reduce_sum((deltah - delta) ** 2)
        assert_shape(res, ())
        return res


#2-dimensional peak case, no boundary data
#delta(u(x,y)) = 2*alpha^2 * ((x-xc)^2 + (y-yc)^2) * (1 + 2*alpha^2 * (x^2 + y^2)) +
#                      + 4*alpha^2 * (x^2 + y^2) - pi^2 * sin(pi*x)
class ProblemPeak(NNPDE):
    def __init__(self, batch_size, N, refn):
        self.alpha = 1000
        self.xc = 0.5
        self.yc = 0.5
        NNPDE.__init__(self,batch_size, N, refn)

    # data to be modified
    def exactsol(self,x,y):
        return np.exp(-1000 * ((x - self.xc) ** 2 + (y - self.yc) ** 2))

    def A(self, x):
        return tf.exp(-1000 * ((x[:,0] - self.xc) ** 2 + (x[:,1] - self.yc) ** 2)) +tf.sin(np.pi * x[:,0]) * tf.sin(np.pi * x[:,1])

    def B(self, x):
        return x[:, 0] * (1 - x[:, 0]) * x[:, 1] * (1 - x[:, 1])



    def tfexactsol(self, x):
        return tf.exp(-1000 * ((x[:,0] - self.xc) ** 2 + (x[:,1] - self.yc) ** 2))

    def f(self, x):
        return -4*self.alpha*self.tfexactsol(self.x) + 4*self.alpha**2*self.tfexactsol(self.x)* \
                                                       ((x[:, 0] - self.xc) ** 2 + (x[:, 1] - self.yc) ** 2)

    def loss_function(self):
        deltah = compute_delta(self.u, self.x)
        delta = self.f(self.x)
        res = tf.reduce_sum((deltah - delta) ** 2)
        assert_shape(res, ())
        return res

    def train(self, sess, i=-1):
        # self.X = rectspace(0,0.5,0.,0.5,self.n)
        X = np.random.rand(self.batch_size, 2)
        # X = np.concatenate([X, rectspace(0.4, 0.5, 0.4, 0.5, 5)], axis=0)
        _, loss = sess.run([self.opt, self.loss], feed_dict={self.x: X})
        if i % 10 == 0:
            print("Iteration={}, loss= {}".format(i, loss))
						
				########## record loss ############
        self.rbloss.append(bloss)
        self.rloss.append(loss)
        uh = sess.run(self.u, feed_dict={self.x: self.refX}) #approximate solutions at each iteration
        Z = uh.reshape((self.refn, self.refn))
        uhref = self.exactsol(self.X, self.Y)
        self.rl2.append( np.sqrt(np.mean((Z-uhref)**2)) )
        ########## record loss ############


#2-dimensional case with singularity, no boundary data
#delta(u(x,y)) = -0.24*x^(-1.4)
class ProblemBLSingularity(NNPDE):
    # data to be modified
    def exactsol(self,x,y):
        return x**0.6

    def A(self, x):
        return x[:,0]**0.6+tf.sin(np.pi * x[:,0]) * tf.sin(np.pi * x[:,1])

    def B(self, x):
        return x[:, 0] * (1 - x[:, 0]) * x[:, 1] * (1 - x[:, 1])

    def f(self, x):
        return 0.6*(0.6-1)*x[:,0]**(0.6-2)



#d-dimensional case
#delta(u(x1,...,xd)) = -pi^2 * d * (sin(pi*x1) * ... * sin(pi*xd))
class HighDimensionSmooth(NNPDE_ND):
    def tfexactsol(self,x):
        return tf.reduce_prod(tf.sin(np.pi * x), axis=1)

    def exactsol(self, x):
        return np.prod(np.sin(np.pi * x), axis=1)

    def f(self, x):
        return -np.pi**2*self.d* self.tfexactsol(x)

    def B(self, x):
        return tf.reduce_prod(x*(1-x),axis=1)

    def train(self, sess, i):
        self.rbloss = []
        self.rloss = []
        self.rl2 = []
        # self.X = rectspace(0,0.5,0.,0.5,self.n)
				
				# boundary points
        bX = np.random.rand(2*self.d*self.batch_size, self.d)
        for j in range(self.d):
            bX[2*j*self.batch_size:(2*j+1)*self.batch_size, j] = 1.0
            bX[(2 * j+1) * self.batch_size:(2 * j + 2) * self.batch_size, j] = 0.0

        bloss = sess.run([self.bloss], feed_dict={self.x_b: bX})[0]
        # if the loss is small enough, stop training on the boundary
        if bloss>1e-5:
            for _ in range(5):
                _, bloss = sess.run([self.opt1, self.bloss], feed_dict={self.x_b: bX})
								
				# interior points
        X = np.random.rand(self.batch_size, self.d)
        _, loss = sess.run([self.opt2, self.loss], feed_dict={self.x: X})


        # ######### record loss ############
        self.rbloss.append(bloss)
        self.rloss.append(loss)
        self.rl2.append( self.compute_L2(sess, self.X_test) )
        # ######### record loss ############
        # loss = np.inf

        if i % 10 == 0:
        	pass
            #print("Iteration={}, bloss = {}, loss= {}, L2={}".format(i, bloss, loss, self.rl2[-1]))

#d-dimensional case
#delta(u(x1,...,xd)) = -pi^2 * d * (sin(pi*x1) * ... * sin(pi*xd))
class HighDimensionPeak(NNPDE_ND):
	
    def tfexactsol(self,x):
        return tf.exp(-1000 * (tf.reduce_sum((x - 0.5) **2)))

    def exactsol(self, x):
        return np.exp(-1000 * (np.sum((x - 0.5) **2)))

    def f(self, x):
        return -4*1000*self.tfexactsol(x) + 4*1000**2*self.tfexactsol(x)* \
                                                       tf.reduce_sum((x - 0.5) ** 2)
																											 
    def B(self, x):
        return tf.reduce_prod(x*(1-x),axis=1)

    def train(self, sess, i):
        self.rbloss = []
        self.rloss = []
        self.rl2 = []
        # self.X = rectspace(0,0.5,0.,0.5,self.n)
				
				# boundary points
        bX = np.random.rand(2*self.d*self.batch_size, self.d)
        for j in range(self.d):
            bX[2*j*self.batch_size:(2*j+1)*self.batch_size, j] = 1.0
            bX[(2 * j+1) * self.batch_size:(2 * j + 2) * self.batch_size, j] = 0.0

        bloss = sess.run([self.bloss], feed_dict={self.x_b: bX})[0]
        # if the loss is small enough, stop training on the boundary
        if bloss>1e-5:
            for _ in range(5):
                _, bloss = sess.run([self.opt1, self.bloss], feed_dict={self.x_b: bX})
								
				# interior points
        X = np.random.rand(self.batch_size, self.d)
        _, loss = sess.run([self.opt2, self.loss], feed_dict={self.x: X})


        # ######### record loss ############
        self.rbloss.append(bloss)
        self.rloss.append(loss)
        self.rl2.append( self.compute_L2(sess, self.X_test) )
        # ######### record loss ############
        # loss = np.inf

        if i % 10 == 0:
        	pass
            #print("Iteration={}, bloss = {}, loss= {}, L2={}".format(i, bloss, loss, self.rl2[-1]))


class HighDimensionSingularity(NNPDE_ND):
	
    def tfexactsol(self,x):
        return tf.pow(x[:,1],0.6)

    def exactsol(self, x):
        return x**0.6

    def f(self, x):
        return 0.6*(0.6-1)*x[:,1]**(0.6-2)
																											 
    def B(self, x):
        return tf.reduce_prod(x*(1-x),axis=1)

    def train(self, sess, i):
        self.rbloss = []
        self.rloss = []
        self.rl2 = []
        # self.X = rectspace(0,0.5,0.,0.5,self.n)
				
				# boundary points
        bX = np.random.rand(2*self.d*self.batch_size, self.d)
        for j in range(self.d):
            bX[2*j*self.batch_size:(2*j+1)*self.batch_size, j] = 1.0
            bX[(2 * j+1) * self.batch_size:(2 * j + 2) * self.batch_size, j] = 0.0

        bloss = sess.run([self.bloss], feed_dict={self.x_b: bX})[0]
        # if the loss is small enough, stop training on the boundary
        if bloss>1e-5:
            for _ in range(5):
                _, bloss = sess.run([self.opt1, self.bloss], feed_dict={self.x_b: bX})
								
				# interior points
        X = np.random.rand(self.batch_size, self.d)
        _, loss = sess.run([self.opt2, self.loss], feed_dict={self.x: X})


        # ######### record loss ############
        self.rbloss.append(bloss)
        self.rloss.append(loss)
        self.rl2.append( self.compute_L2(sess, self.X_test) )
        # ######### record loss ############
        # loss = np.inf

        if i % 10 == 0:
        	pass
            #print("Iteration={}, bloss = {}, loss= {}, L2={}".format(i, bloss, loss, self.rl2[-1]))

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 11:18:24 2020

@author: beacr
"""

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D
from drawnow import drawnow, figure

#assert_shape: error messages for shape mismatch between the given tensor and a desired one
def assert_shape(x, shape):
    S = x.get_shape().as_list()						 #tensor dimensions as int values
    if len(S)!=len(shape):
        raise Exception("Shape mismatch: {} -- {}".format(S, shape))
    for i in range(len(S)):
        if S[i]!=shape[i]:
            raise Exception("Shape mismatch: {} -- {}".format(S, shape))

#compute_delta: approximate laplacian
def compute_delta(u, x):
    grad = tf.gradients(u, x)[0]
    g1 = tf.gradients(grad[:,0], x)[0]
    g2 = tf.gradients(grad[:,1], x)[0]
    delta = g1[:,0] + g2[:,1]
    assert_shape(delta, (None,))
    return delta

#compute_delta_nd: approximate laplacian in n dimensions
def compute_delta_nd(u, x, n):
    grad = tf.gradients(u, x)[0]
    g1 = tf.gradients(grad[:, 0], x)[0]
    delta = g1[:,0]
    for i in range(1,n):
        g = tf.gradients(grad[:,i], x)[0]
        delta += g[:,i]
    assert_shape(delta, (None,))
    return delta
						
#compute_dx: x-derivative
def compute_dx(u,x):
    grad = tf.gradients(u, x)[0]
    dudx = grad[:,0]
    assert_shape(dudx, (None,))
    return dudx

#compute_dy: y-derivative 
def compute_dy(u,x):
    grad = tf.gradients(u, x)[0]
    dudy = grad[:,1]
    assert_shape(dudy, (None,))
    return dudy
	
#compute_dt: time derivative
def compute_dt(u,x):
    grad = tf.gradients(u, x)[0]
    dudt = grad[:,2]
    assert_shape(dudt, (None,))
    return dudt
	
#class definition of d-dimensional NNPDE
class NNPDE_ND:
    def __init__(self,batch_size, L, N, R, d): # d- dimension, N-number of layers
        self.d = d
        self.batch_size = batch_size
        self.N = N
        self.R = R
        self.L = L

        self.x = tf.placeholder(tf.float64, (None, d))  # inner data
        self.x_b = tf.placeholder(tf.float64, (None, d))  # boundary data

        self.u_b = self.bsubnetwork(self.x_b, False)  #solution on the boundary given by
																											#subnetwork on the boundary
        self.u = self.bsubnetwork(self.x, True) + self.B(self.x) * self.subnetwork(self.x, False)
																											#overall solution given by
																											#subnetwork in the inner domain
																											#lifted by subnetwork on the boundary

        self.bloss = tf.reduce_sum((self.tfexactsol(self.x_b) - self.u_b) ** 2)
        self.loss = self.loss_function()

				#training options and initial values (minimisation of loss)
        var_list1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "boundary")
																												#get_collection: collection of data named "boundary"
																												#TRAINABLE_VARIABLES: constuctor that automatically
																												# returns a list of new variables
        self.opt1 = tf.train.AdamOptimizer(learning_rate=self.R).minimize(self.bloss, var_list=var_list1)
																												# definition of the training algorithm and
																												# initialization of the learning rate as self.R
        var_list2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "inner")
        self.opt2 = tf.train.AdamOptimizer(learning_rate=self.R).minimize(self.loss, var_list=var_list2)
        self.init = tf.global_variables_initializer()		#Only after running tf.global_variables_initializer()
																												#in a session your variables hold the values
																												#you told them to hold at declaration time
				
				#test values
        self.X_test = np.array([0.5]*self.d)[np.newaxis,:]

		# exact solution in terms of tensors
    def tfexactsol(self, x):
        raise NotImplementedError

		# exact solution in terms of arrays
    def exactsol(self, x):
        raise NotImplementedError
		
		# B = 0 on the boundary 
    def B(self, x):
        raise NotImplementedError

		# right-hand-side of the PDE
    def f(self, x):
        raise NotImplementedError

		# subnetwork defines a dense neural network on inner points with tanh activation,
		# number of hidden layers = self.L, number of neurons per layer = self.N
    def subnetwork(self, x, reuse = False):
        with tf.variable_scope("inner"):            #variable_scope: create new variables and share
																										#already created ones while providing checks
																										#to not create or share by accident
            for i in range(self.L):
                x = tf.layers.dense(x, self.N, activation=tf.nn.tanh, name="dense{}".format(i), reuse=reuse)
            x = tf.layers.dense(x, 1, activation=None, name="last", reuse=reuse)
            x = tf.squeeze(x, axis=1)								#squeeze removes element in position 1
            assert_shape(x, (None,))
        return x

		#subnetwork defines a dense neural network on boundary points with tanh activation and output dimensionality 256
    def bsubnetwork(self, x, reuse = False):
        with tf.variable_scope("boundary"):
            for i in range(self.L):
                x = tf.layers.dense(x, self.N, activation=tf.nn.tanh, name="bdense{}".format(i), reuse=reuse)
            x = tf.layers.dense(x, 1, activation=None, name="blast", reuse=reuse)
            x = tf.squeeze(x, axis=1)
            assert_shape(x, (None,))
        return x

		#compute SSE (sum of squared errors)
    def loss_function(self):
        deltah = compute_delta_nd(self.u, self.x, self.d) #- compute_dx(self.u, self.x)	- compute_dy(self.u, self.x)	#laplacian of u(x)
        delta = self.f(self.x)														  
        res = tf.reduce_sum((deltah - delta) ** 2)
        assert_shape(res, ())
        return res


    def compute_L2(self, sess, x):
        u0 = self.exactsol(x)
        u1 = sess.run(self.u, feed_dict={self.x: x})[0]   #evaluation of the graph of u on the data given by x
																													## feed_dict is a dictionary where every variable
																													## is a place_holder for given type
        return np.sqrt(np.mean((u0-u1)**2))
			
			## REMARK: need to define a train function member in the inherited classes !!



# class definition of 2-dimensional (x,y) NNPDE with boundary data learned by an ANN
# for the Poisson problem
class NNPDE:
    def __init__(self, batch_size, N, refn):
				# measures of error
        self.rloss = []			# interior loss
        self.rbloss = []		# bounding loss
        self.rl2 = []				# err_2

				# grid for the error computation
        self.refn = refn  	# reference points for the error computation
        x = np.linspace(0, 1, refn)
        y = np.linspace(0, 1, refn)
        self.X, self.Y = np.meshgrid(x, y)
        self.refX = np.concatenate([self.X.reshape((-1, 1)), self.Y.reshape((-1, 1))], axis=1)

        self.batch_size = batch_size  # batchsize (input dimensionality)
        self.N = N 	                  # number of hidden layers

        self.x = tf.placeholder(tf.float64, (None, 2)) # inner data
        self.x_b = tf.placeholder(tf.float64, (None, 2)) # boundary data

        self.u_b = self.bsubnetwork(self.x_b, False)
        self.u = self.bsubnetwork(self.x, True) + self.B(self.x) * self.subnetwork(self.x, False)

        self.bloss = tf.reduce_sum((self.tfexactsol(self.x_b)-self.u_b)**2)
        self.loss = self.loss_function()
				
        var_list1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "boundary")
        self.opt1 = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.bloss,var_list=var_list1)
        var_list2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "inner")
        self.opt2 = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss, var_list=var_list2)
        self.init = tf.global_variables_initializer()

		# B = 0 on the boundary
    def B(self, x):
        return (0 - x[:, 0]) * (1 - x[:, 0]) * (0 - x[:, 1]) * (1 - x[:, 1])

		# exact solution in terms of arrays
    def exactsol(self, x, y):
        raise NotImplementedError

		# exact solution in terms of tensors
    def tfexactsol(self, x):
        raise NotImplementedError

		# right-hand-side of the PDE
    def f(self, x):
        raise NotImplementedError

    def loss_function(self):
        deltah = compute_delta(self.u, self.x)
        delta = self.f(self.x)
        res = tf.reduce_sum((deltah - delta) ** 2)
        assert_shape(res, ())
        return res

		#subnetwork defines a dense neural network on inner points with tanh activation
		# number of layers = self.N, number of neurons per layer = 256
    def subnetwork(self, x, reuse = False):
        with tf.variable_scope("inner"):
            for i in range(self.N):
                x = tf.layers.dense(x, 256, activation=tf.nn.tanh, name="dense{}".format(i), reuse=reuse) #reuse: boolean indicating
            x = tf.layers.dense(x, 1, activation=None, name="last", reuse=reuse)          # whether to reuse weight of a previous layer
            x = tf.squeeze(x, axis=1)
            assert_shape(x, (None,))
        return x

		#subnetwork defines a dense neural network on boundary points with tanh activation
		# number of layers = self.N, number of neurons per layer = 256
    def bsubnetwork(self, x, reuse = False):
        with tf.variable_scope("boundary"):
            for i in range(self.N):
                x = tf.layers.dense(x, 256, activation=tf.nn.tanh, name="bdense{}".format(i), reuse=reuse)
            x = tf.layers.dense(x, 1, activation=None, name="blast", reuse=reuse)
            x = tf.squeeze(x, axis=1)
            assert_shape(x, (None,))
        return x

    def plot_exactsol(self):
        Z = self.exactsol(self.X, self.Y)
        ax = self.fig.gca(projection='3d')
        ax.plot_surface(self.X, self.Y, Z, rstride=1, cstride=1, cmap=cm.summer,
                        linewidth=0, antialiased=False, alpha=1.0)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
				
    def visualize(self, sess, showonlysol=False, i=None, savefig=None):
        x = np.linspace(0, 1, self.refn)
        y = np.linspace(0, 1, self.refn)
        [X, Y] = np.meshgrid(x, y)

        uh = sess.run(self.u, feed_dict={self.x: self.refX})
        Z = uh.reshape((self.refn, self.refn))

        uhref = self.exactsol(X, Y)

        def draw():
            self.fig = plt.figure()
            ax = self.fig.gca(projection='3d')

            if not showonlysol:
                ax.plot_surface(X, Y, uhref, rstride=1, cstride=1, cmap=cm.autumn,
                                 linewidth=0, antialiased=False, alpha=0.3)

            ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.summer,
                             linewidth=0, antialiased=False, alpha=0.5)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_zlim(0, 1.1)

            ax.set_xlabel('$x$')
            ax.set_ylabel('$y$')
            if i:
                plt.title("Iteration {}".format(i))
            if savefig:
                plt.savefig("{}/fig{}".format(savefig,0 if i is None else i))			

        drawnow(draw)

    def train(self, sess, i=-1):
			  #random (boundary) coordinates
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

				# training of the boudnary network
        bloss = sess.run([self.bloss], feed_dict={self.x_b: bX})[0]
        # if the loss is small enough, stop training on the boundary
        if bloss>1e-5:
            for _ in range(5):
                _, bloss = sess.run([self.opt1, self.bloss], feed_dict={self.x_b: bX})

				# random coordinates
        X = np.random.rand(self.batch_size, 2)
#        X=rectspace(0,1,0,1,self.batch_size)
				
				# training of the PDE network
        _, loss = sess.run([self.opt2, self.loss], feed_dict={self.x: X})

        ########## record loss ############
        self.rbloss.append(bloss)
        self.rloss.append(loss)
        uh = sess.run(self.u, feed_dict={self.x: self.refX}) #approximate solutions at each iteration
        Z = uh.reshape((self.refn, self.refn))
        uhref = self.exactsol(self.X, self.Y)
        self.rl2.append( np.sqrt(np.mean((Z-uhref)**2)) )
        ########## record loss ############
	

			
# class definition of 2-dimensional (x,y) NNPDE with boundary data learned by an ANN
# for the stationary diffusion-transport problem, beta=[1,1]
class NNPDE_transport(NNPDE):
		# only need to redefine the loss function computation
    def loss_function(self):
        deltah = compute_delta(self.u, self.x) - compute_dx(self.u, self.x) -  compute_dy(self.u,self.x)
        delta = self.f(self.x)
        res = tf.reduce_sum((deltah - delta) ** 2)
        assert_shape(res, ())
        return res
			
			
#class definition of NNPDE for the heat equation
class Heat_equation:
    def __init__(self, batch_size, N, refn):
				# measures of error
        self.rloss = []				# interior loss
        self.rbloss = []			# bounding loss
        self.rl2 = []					# err_2

				# grid for the error computation
        self.refn = refn  # reference points for the error computation
        x = np.linspace(0, 1, refn)   #x-variable
        y = np.linspace(0, 1, refn)   #y-variable
        t = np.linspace(0, 1, refn)   #time variable
        self.X, self.Y, self.T = np.meshgrid(x, y, t)
        self.refX = np.concatenate([self.X.reshape((-1, 1)), self.Y.reshape((-1, 1)), self.T.reshape((-1, 1))], axis=1)
        self.Z = []
				
        self.i = -1
				
        self.batch_size = batch_size  # batchsize (input dimensionality)
        self.N = N                    # number of hidden layers

        self.x = tf.placeholder(tf.float64, (None, 3)) # inner data
        self.x_b = tf.placeholder(tf.float64, (None, 3)) # boundary data

        self.u_b = self.bsubnetwork(self.x_b, False)
        self.u = self.bsubnetwork(self.x, True) + self.B(self.x) * self.subnetwork(self.x, False)

        self.bloss = tf.reduce_sum((self.g(self.x_b)-self.u_b)**2)
        self.loss = self.loss_function()

        var_list1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "boundary")
        self.opt1 = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.bloss,var_list=var_list1)
        var_list2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "inner")
        self.opt2 = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss, var_list=var_list2)
        self.init = tf.global_variables_initializer()

		# exact solution in terms of arrays
    def exactsol(self, x,y,t):
        raise NotImplementedError

		# B = 0 on the boundary for every t			
    def B(self, x):
        raise NotImplementedError

		# right-hand-side of the PDE
    def f(self, x):
        raise NotImplementedError

		# (Dirichlet) boundary condition
    def g(self, x):
        raise NotImplementedError

		# initial condition
    def u_0(self, x):
        raise NotImplementedError 
				
				
    def loss_function(self):
        deltah = compute_dt(self.u, self.x) - compute_delta(self.u, self.x)
        delta = self.f(self.x)
        res = tf.reduce_sum((deltah - delta) ** 2) + tf.reduce_sum((self.u_0(self.x) - self.u)**2)
        assert_shape(res, ())
        return res

		# subnetwork defines a dense neural network on inner points with tanh activation
		# number of hidden layers = self.N, number of neurons per layer = 256
    def subnetwork(self, x, reuse = False):
        with tf.variable_scope("inner"):
            for i in range(self.N):
                x = tf.layers.dense(x, 256, activation=tf.nn.tanh, name="dense{}".format(i), reuse=reuse) #reuse: boolean indicating
            x = tf.layers.dense(x, 1, activation=None, name="last", reuse=reuse)          # whether to reuse weight of a previous layer
            x = tf.squeeze(x, axis=1)
            assert_shape(x, (None,))
        return x

		#subnetwork defines a dense neural network on boundary points with tanh activation
		# number of hidden layers = self.N, number of neurons per layer = 256
    def bsubnetwork(self, x, reuse = False):
        with tf.variable_scope("boundary"):
            for i in range(self.N):
                x = tf.layers.dense(x, 256, activation=tf.nn.tanh, name="bdense{}".format(i), reuse=reuse)
            x = tf.layers.dense(x, 1, activation=None, name="blast", reuse=reuse)
            x = tf.squeeze(x, axis=1)
            assert_shape(x, (None,))
        return x

    def train(self, sess, i=-1):
			  # number of training iterations
        self.i = i
			
			  #random (boundary) coordinates
        # self.X = rectspace(0,0.5,0.,0.5,self.n)
        bX = np.random.rand(4*self.batch_size, 3)
        bX[:self.batch_size,0] = np.random.rand(self.batch_size)
        bX[:self.batch_size,1] = 0.0
        bX[self.batch_size:2*self.batch_size, 0] = np.random.rand(self.batch_size)
        bX[self.batch_size:2*self.batch_size, 1] = 1.0
        bX[2*self.batch_size:3*self.batch_size, 0] = 0.0
        bX[2*self.batch_size:3*self.batch_size, 1] = np.random.rand(self.batch_size)
        bX[3*self.batch_size:4*self.batch_size, 0] = 1.0
        bX[3 * self.batch_size:4 * self.batch_size, 1] = np.random.rand(self.batch_size)
        bloss = sess.run([self.bloss], feed_dict={self.x_b: bX})[0]
				
				# training of the boundary network
        # if the loss is small enough, stop training on the boundary
        if bloss>1e-5:
            for _ in range(5):
                _, bloss = sess.run([self.opt1, self.bloss], feed_dict={self.x_b: bX})

				# random coordinates
        X = np.random.rand(self.batch_size, 3)
#        X[:8,2] = np.zeros(8)
				
				# training of the PDE network
        _, loss = sess.run([self.opt2, self.loss], feed_dict={self.x: X})


        ########## record loss ############
        self.rbloss.append(bloss)
        self.rloss.append(loss)
        uh = sess.run(self.u, feed_dict={self.x: self.refX}) #approximate solutions at each iteration
        self.Z = uh.reshape((self.refn, self.refn, self.refn))
        uhref = self.exactsol(self.X, self.Y, self.T)
        self.rl2.append( np.sqrt(np.mean((self.Z-uhref)**2)) )
        ########## record loss ############
				
    def visualize(self, sess, showonlysol=False, i=None, savefig=None):
		 
        x = np.linspace(0, 1, self.refn)
        y = np.linspace(0, 1, self.refn)
        [X, Y] = np.meshgrid(x, y)

#        uh = sess.run(self.u, feed_dict={self.x: self.refX})
#        Z = uh.reshape((self.refn, self.refn, self.refn))
				
        def draw():
            for t in range (3):
                self.fig = plt.figure()
                ax = self.fig.gca(projection='3d')
                if t == 0:
                    ax.plot_surface(X, Y, self.Z[:,:,0], rstride=1, cstride=1, cmap=cm.summer,
											  linewidth=0, antialiased=False, alpha=0.5)
                    uhref = self.u_0(self.refX)
                    ZZ = uhref.eval(session=sess).reshape(self.refn, self.refn, self.refn)
                    ax.plot_surface(X, Y, ZZ[:,:,0], rstride=1, cstride=1, cmap=cm.autumn,
																linewidth=0, antialiased=False, alpha=0.3)
                else:
                    uex = self.exactsol(X, Y, t/2)				
                    ax.plot_surface(X, Y, uex, rstride=1, cstride=1, cmap=cm.autumn,
                                 linewidth=0, antialiased=False, alpha=0.3)
                    ax.plot_surface(X, Y, self.Z[:,:,t*10-1], rstride=1, cstride=1, cmap=cm.summer,
											  linewidth=0, antialiased=False, alpha=0.5)

                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_zlim(0, 1.1)

                ax.set_xlabel('$x$')
                ax.set_ylabel('$y$')
                plt.title("Iteration {}, time {}".format(i, t/2))
                if savefig:
                   plt.savefig("{}/fig{}{}".format(savefig,0 if i is None else i,t))			

        drawnow(draw)	


# class definition of 2-dimensional (x,y) NNPDE with boundary data learned by an ANN
# for the advection-diffusion problem, beta=[1,1]
class Advection_diffusion_equation(Heat_equation):
		# only need to redefine the interior loss
    def loss_function(self):
        deltah = compute_dt(self.u, self.x) - compute_delta(self.u, self.x) + compute_dx(self.u, self.x) + compute_dy(self.u, self.x)
        delta = self.f(self.x)
        res = tf.reduce_sum((deltah - delta) ** 2) + tf.reduce_sum((self.u_0(self.x) - self.u)**2)
        assert_shape(res, ())
        return res
	

# class definition of 2-dimensional (x,y) NNPDE with boundary data learned by an ANN
# for the transport problem, beta=[1,1]
class NNPDE_hyperbolic(Heat_equation):
		#compute SSE (sum of squared errors)
    def loss_function(self):
        deltah = compute_dt(self.u, self.x) + compute_dx(self.u, self.x) + compute_dy(self.u, self.x) 	#laplacian of u(x)
        delta = self.f(self.x)														  #data f(x)
        res = tf.reduce_sum((deltah - delta) ** 2) + tf.reduce_sum((self.u - self.u_0(self.x))**2)
        assert_shape(res, ())
        return res
			

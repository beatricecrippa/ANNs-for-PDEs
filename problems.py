# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 11:02:31 2020

@author: beacr
"""

from pdebase import *

###################################### Poisson #######################################################

#2-dimensional smooth problem
class Smooth_poisson(NNPDE):
    def exactsol(self,x,y):
        return np.sin(np.pi * x) * np.sin(np.pi * y)

    def tfexactsol(self,x):
        return tf.sin(np.pi * x[:,0]) * tf.sin(np.pi * x[:,1])

    def f(self, x):
        return -2 * np.pi ** 2 * tf.sin(np.pi * x[:, 0]) * tf.sin(np.pi * x[:, 1])

#2-dimensional problem with peak
class Peak_poisson(NNPDE):
    def __init__(self, batch_size, N, refn):
        self.alpha = 1000
        self.xc = 0.5
        self.yc = 0.5
        NNPDE2.__init__(self,batch_size, N, refn)

    def exactsol(self, x, y):
        return np.exp(-self.alpha*((x-self.xc)**2+(y-self.yc)**2)) # +np.sin(np.pi*x) # smooth correction

    def tfexactsol(self, x):
        return tf.exp(-1000 * ((x[:,0] - self.xc) ** 2 + (x[:,1] - self.yc) ** 2)) # +tf.sin(np.pi*x[:,0]) # smooth correction

    def f(self, x):
        return -4*self.alpha*self.tfexactsol(self.x) + 4*self.alpha**2*self.tfexactsol(self.x)* \
                                                       ((x[:, 0] - self.xc) ** 2 + (x[:, 1] - self.yc) ** 2) #- np.pi**2*tf.sin(np.pi*x[:,0]) # smooth correction

		# redefinition of the training method
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
#				# gather the input coordinates around the peak
#        if i>50:
#        X = np.concatenate([X,rectspace(0.4,0.5,0.4,0.5,5)], axis=0)
				
        _, loss = sess.run([self.opt2, self.loss], feed_dict={self.x: X})
				
				########## record loss ############
        self.rbloss.append(bloss)
        self.rloss.append(loss)
        uh = sess.run(self.u, feed_dict={self.x: self.refX}) #approximate solutions at each iteration
        Z = uh.reshape((self.refn, self.refn))
        uhref = self.exactsol(self.X, self.Y)
        self.rl2.append( np.sqrt(np.mean((Z-uhref)**2)) )
        ########## record loss ############


#2-dimensional problem with singularity for y=0
class Singularity_poisson(NNPDE):
    def __init__(self, batch_size, N, refn):
        self.alpha = 0.6
        NNPDE2.__init__(self,batch_size, N, refn)

    def exactsol(self, x, y):
        return y**0.6

    def tfexactsol(self, x):
        return tf.pow(x[:,1],0.6)

    def f(self, x):
        return self.alpha*(self.alpha-1)*x[:,1]**(self.alpha-2)


#2-dimensional problem with singularity for (x,y)=(0,0)
class Singularity2_poisson(NNPDE):
    def exactsol(self, x, y):
        return np.sin((3*np.arcsin(y*(1-(x+y==0.0))*(1/((x+y==0.0) + x**2 + y**2)**(1/2))))/2)*(x**2 + y**2)**(1/3)

    def tfexactsol(self, x):
        return tf.sin((3*tf.asin(x[:,1]*(1-(x==0.0))*(1/((x==0.0) + x[:,0]**2 + x[:,1]**2)**(1/2))))/2)*(x[:,0]**2 + x[:,1]**2)**(1/3)

    def f(self, x):
        return 65*tf.sin((3*tf.asin(x[:,1]*(1-(x==0.0))*(1/((x==0.0) + x[:,0]**2 + x[:,1]**2)**(1/2))))/2)*(1-(x==0.0))*(1/(36*((x==0.0) + x[:,0]**2 + x[:,1]**2)**(2/3)))


#################### Stationary advection-diffusion ####################################################

#2-dimensional transport smooth problem:
class Smooth_transport(NNPDE_transport):
    def exactsol(self,x,y):
        return np.sin(np.pi * x) * np.sin(np.pi * y)
			
    def tfexactsol(self,x):
        return tf.sin(np.pi * x[:,0]) * tf.sin(np.pi * x[:,1])

    def B(self, x):
        return x[:, 0] * (1 - x[:, 0]) * x[:, 1] * (1 - x[:, 1])

    def f(self, x):
        return -2 * np.pi ** 2 * tf.sin(np.pi * x[:, 0]) * tf.sin(np.pi * x[:, 1]) - np.pi*tf.cos(np.pi*x[:,0])* tf.sin(np.pi * x[:,1]) - np.pi*tf.cos(np.pi*x[:,1])* tf.sin(np.pi * x[:,0])


#2-dimensional transport problem with peak:
class Peak_transport(NNPDE_transport):
    def exactsol(self,x,y):
        return np.exp(-1000 * ((x - 0.5) ** 2 + (y - 0.5) ** 2))
			
    def tfexactsol(self,x):
        return tf.exp(-1000 * ((x[:,0] - 0.5) ** 2 + (x[:,1] - 0.5) ** 2))

    def B(self, x):
        return x[:, 0] * (1 - x[:, 0]) * x[:, 1] * (1 - x[:, 1])

    def f(self, x):
        return -4000*self.tfexactsol(self.x) + 4*1000**2*self.tfexactsol(self.x)* \
                                                       ((x[:, 0] - 0.5) ** 2 + (x[:, 1] - 0.5) ** 2) + 2000*(x[:,0]+x[:,1]-1)*self.tfexactsol(self.x)

#2-dimensional transport problem with singularity for y=0:
class Singularity_transport(NNPDE_transport):
    def exactsol(self,x,y):
        return y**0.6
			
    def tfexactsol(self,x):
        return tf.pow(x[:,1],0.6)

    def B(self, x):
        return x[:, 0] * (1 - x[:, 0]) * x[:, 1] * (1 - x[:, 1])

    def f(self, x):
        return 0.6*(0.6-1)*x[:,1]**(0.6-2) - 0.6*x[:,1]**(0.6-1)

#2-dimensional transport problem with singularity for (x,y)=(0,0)
class Singularity2_transport(NNPDE_transport):
    def exactsol(self, x, y):
        return np.sin((3*np.arcsin(y*(1-(x+y==0.0))*(1/((x+y==0.0) + x**2 + y**2)**(1/2))))/2)*(x**2 + y**2)**(1/3)

    def tfexactsol(self, x):
        return tf.sin((3*tf.asin(x[:,1]*(1-(x==0.0))*(1/((x==0.0) + x[:,0]**2 + x[:,1]**2)**(1/2))))/2)*(x[:,0]**2 + x[:,1]**2)**(1/3)

    def B(self, x):
        return x[:, 0] * (1 - x[:, 0]) * x[:, 1] * (1 - x[:, 1])
			
    def f(self, x):
        return -((54*x[:,0]**2-54*x[:,0]*x[:,1])*tf.cos(3*tf.asin(x[:,1]/(x[:,0]**2+x[:,1]**2)**(1/2))/2)+(64+24*x[:,0]+24*x[:,1])*tf.abs(x[:,0])*tf.sin(3*tf.asin(x[:,1]/(x[:,0]**2+x[:,1]**2)**(1/2))/2))/(36*(x[:,0]**2+x[:,1]**2)**(2/3)*tf.abs(x[:,0]))


################################# Heat equation ####################################################

#2-dimensional smooth heat equation
class Smooth_parabolic(Heat_equation):
    def exactsol(self, x,y,t):
        return np.sin(np.pi*x)*np.sin(np.pi*y)*np.exp(-t)
			
    def B(self, x):
        return x[:,0]*(1-x[:,0])*x[:,1]*(1-x[:,1])

    def f(self, x):
        return (-1 + 2* np.pi **2)* tf.sin(np.pi*x[:,0]) * tf.sin(np.pi*x[:,1]) * tf.exp(-x[:,2])

    def g(self, x):
        return 0

    def u_0(self, x):
        return tf.sin(np.pi*x[:,0]) * tf.sin(np.pi*x[:,1]) 
		
#2-dimensional heat equation with peak
class Peak_parabolic(Heat_equation):
    def __init__(self, batch_size, N, refn):
        self.alpha = 1000
        self.xc = 0.5
        self.yc = 0.5
        Heat_equation.__init__(self,batch_size, N, refn)

    def exactsol(self, x, y, t):
        return np.exp(-self.alpha*((x-self.xc)**2+(y-self.yc)**2)-t)
			
    def g(self, x):
        return tf.exp(-self.alpha*((x[:,0]-self.xc)**2+(x[:,1]-self.yc)**2)-x[:,2])

    def f(self, x):
        return (4*self.alpha-1 - 4*self.alpha**2*((x[:, 0] - self.xc) ** 2 + (x[:, 1] - self.yc) ** 2))*self.g(x)		

    def B(self, x):
        return x[:,0]*(1-x[:,0])*x[:,1]*(1-x[:,1])

    def u_0(self, x):
        return tf.exp(-self.alpha*((x[:,0]-self.xc)**2+(x[:,1]-self.yc)**2))
			
#2-dimensional heat equation with singularity for y=0
class Parabolic_singularity(Heat_equation):
    def exactsol(self, x, y, t):
        return y**0.6*np.exp(-t)
			
    def g(self, x):
        return x[:,1]**0.6*tf.exp(-x[:,2])

    def f(self, x):
        return (0.24*x[:,1]**(-1.4) - x[:,1]**0.6)*tf.exp(-x[:,2])
			
    def C(self, x):
        return x[:,2]
			
    def B(self, x):
        return x[:,0]*(1-x[:,0])*x[:,1]*(1-x[:,1])

    def u_0(self, x):
        return tf.pow(x[:,1], 0.6)

#2-dimensional heat equation with singularity for (x,y)=(0,0)
class Singularity2_parabolic(Heat_equation):
    def exactsol(self, x,y,t):
        return np.sin((3*np.arcsin(y*(1-(x+y==0.0))*(1/((x+y==0.0) + x**2 + y**2)**(1/2))))/2)*(x**2 + y**2)**(1/3)*np.exp(-t)
			
    def B(self, x):
        return x[:,0]*(1-x[:,0])*x[:,1]*(1-x[:,1])

    def f(self, x):
        return -65*tf.sin((3*tf.asin(x[:,1]*(1-(x[:,0]+x[:,1]==0.0))*(1/((x[:,0]+x[:,1]==0.0) + x[:,0]**2 + x[:,1]**2)**(1/2))))/2)*(1-(x[:,0]+x[:,1]==0.0))*(1/(36*((x[:,0]+x[:,1]==0.0) + x[:,0]**2 + x[:,1]**2)**(2/3)))*tf.exp(-x[:,2])-self.u_0(x)*tf.exp(-x[:,2])

    def g(self, x):
        return self.u_0(x)*tf.exp(-x[:,2])

    def u_0(self, x):
        return tf.sin((3*tf.asin(x[:,1]*(1-(x[:,0]+x[:,1]==0.0))*(1/((x[:,0]+x[:,1]==0.0) + x[:,0]**2 + x[:,1]**2)**(1/2))))/2)*(x[:,0]**2 + x[:,1]**2)**(1/3)
				
			
################################ Advection-diffusion ###############################################
	
#2-dimensional smooth advection-diffusion problem			
class Smooth_ad(Advection_diffusion_equation):
    def exactsol(self, x,y,t):
        return np.sin(np.pi*x)*np.sin(np.pi*y)*np.exp(-t)
			
    def B(self, x):
        return x[:,0]*(1-x[:,0])*x[:,1]*(1-x[:,1])

    def f(self, x):
        return (-1 + 2* np.pi **2)* tf.sin(np.pi*x[:,0]) * tf.sin(np.pi*x[:,1]) * tf.exp(-x[:,2]) + np.pi*tf.cos(np.pi*x[:,0])* tf.sin(np.pi * x[:,1]) + np.pi*tf.cos(np.pi*x[:,1])* tf.sin(np.pi * x[:,0])

    def g(self, x):
        return 0

    def u_0(self, x):
        return tf.sin(np.pi*x[:,0]) * tf.sin(np.pi*x[:,1]) 

#2-dimensional advection-diffusion problem with peak			
class Peak_ad(Advection_diffusion_equation):
    def __init__(self, batch_size, N, refn):
        self.alpha = 1000
        self.xc = 0.5
        self.yc = 0.5
        Heat_equation.__init__(self,batch_size, N, refn)

    def exactsol(self, x, y, t):
        return np.exp(-self.alpha*((x-self.xc)**2+(y-self.yc)**2)-t)
			
    def g(self, x):
        return tf.exp(-self.alpha*((x[:,0]-self.xc)**2+(x[:,1]-self.yc)**2)-x[:,2])

    def f(self, x):
        return (4*self.alpha-1 - 4*self.alpha**2*((x[:, 0] - self.xc) ** 2 + (x[:, 1] - self.yc) ** 2) - 2000*(x[:,0]+x[:,1]-1))*self.g(x)		
			
    def B(self, x):
        return x[:,0]*(1-x[:,0])*x[:,1]*(1-x[:,1])

    def u_0(self, x):
        return tf.exp(-self.alpha*((x[:,0]-self.xc)**2+(x[:,1]-self.yc)**2))
	
#2-dimensional advection-diffusion problem with singularity for y=0		
class Singularity_ad(Advection_diffusion_equation):
    def exactsol(self, x, y, t):
        return y**0.6*np.exp(-t)
			
    def g(self, x):
        return x[:,1]**0.6*tf.exp(-x[:,2])

    def f(self, x):
        return (0.24*x[:,1]**(-1.4)+0.4*x[:,1]**(-0.4) - x[:,1]**0.6)*tf.exp(-x[:,2])
			
    def B(self, x):
        return x[:,0]*(1-x[:,0])*x[:,1]*(1-x[:,1])

    def u_0(self, x):
        return tf.pow(x[:,1], 0.6)
			
#2-dimensional advection-diffusion problem with singularity for (x,y)=(0,0)
class Singularity2_ad(Advection_diffusion_equation):
    def exactsol(self, x, y, t):
        return np.sin((3*np.arcsin(y*(1-(x+y==0.0))*(1/((x+y==0.0) + x**2 + y**2)**(1/2))))/2)*(x**2 + y**2)**(1/3)*np.exp(-t)

    def u_0(self, x):
        return tf.sin((3*tf.asin(x[:,1]/( x[:,0]**2 + x[:,1]**2)**(1/2)))/2)*(x[:,0]**2 + x[:,1]**2)**(1/3)*tf.exp(-x[:,2])

    def B(self, x):
        return x[:, 0] * (1 - x[:, 0]) * x[:, 1] * (1 - x[:, 1])
			
    def g(self, x):
        return self.u_0(x)*tf.exp(-x[:,2])
			
    def f(self, x):
        return ((54*x[:,0]**2-54*x[:,0]*x[:,1])*tf.cos(3*tf.asin(x[:,1]/(x[:,0]**2+x[:,1]**2)**(1/2))/2)+(64+24*x[:,0]+24*x[:,1])*tf.abs(x[:,0])*tf.sin(3*tf.asin(x[:,1]/(x[:,0]**2+x[:,1]**2)**(1/2))/2))/(36*(x[:,0]**2+x[:,1]**2)**(2/3)*tf.abs(x[:,0]))*tf.exp(-x[:,2])-self.u_0(x)*tf.exp(-x[:,2])
			

################################# Linear transport ###################################################
			
#2-dimensional smooth hyperbolic problem
class Smooth_hyperbolic(NNPDE_hyperbolic):
    def exactsol(self, x,y,t):
        return np.sin(np.pi*x)*np.sin(np.pi*y)*np.exp(-t)
		
    def B(self, x):
        return x[:,0]*(1-x[:,0])*x[:,1]*(1-x[:,1])

    def f(self, x):
        return (- tf.sin(np.pi*x[:,0]) * tf.sin(np.pi*x[:,1]) + np.pi*tf.cos(np.pi*x[:,0])* tf.sin(np.pi * x[:,1]) + np.pi*tf.cos(np.pi*x[:,1])* tf.sin(np.pi * x[:,0]))  * tf.exp(-x[:,2]) 

    def g(self, x):
        return 0

    def u_0(self, x):
        return tf.sin(np.pi*x[:,0]) * tf.sin(np.pi*x[:,1]) 

#2-dimensional hyperbolic problem with peak			
class Peak_hyperbolic(NNPDE_hyperbolic):
    def __init__(self, batch_size, N, refn):
        self.alpha = 1000
        self.xc = 0.5
        self.yc = 0.5
        NNPDE_hyperbolic.__init__(self, batch_size, N, refn)

    def exactsol(self, x, y, t):
        return np.exp(-self.alpha*((x-self.xc)**2+(y-self.yc)**2)-t)
			
    def g(self, x):
        return tf.exp(-self.alpha*((x[:,0]-self.xc)**2+(x[:,1]-self.yc)**2)-x[:,2])

    def f(self, x):
        return (-1 - 2000*(x[:,0]+x[:,1]-1))*self.g(x)		
			
    def B(self, x):
        return x[:,0]*(1-x[:,0])*x[:,1]*(1-x[:,1])

    def u_0(self, x):
        return tf.exp(-self.alpha*((x[:,0]-self.xc)**2+(x[:,1]-self.yc)**2))
			
#2-dimensional hyperbolic problem with singularity for y=0		
class Singularity_hyperbolic(NNPDE_hyperbolic):
    def exactsol(self, x, y, t):
        return y**0.6*np.exp(-t)
			
    def g(self, x):
        return x[:,1]**0.6*tf.exp(-x[:,2])

    def f(self, x):
        return (0.6*x[:,1]**(-0.4) - x[:,1]**0.6)*tf.exp(-x[:,2])
			
    def B(self, x):
        return x[:,0]*(1-x[:,0])*x[:,1]*(1-x[:,1])

    def u_0(self, x):
        return tf.pow(x[:,1],0.6)
	
#2-dimensional hyperbolic problem with singularity for (x,y)=(0,0)		
class Singularity2_hyperbolic(NNPDE_hyperbolic):
    def exactsol(self, x, y, t):
        return np.sin((3*np.arcsin(y*(1-(x+y==0.0))*(1/((x+y==0.0) + x**2 + y**2)**(1/2))))/2)*(x**2 + y**2)**(1/3)*np.exp(-t)

    def u_0(self, x):
        return tf.sin((3*tf.asin(x[:,1]/( x[:,0]**2 + x[:,1]**2)**(1/2)))/2)*(x[:,0]**2 + x[:,1]**2)**(1/3)*tf.exp(-x[:,2])

    def B(self, x):
        return x[:, 0] * (1 - x[:, 0]) * x[:, 1] * (1 - x[:, 1])
			
    def g(self, x):
        return self.u_0(x)*tf.exp(-x[:,2])
			
    def f(self, x):
        return (9*x[:,0]**2*tf.cos((3*tf.asin(x[:,1]/(x[:,0]**2 + x[:,1]**2)**(1/2)))/2) + 4*x[:,1]*tf.sin((3*tf.asin(x[:,1]/(x[:,0]**2 + x[:,1]**2)**(1/2)))/2)*tf.abs(x[:,0]))/(6*tf.abs(x[:,0])*(x[:,0]**2 + x[:,1]**2)**(2/3)) + (2*x[:,0]*tf.sin((3*tf.asin(x[:,1]/(x[:,0]**2 + x[:,1]**2)**(1/2)))/2))/(3*(x[:,0]**2 + x[:,1]**2)**(2/3)) - (3*x[:,0]*x[:,1]*tf.cos((3*tf.asin(x[:,1]/(x[:,0]**2 + x[:,1]**2)**(1/2)))/2))/(2*tf.abs(x[:,0])*(x[:,0]**2 + x[:,1]**2)**(2/3))*tf.exp(-x[:,2])-self.u_0(x)*tf.exp(-x[:,2])


############################ High-dimensional problems ################################################
				
#N-dimensional smooth Poisson problem
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

#N-dimensional Poisson problem with peak
class HighDimensionPeak(NNPDE_ND):	
    def tfexactsol(self,x):
        return tf.exp(-1000 * (tf.reduce_sum((x - 0.5) **2)))

    def exactsol(self, x):
        return np.exp(-1000 * (np.sum((x - 0.5) **2)))

    def f(self, x):
        return (-4000 + 2000**2 * (tf.reduce_sum((x-0.5)**2)))*self.tfexactsol(x)
																											 
    def B(self, x):
        return tf.reduce_prod(x*(1-x),axis=1)

    def train(self, sess, i):
        self.rbloss = []
        self.rloss = []
        self.rl2 = []
				
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

#N-dimensional Poisson problem with singularity for x_2 = 0
class HighDimensionSingularity(NNPDE_ND):	
    def tfexactsol(self, x):
        return tf.pow(x[:,1],0.6)

    def exactsol(self, x):
        return x[:,1]**0.6

    def f(self, x):
        return 0.6*(0.6-1)*x[:,1]**(0.6-2)
																											 
    def B(self, x):
        return tf.reduce_prod(x*(1-x),axis=1)

    def train(self, sess, i):
        self.rbloss = []
        self.rloss = []
        self.rl2 = []
				
				# boundary points
        bX = np.random.rand(2*self.d*self.batch_size, self.d) * 0.99 + 0.01
        for j in range(self.d):
            bX[2*j*self.batch_size:(2*j+1)*self.batch_size, j] = 1.0 * 0.99 + 0.01
            bX[(2 * j+1) * self.batch_size:(2 * j + 2) * self.batch_size, j] = 0.01

        bloss = sess.run([self.bloss], feed_dict={self.x_b: bX})[0]
        # if the loss is small enough, stop training on the boundary
        if bloss>1e-5:
            for _ in range(5):
                _, bloss = sess.run([self.opt1, self.bloss], feed_dict={self.x_b: bX})
								
				# interior points
        X = np.random.rand(self.batch_size, self.d) * 0.99 + 0.01
        _, loss = sess.run([self.opt2, self.loss], feed_dict={self.x: X})

        # ######### record loss ############
        self.rbloss.append(bloss)
        self.rloss.append(loss)
        self.rl2.append( self.compute_L2(sess, self.X_test) )
        # ######### record loss ############

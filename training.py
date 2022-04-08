# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 12:15:32 2020

@author: beacr
"""

from problems import *

import time
start_time = time.time()

dir = 'p1'

# Modify the following row with the desired problem defined in problems.py
npde = Smooth_poisson(64, 3, 20)   # 64 input points, 3 hidden layers, 20X20 grid

with tf.Session() as sess:
    sess.run(npde.init)
    for i in range(400):
        npde.train(sess, i)
        if i%100==0:
            npde.visualize(sess, False, i=i, savefig=dir)
#    # test and compute final error
#    uh = sess.run(npde.u, feed_dict={npde.x: npde.refX})   #approximate solutions at each iteration
#    Z = uh.reshape((npde.refn, npde.refn))
#    uhref = npde.exactsol(npde.X, npde.Y)
#    npde.rl2.append( np.sqrt(np.mean((Z-uhref)**2)) )

print("--- %s seconds ---" % (time.time() - start_time)) 		#execution time
	 
plt.close('all')
plt.semilogy(npde.rbloss)
plt.xlabel('Iteration')
plt.ylabel('$L_b$')
plt.savefig(dir + '/lb.png')

plt.close('all')
plt.semilogy(npde.rloss)
plt.xlabel('Iteration')
plt.ylabel('$L_i$')
plt.savefig(dir + '/li.png')

plt.close('all')
plt.semilogy(npde.rl2)
plt.xlabel('Iteration')
plt.ylabel('$|err_2$')
plt.savefig(dir + '/l2.png')
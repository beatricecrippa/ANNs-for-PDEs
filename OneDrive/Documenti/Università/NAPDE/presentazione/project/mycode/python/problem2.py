# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 12:13:34 2020

@author: beacr
"""

from problems import *

dir = 'p2'
npde = ProblemPeak_BD(64, 3, 50) # works very well
#npde.plot_exactsol()
#plt.show()
#exit(0)
with tf.Session() as sess:
    sess.run(npde.init)
    for i in range(10000):
        if( i>1000 ):
            break
        npde.train(sess, i)
        if i%100==0:
            npde.visualize(sess, False, i=i, savefig=dir)


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
plt.ylabel('$||u-u_h||_2$')
plt.savefig(dir + '/l2.png')
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 16:41:49 2020

@author: beacr
"""

from problems import *

from matplotlib import pyplot as plt

dir = './high/p1/'

layers = np.array ([1, 2, 3])				 #modify the number of layers and see what happens for different dimensions

for dim in [2, 5, 7]:			 #loop on different dimensions

    rblossFull = []
    rlossFull = []
    rl2Full = []
		
		# training on each layer
    for layer in layers:
        #print(layer)
        npde = HighDimensionSmooth(64, layer, dim) # batch-size, number of layers, dimension
        rbloss = []
        rloss = []
        rl2 = []
        with tf.Session() as sess:				 			# a session encapsulates the environment where
																								# operations are excuted and tensors are evaluated
            sess.run(npde.init)
            for i in range(20000):
                if( i % 1000 == 0):
                    print('Now step'+ str(i))
                npde.train(sess, i)
                rbloss.append(npde.rbloss)
                rloss.append(npde.rloss)
                rl2.append(npde.rl2)
								
        print(rbloss)
#        plt.close('all')
#        plt.figure(1)			
#        plt.xlabel('Iteration')
#        plt.ylabel('$L_b$' + layer)
#        plt.savefig(dir + '/lb.png')
				
        tf.reset_default_graph()
        rblossFull.append(rbloss)
        rlossFull.append(rloss)
        rl2Full.append(rl2)

		# record and save loss in a file
    with open(dir + str(dim) + '.txt', 'w') as file:
        for l in rblossFull:
            for item in l:
                file.write("%s," % item)
            file.write("\t")
        file.write("\n")
        for l in rlossFull:
            for item in l:
                file.write("%s," % item)
            file.write("\t")
        file.write("\n")
        for l in rl2Full:
            for item in l:
                file.write("%s," % item)
            file.write("\t")


    plt.close('all')
    plt.figure(1)
    for i in range(len(rblossFull)):
        plt.semilogy(rblossFull[i],label="nLayer=%d" %layers[i])
    plt.xlabel('Iteration')
    plt.ylabel('$L_b$')
    plt.legend()
    plt.savefig(dir + str(dim) + '_' 'lb.png')
    
    plt.figure(2)
    for i in range(len(rlossFull)):
        plt.semilogy(rlossFull[i],label="nLayer=%d"%layers[i])
    plt.xlabel('Iteration')
    plt.ylabel('$L_i$')
    plt.legend()
    plt.savefig(dir + str(dim) + '_' 'li.png')
    
    plt.figure(3)
    for i in range(len(rl2Full)):
        plt.semilogy(rl2Full[i],label="nLayer=%d"%layers[i])
    plt.xlabel('Iteration')
    plt.ylabel('$L_2 = ||u-u_h||_2$')
    plt.legend()
    plt.savefig(dir + str(dim) + '_' 'l2.png')
    plt.show()

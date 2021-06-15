#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
This work partially relies on my paper
https://doi.org/10.1109/ACIT49673.2020.9208998

We define here functions related to the equation itself and then functions
for calculating parts of the Hessian matrix
"""


import numpy as np 

import tensorflow as tf
#tf.enable_v2_behavior()

#%%
"""creating inputs (focusing on the interval around x=4t)"""
def create_input(n,pu):    
    t1=tf.random.uniform([int(0.98*n),1],minval=-1,maxval=1)
    t2=tf.random.uniform([int(0.01*n),1],minval=-1,maxval=1)
    t3=tf.random.uniform([int(0.01*n),1],minval=-1,maxval=1)
    t=tf.concat([t2,t1,t3],0)
    
    t_zeros=tf.zeros((int((1-pu)*n),1))
    t_ones=tf.ones((int(pu*n),1))
    t_ind=tf.concat([t_zeros,t_ones],0)
    t_rand_ind=tf.random.shuffle(t_ind)
    
    indices2=tf.concat([t_rand_ind,t_rand_ind],1)
    #indices2=tf.Variable(indices2, name='indices2')
    
    t=(1-t_rand_ind)*t #puts zeros on places where t_rand_ind (and ind2) is one
    #t=tf.Variable(t, name='t')
    
    xm=4*t1+tf.random.uniform([int(0.98*n),1],minval=-5,maxval=5)
    x1=4*t2+tf.random.uniform([int(0.01*n),1],minval=-16,maxval=-5)
    x2=4*t3+tf.random.uniform([int(0.01*n),1],minval=5,maxval=16)

    x=tf.concat([x1,xm,x2],0)
    #x=tf.Variable(x, name='x')
        
    return [x,t,indices2]




#%%
"""initialization weights and biases"""
def create_weights(s,p,r, mean, stdev):
    W11shape=[1,s]
    b1shape=[1,s]
    W12shape=[1,s]
    
    W2shape=[s,p]
    b2shape=[1,p]

    W3shape=[p,r]
    b3shape=[1,r]
    W4shape=[r,r]
    b4shape=[1,r]
    W5shape=[r,r]
    b5shape=[1,r]
    
    W6shape=[r,r]
    b6shape=[1,r]
    
    W7shape=[r,2]
    
    b7shape=[1,2]

    W11=tf.Variable(tf.random.truncated_normal(W11shape, mean=mean, stddev=stdev, dtype=tf.float32))
    b1=tf.Variable(tf.random.truncated_normal(b1shape, mean=mean, stddev=stdev, dtype=tf.float32))
    W12=tf.Variable(tf.random.truncated_normal(W12shape, mean=mean, stddev=stdev, dtype=tf.float32))


    W2=tf.Variable(tf.random.truncated_normal(W2shape, stddev=stdev, dtype=tf.float32))
    b2=tf.Variable(tf.random.truncated_normal(b2shape, stddev=stdev, dtype=tf.float32))
    W3=tf.Variable(tf.random.truncated_normal(W3shape, stddev=stdev, dtype=tf.float32))
    b3=tf.Variable(tf.random.truncated_normal(b3shape, stddev=stdev, dtype=tf.float32))
    W4=tf.Variable(tf.random.truncated_normal(W4shape, stddev=stdev, dtype=tf.float32))
    b4=tf.Variable(tf.random.truncated_normal(b4shape, stddev=stdev, dtype=tf.float32))
    W5=tf.Variable(tf.random.truncated_normal(W5shape, stddev=stdev, dtype=tf.float32))
    b5=tf.Variable(tf.random.truncated_normal(b5shape, stddev=stdev, dtype=tf.float32))
    W6=tf.Variable(tf.random.truncated_normal(W6shape, stddev=stdev, dtype=tf.float32))
    b6=tf.Variable(tf.random.truncated_normal(b6shape, stddev=stdev, dtype=tf.float32))
    W7=tf.Variable(tf.random.truncated_normal(W7shape, mean=mean, stddev=stdev, dtype=tf.float32))
    b7=tf.Variable(tf.random.truncated_normal(b7shape, mean=mean, stddev=stdev, dtype=tf.float32))
    trainable_vars=[W11, W12, W2,W3,W4,W5,W6,W7, b1,b2,b3,b4,b5,b6,b7]
    #trainable_vars=[W11,W12, W2,W3, W7,b1,b2,b3, b7]
    return trainable_vars

#%%
"""functions related to the equation"""

def delta(x):
    d1=10*tf.exp(-tf.square(x)*100)/tf.sqrt(np.pi)
    d2=tf.zeros((len(x),1))
    d=tf.concat([d1,d2],1)
    return d



def a(x): #inital datum
    n=tf.shape(x)[0]
    a1=tf.cos(2*x+np.pi/2)*1/tf.cosh(x)
    a2=-tf.sin(2*x+np.pi/2)*1/tf.cosh(x)
    a1=tf.reshape(a1, (n,1))
    a2=tf.reshape(a2, (n,1))
    a=tf.concat((a1,a2), axis=1)
    return a


def run_network(x,t,indices2,trains):
    W11=trains[0]
    W12=trains[1]
    W2=trains[2]
    W3=trains[3]
    W4=trains[4]
    W5=trains[5]
    W6=trains[6]
    W7=trains[7]
    b1=trains[8]
    b2=trains[9]
    b3=trains[10]
    b4=trains[11]
    b5=trains[12]
    b6=trains[13]
    b7=trains[14]
    h1=tf.tanh(x*W11+t*W12+b1) #in order to take derivatives in x and t
    #run_network must be a function of both (separately)
    h2=tf.tanh(tf.matmul(h1,W2)+b2)
    h3=tf.tanh(tf.matmul(h2,W3)+b3)
    h4=tf.tanh(tf.matmul(h3,W4)+b4)
    h5=tf.tanh(tf.matmul(h4,W5)+b5)
    h6=tf.tanh(tf.matmul(h5,W6)+b6)
    u=tf.tanh(tf.matmul(h6,W7) + b7)
    return u

#derivatives needed for the equation, using autodiff:

def laplacian1(x,t,indices2, trains):
    with tf.GradientTape() as g:
        g.watch(x)
        with tf.GradientTape() as gg:
            gg.watch(x)
            u=run_network(x,t,indices2,trains)
            u1=u[:,0:1]
        du1_dx = gg.gradient(u1, x)
    d2u1_dx2 = g.gradient(du1_dx, x)
    return d2u1_dx2

def laplacian2(x,t,indices2,trains):
    with tf.GradientTape() as g:
        g.watch(x)
        with tf.GradientTape() as gg:
            gg.watch(x)
            u=run_network(x,t,indices2, trains)
            u2=u[:,1:]
        du2_dx = gg.gradient(u2, x)
    d2u2_dx2 = g.gradient(du2_dx, x)
    return d2u2_dx2

def u1_t(x,t,indices2, trains):
    with tf.GradientTape() as g:
        g.watch(t)
        u=run_network(x,t,indices2, trains)
        u1=u[:,0:1]
    du1dX=g.gradient(u1,t)
    return du1dX

def u2_t(x,t,indices2, trains):
    with tf.GradientTape() as g:
        g.watch(t)
        u=run_network(x,t,indices2,trains)
        u2=u[:,1:]
    du2dX=g.gradient(u2,t)
    return du2dX

#now the loss function:

def calc_loss(x,t,indices2,trains):
    u=run_network(x,t,indices2,trains)
    u1=u[:,0:1]
    u2=u[:,1:]
    u2t=u2_t(x,t, indices2,trains)
    l1=laplacian1(x,t,indices2, trains)
    u1t=u1_t(x,t,indices2,trains)
    l2=laplacian2(x,t,indices2, trains)
    deltas1=-u2t-l1-2*u1*(tf.pow(u1,2)+tf.pow(u2,2))
    deltas2=u1t-l2-2*u2*(tf.pow(u1,2)+tf.pow(u2,2))
    deltas=tf.concat([deltas1,deltas2],1)
    squared_deltas=tf.square(deltas)
    dd=a(x)
    jot1=tf.square(u-dd)
    #jot2=tf.square(u)
    error=squared_deltas+jot1*indices2#+jot2*indices1
    #sq_deltas_var.assign(sq_deltas)
    loss1=tf.reduce_sum(error, axis=1)
    loss1=tf.reshape(loss1, [len(x),1])
    loss=tf.reduce_mean(error) 
    return loss, loss1


#%%
"""the exact solution to be compared with"""
def resenje(x,t,n):
    res1=np.cos(2*x-3*t+np.pi/2)*1/np.cosh(x-4*t)
    res2=-np.sin(2*x-3*t+np.pi/2)*1/np.cosh(x-4*t)
    res1=np.reshape(res1, (n,1))
    res2=np.reshape(res2, (n,1))
    resenje=np.concatenate((res1,res2), axis=1)
    return resenje

#%%
"""gradient (needed later) and function needed for parts of Hessian"""

def grad1(x,t,indices2,trains, v):
    with tf.GradientTape() as g:
        g.watch(v)
        u=calc_loss(x,t,indices2,trains)[0]#ovde nema s i to mozda pravi problem
    du1dX=g.gradient(u,v)
    return du1dX    
def Hes1(x,t,indices2,trains,s,v):
    with tf.GradientTape() as g:
        g.watch(v)
        with tf.GradientTape() as gg:
             gg.watch(s)
             u=calc_loss(x,t,indices2,trains)[0]
        du1dX=gg.gradient(u,s)
    jacobian = g.jacobian(du1dX, v)
    return jacobian

#%%
""" if we want to sort matrices form trains in some way;
here the first part contains all 1xm,
the first 2 matrices being the most difficult;
the result is a list containing lists of columns"""

def sort_shapes_1xm(trains):    
    trains1=[]
    redni_br=[]
    k=0
    for v in trains:       
        if tf.shape(v)[0].numpy()==1:
            trains1.append(v)
            redni_br.append(k)
        k=k+1
    return trains1, redni_br


"""all not 1xm, result is a list of np. matrices,
which can be flattended to columns"""

def sort_shapes_mxr(trains):    
    trains1=[]
    redni_br=[]
    k=0
    for v in trains:       
        if tf.shape(v)[0].numpy()!=1:
            trains1.append(v)
            redni_br.append(k)
        k=k+1
    return trains1, redni_br
#part_of_trains=sort_shapes_mxr(trains)[0]
#indices=sort_shapes_mxr(trains)[1]
#%%
"""returns a part of hessian: every element of return is a np matrix,
and every element of that matrix is a column of the hessian
but not the whole column - for the whole hessian, a symmetric copying needs to be done"""

def hes_part(x,t,indices2,trains,part_of_trains,indices):
    k=0
    hes_part=[]
    #hes_part=tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    for v in part_of_trains:
        br=indices[k]
        trainable_variables=trains[br+1:]
        dim1=tf.shape(v)[0]
        dim2=tf.shape(v)[1]
        izvod=Hes1(x,t,indices2,trains,v,v)
        izvod=tf.reshape(izvod, (dim1,dim2, dim1*dim2,1)) #matrice unutra se reshapuju u kolone, tj. izvod[i][j] je kolona
        deo=tf.TensorArray(tf.float32,size=0, dynamic_size=True)
        s=0
        for i in tf.range(dim1):
            for j in tf.range(dim2):
                deo=deo.write(s,izvod[i][j])
                s=s+1
                #deo=deo.write(2**(2*i+1)*(4*j+3),izvod[i][j]) #bijection from ZxZ to N
                #https://math.stackexchange.com/questions/187751/cardinality-of-the-set-of-all-pairs-of-integers
                #holes will be filled with zeros which are never used
        novi_deo=tf.TensorArray(tf.float32,size=0, dynamic_size=True)
        for z in trainable_variables:
            d1=tf.shape(z)[0]
            d2=tf.shape(z)[1]            
            iz=Hes1(x,t,indices2,trains,v,z)#spoljasnje dim su uvek dim1 x dim2, kao i v
            iz=tf.reshape(iz, (dim1,dim2,d1*d2,1))
            s=0
            for i in tf.range(dim1):
                for j in tf.range(dim2):
                    drugi=iz[i][j] #w11 po w12 npr.
                    deo_tensor=deo.read(s)
                    novi_deo=novi_deo.write(s,tf.concat([deo_tensor,drugi],0))
                    s=s+1
                    #deo_tensor=deo.read(2**(2*i+1)*(4*j+3))
                    #novi_deo=novi_deo.write(2**(2*i+1)*(4*j+3),tf.concat([deo_tensor,drugi],0))
            deo=novi_deo
            novi_deo=tf.TensorArray(tf.float32,size=0, dynamic_size=True)
        deo1=deo.stack()
        hes_part.append(deo1)
        #hes_part.write(k,deo1)
        k=k+1
    return hes_part


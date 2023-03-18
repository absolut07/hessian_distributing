#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 19:27:47 2020

This was an attempt of applying the DFIX (https://arxiv.org/pdf/2001.03968.pdf)
optimization method to approximately solving a PDE. The main difficulty of this
method is that it uses the Hessian matrix, meaning second order derivatives 
of the loss function. This code is devoted to distributing these calculations to
2 GPUs, the DFIX method itself is mainly omitted (and at the end was not used 
                                                  for this purpose).
All this is also done using neural networks, specifically, our approximation
to the solution of the PDE is a neural network and the loss function 
is created based on the equation.

"""
import distributed_nn_functions as dnn
import tensorflow as tf 
import pickle
import time


#%%
"""functions, transformations and gradient"""

#turn W to x
def W_to_x(trainable_vars):
    sh=tf.shape(trainable_vars[0])[0]*tf.shape(trainable_vars[0])[1]
    x=tf.reshape(trainable_vars[0],[sh,1])
    for v in trainable_vars[1:]:
        sh=tf.shape(v)[0]*tf.shape(v)[1]
        v=tf.reshape(v, [sh,1])
        x=tf.concat([x,v],0)
    x=tf.Variable(x, dtype=tf.float32)
    return x

#turn x to W
def x_to_W(x, shapes):
    trains=[]
    start=0
    for sh in shapes:
        dd=sh[0]*sh[1]+start
        xi=tf.reshape(x[start:dd],sh)
        trains.append(xi)
        start=dd
    return trains

#gradient
def grad(trainable_variables,x,t,indices2):
    g=dnn.grad1(x,t,indices2,trainable_variables,trainable_variables[0])
    g=tf.reshape(g,(tf.shape(g)[0]*tf.shape(g)[1],1))
    for v in trainable_variables[1:]:
        bb=dnn.grad1(x,t,indices2,trainable_variables,v)
        sh1=tf.shape(bb)[0]
        sh2=tf.shape(bb)[1]
        b=tf.reshape(bb,(sh1*sh2,1))
        g=tf.concat([g,b],0)

    return g

"""hessian"""
#not used here, but if needed, this is how to use fje1graph to 
#create the whole hessian matrix
 
def Hes(train,x,t,indices2):
    m=2#how to divide trains and hence the hessian
    part_of_trains=train[:m]
    indices=[ind for ind in range(m)]
    pr=dnn.hes_part(x,t,indices2,train,part_of_trains,indices)
    
    part_of_trains=train[m:]
    indices=[ind for ind in range(m,len(train))]
    dr=dnn.hes_part(x,t,indices2,train,part_of_trains,indices)
    
    hes=tf.TensorArray(tf.float32,size=0, dynamic_size=True)
    dim1=tf.shape(train[0])[0]
    dim2=tf.shape(train[0])[1]
    s=0
    for i in tf.range(dim1):
        for j in tf.range(dim2):
            hes=hes.write(s, pr[0][2**(2*i+1)*(4*j+3)])
            s=s+1
    #pr[0] was the start, now we have to fill up so we can concat
    
    d=0
    for k in range(1,len(pr)):
        d=d+dim1*dim2
        nule=tf.zeros([d,1])
        dim1=tf.shape(train[k])[0]
        dim2=tf.shape(train[k])[1]
        for i in tf.range(dim1):
            for j in tf.range(dim2):
                aa=tf.concat([nule,pr[k][2**(2*i+1)*(4*j+3)]],0)
                hes=hes.write(s,aa)
                s=s+1

    for k in range(len(dr)):
        d=d+dim1*dim2
        nule=tf.zeros([d,1])
        dim1=tf.shape(train[m+k])[0]
        dim2=tf.shape(train[m+k])[1]
        for i in tf.range(dim1):
            for j in tf.range(dim2):
                aa=tf.concat([nule,dr[k][2**(2*i+1)*(4*j+3)]],0)
                hes=hes.write(s,aa)
                s=s+1
    
    hes=hes.stack()
    n=tf.shape(hes)[0]
    hes=tf.reshape(hes,(n,n))
    hes=tf.transpose(tf.linalg.band_part(hes, 0, -1))+tf.linalg.band_part(hes, 0, -1)-tf.linalg.band_part(hes, 0, 0)
    return hes


@tf.function
def test_update(trainable_variables,X,d,a):
    with tf.device("/gpu:0"):
        m=1
        part_of_trains=trainable_variables[:m]#the first is the most demanding
        indices=[ind for ind in range(m)]
        pr=dnn.hes_part(x,t,indices2,trainable_variables,part_of_trains,indices)
    with tf.device("/gpu:1"):
        m=1
        part_of_trains=trainable_variables[m:]
        indices=[ind for ind in range(m,len(trainable_variables))]
        dr=dnn.hes_part(x,t,indices2,trainable_variables,part_of_trains,indices)
    X.assign(X+a*d)
    tt=x_to_W(X,shapes)
    i=0
    for tr in tt:
        trainable_variables[i].assign(tr)
        i=i+1
    return pr, dr

#%%
"""input for neural network"""
input_x_t_ind=dnn.create_input(300,0.5)
x=input_x_t_ind[0]
t=input_x_t_ind[1]
indices2=input_x_t_ind[2]
trainable_variables=dnn.create_weights(15,15,0,0.1)

"""DFIX: main algorithm, returnig trainable vars"""

#loading a start matrix
with open('W', 'rb') as f:
    [W]=pickle.load(f) 

shapes=[]
for v in trainable_variables:
    shapes.append(tf.shape(v).numpy())

X=W_to_x(trainable_variables)
X=tf.Variable(X)
n=tf.shape(X)[0].numpy()

granice=tf.constant([0,15,30,n]) #nods
#%%


"""vars that don't change"""
eps=10**(-5)
kmax=300
reg=10**(-4)
kumax=1000 #no calculation of derivatives
amin=tf.constant(10**(-4))

W=tf.Variable(W ,dtype=tf.float32)
nt=len(W.numpy())

"""ff"""
deg=tf.reduce_sum(tf.sign(W), axis=0) #1/0; nod with the largest number of neighbours
maxd=max(deg) #maximum
glavni=0
while deg[glavni]<maxd:
    glavni=glavni+1

k=0
#exit criterion:
izlazni=0
#first Newton step
d0=tf.zeros((n,1))
#d0=tf.Variable(d0, dtype=tf.float32)
d=d0
gniz=[]
fniz=[]

a=tf.constant(1.)
        
#%%
"""main loop"""


while k<10 and izlazni==0:
    start=time.time()
    [g,H]=test_update(trainable_variables,X,d,a)
    end=time.time()
    print(end-start)
    d=d+1
    a=a+1
    #in the real algorithm a and d are calculated using g and H, but this is 
    #a simple version just to show how the distributing works
    k=k+1
    

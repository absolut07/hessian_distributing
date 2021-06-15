# hessian_distributing
This code is deddicated to distributing calculations of second order derivatives on more GPUs and also on a cluster.
It is done in TensorFlow 2 and it uses neural networks. 
The goal was to combine the concepts of neural networks and optimization in order to solve a PDE,
but here the focus is on distributing. More detailed descriptions are in "github_distribute2gpu.py" and "fje1graph.py", whereas "github_cluster_distribute.py" is a continuation
of those works, but on a cluster.

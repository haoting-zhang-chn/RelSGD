# Relativistic Stochastic Gradient Descent

Relativistic Stochastic Gradient Descent (RSGD) is an optimization method (or more specifically, an SGD method) extended from the Relativistic Hamiltonian Monte Carlo, developed in the paper [Relativistic Monte Carlo](https://arxiv.org/abs/1609.04388), by Xiaoyu Lu, Valerio Perrone, Leonard Hasenclever, Yee Whye Teh and Sebastian J. Vollmer. Its formulae have similarities with other popular SGD methods like Adam and RMSProp, but the difference is that it is derived from the relativistic dynamics of a Hamiltonian system, and is a stochastic optimization algorithm starting from a Bayesian setup.

The script *RSGD.py* (yet to be double checked) contains the implementation of RSGD. By importing *RSGD.py*, the RSGD optimizer will be able to be called in the same way as other optimization algorithms in TensorFlow. For example, *train_step = tf.train.RSGDOptimizer().minimize(cross_entropy)* would work, similar to *train_step = tf.train.AdamOptimizer().minimize(cross_entropy)*.

*MNIST_cnn.py* and *CIFAR_vgg16.py* are implementations of RSGD on MNIST (with the c32-c64-1024 CNN architecture) and CIFAR10 (with the VGG16 architecture). More codes in finer style and a report comparing the performance of RSGD to Adam and RMSProp are to be uploaded.

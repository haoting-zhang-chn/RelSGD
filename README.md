# Relativistic Stochastic Gradient Descent

Relativistic Stochastic Gradient Descent (RSGD) is an optimization method (or more specifically, an SGD method) extended from the Relativistic Hamiltonian Monte Carlo, developed in the paper [Relativistic Monte Carlo](https://arxiv.org/abs/1609.04388), by Xiaoyu Lu, Valerio Perrone, Leonard Hasenclever, Yee Whye Teh and Sebastian J. Vollmer. Its formulae have similarities with other popular SGD methods like Adam and RMSProp, but the difference is that it is derived from the relativistic dynamics of a Hamiltonian system, and is a stochastic optimization algorithm starting from a Bayesian setup.

The script **RSGD.py** contains the implementation of RSGD. By importing **RSGD.py**, the RSGD optimizer will be able to be called in the same way as other optimization algorithms in TensorFlow. For example, **train_step = tf.train.RSGDOptimizer().minimize(cross_entropy)** would work, similar to **train_step = tf.train.AdamOptimizer().minimize(cross_entropy)**.

The script **RSGDc.py** contains the componentwise/elementwise version of RSGD, where all arithmetic operations are interpreted as element-wise operations. This could be useful in very high dimensional problems, so that the maximum speed imposed on the system does not need to be very large and is easier to tune.

The script **RSHD.py** is the TensorFlow implementation of the optimization algorithm in another relevant paper, namely [Hamiltonian Descent Methods](https://arxiv.org/abs/1809.05042) by Chris J. Maddison, Daniel Paulin, Yee Whye Teh, Brendan O'Donoghue and Arnaud Doucet. In the paper, a family of optimization methods is introduced, namely Hamiltonian Descent. What this script implements is the version of Hamiltonian Descent with relativistic kinetic energy, which we call Relativistic Stochastic Hamiltonian Descent (RSHD). It has similarities with RSGD, but has different update rules.

**MNIST_cnn.py** and **CIFAR_vgg16.py** are implementations of RSGD on MNIST (with the c32-c64-1024 CNN architecture) and CIFAR10 (with the VGG16 architecture). More codes and an experiment report are to be uploaded.

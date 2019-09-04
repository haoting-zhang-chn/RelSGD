# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
from tensorflow.python.ops import resource_variable_ops  ##for sparse implementation

class RSHDOptimizer(optimizer.Optimizer):
    def __init__(self, learning_rate=0.5, gamma = 0.5, m=1, c=1, use_locking=False, name="RSHD"):
        super(RSHDOptimizer, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._m = m
        self._c = c
        self._gamma = gamma
        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._m_t = None
        self._c_t = None
        self._gamma_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._m_t = ops.convert_to_tensor(self._m, name="m")
        self._c_t = ops.convert_to_tensor(self._c, name="c")
        self._gamma_t = ops.convert_to_tensor(self._gamma, name="gamma")

    def _create_slots(self, var_list):
        # Create slot for p
        for v in var_list:
            self._zeros_slot(v, "p", self._name)

    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        m_t = math_ops.cast(self._m_t, var.dtype.base_dtype)
        c_t = math_ops.cast(self._c_t, var.dtype.base_dtype)
        gamma_t = math_ops.cast(self._gamma_t, var.dtype.base_dtype)
        ### Update p: p_(t+1) = p_t / (1+gamma*lr) - lr / (1+gamma*lr) * gradient
        p = self.get_slot(var, "p")
        a = 1 / (1+gamma_t*lr_t)
        p_t = p.assign(a*p - lr_t*a*grad)
        ### Update theta: theta_(t+1) = theta_t + lr * p_(t+1) / sqrt(p_(t+1) * p_(t+1) / c^2 + m^2)
        g_t = - p_t / tf.sqrt(tf.matmul(tf.reshape(p_t, [1, -1]), tf.reshape(p_t, [-1, 1])) / tf.pow(c_t, 2) + tf.pow(m_t, 2))
        var_update = state_ops.assign_sub(var, tf.reshape(lr_t * g_t, tf.shape(var)))
        return control_flow_ops.group(*[var_update, p_t])

    def _resource_apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, grad.dtype.base_dtype)
        m_t = math_ops.cast(self._m_t, grad.dtype.base_dtype)
        c_t = math_ops.cast(self._c_t, grad.dtype.base_dtype)
        gamma_t = math_ops.cast(self._gamma_t, grad.dtype.base_dtype)
        ### Update p: p_(t+1) = p_t / (1+gamma*lr) - lr / (1+gamma*lr) * gradient
        p = self.get_slot(var, "p")
        a = 1 / (1+gamma_t*lr_t)
        p_t = p.assign(a*p - lr_t*a*grad)
        ### Update theta: theta_(t+1) = theta_t + lr * p_(t+1) / sqrt(p_(t+1) * p_(t+1) / c^2 + m^2)
        g_t = - p_t / tf.sqrt(tf.matmul(tf.reshape(p_t, [1, -1]), tf.reshape(p_t, [-1, 1])) / tf.pow(c_t, 2) + tf.pow(m_t, 2))
        var_update = state_ops.assign_sub(var, tf.reshape(lr_t * g_t, tf.shape(var)))
        return control_flow_ops.group(*[var_update, p_t])

    def _apply_sparse(self, grad, var):
        return self._apply_sparse_shared(
            grad.values, var, grad.indices,
            lambda x, i, v: state_ops.scatter_add(  # pylint: disable=g-long-lambda
                x, i, v, use_locking=self._use_locking))

    def _apply_sparse_shared(self, grad, var, indices, scatter_add):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        m_t = math_ops.cast(self._m_t, var.dtype.base_dtype)
        c_t = math_ops.cast(self._c_t, var.dtype.base_dtype)
        gamma_t = math_ops.cast(self._gamma_t, var.dtype.base_dtype)
        ### Update p: p_(t+1) = p_t - lr * (gradient + D * p_t / sqrt(p_t * p_t / c^2 + m^2))
        p = self.get_slot(var, "p")
        a = 1 / (1+gamma_t*lr_t)
        p_scaled_g_values = - lr_t * a * grad
        p_t = p.assign(a*p)
        with ops.control_dependencies([p_t]):
            p_t = scatter_add(p_t, indices, p_scaled_g_values)
        ### Update theta: theta_(t+1) = theta_t + lr * p_(t+1) / sqrt(p_(t+1) * p_(t+1) / c^2 + m^2)
        g_t = - p_t / tf.sqrt(tf.matmul(tf.reshape(p_t, [1, -1]), tf.reshape(p_t, [-1, 1])) / tf.pow(c_t, 2) + tf.pow(m_t, 2))
        var_update = state_ops.assign_sub(var, tf.reshape(lr_t * g_t, tf.shape(var)))
        return control_flow_ops.group(*[var_update, p_t])

    def _resource_scatter_add(self, x, i, v):
        with ops.control_dependencies(
           [resource_variable_ops.resource_scatter_add(
                x.handle, i, v)]):
           return x.value()

    def _resource_apply_sparse(self, grad, var, indices):
        return self._apply_sparse_shared(
            grad, var, indices, self._resource_scatter_add)

tf.train.RSHDOptimizer = RSHDOptimizer

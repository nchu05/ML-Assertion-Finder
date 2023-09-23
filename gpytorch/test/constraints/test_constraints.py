import math
import unittest
import torch
from torch import sigmoid
from torch.nn.functional import softplus
import gpytorch
from gpytorch.test.base_test_case import BaseTestCase
import torch
import numpy as np
import pytest

def complexObj(string):
    try:
        if isinstance(string, torch.Tensor):
            return np.array2string(string.detach().numpy(), precision=50, separator=',').replace('\n', '')
    except:
        pass
    if isinstance(string, np.ndarray):
        return np.array2string(string, precision=50, separator=',').replace('\n', '')
    if isinstance(string, list):
        for i in string:
            return '[{0}]'.format(','.join([complexObj(i)]))
    try:
        if isinstance(string, tf.Tensor):
            return np.array2string(string.eval(), precision=50, separator=',').replace('\n', '')
    except:
        pass
    return str(string)

class ExactGPModel(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class TestInterval(unittest.TestCase, BaseTestCase):

    def test_transform_float_bounds(self):
        constraint = gpytorch.constraints.Interval(1.0, 5.0)
        v = torch.tensor((- 3.0))
        value = constraint.transform(v)
        actual_value = (((5.0 - 1.0) * sigmoid(v)) + 1.0)
        val_1 = value
        print(('log>>>%s' % complexObj(val_1)))
        val_2 = actual_value
        print(('log>>>%s' % complexObj(val_2)))
        self.assertAllClose(val_1, val_2)

    def test_inverse_transform_float_bounds(self):
        constraint = gpytorch.constraints.Interval(1.0, 5.0)
        v = torch.tensor((- 3.0))
        value = constraint.inverse_transform(constraint.transform(v))
        val_3 = v
        print(('log>>>%s' % complexObj(val_3)))
        val_4 = value
        print(('log>>>%s' % complexObj(val_4)))
        self.assertAllClose(val_3, val_4)

    def test_transform_tensor_bounds(self):
        constraint = gpytorch.constraints.Interval(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
        v = torch.tensor([(- 3.0), (- 2.0)])
        value = constraint.transform(v)
        actual_value = v.clone()
        actual_value[0] = (((3.0 - 1.0) * sigmoid(v[0])) + 1.0)
        actual_value[1] = (((4.0 - 2.0) * sigmoid(v[1])) + 2.0)
        val_5 = value
        print(('log>>>%s' % complexObj(val_5)))
        val_6 = actual_value
        print(('log>>>%s' % complexObj(val_6)))
        self.assertAllClose(val_5, val_6)

    def test_inverse_transform_tensor_bounds(self):
        constraint = gpytorch.constraints.Interval(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
        v = torch.tensor([(- 3.0), (- 2.0)])
        value = constraint.inverse_transform(constraint.transform(v))
        val_7 = v
        print(('log>>>%s' % complexObj(val_7)))
        val_8 = value
        print(('log>>>%s' % complexObj(val_8)))
        self.assertAllClose(val_7, val_8)

    def test_initial_value(self):
        constraint = gpytorch.constraints.Interval(1.0, 5.0, transform=None, initial_value=3.0)
        lkhd = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=constraint)
        self.assertEqual(lkhd.noise.item(), 3.0)

    def test_error_on_infinite(self):
        err_msg = 'Cannot make an Interval directly with non-finite bounds'
        with self.assertRaisesRegex(ValueError, err_msg):
            gpytorch.constraints.Interval(0.0, math.inf)
        with self.assertRaisesRegex(ValueError, err_msg):
            gpytorch.constraints.Interval((- math.inf), 0.0)

class TestGreaterThan(unittest.TestCase, BaseTestCase):

    def test_transform_float_greater_than(self):
        constraint = gpytorch.constraints.GreaterThan(1.0)
        v = torch.tensor((- 3.0))
        value = constraint.transform(v)
        actual_value = (softplus(v) + 1.0)
        val_9 = value
        print(('log>>>%s' % complexObj(val_9)))
        val_10 = actual_value
        print(('log>>>%s' % complexObj(val_10)))
        self.assertAllClose(val_9, val_10)

    def test_transform_tensor_greater_than(self):
        constraint = gpytorch.constraints.GreaterThan([1.0, 2.0])
        v = torch.tensor([(- 3.0), (- 2.0)])
        value = constraint.transform(v)
        actual_value = v.clone()
        actual_value[0] = (softplus(v[0]) + 1.0)
        actual_value[1] = (softplus(v[1]) + 2.0)
        val_11 = value
        print(('log>>>%s' % complexObj(val_11)))
        val_12 = actual_value
        print(('log>>>%s' % complexObj(val_12)))
        self.assertAllClose(val_11, val_12)

    def test_inverse_transform_float_greater_than(self):
        constraint = gpytorch.constraints.GreaterThan(1.0)
        v = torch.tensor((- 3.0))
        value = constraint.inverse_transform(constraint.transform(v))
        val_13 = value
        print(('log>>>%s' % complexObj(val_13)))
        val_14 = v
        print(('log>>>%s' % complexObj(val_14)))
        self.assertAllClose(val_13, val_14)

    def test_inverse_transform_tensor_greater_than(self):
        constraint = gpytorch.constraints.GreaterThan([1.0, 2.0])
        v = torch.tensor([(- 3.0), (- 2.0)])
        value = constraint.inverse_transform(constraint.transform(v))
        val_15 = value
        print(('log>>>%s' % complexObj(val_15)))
        val_16 = v
        print(('log>>>%s' % complexObj(val_16)))
        self.assertAllClose(val_15, val_16)

class TestLessThan(unittest.TestCase, BaseTestCase):

    def test_transform_float_less_than(self):
        constraint = gpytorch.constraints.LessThan(1.0)
        v = torch.tensor((- 3.0))
        value = constraint.transform(v)
        actual_value = ((- softplus((- v))) + 1.0)
        val_17 = value
        print(('log>>>%s' % complexObj(val_17)))
        val_18 = actual_value
        print(('log>>>%s' % complexObj(val_18)))
        self.assertAllClose(val_17, val_18)

    def test_transform_tensor_less_than(self):
        constraint = gpytorch.constraints.LessThan([1.0, 2.0])
        v = torch.tensor([(- 3.0), (- 2.0)])
        value = constraint.transform(v)
        actual_value = v.clone()
        actual_value[0] = ((- softplus((- v[0]))) + 1.0)
        actual_value[1] = ((- softplus((- v[1]))) + 2.0)
        val_19 = value
        print(('log>>>%s' % complexObj(val_19)))
        val_20 = actual_value
        print(('log>>>%s' % complexObj(val_20)))
        self.assertAllClose(val_19, val_20)

    def test_inverse_transform_float_less_than(self):
        constraint = gpytorch.constraints.LessThan(1.0)
        v = torch.tensor((- 3.0))
        value = constraint.inverse_transform(constraint.transform(v))
        self.assertAllClose(value, v)

    def test_inverse_transform_tensor_less_than(self):
        constraint = gpytorch.constraints.LessThan([1.0, 2.0])
        v = torch.tensor([(- 3.0), (- 2.0)])
        value = constraint.inverse_transform(constraint.transform(v))
        self.assertAllClose(value, v)

class TestPositive(unittest.TestCase, BaseTestCase):

    def test_transform_float_positive(self):
        constraint = gpytorch.constraints.Positive()
        v = torch.tensor((- 3.0))
        value = constraint.transform(v)
        actual_value = softplus(v)
        self.assertAllClose(value, actual_value)

    def test_transform_tensor_positive(self):
        constraint = gpytorch.constraints.Positive()
        v = torch.tensor([(- 3.0), (- 2.0)])
        value = constraint.transform(v)
        actual_value = v.clone()
        actual_value[0] = softplus(v[0])
        actual_value[1] = softplus(v[1])
        self.assertAllClose(value, actual_value)

    def test_inverse_transform_float_positive(self):
        constraint = gpytorch.constraints.Positive()
        v = torch.tensor((- 3.0))
        value = constraint.inverse_transform(constraint.transform(v))
        self.assertAllClose(value, v)

    def test_inverse_transform_tensor_positive(self):
        constraint = gpytorch.constraints.Positive()
        v = torch.tensor([(- 3.0), (- 2.0)])
        value = constraint.inverse_transform(constraint.transform(v))
        self.assertAllClose(value, v)

class TestConstraintNaming(unittest.TestCase, BaseTestCase):

    def test_constraint_by_name(self):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(None, None, likelihood)
        constraint = model.constraint_for_parameter_name('likelihood.noise_covar.raw_noise')
        self.assertIsInstance(constraint, gpytorch.constraints.GreaterThan)
        constraint = model.constraint_for_parameter_name('covar_module.base_kernel.raw_lengthscale')
        self.assertIsInstance(constraint, gpytorch.constraints.Positive)
        constraint = model.constraint_for_parameter_name('mean_module.constant')
        self.assertIsNone(constraint)

    def test_named_parameters_and_constraints(self):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(None, None, likelihood)
        for (name, _param, constraint) in model.named_parameters_and_constraints():
            if (name == 'likelihood.noise_covar.raw_noise'):
                self.assertIsInstance(constraint, gpytorch.constraints.GreaterThan)
            elif (name == 'mean_module.constant'):
                self.assertIsNone(constraint)
            elif (name == 'covar_module.raw_outputscale'):
                self.assertIsInstance(constraint, gpytorch.constraints.Positive)
            elif (name == 'covar_module.base_kernel.raw_lengthscale'):
                self.assertIsInstance(constraint, gpytorch.constraints.Positive)
if (__name__ == '__main__'):
    unittest.main()
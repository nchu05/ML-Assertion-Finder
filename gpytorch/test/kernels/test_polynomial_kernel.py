import unittest
import torch
from gpytorch.kernels import PolynomialKernel
from gpytorch.priors import NormalPrior
from gpytorch.test.base_kernel_test_case import BaseKernelTestCase
import pytest
import numpy as np

def sample_to_string(mystr):
    if isinstance(mystr, list):
        return '[{0}]'.format(','.join([sample_to_string(i) for i in mystr]))
    if isinstance(mystr, np.ndarray):
        return np.array2string(mystr, max_line_width=np.inf, precision=100, separator=',', threshold=np.inf).replace('\n', '')
    try:
        if isinstance(mystr, torch.Tensor):
            return np.array2string(mystr.detach().numpy(), max_line_width=np.inf, precision=100, separator=',', threshold=np.inf).replace('\n', '')
    except:
        pass
    try:
        if isinstance(mystr, tf.Tensor):
            with tf.Session():
                return np.array2string(mystr.eval(), max_line_width=np.inf, precision=100, separator=',', threshold=np.inf).replace('\n', '')
    except:
        pass
    return str(mystr)

class TestPolynomialKernel(unittest.TestCase, BaseKernelTestCase):

    def create_kernel_no_ard(self, **kwargs):
        return PolynomialKernel(power=2, **kwargs)

    @pytest.mark.usefixtures('show_guts')
    def test_computes_quadratic_kernel(self):
        a = torch.tensor([[4, 1], [2, 2], [8, 0]], dtype=torch.float)
        b = torch.tensor([[0, 0], [2, 1], [1, 0]], dtype=torch.float)
        kernel = PolynomialKernel(power=2)
        kernel.eval()
        actual = torch.zeros(3, 3)
        for i in range(3):
            for j in range(3):
                actual[(i, j)] = (a[i].matmul(b[j]) + kernel.offset).pow(kernel.power)
        res = kernel(a, b).to_dense()
        val_1 = torch.norm((res - actual))
        print(('log>>>%s' % sample_to_string(val_1)))
        val_2 = 1e-05
        print(('log>>>%s' % sample_to_string(val_2)))
        self.assertLess(val_1, val_2)
        res = kernel(a, b).diagonal(dim1=(- 1), dim2=(- 2))
        actual = actual.diagonal(dim1=(- 1), dim2=(- 2))
        val_3 = torch.norm((res - actual))
        print(('log>>>%s' % sample_to_string(val_3)))
        val_4 = 1e-05
        print(('log>>>%s' % sample_to_string(val_4)))
        self.assertLess(val_3, val_4)
        actual = torch.zeros(2, 3, 3)
        for i in range(2):
            actual[i] = kernel(a[(:, i)].unsqueeze((- 1)), b[(:, i)].unsqueeze((- 1))).to_dense()
        res = kernel(a, b, last_dim_is_batch=True).to_dense()
        val_5 = torch.norm((res - actual))
        print(('log>>>%s' % sample_to_string(val_5)))
        val_6 = 1e-05
        print(('log>>>%s' % sample_to_string(val_6)))
        self.assertLess(val_5, val_6)
        res = kernel(a, b, last_dim_is_batch=True).diagonal(dim1=(- 1), dim2=(- 2))
        actual = torch.cat([actual[i].diagonal(dim1=(- 1), dim2=(- 2)).unsqueeze(0) for i in range(actual.size(0))])
        val_7 = torch.norm((res - actual))
        print(('log>>>%s' % sample_to_string(val_7)))
        val_8 = 1e-05
        print(('log>>>%s' % sample_to_string(val_8)))
        self.assertLess(val_7, val_8)

    def test_computes_cubic_kernel(self):
        a = torch.tensor([[4, 1], [2, 2], [8, 0]], dtype=torch.float)
        b = torch.tensor([[0, 0], [2, 1], [1, 0]], dtype=torch.float)
        kernel = PolynomialKernel(power=3)
        kernel.eval()
        actual = torch.zeros(3, 3)
        for i in range(3):
            for j in range(3):
                actual[(i, j)] = (a[i].matmul(b[j]) + kernel.offset).pow(kernel.power)
        res = kernel(a, b).to_dense()
        val_9 = torch.norm((res - actual))
        print(('log>>>%s' % sample_to_string(val_9)))
        val_10 = 1e-05
        print(('log>>>%s' % sample_to_string(val_10)))
        self.assertLess(val_9, val_10)
        res = kernel(a, b).diagonal(dim1=(- 1), dim2=(- 2))
        actual = actual.diagonal(dim1=(- 1), dim2=(- 2))
        val_11 = torch.norm((res - actual))
        print(('log>>>%s' % sample_to_string(val_11)))
        val_12 = 1e-05
        print(('log>>>%s' % sample_to_string(val_12)))
        self.assertLess(val_11, val_12)
        actual = torch.zeros(2, 3, 3)
        for i in range(2):
            actual[i] = kernel(a[(:, i)].unsqueeze((- 1)), b[(:, i)].unsqueeze((- 1))).to_dense()
        res = kernel(a, b, last_dim_is_batch=True).to_dense()
        val_13 = torch.norm((res - actual))
        print(('log>>>%s' % sample_to_string(val_13)))
        val_14 = 1e-05
        print(('log>>>%s' % sample_to_string(val_14)))
        self.assertLess(val_13, val_14)
        res = kernel(a, b, last_dim_is_batch=True).diagonal(dim1=(- 1), dim2=(- 2))
        actual = torch.cat([actual[i].diagonal(dim1=(- 1), dim2=(- 2)).unsqueeze(0) for i in range(actual.size(0))])
        val_15 = torch.norm((res - actual))
        print(('log>>>%s' % sample_to_string(val_15)))
        val_16 = 1e-05
        print(('log>>>%s' % sample_to_string(val_16)))
        self.assertLess(val_15, val_16)

    def test_quadratic_kernel_batch(self):
        a = torch.tensor([[4, 2, 8], [1, 2, 3]], dtype=torch.float).view(2, 3, 1)
        b = torch.tensor([[0, 2, 1], [(- 1), 2, 0]], dtype=torch.float).view(2, 3, 1)
        kernel = PolynomialKernel(power=2, batch_shape=torch.Size([2])).initialize(offset=torch.rand(2, 1))
        kernel.eval()
        actual = torch.zeros(2, 3, 3)
        for k in range(2):
            for i in range(3):
                for j in range(3):
                    actual[(k, i, j)] = (a[(k, i)].matmul(b[(k, j)]) + kernel.offset[k]).pow(kernel.power)
        res = kernel(a, b).to_dense()
        val_17 = torch.norm((res - actual))
        print(('log>>>%s' % sample_to_string(val_17)))
        val_18 = 1e-05
        print(('log>>>%s' % sample_to_string(val_18)))
        self.assertLess(val_17, val_18)

    def test_cubic_kernel_batch(self):
        a = torch.tensor([[4, 2, 8], [1, 2, 3]], dtype=torch.float).view(2, 3, 1)
        b = torch.tensor([[0, 2, 1], [(- 1), 2, 0]], dtype=torch.float).view(2, 3, 1)
        kernel = PolynomialKernel(power=3, batch_shape=torch.Size([2])).initialize(offset=torch.rand(2, 1))
        kernel.eval()
        actual = torch.zeros(2, 3, 3)
        for k in range(2):
            for i in range(3):
                for j in range(3):
                    actual[(k, i, j)] = (a[(k, i)].matmul(b[(k, j)]) + kernel.offset[k]).pow(kernel.power)
        res = kernel(a, b).to_dense()
        val_19 = torch.norm((res - actual))
        print(('log>>>%s' % sample_to_string(val_19)))
        val_20 = 1e-05
        print(('log>>>%s' % sample_to_string(val_20)))
        self.assertLess(val_19, val_20)

    def create_kernel_with_prior(self, offset_prior):
        return self.create_kernel_no_ard(offset_prior=offset_prior)

    def test_prior_type(self):
        '\n        Raising TypeError if prior type is other than gpytorch.priors.Prior\n        '
        self.create_kernel_with_prior(None)
        self.create_kernel_with_prior(NormalPrior(0, 1))
        self.assertRaises(TypeError, self.create_kernel_with_prior, 1)
if (__name__ == '__main__'):
    unittest.main()
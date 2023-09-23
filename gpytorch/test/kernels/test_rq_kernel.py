import math
import unittest
import torch
from gpytorch.kernels import RQKernel
from gpytorch.test.base_kernel_test_case import BaseKernelTestCase
import torch
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

class TestRQKernel(unittest.TestCase, BaseKernelTestCase):

    def create_kernel_no_ard(self, **kwargs):
        return RQKernel(**kwargs)

    def create_kernel_ard(self, num_dims, **kwargs):
        return RQKernel(ard_num_dims=num_dims, **kwargs)

    @pytest.mark.usefixtures('show_guts')
    def test_ard(self):
        a = torch.tensor([[1, 2], [2, 4]], dtype=torch.float)
        b = torch.tensor([[1, 3], [0, 4]], dtype=torch.float)
        lengthscales = torch.tensor([1, 2], dtype=torch.float).view(1, 2)
        kernel = RQKernel(ard_num_dims=2)
        kernel.initialize(lengthscale=lengthscales)
        kernel.initialize(alpha=3.0)
        kernel.eval()
        scaled_a = a.div(lengthscales)
        scaled_b = b.div(lengthscales)
        dist = (scaled_a.unsqueeze((- 2)) - scaled_b.unsqueeze((- 3))).pow(2).sum(dim=(- 1))
        actual = dist.div_((2 * kernel.alpha)).add_(1.0).pow((- kernel.alpha))
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
        diff = (scaled_a.transpose((- 1), (- 2)).unsqueeze((- 1)) - scaled_b.transpose((- 1), (- 2)).unsqueeze((- 2)))
        actual = diff.pow(2).div_((2 * kernel.alpha)).add_(1.0).pow((- kernel.alpha))
        res = kernel(a, b, last_dim_is_batch=True).to_dense()
        val_5 = torch.norm((res - actual))
        print(('log>>>%s' % sample_to_string(val_5)))
        val_6 = 1e-05
        print(('log>>>%s' % sample_to_string(val_6)))
        self.assertLess(val_5, val_6)
        res = kernel(a, b, last_dim_is_batch=True).diagonal(dim1=(- 1), dim2=(- 2))
        actual = actual.diagonal(dim1=(- 1), dim2=(- 2))
        val_7 = torch.norm((res - actual))
        print(('log>>>%s' % sample_to_string(val_7)))
        val_8 = 1e-05
        print(('log>>>%s' % sample_to_string(val_8)))
        self.assertLess(val_7, val_8)

    def test_ard_batch(self):
        a = torch.tensor([[[1, 2, 3], [2, 4, 0]], [[(- 1), 1, 2], [2, 1, 4]]], dtype=torch.float)
        b = torch.tensor([[[1, 3, 1]], [[2, (- 1), 0]]], dtype=torch.float).repeat(1, 2, 1)
        lengthscales = torch.tensor([[[1, 2, 1]]], dtype=torch.float)
        kernel = RQKernel(batch_shape=torch.Size([2]), ard_num_dims=3)
        kernel.initialize(lengthscale=lengthscales)
        kernel.initialize(alpha=3.0)
        kernel.eval()
        scaled_a = a.div(lengthscales)
        scaled_b = b.div(lengthscales)
        dist = (scaled_a.unsqueeze((- 2)) - scaled_b.unsqueeze((- 3))).pow(2).sum(dim=(- 1))
        actual = dist.div_((2 * kernel.alpha)).add_(1.0).pow((- kernel.alpha))
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
        double_batch_a = scaled_a.transpose((- 1), (- 2)).unsqueeze((- 1))
        double_batch_b = scaled_b.transpose((- 1), (- 2)).unsqueeze((- 2))
        actual = (double_batch_a - double_batch_b)
        alpha = kernel.alpha.view(2, 1, 1, 1)
        actual = actual.pow_(2).div_((2 * alpha)).add_(1.0).pow((- alpha))
        res = kernel(a, b, last_dim_is_batch=True).to_dense()
        val_13 = torch.norm((res - actual))
        print(('log>>>%s' % sample_to_string(val_13)))
        val_14 = 1e-05
        print(('log>>>%s' % sample_to_string(val_14)))
        self.assertLess(val_13, val_14)
        res = kernel(a, b, last_dim_is_batch=True).diagonal(dim1=(- 1), dim2=(- 2))
        actual = actual.diagonal(dim1=(- 2), dim2=(- 1))
        val_15 = torch.norm((res - actual))
        print(('log>>>%s' % sample_to_string(val_15)))
        val_16 = 1e-05
        print(('log>>>%s' % sample_to_string(val_16)))
        self.assertLess(val_15, val_16)

    def test_ard_separate_batch(self):
        a = torch.tensor([[[1, 2, 3], [2, 4, 0]], [[(- 1), 1, 2], [2, 1, 4]]], dtype=torch.float)
        b = torch.tensor([[[1, 3, 1]], [[2, (- 1), 0]]], dtype=torch.float).repeat(1, 2, 1)
        lengthscales = torch.tensor([[[1, 2, 1]], [[2, 1, 0.5]]], dtype=torch.float)
        kernel = RQKernel(batch_shape=torch.Size([2]), ard_num_dims=3)
        kernel.initialize(lengthscale=lengthscales)
        kernel.initialize(alpha=3.0)
        kernel.eval()
        scaled_a = a.div(lengthscales)
        scaled_b = b.div(lengthscales)
        dist = (scaled_a.unsqueeze((- 2)) - scaled_b.unsqueeze((- 3))).pow(2).sum(dim=(- 1))
        actual = dist.div_((2 * kernel.alpha)).add_(1.0).pow((- kernel.alpha))
        res = kernel(a, b).to_dense()
        val_17 = torch.norm((res - actual))
        print(('log>>>%s' % sample_to_string(val_17)))
        val_18 = 1e-05
        print(('log>>>%s' % sample_to_string(val_18)))
        self.assertLess(val_17, val_18)
        res = kernel(a, b).diagonal(dim1=(- 1), dim2=(- 2))
        actual = actual.diagonal(dim1=(- 1), dim2=(- 2))
        val_19 = torch.norm((res - actual))
        print(('log>>>%s' % sample_to_string(val_19)))
        val_20 = 1e-05
        print(('log>>>%s' % sample_to_string(val_20)))
        self.assertLess(val_19, val_20)

    def test_computes_rational_quadratic(self):
        a = torch.tensor([4, 2, 8], dtype=torch.float).view(3, 1)
        b = torch.tensor([0, 2, 4], dtype=torch.float).view(3, 1)
        lengthscale = 2
        kernel = RQKernel().initialize(lengthscale=lengthscale)
        kernel.eval()
        dist = torch.tensor([[16, 4, 0], [4, 0, 4], [64, 36, 16]], dtype=torch.float).div((lengthscale ** 2))
        actual = dist.div_((2 * kernel.alpha)).add_(1.0).pow((- kernel.alpha))
        res = kernel(a, b).to_dense()
        val_21 = torch.norm((res - actual))
        print(('log>>>%s' % sample_to_string(val_21)))
        val_22 = 1e-05
        print(('log>>>%s' % sample_to_string(val_22)))
        self.assertLess(val_21, val_22)
        res = kernel(a, b).diagonal(dim1=(- 1), dim2=(- 2))
        actual = actual.diagonal(dim1=(- 1), dim2=(- 2))
        val_23 = torch.norm((res - actual))
        print(('log>>>%s' % sample_to_string(val_23)))
        val_24 = 1e-05
        print(('log>>>%s' % sample_to_string(val_24)))
        self.assertLess(val_23, val_24)

    def test_computes_rational_quadratic_gradient(self):
        softplus = torch.nn.functional.softplus
        a = torch.tensor([4, 2, 8], dtype=torch.float).view(3, 1)
        b = torch.tensor([0, 2, 2], dtype=torch.float).view(3, 1)
        kernel = RQKernel()
        kernel.initialize(lengthscale=2.0)
        kernel.initialize(alpha=3.0)
        kernel.eval()
        raw_lengthscale = torch.tensor(math.log((math.exp(2.0) - 1)))
        raw_lengthscale.requires_grad_()
        raw_alpha = torch.tensor(math.log((math.exp(3.0) - 1)))
        raw_alpha.requires_grad_()
        (lengthscale, alpha) = (softplus(raw_lengthscale), softplus(raw_alpha))
        dist = (a.expand(3, 3) - b.expand(3, 3).transpose(0, 1)).div(lengthscale).pow(2)
        actual_output = dist.div((2 * alpha)).add(1).pow((- alpha))
        actual_output.backward(gradient=torch.eye(3))
        output = kernel(a, b).to_dense()
        output.backward(gradient=torch.eye(3))
        res = kernel.raw_lengthscale.grad
        val_25 = torch.norm((res - raw_lengthscale.grad))
        print(('log>>>%s' % sample_to_string(val_25)))
        val_26 = 1e-05
        print(('log>>>%s' % sample_to_string(val_26)))
        self.assertLess(val_25, val_26)
        res = kernel.raw_alpha.grad
        val_27 = torch.norm((res - raw_alpha.grad))
        print(('log>>>%s' % sample_to_string(val_27)))
        val_28 = 1e-05
        print(('log>>>%s' % sample_to_string(val_28)))
        self.assertLess(val_27, val_28)

    def test_subset_active_compute_rational_quadratic(self):
        a = torch.tensor([4, 2, 8], dtype=torch.float).view(3, 1)
        a_p = torch.tensor([1, 2, 3], dtype=torch.float).view(3, 1)
        a = torch.cat((a, a_p), 1)
        b = torch.tensor([0, 2, 4], dtype=torch.float).view(3, 1)
        lengthscale = 2
        kernel = RQKernel(active_dims=[0])
        kernel.initialize(lengthscale=lengthscale)
        kernel.initialize(alpha=3.0)
        kernel.eval()
        actual = torch.tensor([[16, 4, 0], [4, 0, 4], [64, 36, 16]], dtype=torch.float)
        actual.div_((lengthscale ** 2)).div_((2 * kernel.alpha)).add_(1).pow_((- kernel.alpha))
        res = kernel(a, b).to_dense()
        val_29 = torch.norm((res - actual))
        print(('log>>>%s' % sample_to_string(val_29)))
        val_30 = 1e-05
        print(('log>>>%s' % sample_to_string(val_30)))
        self.assertLess(val_29, val_30)
        res = kernel(a, b).diagonal(dim1=(- 1), dim2=(- 2))
        actual = actual.diagonal(dim1=(- 1), dim2=(- 2))
        val_31 = torch.norm((res - actual))
        print(('log>>>%s' % sample_to_string(val_31)))
        val_32 = 1e-05
        print(('log>>>%s' % sample_to_string(val_32)))
        self.assertLess(val_31, val_32)

    def test_subset_active_computes_rational_quadratic_gradient(self):
        softplus = torch.nn.functional.softplus
        a_1 = torch.tensor([4, 2, 8], dtype=torch.float).view(3, 1)
        a_p = torch.tensor([1, 2, 3], dtype=torch.float).view(3, 1)
        a = torch.cat((a_1, a_p), 1)
        b = torch.tensor([0, 2, 2], dtype=torch.float).view(3, 1)
        kernel = RQKernel(active_dims=[0])
        kernel.initialize(lengthscale=2.0)
        kernel.initialize(alpha=3.0)
        kernel.eval()
        raw_lengthscale = torch.tensor(math.log((math.exp(2.0) - 1)))
        raw_lengthscale.requires_grad_()
        raw_alpha = torch.tensor(math.log((math.exp(3.0) - 1)))
        raw_alpha.requires_grad_()
        (lengthscale, alpha) = (softplus(raw_lengthscale), softplus(raw_alpha))
        dist = (a_1.expand(3, 3) - b.expand(3, 3).transpose(0, 1)).div(lengthscale).pow(2)
        actual_output = dist.div((2 * alpha)).add(1).pow((- alpha))
        actual_output.backward(gradient=torch.eye(3))
        output = kernel(a, b).to_dense()
        output.backward(gradient=torch.eye(3))
        res = kernel.raw_lengthscale.grad
        val_33 = torch.norm((res - raw_lengthscale.grad))
        print(('log>>>%s' % sample_to_string(val_33)))
        val_34 = 1e-05
        print(('log>>>%s' % sample_to_string(val_34)))
        self.assertLess(val_33, val_34)
        res = kernel.raw_alpha.grad
        val_35 = torch.norm((res - raw_alpha.grad))
        print(('log>>>%s' % sample_to_string(val_35)))
        val_36 = 1e-05
        print(('log>>>%s' % sample_to_string(val_36)))
        self.assertLess(val_35, val_36)

    def test_initialize_lengthscale(self):
        kernel = RQKernel()
        kernel.initialize(lengthscale=3.14)
        actual_value = torch.tensor(3.14).view_as(kernel.lengthscale)
        val_37 = torch.norm((kernel.lengthscale - actual_value))
        print(('log>>>%s' % sample_to_string(val_37)))
        val_38 = 1e-05
        print(('log>>>%s' % sample_to_string(val_38)))
        self.assertLess(val_37, val_38)

    def test_initialize_lengthscale_batch(self):
        kernel = RQKernel(batch_shape=torch.Size([2]))
        ls_init = torch.tensor([3.14, 4.13])
        kernel.initialize(lengthscale=ls_init)
        actual_value = ls_init.view_as(kernel.lengthscale)
        val_39 = torch.norm((kernel.lengthscale - actual_value))
        print(('log>>>%s' % sample_to_string(val_39)))
        val_40 = 1e-05
        print(('log>>>%s' % sample_to_string(val_40)))
        self.assertLess(val_39, val_40)

    def test_initialize_alpha(self):
        kernel = RQKernel()
        kernel.initialize(alpha=3.0)
        actual_value = torch.tensor(3.0).view_as(kernel.alpha)
        val_41 = torch.norm((kernel.alpha - actual_value))
        print(('log>>>%s' % sample_to_string(val_41)))
        val_42 = 1e-05
        print(('log>>>%s' % sample_to_string(val_42)))
        self.assertLess(val_41, val_42)
if (__name__ == '__main__'):
    unittest.main()
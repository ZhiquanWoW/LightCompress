#!/usr/bin/env python
# coding=utf-8

import math
import argparse
import configparser 
from numba import cuda
# import numpy as np
import cupy as np
import struct
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use('webagg')
import matplotlib.pyplot as plt

class cxn1_lut_t():
    def __init__(self, index_nbit, dx_nbit, input_table = None):
        self.entry_num = (1 << index_nbit) + 1
        self.dx_nbit = dx_nbit
        # self.table = self._create_table(self.entry_num)

    def create_exp2_table(self):
        self.bias = np.float32(0)
        self.scale = np.float32(65504)

        interval = 1 / (self.entry_num)
        x = np.arange(self.entry_num, dtype=np.float16) * interval
        table = np.power(2, x).astype(np.float16)
        
        self.table =  table

    def create_silu_table(self):
        l = -8
        r = 8

        self.bias = np.float32(r)
        self.scale = np.float32(4096)

        interval = (r - l) / (self.entry_num)
        x = np.arange(self.entry_num, dtype=np.float16) * interval - r
        table = F.silu(torch.tensor(x)).numpy().astype(np.float16)

        self.table =  table

    def _lut_pre(self, x : np.float16):
        a = ((np.float32(x) + self.bias) * self.scale).astype(np.uint16)
        index = a >> self.dx_nbit
        dx = a & 0x7f

        return index, dx
    
    def _lut_intp(self, y0 : np.float16, y1 : np.float16, dx : int, dx_nbit : int):
        slope = y1.astype(np.float32) - y0.astype(np.float32)
        result = (y0 * np.float32(1 << dx_nbit) + slope * np.float32(dx)) / 128

        return result.astype(np.float32)

    def _calc_exp_result(self, exponent, y):
        return np.power(2, exponent.astype(np.float16)) * y.astype(np.float32)

    def _find_table(self, index):
        return self.table[index], self.table[index+1]

    def exp_forward(self, x):
        exponent = np.floor(x)
        fractional = x - exponent
        
        index,dx = self._lut_pre(fractional)

        y0,y1 = self._find_table(index)
        y = self._lut_intp(y0, y1, dx, self.dx_nbit)
        result = self._calc_exp_result(exponent, y)
        return result

    def silu_forward(self, x):
        index,dx = self._lut_pre(x)

        y0,y1 = self._find_table(index)
        y = self._lut_intp(y0, y1, dx, self.dx_nbit)
        return y

class vsi_pwl_t:
    # entry_shape: (1,2,5) signed, exp_bits, mant_bits
    def __init__(self, entry_shape = (1, 2, 4)):
        self.sig_bit, self.exp_nbit, self.mant_nbit = entry_shape
        
        self.entry_num = pow(2, self.sig_bit) * pow(2,self.exp_nbit) * pow(2,self.mant_nbit)
        self.index_nbit = self.sig_bit + self.exp_nbit + self.mant_nbit
        # self.k_tab,self.b_tab = self._create_table(self.entry_num)

    def create_inputs(self, scale):
        fraction = np.zeros(1 << self.mant_nbit, dtype = np.float32)
        tmp = np.arange(1 << self.mant_nbit)
        fraction = 1.0 + tmp * (1 / (1 << self.mant_nbit))

        exps = np.arange(1 << self.exp_nbit)
        exps = exps - ((1 << self.exp_nbit) / 2) + 1
        
        x = np.zeros(self.entry_num, dtype = np.float32)

        i = 0
        for sig in [1.0, -1.0]:
            for exp in exps:
                x[i : i+(1<<self.mant_nbit)] = sig * np.power(2.0, exp) * fraction
                i = i + (1 << self.mant_nbit)

        # for mant in range(1 << self.mant_nbit):
        #     fraction = 1 + 1 / (1 << self.mant_nbit) * i

        # i = 0
        # for sig in [0, 1]:
        #     for exp in range(1 << self.exp_nbit):
        #         for mant in range(1 << self.mant_nbit):
        #             if exp >> (self.exp_nbit - 1):
        #                 x[i] = (sig << 31) | (1<<30) | (exp << 23) | (mant << (23 - self.mant_nbit))
        #             else:
        #                 x[i] = (sig << 31) | (0<<30) | ((((0x7f >> self.exp_nbit) << self.exp_nbit) | exp) << 23) |  (mant << (23 - self.mant_nbit))
        #             i = i + 1

        x_list = x.view(np.float32)
        return x_list

    def create_silu_table(self):
        self.scale = 1
        self.k_tab = None
        self.b_tab = None

        x_list = self.create_inputs(self.scale)
        y_list = F.silu(torch.tensor(x_list)).to(torch.float16).numpy()

        # 这里少一个表项，x_list 256个数只能生成255个 k,b
        k_tab = np.zeros(self.entry_num, dtype=np.float32)
        b_tab = np.zeros(self.entry_num, dtype=np.float32)
        for i in range(self.entry_num - 1):
            k_tab[i] = (y_list[i + 1] - y_list[i]) / (x_list[i + 1] - x_list[i])
            b_tab[i] = y_list[i] - x_list[i] * k_tab[i]

        # 处理2个边界情况
        r_edge = int(self.entry_num / 2 - 1)
        k_tab[r_edge] = (F.silu(torch.tensor(x_list[r_edge] + 10000)) - F.silu(torch.tensor(x_list[r_edge]))) / 10000
        b_tab[r_edge] = F.silu(torch.tensor(x_list[r_edge])).numpy() - k_tab[r_edge] * x_list[r_edge]
        b_tab[-1] = F.silu(torch.tensor(x_list[-1] - 100)).numpy()
        k_tab[-1] = (F.silu(torch.tensor(x_list[-1] - 100)) - F.silu(torch.tensor(x_list[-1])))

        k_tab = k_tab.astype(np.float16)
        b_tab = b_tab.astype(np.float16)
        self.k_tab,self.b_tab = k_tab,b_tab
        np.cuda.Stream.null.synchronize()


    def create_tanh_table(self):
        self.scale = 1
        self.k_tab = None
        self.b_tab = None

        x_list = self.create_inputs(self.scale)
        y_list = F.tanh(torch.tensor(x_list)).to(torch.float16).numpy()

        # 这里少一个表项，x_list 256个数只能生成255个 k,b
        k_tab = np.zeros(self.entry_num, dtype=np.float32)
        b_tab = np.zeros(self.entry_num, dtype=np.float32)
        for i in range(self.entry_num - 1):
            k_tab[i] = (y_list[i + 1] - y_list[i]) / (x_list[i + 1] - x_list[i])
            b_tab[i] = y_list[i] - x_list[i] * k_tab[i]

        # 处理2个边界情况
        r_edge = int(self.entry_num / 2 - 1)
        k_tab[r_edge] = (F.tanh(torch.tensor(x_list[r_edge] + 10000)) - F.tanh(torch.tensor(x_list[r_edge]))) / 10000
        b_tab[r_edge] = F.tanh(torch.tensor(x_list[r_edge])).numpy() - k_tab[r_edge] * x_list[r_edge]

        b_tab[-1] = F.tanh(torch.tensor(x_list[-1] - 10000)).numpy()
        k_tab[-1] = (F.tanh(torch.tensor(x_list[-1] - 10000)) - F.tanh(torch.tensor(x_list[-1])))

        k_tab = k_tab.astype(np.float16)
        b_tab = b_tab.astype(np.float16)
        self.k_tab,self.b_tab = k_tab,b_tab

    def create_exp2_table(self):
        self.scale = np.float16(1)
        self.bias = np.float16(0)

        x_list = self.create_inputs(self.scale)
        y_list = np.pow(2, x_list, dtype=np.float32)

        k_tab = np.zeros(self.entry_num, dtype=np.float32)
        b_tab = np.zeros(self.entry_num, dtype=np.float32)
        for i in range(self.entry_num - 1):
            k_tab[i] = (y_list[i + 1] - y_list[i]) / (x_list[i+1] - x_list[i])
            b_tab[i] = y_list[i] - x_list[i] * k_tab[i]

        b_tab[-1] = F.silu(torch.tensor(x_list[-1] - 10)).numpy()
        k_tab[-1] = (F.silu(torch.tensor(x_list[-1] - 10)) - F.silu(torch.tensor(x_list[-1])))

        k_tab = k_tab.astype(np.float16)
        b_tab = b_tab.astype(np.float16)
        self.k_tab,self.b_tab = k_tab,b_tab
    
    def _calc_exp2_result(self, exponent, pwl_y):
        return (np.power(2, exponent.astype(np.float32)) * pwl_y.astype(np.float32)).astype(np.float16)

    def split(self, fp32, exp_nbit, mant_nbit):
        x = fp32.view(np.uint32)
        signed = x >> 31
        exps = (x >> 23) & 0xff
        # 8位指数位
        exps = exps.astype(np.uint8)

        # 得到指index
        exp_idx = ((exps & 0x80) >> (8-exp_nbit)) | ((exps <<  (8 - exp_nbit + 1)) >> (8 - exp_nbit + 1))

        '''
        for i,val in enumerate(exp):
            if 1 == (val >> 7):
                exp[i] = (val & 0x7f) + 1
            else:
                exp[i] = (((val&0x7f).view(np.uint8) + 1) | 0x80).view(np.int8)
        exp = exp + (1 << (exp_nbit - 1))
        '''

        # 得到尾数位
        mant = x & 0x7fffff
        mant_idx = mant >> (23 - mant_nbit)

        # 越界处理
        for i,exp in enumerate(exps):
            if 0 == (exp >> 7):     # exp 负边界
                if (1 << exp_nbit-1) -1 < (~exp & 0x7f):
                    exp_idx[i] = 0x00
                    mant_idx[i] = 0x00
            else:                   # exp 正边界
                if (1 << exp_nbit-1) < ((exp & 0x7f) + 1):
                    exp_idx[i] = 0xff >> (8 - exp_nbit)
                    mant_idx[i] = 0xff >> (8 - mant_nbit)

        '''
        for i,val in enumerate(exp):
            if exp[i] < 0:
                exp[i] = 0
                mant[i] = 0
            if exp[i] > (1 << exp_nbit):
                exp[i] = 1 << exp_nbit
                mant[i] = 0xffffffff >> (23 - mant_nbit)
        '''

        return signed, exp_idx, mant_idx

    def _find_table(self, index):
        return self.k_tab[index], self.b_tab[index]

    def forward(self, x):

        x = x.astype(np.float32) * np.float32(self.scale)

        sig_idx,exp_idx,mant_idx = self.split(x, self.exp_nbit, self.mant_nbit)
        index = (sig_idx << (self.exp_nbit + self.mant_nbit)) | (exp_idx << self.mant_nbit) | mant_idx
        
        k,b = self._find_table(index)

        y = np.float32(k) * np.float32(x) + np.float32(b) 
        np.cuda.Stream.null.synchronize()

        return y.astype(np.float16)
    
    def exp_forward(self, x):
        x = x.astype(np.float32)
        exponent = np.floor(x)
        fractional = (x.astype(np.float32) - exponent).astype(np.float32)

        y = self.forward(fractional)
        exp_result = self._calc_exp2_result(exponent, y)
        return exp_result


class pwl_test1_t:
    def _create_table(self, entry_num):
        k = np.arange(entry_num, dtype=np.float16)
        b = np.arange(entry_num, dtype=np.float16)
        return k,b
    
    # input remap: (-12,12) -> (1,2)
    # scale = 0.03125, bias = 48
    def create_exp_table(self):
        self.scale = 0.03125 # 
        self.bias = 48

        range_l = -16
        range_r = 16
        interval = (range_r - range_l) / (self.entry_num + 1)

        x_list = np.arange(self.entry_num + 1, dtype=np.float16) * interval + range_l
        y_list = np.exp(x_list)

        k_tab = np.zeros(self.entry_num, dtype=np.float16)
        b_tab = np.zeros(self.entry_num, dtype=np.float16)
        for i in range(self.entry_num):
            k_tab[i] = (y_list[i + 1] - y_list[i]) / interval
            b_tab[i] = y_list[i] - x_list[i] * k_tab[i]
        
        self.k_tab,self.b_tab = k_tab,b_tab

    def create_exp2_table(self):
        self.scale = np.float16(1)
        self.bias = np.float16(1)
        range_l = 0
        range_r = 1

        interval = (range_r - range_l) / (self.entry_num + 1)

        x_list = np.arange(self.entry_num + 1, dtype=np.float32) * interval + range_l
        y_list = np.pow(2, x_list, dtype=np.float32)

        k_tab = np.zeros(self.entry_num, dtype=np.float32)
        b_tab = np.zeros(self.entry_num, dtype=np.float32)
        for i in range(self.entry_num):
            k_tab[i] = (y_list[i + 1] - y_list[i]) / interval
            b_tab[i] = y_list[i] - x_list[i] * k_tab[i]

        k_tab = k_tab.astype(np.float16)
        b_tab = b_tab.astype(np.float16)
        self.k_tab,self.b_tab = k_tab,b_tab

    def create_silu_table(self):
        range_l = -8
        range_r = 8
        self.scale = np.float16(1 / (range_r - range_l))
        self.bias = 1 - range_l * self.scale

        interval = (range_r - range_l) / (self.entry_num + 1)

        x_list = np.arange(self.entry_num + 1, dtype=np.float32) * interval + range_l
        x_list[0] = x_list[1] - 40000
        x_list[-1] = x_list[-2] + 40000
        y_list = F.silu(torch.tensor(x_list)).numpy()

        k_tab = np.zeros(self.entry_num, dtype=np.float32)
        b_tab = np.zeros(self.entry_num, dtype=np.float32)
        for i in range(self.entry_num):
            k_tab[i] = (y_list[i + 1] - y_list[i]) / (x_list[i+1] - x_list[i])
            b_tab[i] = y_list[i] - x_list[i] * k_tab[i]

        k_tab = k_tab.astype(np.float16)
        b_tab = b_tab.astype(np.float16)

        self.k_tab,self.b_tab = k_tab,b_tab

    def create_tanh_table(self, range_l=-8, range_r=8):
        self.scale = np.float16(1 / (range_r - range_l))
        self.bias = 1 - range_l * self.scale

        interval = (range_r - range_l) / (self.entry_num + 1)

        x_list = np.arange(self.entry_num + 1, dtype=np.float32) * interval + range_l
        x_list[0] = x_list[0] - 1000
        x_list[-1] = x_list[-2] + 1000
        y_list = F.tanh(torch.tensor(x_list)).numpy()

        k_tab = np.zeros(self.entry_num, dtype=np.float32)
        b_tab = np.zeros(self.entry_num, dtype=np.float32)
        for i in range(self.entry_num):
            k_tab[i] = (y_list[i + 1] - y_list[i]) / (x_list[i+1] - x_list[i])
            b_tab[i] = y_list[i] - x_list[i] * k_tab[i]

        k_tab = k_tab.astype(np.float16)
        b_tab = b_tab.astype(np.float16)

        self.k_tab,self.b_tab = k_tab,b_tab

    def _calc_exp2_result(self, exponent, pwl_y):
        return (np.power(2, exponent.astype(np.float32)) * pwl_y.astype(np.float32)).astype(np.float16)

    def exp_forward(self, x):
        x = x.astype(np.float32)
        exponent = np.floor(x)
        fractional = (x.astype(np.float32) - exponent).astype(np.float32)

        _,_,index = self._lut_pre(fractional)
        k,b = self._find_table(index)
        y = k.astype(np.float32) * fractional.astype(np.float32) + b.astype(np.float32)

        exp_result = self._calc_exp2_result(exponent, y)
        return exp_result

    def forward(self, x):
        _, _, index = self._lut_pre(x)
        k,b = self._find_table(index)

        y = np.float32(k) * np.float32(x) + np.float32(b) 
        return y.astype(np.float16)

    def __init__(self, index_nbist, in_k_table = None, in_b_table = None):
        self.index_nbit = index_nbist
        self.entry_num = 1 << self.index_nbit

        self.k_tab,self.b_tab = self._create_table(self.entry_num)

    # bf16 : 1,8,7
    # fp16 : 1,5,10
    # scale. 令 1 <= x * scale < 2
    def _lut_exp_pre(self, x : np.float16):
        x = x.astype(np.float16)
        exponent = np.floor(x)
        fractional = (x - exponent).astype(np.float16)
        tmp = (fractional * self.scale) + self.bias
        index = (np.float16(tmp).view(np.uint16) & 0x3ff) >> (10 - self.index_nbit)

        for i in range(len(x)):
            if 0x10 == ((np.float16(tmp[i]).view(np.uint16) >> 10) & 0x1f):
                index[i] = self.entry_num - 1
        return exponent,fractional,index

    def _lut_pre(self, x : np.float16):
        x = x.astype(np.float32)
        tmp = (x * self.scale) + self.bias
        index = (np.float16(tmp).view(np.uint16) & 0x3ff) >> (10 - self.index_nbit)

        # 边界处理
        index[np.where(tmp < 1)] = 0
        for i,val in enumerate(tmp):
            if 0 == np.float16(val).view(np.uint16) >> 15:  # 正边界
                if 0x10 <= (np.float16(val).view(np.uint16) & 0x7fff) >> 10:
                    index[i] = self.entry_num - 1

        return None,None,index

    def _find_table(self, index):
        return self.k_tab[index], self.b_tab[index]
    
    def _calc_result(self, x, k, b):
        return x * k + b

def print_err(label,result, golden):
    abs_err = np.abs(golden - result)
    rel_err = np.abs(abs_err.astype(np.float32) / golden)
    mae = np.mean(np.abs(golden - result))
    mse = np.mean((golden - result) ** 2)
    print(f'  {label} err_max \t{abs_err.max():.5f} \trela_err_max \t{rel_err.max():.4%} \t mae \t{mae:.4} \tmse \t{mse:.4}')

def eva_exp_precision(x_min, x_max, N):
    print(f'输入[{x_min},{x_max}] \t 参数量 {N}')

    x = np.random.uniform(low=x_min, high=x_max, size=N).astype(np.float16)
    x = np.sort(x)

    py_result = np.exp(x.astype(np.float16))
    x2 = x.astype(np.float16) * np.float16(1.4426950408889)

    # 现有lut， scale使用 2^n
    cxn1_exp_lut2 = cxn1_lut_t(9, 7, input_table = None)
    cxn1_exp_lut2.create_exp2_table()
    lut_result = cxn1_exp_lut2.exp_forward(x2)
    print_err('lut', lut_result, py_result)

    name = f'lut_exp_{x_min}_{x_max}_N_{N}'
    plot_put_pkg = (x, lut_result, "lut", py_result, 'torch', name)

    # pwl 均匀分段
    cxn1_pwl_exp = pwl_test1_t(7)
    cxn1_pwl_exp.create_exp2_table()
    pwl_out = cxn1_pwl_exp.exp_forward(x2)
    print_err('pwl', pwl_out, py_result)

    name = f'pwl_exp_{x_min}_{x_max}_N_{N}'
    plot_pwl_pkg = (x, pwl_out, "pwl", py_result, 'torch', name)

    # vsi 1,4,2
    vsi_142 = vsi_pwl_t((1,4,2))
    vsi_142.create_exp2_table()
    vsi_142_out = vsi_142.exp_forward(x2)
    print_err('142', vsi_142_out, py_result)
    name = f'vsi_142_exp_{x_min}_{x_max}_N_{N}'
    vsi_142_pkg = (x, vsi_142_out, "vsi_142", py_result, 'torch', name)

    # vsi 1,3,3
    vsi_133 = vsi_pwl_t((1,3,3))
    vsi_133.create_exp2_table()
    vsi_133_out = vsi_133.exp_forward(x2)
    print_err('133', vsi_133_out, py_result)
    name = f'vsi_133_exp_{x_min}_{x_max}_N_{N}'
    vsi_133_pkg = (x, vsi_133_out, "vsi_133", py_result, 'torch', name)

    name = f'exp_{x_min}_{x_max}_N_{N}'
    plot_compare(plot_put_pkg, plot_pwl_pkg, vsi_142_pkg, vsi_133_pkg, name)

def eva_tanh_precision(x_min, x_max, N):
    print(f'输入[{x_min},{x_max}] \t 参数量 {N}')

    x = np.random.uniform(low=x_min, high=x_max, size=N).astype(np.float16)
    x = np.sort(x)

    py_result = F.tanh(torch.tensor(x).to(torch.float16)).numpy()
    
    # 现有lut， scale使用 2^n
    # lut = cxn1_lut_t(9, 7, input_table = None)
    # lut.create_tanh_table()
    # lut_result = lut.tanh_forward(x)
    # print_err("lut", lut_result, py_result)

    # name = f'lut_tanh_{x_min}_{x_max}_N_{N}'
    # plot_put_pkg = (x, lut_result, "lut", py_result, 'torch', name)
    # plot(x, lut_result, "lut", py_result, 'torch', name)

    # pwl 均匀分段
    pwl = pwl_test1_t(7)
    pwl.create_tanh_table(-8,8)
    pwl_out = pwl.forward(x)
    print_err("pwl", pwl_out, py_result)
    
    name = f'pwl7_tanh_{x_min}_{x_max}_N_{N}'
    # plot(x, pwl_out, "pwl", py_result, 'torch', name)
    plot_pwl_pkg = (x, pwl_out, "pwl", py_result, 'torch', name)

    # vsi pwl 143
    vsi_pwl_143 = vsi_pwl_t((1, 4, 3))
    vsi_pwl_143.create_tanh_table()
    vsi_out = vsi_pwl_143.forward(x)
    print_err("143", vsi_out, py_result)
    name = f'vsi_tanh_143{x_min}_{x_max}_N_{N}'
    plot_vsi_152_pkg = (x, vsi_out, "vsi", py_result, 'torch', name)
    # plot_compare(plot_vsi_pkg, plot_pwl_pkg, name)
    # plot(x, vsi_out, 'vsi', py_result, 'torch', name)

    # vsi pwl 133
    vsi_pwl_133 = vsi_pwl_t((1, 3, 3))
    vsi_pwl_133.create_tanh_table()
    vsi_out = vsi_pwl_133.forward(x)
    print_err("133", vsi_out, py_result)
    name = f'vsi_tanh_133{x_min}_{x_max}_N_{N}'
    plot_vsi_133_pkg = (x, vsi_out, "vsi", py_result, 'torch', name)

    # vsi 142
    vsi_pwl_142 = vsi_pwl_t((1, 4, 2))
    vsi_pwl_142.create_tanh_table()
    vsi_out = vsi_pwl_142.forward(x)
    print_err("142", vsi_out, py_result)
    name = f'vsi_tanh_142{x_min}_{x_max}_N_{N}'
    plot_vsi_142_pkg = (x, vsi_out, "vsi", py_result, 'torch', name)

    name = f'tanh_{x_min}_{x_max}_N_{N}'
    plot_compare(plot_pwl_pkg, plot_vsi_152_pkg, plot_vsi_133_pkg, plot_vsi_142_pkg, name)

    # vsi_pwl_256_2 = vsi_pwl_t((1,3,4))    # 512个表项

def plot(x, y1, name1, y2, name2, title):
    filename = title.replace("-","neg").replace(".","")
    
    plt.clf()
    plt.title(title)

    plt.plot(x, y1, label = name1)
    plt.plot(x, y2, label = name2)
    plt.legend()
    # plt.show()
    plt.savefig(filename)

def plot_compare(pic1_pkg, pic2_pkg, pic3_pkg, pic4_pkg, name):
    filename = name.replace("-","neg").replace(".","")
    plt.clf()
    plt.figure(figsize=(10, 10))
    
    x, y1, name1, y2, name2, title = pic1_pkg
    plt.subplot(2, 2, 1)
    plt.title(title)
    plt.plot(x, y1, label = name1)
    plt.plot(x, y2, label = name2)
    plt.legend()

    plt.subplot(2, 2, 2)
    x, y1, name1, y2, name2, title = pic2_pkg
    plt.title(title)
    plt.plot(x, y1, label = name1)
    plt.plot(x, y2, label = name2)
    plt.legend()
    
    if pic3_pkg is not None:
        plt.subplot(2, 2, 3)
        x, y1, name1, y2, name2, title = pic3_pkg
        plt.title(title)
        plt.plot(x, y1, label = name1)
        plt.plot(x, y2, label = name2)
        plt.legend()

    if pic4_pkg is not None:
        plt.subplot(2, 2, 4)
        x, y1, name1, y2, name2, title = pic4_pkg
        plt.title(title)
        plt.plot(x, y1, label = name1)
        plt.plot(x, y2, label = name2)
        plt.legend()

    # plt.show()
    plt.savefig(filename)

def eva_silu_precision(x_min, x_max, N):
    print(f'输入[{x_min},{x_max}] \t 参数量 {N}')

    x = np.random.uniform(low=x_min, high=x_max, size=N).astype(np.float16)
    x = np.sort(x)

    py_result = F.silu(torch.tensor(x).to(torch.float16)).numpy()
    
    # 现有lut， scale使用 2^n
    # lut = cxn1_lut_t(9, 7, input_table = None)
    # lut.create_silu_table()
    # lut_result = lut.silu_forward(x)
    # print_err("lut", lut_result, py_result)

    # name = f'lut_silu_{x_min}_{x_max}_N_{N}'
    # plot_put_pkg = (x, lut_result, "lut", py_result, 'torch', name)
    # plot(x, lut_result, "lut", py_result, 'torch', name)

    # pwl 均匀分段
    pwl = pwl_test1_t(7)
    pwl.create_silu_table()
    pwl_out = pwl.forward(x)
    print_err("pwl", pwl_out, py_result)
    
    name = f'pwl7_silu_{x_min}_{x_max}_N_{N}'
    # plot(x, pwl_out, "pwl", py_result, 'torch', name)
    plot_pwl_pkg = (x, pwl_out, "pwl", py_result, 'torch', name)

    # vsi pwl 143
    vsi_pwl_143 = vsi_pwl_t((1, 4, 3))
    vsi_pwl_143.create_silu_table()
    vsi_out = vsi_pwl_143.forward(x)
    print_err("143", vsi_out, py_result)
    name = f'vsi_silu_143{x_min}_{x_max}_N_{N}'
    plot_vsi_152_pkg = (x, vsi_out, "vsi", py_result, 'torch', name)
    # plot_compare(plot_vsi_pkg, plot_pwl_pkg, name)
    # plot(x, vsi_out, 'vsi', py_result, 'torch', name)

    # vsi pwl 133
    vsi_pwl_133 = vsi_pwl_t((1, 3, 3))
    vsi_pwl_133.create_silu_table()
    vsi_out = vsi_pwl_133.forward(x)
    print_err("133", vsi_out, py_result)
    name = f'vsi_silu_133{x_min}_{x_max}_N_{N}'
    plot_vsi_133_pkg = (x, vsi_out, "vsi", py_result, 'torch', name)

    # vsi 142
    vsi_pwl_142 = vsi_pwl_t((1, 4, 2))
    vsi_pwl_142.create_silu_table()
    vsi_out = vsi_pwl_142.forward(x)
    print_err("142", vsi_out, py_result)
    name = f'vsi_silu_142{x_min}_{x_max}_N_{N}'
    plot_vsi_142_pkg = (x, vsi_out, "vsi", py_result, 'torch', name)

    name = f'silu_{x_min}_{x_max}_N_{N}'
    plot_compare(plot_pwl_pkg, plot_vsi_152_pkg, plot_vsi_133_pkg, plot_vsi_142_pkg, name)

    # vsi_pwl_256_2 = vsi_pwl_t((1,3,4))    # 512个表项

def main():
    # print("exp精度评估 : ")
    # eva_exp_precision(-8, 8, 100000)
    # eva_exp_precision(-8, 0, 100000)
    # eva_exp_precision(-8, -1, 100000)

    # print("silu精度评估 : ")
    # eva_silu_precision(-6, 6, 100000)
    # eva_silu_precision(-2, 0.2, 100000)
    # eva_silu_precision(-60000, 60000, 100000)

    # print("tanh精度评估 : ")
    # eva_tanh_precision(-6, 6, 100000)
    # eva_tanh_precision(-2, 2, 100000)
    # eva_tanh_precision(-0.1, 0.1, 100000)
    # eva_tanh_precision(-60000, 60000, 100000)

    x_min = -60000
    x_max = 60000
    N = 100000
    x = np.random.uniform(low=x_min, high=x_max, size=N).astype(np.float16)
    x = np.sort(x)
    
    # python
    py_result = F.silu(torch.tensor(x).to(torch.float16)).numpy()
    
    # pwl 均匀分段
    pwl = pwl_test1_t(7)
    pwl.create_silu_table()
    pwl_out = pwl.forward(x)
    print_err("pwl    ", pwl_out, py_result)
    
    
    # vsi 142
    vsi_pwl_142 = vsi_pwl_t((1, 4, 2))
    vsi_pwl_142.create_silu_table()
    vsi_out = vsi_pwl_142.forward(x)
    print_err("vsi_142", vsi_out, py_result)

if __name__ == "__main__":
    main()

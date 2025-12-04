import numpy as np
import torch
import torch.nn.functional as F
# import sys
# import os
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from .lut_xuesong import pwl_test1_t, vsi_pwl_t, print_err
from .lut_kb_torch import lut_silu_kb
from .vsi_lut_t4_silu_fp16 import *


def main():
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

    # vsi kb
    vsi_kb = lut_silu_kb(5, 2, 2)
    vsi_kb_res = vsi_kb.forward(torch.tensor(x))
    print_err("vsi kb", vsi_kb_res, py_result)

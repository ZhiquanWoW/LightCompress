import torch
from torch import Tensor, nn
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def silu_torch(x):
    """
    Silu函数实现 - PyTorch版本
    """
    x = torch.tensor(x, dtype=torch.float16,device=DEVICE)
    return x * torch.sigmoid(x)


def extract_fp16_components(fp16_tensor):
    """
    从float16张量中提取符号、指数和尾数部分
    使用数学方法而不是位操作，避免CUDA uint16限制
    """

    
    # 转换为float32以便更精确地处理
    fp32_tensor = fp16_tensor.to(torch.float32)
    
    # 提取符号位
    sign = (fp32_tensor < 0).to(torch.int64)
    
    # 处理零和特殊情况
    abs_tensor = torch.abs(fp32_tensor)
    is_zero = abs_tensor == 0
    is_inf = torch.isinf(fp32_tensor)
    is_nan = torch.isnan(fp32_tensor)
    
    # 对于非零、非无穷大、非NaN的值，提取指数和尾数
    normal_mask = ~(is_zero | is_inf | is_nan)
    
    # 初始化指数和尾数
    exp_unbiased = torch.zeros_like(fp32_tensor, dtype=torch.int64)
    mant = torch.zeros_like(fp32_tensor, dtype=torch.int64)
    
    if torch.any(normal_mask):
        normal_values = abs_tensor[normal_mask]
        
        # 计算指数（无偏）
        # 使用对数方法提取指数
        exp_unbiased_normal = torch.floor(torch.log2(normal_values)).to(torch.int64)
        
        # 限制指数范围（根据float16规范）
        exp_unbiased_normal = torch.clamp(exp_unbiased_normal, -14, 15)
        
        # 计算尾数
        # 尾数 = (值 / 2^指数) - 1（对于规范化数）
        mantissa_normal = (normal_values / (2.0 ** exp_unbiased_normal)) - 1.0
        
        # 将尾数转换为10位整数表示
        mant_normal = torch.round(mantissa_normal * 1024).to(torch.int64)
        mant_normal = torch.clamp(mant_normal, 0, 1023)
        
        # 将结果放回原始张量
        exp_unbiased[normal_mask] = exp_unbiased_normal
        mant[normal_mask] = mant_normal
    
    # 处理零值
    if torch.any(is_zero):
        # 对于零，指数和尾数都是0
        exp_unbiased[is_zero] = -15  # float16的零指数
        mant[is_zero] = 0
    
    # 处理无穷大和NaN
    if torch.any(is_inf | is_nan):
        # 对于无穷大和NaN，使用最大指数
        exp_unbiased[is_inf | is_nan] = 16  # 超出正常范围
        mant[is_inf | is_nan] = 0
    
    return sign, exp_unbiased, mant


class lut_silu_kb(nn.Module):
    def __init__(self, exp_bit, mant_bit, exp_offset):
        super().__init__()
        self.levels = 2 ** exp_bit
        self.entries = 2 ** mant_bit
        self.mant_bit = mant_bit
        self.offset = exp_offset
        self.k_table, self.b_table = self.gen_lut_silu_tables_kb()

    def gen_lut_silu_tables_kb(self):
        lut = torch.zeros((2,self.levels, self.entries), dtype=torch.float16,device=DEVICE)
        lut_x = torch.zeros((2,self.levels, self.entries), dtype=torch.float16,device=DEVICE)
        lut_k = torch.zeros((2,self.levels, self.entries), dtype=torch.float16,device=DEVICE)
        lut_b = torch.zeros((2,self.levels, self.entries), dtype=torch.float16,device=DEVICE)

        for sign in [0,1]:
            for level in range(self.levels):
                base = 2.0 ** (level-self.offset) # 2, 4, 7
                for i in range(self.entries):
                    # mantissa[9:4] -> i，表示在该数量级内采样
                    frac = i / self.entries  # 0, 1/64, 2/64, ...
                    x_val =(1-2*sign)* base * (1.0 + frac)   # e.g. level=4 -> [2,4)，即table3           
                    # 计算Silu函数值
                    silu_val = silu_torch(x_val)
                    lut[sign,level, i] = silu_val.to(torch.float16)
                    lut_x[sign,level, i] = torch.tensor(x_val).to(torch.float16)

        for sign in [0, 1]:
            for level in range(self.levels):
                for i in range(self.entries-1):
                    lut_k[sign,level, i] = (lut[sign,level, i+1] - lut[sign,level, i]) / (lut_x[sign,level, i+1] - lut_x[sign,level, i])
                    lut_b[sign,level, i] = lut[sign,level, i] - lut_x[sign,level, i] * lut_k[sign,level, i]
        for sign in [0, 1]:
            for level in range(self.levels-1):
                # 跨level的处理
                lut_k[sign,level, -1] = (lut[sign,level+1, 0] - lut[sign,level, -1]) / (lut_x[sign,level+1, 0] - lut_x[sign,level, -1])
                lut_b[sign,level, -1] = lut[sign,level, -1] - lut_x[sign,level, -1] * lut_k[sign,level, -1]

        # 边界情况处理
        r_max = torch.finfo(torch.float16).max
        l_min = -torch.finfo(torch.float16).max
        r_edge = silu_torch(r_max).to(torch.float16)
        l_edge = silu_torch(l_min).to(torch.float16)
        lut_k[0,-1,-1] = (r_edge - lut[0,-1,-1]) / (r_max - lut_x[0, -1, -1])
        lut_b[0,-1,-1] = r_edge - r_max * lut_k[0, -1, -1]
        lut_k[1,-1,-1] = (l_edge - lut[1,-1,-1]) / (l_min - lut_x[1, -1, -1])
        lut_b[1,-1,-1] = l_edge - l_min * lut_k[1, -1, -1]

        
        return lut_k, lut_b

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        使用 LUT 近似 Silu(x) = x * sigmoid(x) - PyTorch 版本
        返回: torch.float16 近似值
        """
        
        original_shape = input.shape
        # 将输入转换为一维张量处理
        points_flat = input.reshape(-1)
        n_points = len(points_flat)
        
        # 将输入转换为float16并限制范围
        x_float16_ = points_flat.to(torch.float16)
        x_float16 = torch.clamp(x_float16_, -2**(self.levels-self.offset), 2**(self.levels-self.offset))
        
        # 初始化结果张量
        results = torch.zeros(n_points, device=DEVICE, dtype=torch.float16)  # 先用float32计算，最后转float16
        
        # 处理NaN值
        nan_mask = torch.isnan(x_float16_)
        valid_mask = ~nan_mask
        
        if torch.any(nan_mask):
            results[nan_mask] = torch.nan
        
        if not torch.any(valid_mask):
            return results.to(torch.float16).reshape(original_shape)
        
        # 提取有效点
        valid_points = x_float16[valid_mask]
        valid_points_orig = x_float16_[valid_mask]
        
        sign, exp_unbiased, mant = extract_fp16_components(valid_points)
        
        # 调整指数索引以适应LUT表
        lut_exp_idx = exp_unbiased + self.offset
        mant_mask = (1 << self.mant_bit) - 1
        # mant_idx = (mant >> 3) & 0x7F # 128
        mant_idx = (mant >> (10 - self.mant_bit)) & mant_mask # 32

        
        # 限制索引在有效范围内
        lut_exp_idx = torch.clamp(lut_exp_idx, 0, self.k_table.shape[1] - 1)
        mant_idx = torch.clamp(mant_idx, 0, self.k_table.shape[2] - 1)

        k = self.k_table[sign, lut_exp_idx, mant_idx]
        b = self.b_table[sign, lut_exp_idx, mant_idx]
        results = k * x_float16 + b
        return results.reshape(original_shape).to(input.dtype)

if __name__ == "__main__":
    x = torch.rand([10000], dtype=torch.float16, device="cuda") * 1000
    golden = silu_torch(x)
    lut_silu_kb0 = lut_silu_kb(5, 2, 2)
    result = lut_silu_kb0.forward(x)

    mae = torch.mean(torch.abs(result - golden))
    mse = torch.mean((result - golden) ** 2)
    print("item num: ", lut_silu_kb0.k_table.flatten().size())
    print("mae: ", mae)
    print("mse: ", mse)
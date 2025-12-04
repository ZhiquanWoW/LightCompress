import numpy as np
import struct
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

levels = 4
entries = 32
acc = 2

def float16_to_uint16(x_f16):
    """将float16转换为uint16的比特表示"""
    return struct.unpack('<H', struct.pack('<e', x_f16))[0]

def uint16_to_float16(x_u16):
    """将uint16的比特表示转换回float16"""
    return struct.unpack('<e', struct.pack('<H', x_u16))[0]

def approx_using_lut(test_points, lut):
    """
    使用 LUT 近似 Silu(x) = x * sigmoid(x)
    返回: np.float16 近似值（向量化版本）
    """
    original_shape = test_points.shape
    # 将输入转换为一维数组处理
    points_flat = test_points.reshape(-1)
    n_points = len(points_flat)
    
    # 将输入转换为float16并限制范围
    x_float16_ = points_flat.astype(np.float16)
    x_float16 = np.clip(x_float16_, -4, 4)
    
    # 预定义斜率（提前计算好）
    # slop_upper = (silu(10) - silu(3.9844)) / 6.0
    slop_upper = 1
    # slop_lower = (silu(-10) - silu(-3.9844)) / 6.0
    slop_lower = 0
    slop_small = (silu(0.25) - silu(-0.25)) / 0.5
    
    # 初始化结果数组
    results = np.zeros(n_points, dtype=np.float32)  # 先用float32计算，最后转float16
    
    # 处理NaN值
    nan_mask = np.isnan(x_float16_)
    valid_mask = ~nan_mask
    
    if np.any(nan_mask):
        results[nan_mask] = np.nan

    if not np.any(valid_mask):
        return results.astype(np.float16).reshape(original_shape)

    # 提取有效点
    valid_points = x_float16[valid_mask]
    valid_points_orig = x_float16_[valid_mask]
    
    # 使用struct转换而不是.view()来避免0维数组问题
    bits = float16_to_uint16_vectorized(valid_points)
    
    # 分解出 sign exp mant
    sign = (bits >> 15) & 0x1
    exp = (bits >> 10) & 0x1F
    mant = bits & 0x3FF

    exp_unbiased = exp - 15  # 无偏指数
    
    # 调整指数索引以适应LUT表
    lut_exp_idx = exp_unbiased + 2
    # mantissa index [9:3]，使用7位索引（128个条目）
    mant_idx = (mant >> 3) & 0x7F
    
    # 限制索引在有效范围内
    lut_exp_idx = np.clip(lut_exp_idx, 0, lut.shape[1] - 1)
    mant_idx = np.clip(mant_idx, 0, lut.shape[2] - 1)
    
    # 创建各种情况的掩码（相对于valid_points的掩码）
    upper_mask = valid_points_orig >= 3.9844
    lower_mask = valid_points_orig <= -3.9844
    small_mask = (valid_points_orig < 0.25) & (valid_points_orig > -0.25)
    normal_mask = ~(upper_mask | lower_mask | small_mask)
    
    # 创建临时数组存储有效点的结果
    valid_results = np.zeros(len(valid_points), dtype=np.float32)
    
    # 处理上边界情况
    if np.any(upper_mask):
        y0 = lut[0, -1, -1]  # 正数最后一个值
        valid_results[upper_mask] = y0 + slop_upper * (valid_points_orig[upper_mask] - 4)
    
    # 处理下边界情况
    if np.any(lower_mask):
        y0 = lut[1, -1, -1]  # 负数最后一个值
        valid_results[lower_mask] = y0 + slop_lower * (-3.9844 - valid_points_orig[lower_mask])
    
    # 处理小值情况
    if np.any(small_mask):
        y0 = lut[1, 0, 0]
        valid_results[small_mask] = y0 + slop_small * (valid_points_orig[small_mask] - (-0.25))
    
    # 处理正常插值情况
    if np.any(normal_mask):
        # 提取正常点的相关数据
        normal_sign = sign[normal_mask]
        normal_exp_idx = lut_exp_idx[normal_mask]
        normal_mant_idx = mant_idx[normal_mask]
        normal_mant = mant[normal_mask]
        
        # 检查是否需要跨指数边界插值
        cross_exp_mask = normal_mant_idx >= (lut.shape[2] - 1)
        
        # 处理跨指数边界的情况
        if np.any(cross_exp_mask):
            cross_sign = normal_sign[cross_exp_mask]
            cross_exp_idx = normal_exp_idx[cross_exp_mask]
            
            # 确保指数索引不越界
            cross_exp_idx = np.clip(cross_exp_idx, 0, lut.shape[1] - 2)
            
            # 向量化查找y0和y1
            y0 = np.array([lut[s, e, lut.shape[2] - 1] for s, e in zip(cross_sign, cross_exp_idx)])
            y1 = np.array([lut[s, e + 1, 0] for s, e in zip(cross_sign, cross_exp_idx)])
            delta_y = y1 - y0
            
            # mantissa delta
            mant_delta = (normal_mant[cross_exp_mask] & 0x7) * (2 ** -3)
            
            # 创建跨指数边界点的掩码在normal_mask中的位置
            cross_in_normal = np.where(normal_mask)[0][cross_exp_mask]
            valid_results[cross_in_normal] = y0 + delta_y * mant_delta
        
        # 处理普通插值情况
        if np.any(~cross_exp_mask):
            regular_mask_in_normal = ~cross_exp_mask
            regular_sign = normal_sign[regular_mask_in_normal]
            regular_exp_idx = normal_exp_idx[regular_mask_in_normal]
            regular_mant_idx = normal_mant_idx[regular_mask_in_normal]
            regular_mant = normal_mant[regular_mask_in_normal]
            
            # 向量化查找y0和y1
            y0 = np.array([lut[s, e, m] for s, e, m in zip(regular_sign, regular_exp_idx, regular_mant_idx)])
            y1 = np.array([lut[s, e, m + 1] for s, e, m in zip(regular_sign, regular_exp_idx, regular_mant_idx)])
            delta_y = y1 - y0
            
            # mantissa delta
            mant_delta = (regular_mant & 0x7) * (2 ** -3)
            
            # 创建普通插值点在normal_mask中的位置
            regular_in_normal = np.where(normal_mask)[0][regular_mask_in_normal]
            valid_results[regular_in_normal] = y0 + delta_y * mant_delta
    
    # 将有效点结果复制到最终结果数组
    results[valid_mask] = valid_results
    
    return results.astype(np.float16).reshape(original_shape)


def float16_to_uint16_vectorized(arr):
    """
    将float16数组转换为uint16表示的向量化版本
    """
    # 这里需要根据你的具体实现来定义
    # 假设你已经有一个向量化的版本
    return arr.view(np.uint16)


def sigmoid(x):
    """计算sigmoid函数"""
    return 1 / (1 + np.exp(-x))

def silu(x):
    """计算Silu函数: x * sigmoid(x)"""
    return x * sigmoid(x)


def gen_lut_silu_tables():
    """
    生成Silu函数的LUT表
    覆盖范围: [-4, 4]
    8个级别，每个级别128个条目（提供更好的精度）
    """

    lut = np.zeros((2,levels, entries), dtype=np.float16)


    for sign in [0,1]:
        for level in range(levels):
            base = 2.0 ** (level-acc)
            for i in range(entries):
                # mantissa[9:4] -> i，表示在该数量级内采样
                frac = i / entries  # 0, 1/64, 2/64, ...
                x_val =(1-2*sign)* base * (1.0 + frac)   # e.g. level=4 -> [2,4)，即table3           
                # 计算Silu函数值
                silu_val = silu(x_val)
                lut[sign,level, i] = np.float16(silu_val)
    return lut


def approx_using_lut_torch(test_points, lut_tensor):
    """
    使用 LUT 近似 Silu(x) = x * sigmoid(x) - PyTorch 版本
    返回: torch.float16 近似值
    """
    
    original_shape = test_points.shape
    # 将输入转换为一维张量处理
    points_flat = test_points.reshape(-1)
    n_points = len(points_flat)
    
    # 将输入转换为float16并限制范围
    x_float16_ = points_flat.to(torch.float16)
    x_float16 = torch.clamp(x_float16_, -2**(levels-acc), 2**(levels-acc))
    
    # 预定义斜率（提前计算好）
    # slop_upper = (silu_torch(torch.tensor(10.0, device=DEVICE)) - silu_torch(torch.tensor(3.9844, device=DEVICE))) / 6.0
    slop_upper = 1
    # slop_lower = (silu_torch(torch.tensor(-10.0, device=DEVICE)) - silu_torch(torch.tensor(-3.9844, device=DEVICE))) / 6.0
    slop_lower = 0
    slop_small = (silu_torch(torch.tensor(2**(-acc), device=DEVICE)) - silu_torch(torch.tensor(-2**(-acc), device=DEVICE))) / 2**(-acc+1)
    
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
    lut_exp_idx = exp_unbiased + acc
    # mantissa index [9:3]，使用7位索引（128个条目）
    # mant_idx = (mant >> 3) & 0x7F # 128
    mant_idx = (mant >> 5) & 0x1F # 32

    
    # 限制索引在有效范围内
    lut_exp_idx = torch.clamp(lut_exp_idx, 0, lut_tensor.shape[1] - 1)
    mant_idx = torch.clamp(mant_idx, 0, lut_tensor.shape[2] - 1)
    
    # 创建各种情况的掩码（相对于valid_points的掩码）
    upper_mask = valid_points_orig >= 2**(levels-2)
    lower_mask = valid_points_orig <= -2**(levels-2)
    small_mask = (valid_points_orig < 2**(-acc)) & (valid_points_orig > -2**(-acc))
    normal_mask = ~(upper_mask | lower_mask | small_mask)
    
    # 创建临时张量存储有效点的结果
    valid_results = torch.zeros(len(valid_points), device=DEVICE, dtype=torch.float16)
    
    # 处理上边界情况
    if torch.any(upper_mask):
        y0 = lut_tensor[0, -1, -1]  # 正数最后一个值
        valid_results[upper_mask] = y0 + slop_upper * (valid_points_orig[upper_mask] - 2**(levels-2))
    
    # 处理下边界情况
    if torch.any(lower_mask):
        y0 = lut_tensor[1, -1, -1]  # 负数最后一个值
        valid_results[lower_mask] = y0 + slop_lower * (-2**(levels-2) - valid_points_orig[lower_mask])
    
    # 处理小值情况
    if torch.any(small_mask):
        y0 = lut_tensor[1, 0, 0]
        valid_results[small_mask] = y0 + slop_small * (valid_points_orig[small_mask] - (-2**(-acc)))
    
    # 处理正常插值情况
    if torch.any(normal_mask):
        # 提取正常点的相关数据
        normal_sign = sign[normal_mask]
        normal_exp_idx = lut_exp_idx[normal_mask]
        normal_mant_idx = mant_idx[normal_mask]
        normal_mant = mant[normal_mask]
        
        # 检查是否需要跨指数边界插值
        cross_exp_mask = normal_mant_idx >= (lut_tensor.shape[2] - 1)
        
        # 处理跨指数边界的情况
        if torch.any(cross_exp_mask):
            cross_sign = normal_sign[cross_exp_mask]
            cross_exp_idx = normal_exp_idx[cross_exp_mask]
            
            # 确保指数索引不越界
            cross_exp_idx = torch.clamp(cross_exp_idx, 0, lut_tensor.shape[1] - 2)
            
            # 向量化查找y0和y1
            y0 = lut_tensor[cross_sign, cross_exp_idx, lut_tensor.shape[2] - 1]
            y1 = lut_tensor[cross_sign, cross_exp_idx + 1, 0]
            delta_y = y1 - y0
            
            # mantissa delta
            mant_delta = (normal_mant[cross_exp_mask] & 0x7) * (2 ** -3)
            
            # 创建跨指数边界点的掩码在normal_mask中的位置
            cross_in_normal = torch.nonzero(normal_mask, as_tuple=True)[0][cross_exp_mask]
            valid_results[cross_in_normal] = (y0 + delta_y * mant_delta).to(torch.float16)
        
        # 处理普通插值情况
        if torch.any(~cross_exp_mask):
            regular_mask_in_normal = ~cross_exp_mask
            regular_sign = normal_sign[regular_mask_in_normal]
            regular_exp_idx = normal_exp_idx[regular_mask_in_normal]
            regular_mant_idx = normal_mant_idx[regular_mask_in_normal]
            regular_mant = normal_mant[regular_mask_in_normal]
            
            # 向量化查找y0和y1
            y0 = lut_tensor[regular_sign, regular_exp_idx, regular_mant_idx]
            y1 = lut_tensor[regular_sign, regular_exp_idx, regular_mant_idx + 1]
            delta_y = y1 - y0
            
            # mantissa delta
            # mant_delta = (regular_mant & 0x7) * (2 ** -3) # 128
            mant_delta = (regular_mant & 0x1F) * (2 ** -5) # 32
            
            # 创建普通插值点在normal_mask中的位置
            regular_in_normal = torch.nonzero(normal_mask, as_tuple=True)[0][regular_mask_in_normal]
            valid_results[regular_in_normal] = (y0 + delta_y * mant_delta).to(torch.float16)
    
    # 将有效点结果复制到最终结果张量
    results[valid_mask] = valid_results
    
    return results.reshape(original_shape)

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

def silu_torch(x):
    """
    Silu函数实现 - PyTorch版本
    """
    x = torch.tensor(x, dtype=torch.float16,device=DEVICE)
    return x * torch.sigmoid(x)
def gen_lut_silu_tables_torch():
    """
    生成Silu函数的LUT表
    覆盖范围: [-4, 4]
    8个级别，每个级别128个条目（提供更好的精度）
    """
    # levels = 4  # 增加级别数以提高精度 exp bits 16
    # entries = 128 # mant bit 

    lut = torch.zeros((2,levels, entries), dtype=torch.float16,device=DEVICE)

    for sign in [0,1]:
        for level in range(levels):
            base = 2.0 ** (level-acc) # 2, 4, 7
            for i in range(entries):
                # mantissa[9:4] -> i，表示在该数量级内采样
                frac = i / entries  # 0, 1/64, 2/64, ...
                x_val =(1-2*sign)* base * (1.0 + frac)   # e.g. level=4 -> [2,4)，即table3           
                # 计算Silu函数值
                silu_val = silu_torch(x_val)
                lut[sign,level, i] = silu_val.to(torch.float16)
                 

    return lut

def gen_lut_silu_tables_torch_kb():
    lut = torch.zeros((2,levels, entries), dtype=torch.float16,device=DEVICE)
    lut_x = torch.zeros((2,levels, entries), dtype=torch.float16,device=DEVICE)
    lut_k = torch.zeros((2,levels, entries), dtype=torch.float16,device=DEVICE)
    lut_b = torch.zeros((2,levels, entries), dtype=torch.float16,device=DEVICE)

    for sign in [0,1]:
        for level in range(levels):
            base = 2.0 ** (level-acc) # 2, 4, 7
            for i in range(entries):
                # mantissa[9:4] -> i，表示在该数量级内采样
                frac = i / entries  # 0, 1/64, 2/64, ...
                x_val =(1-2*sign)* base * (1.0 + frac)   # e.g. level=4 -> [2,4)，即table3           
                # 计算Silu函数值
                silu_val = silu_torch(x_val)
                lut[sign,level, i] = silu_val.to(torch.float16)
                lut_x[sign,level, i] = torch.tensor(x_val).to(torch.float16)

    for sign in [0, 1]:
        for level in range(levels):
            for i in range(entries-1):
                lut_k[sign,level, i] = (lut[sign,level, i+1] - lut[sign,level, i]) / (lut_x[sign,level, i+1] - lut_x[sign,level, i])
                lut_b[sign,level, i] = lut[sign,level, i] - lut_x[sign,level, i] * lut_k[sign,level, i]
    for sign in [0, 1]:
        for level in range(levels-1):
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

def lut_silu_torch_kb(test_points, k_table, b_table):
    """
    使用 LUT 近似 Silu(x) = x * sigmoid(x) - PyTorch 版本
    返回: torch.float16 近似值
    """
    
    original_shape = test_points.shape
    # 将输入转换为一维张量处理
    points_flat = test_points.reshape(-1)
    n_points = len(points_flat)
    
    # 将输入转换为float16并限制范围
    x_float16_ = points_flat.to(torch.float16)
    x_float16 = torch.clamp(x_float16_, -2**(levels-acc), 2**(levels-acc))
    
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
    lut_exp_idx = exp_unbiased + acc
    # mantissa index [9:3]，使用7位索引（128个条目）
    # mant_idx = (mant >> 3) & 0x7F # 128
    mant_idx = (mant >> 5) & 0x1F # 32

    
    # 限制索引在有效范围内
    lut_exp_idx = torch.clamp(lut_exp_idx, 0, k_table.shape[1] - 1)
    mant_idx = torch.clamp(mant_idx, 0, k_table.shape[2] - 1)

    k = k_table[sign, lut_exp_idx, mant_idx]
    b = b_table[sign, lut_exp_idx, mant_idx]
    results = k * x_float16 + b
    return results.reshape(original_shape)

if __name__ == "__main__":
    x = torch.rand([1024], dtype=torch.float16, device="cuda")
    golden = silu_torch(x)
    lut_k, lut_b = gen_lut_silu_tables_torch_kb()
    result = lut_silu_torch_kb(x, k_table=lut_k, b_table=lut_b)

    print(lut_k)
    print(lut_b)

    mae = torch.mean(torch.abs(result - golden))
    mse = torch.mean((result - golden) ** 2)
    print("mae: ", mae)
    print("mse: ", mse)
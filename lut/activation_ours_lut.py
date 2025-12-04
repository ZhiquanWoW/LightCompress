import torch
import numpy as np
import math

def nearest_power_of_two(x):
    if x == 0:
        return 1.0
    return 2 ** math.ceil(math.log2(x))

class LUTTensor:
    def __init__(
        self, func_name, lut_bit=9, mirror=False, value_offset=0, device="cuda"
    ):
        self.min = 0
        self.max = 1.0
        self.use_dynamic_lut = False
        self.lut_items_in_bits = lut_bit
        self.alpha = 1
        self.device = device
        self.func_name = func_name
        # ---------------------
        self.lut_items_in_bits = 9
        self.q_mode_activation = "per_tensor_symmetric_full_range"
        self.out_sign = False
        self.value_offset = value_offset

        # fixed type
        self.fixd_type = False
        self.cfg_bias = None
        self.cfg_shift = None

        self.func = self._build_func()
        self._build_lut()

        self.mirror = mirror

    def _build_func(self):
        if self.func_name == "silu" and self.lut_items_in_bits == 9:
            return self._sigmoid_func

    def _sigmoid_func(self, x):
        alpha = self.alpha
        func = torch.nn.functional.sigmoid(x * alpha)
        return func

    def _build_lut(self):
        if self.func_name == "exp" or self.func_name == "softmax":
            self._build_pow2_lut()

        elif self.func_name == "silu" or self.func_name == "sigmoid":
            self._build_sigmoid_lut()

    def _build_sigmoid_lut(self):
        min_x = 0
        max_x = 8

        lut_index = torch.linspace(min_x, max_x, 2**self.lut_items_in_bits + 1).to(
            self.device
        )

        value_offset = torch.tensor(
            self.value_offset, dtype=torch.float16, device=self.device
        )

        func = self.func
        if func is not None:
            lut = func(lut_index) + value_offset
        else:
            NotImplementedError(
                f"{self.func_name} Activation op method=UNKNOWN does not support float forward, and the output.tensor will directly use input.tensor"
            )

        lut = lut.to(torch.float16)

        index_scale = 65535 / (max_x - min_x)
        M = torch.tensor(nearest_power_of_two(index_scale))

        index_zp = min_x
        index_scale = torch.tensor(index_scale, dtype=torch.float32).to(self.device)
        index_zp = torch.tensor(index_zp, dtype=torch.float32).to(self.device)
        index_zp = (-index_scale * index_zp).to(torch.int32)

        left_slope = (lut[1] - lut[0]) / index_scale
        right_slope = (lut[-1] - lut[-2]) / index_scale

        lut_data = lut
        # lut_data = torch.nn.functional.pad(lut, (0, 1), value=lut[-1])

        # lut dtype float16
        setattr(self, "lut", lut_data.to(torch.float16))
        # index_scale_value float32
        # setattr(self, "index_scale_value", index_scale.to(torch.float32))
        setattr(self, "index_scale_value_M", M.to(torch.int32))
        # setattr(self, "index_scale_value_N", N.to(torch.int32))
        # index_offset_value int32
        setattr(self, "index_offset_value", index_zp.to(torch.int32))
        # left slope FP32
        setattr(self, "left_slope", left_slope.to(torch.float32))
        # right slope FP32
        setattr(self, "right_slope", right_slope.to(torch.float32))
        # FP16
        setattr(self, "value_offset_value", value_offset.to(torch.float16))

    def _build_pow2_lut(self):
        min_x = 0
        max_x = 1
        lut_index = torch.linspace(min_x, max_x, 2**self.lut_items_in_bits + 1)
        lut_data = (2**lut_index).to(torch.float16).to(device=self.device)
        index_scale = 65535 / (max_x - min_x)
        M = torch.tensor(nearest_power_of_two(index_scale))
        index_zp = min_x

        index_scale = torch.tensor(index_scale, dtype=torch.float32).to(self.device)
        index_zp = torch.tensor(index_zp, dtype=torch.float32).to(self.device)
        # setattr(self, "lut_index", lut_index.to(data_type))
        index_zp = (-index_scale * index_zp).to(torch.int32)

        left_slope = (lut_data[1] - lut_data[0]) / index_scale
        right_slope = (lut_data[-1] - lut_data[-2]) / index_scale

        # lut dtype float16
        setattr(self, "lut", lut_data.to(torch.float16))
        # index_scale_value float32
        # setattr(self, "index_scale_value", index_scale.to(torch.float32))
        setattr(self, "index_scale_value_M", M.to(torch.int32))
        # setattr(self, "index_scale_value_N", N.to(torch.int32))
        # index_offset_value int32
        setattr(self, "index_offset_value", index_zp.to(torch.int32))
        # left slope
        setattr(self, "left_slope", left_slope.to(torch.float32))
        # right slope
        setattr(self, "right_slope", right_slope.to(torch.float32))

    def _pre_process(self):
        pass

    def _post_process(self):
        pass

    def _lut_data(self, x):
        if self.fixd_type:
            NotImplementedError()
        else:

            def _lut(x_t):

                # x_t = x.floor().to(torch.int32)

                table_size = self.lut.numel()

                # index = torch.clamp(x_t, 0, table_size - 1).floor().long()

                index = x_t >> 7
                interp_bit = x_t & 0x7F

                left_slope = self.left_slope
                right_slope = self.right_slope

                left_mask = x_t < 0
                right_mask = x_t > 65535  # table size的长度问题

                # left slope
                y_left = torch.where(
                    left_mask, self.lut[0] - left_slope * (0 - x_t), x_t
                )
                y_right = torch.where(
                    right_mask,
                    self.lut[-1] + right_slope * (x_t - 65535),
                    x_t,
                )

                index = torch.clamp(index, 0, table_size - 2).floor().to(torch.int32)
                diff = self.lut[index + 1] - self.lut[index]
                y_middle = self.lut[index] + (diff * interp_bit) / (2.0**7)

                y = torch.where(left_mask, y_left, y_middle)
                y = torch.where(right_mask, y_right, y)

                return y

            x = x.to(torch.float32)
            x_t = (x * self.index_scale_value_M).to(torch.int32)  + self.index_offset_value
            if self.mirror:
                x_t = x_t.to(torch.int32)
                x_t_1 = torch.clamp(x_t,0,65535)  # Uint16
                x_t_2 = torch.clamp(x_t.negative(),0,65535)  # Uint16
                y_pos = _lut(x_t_1)
                y_neg = _lut(x_t_2).negative()
                y = torch.where(x >= 0.0, y_pos, y_neg) - self.value_offset_value
            else:
                x_t = torch.clamp(x_t, 0, 65535)
                x_t = x_t.to(torch.int32) # Uint16
                y = _lut(x_t)
        return y


    def exp_approximated_float_forward_fp32_9bit(self, inp_tensor):
        inp_tensor = inp_tensor * 1.442695
        inp_tensor = inp_tensor.to(torch.float16)
        inp_tensor[inp_tensor.isnan()] = 0
        inp_tensor = torch.clamp(
            inp_tensor, torch.finfo(torch.float16).min, torch.finfo(torch.float16).max
        )

        # # limit input
        # inp_tensor = (inp_tensor < -126.9) * -126.9 + inp_tensor * (
        #     inp_tensor >= -126.9
        # )
        # inp_tensor = (inp_tensor > 126.9) * 126.9 + inp_tensor * (inp_tensor < 126.9)

        exponent = inp_tensor.floor()
        mantisa = inp_tensor - exponent

        # look up table
        out = self._lut_data(mantisa)

        pow2_factor = out * (2.0**exponent)
        return pow2_factor.to(torch.float16)

    def softmax_approximated_float_forward_fp32_9bit(self, inp_tensor, axis):
        inp_tensor[inp_tensor.isnan()] = 0
        inp_tensor = torch.clamp(
            inp_tensor, torch.finfo(torch.float16).min, torch.finfo(torch.float16).max
        )
        inp_tensor_max, _ = inp_tensor.max(axis, keepdim=True)
        inp_fp_tensor = inp_tensor - inp_tensor_max

        yy = self.exp_approximated_float_forward_fp32_9bit(inp_fp_tensor)
        y_sum = yy.sum(axis, keepdim=True)

        # convert to fp24
        score = 1 / y_sum
        # convert yy to fp16
        # yy16 = yy.half()
        score = yy * score
        # convert f32 softmax result to fp16 output
        # f16 = score.half()
        # return f16.reshape(inp_tensor.shape)
        return score.reshape(inp_tensor.shape)
    def silu_approximated_float_forward_fp32_9bit(self, inp_tensor):
        out = self._lut_data(inp_tensor).to(torch.float16)
        out = inp_tensor * out
        return out

    def __call__(self, x, dim=None):
        if self.func_name == "softmax":
            output = self.softmax_approximated_float_forward_fp32_9bit(x, axis=dim)
        elif self.func_name == "exp":
            output = self.exp_approximated_float_forward_fp32_9bit(x)
        elif self.func_name == "silu":
            output = self.silu_approximated_float_forward_fp32_9bit(x)

        return output


if __name__ == "__main__":
    np.random.seed(42)
    device = "cuda"
    lut_bit = 9
    random_input_numpy = np.random.uniform(
        low=-10, high=10, size=(1024 * 1024,)
    ).astype(np.float16)

    random_input_torch = (
        torch.from_numpy(random_input_numpy).to(torch.float16).to(device)
    )

    # ---------------- debug exp ------------------------------
    # func_name = "exp"
    
    # lut_func = LUTTensor(
    #     func_name=func_name,
    #     lut_bit=lut_bit,
    #     mirror=False,
    #     value_offset=0,
    #     device=device,
    # )
    
    # output_lut = lut_func(random_input_torch)
    # output_gt = torch.exp(random_input_torch)
    # diff = output_lut - output_gt
    # print(f"diff min : {diff.abs().min().item()}")
    # print(f"diff max : {diff.abs().max().item()}")

    # errors = torch.abs(output_lut - output_gt)

    # # Find the max error and corresponding index
    # max_error = torch.max(errors)
    # max_error_index = torch.argmax(errors)

    # # Get values at the max error index
    # input_value = random_input_torch[max_error_index]
    # lut_value = output_lut[max_error_index]
    # gt_value = output_gt[max_error_index]

    # print(f"Max Error: {max_error.item()}")
    # print(f"Index: {max_error_index.item()}")
    # print(f"Input at max error: {input_value.item()}")
    # print(f"LUT Output: {lut_value.item()}")
    # print(f"GT Output: {gt_value.item()}")

    # ---------------- debug silu ------------------------------
    func_name = "silu"

    lut_func = LUTTensor(
        func_name=func_name,
        lut_bit=lut_bit,
        mirror=True,
        value_offset=-0.5,
        device=device,
    )

    
    output_lut = lut_func(random_input_torch)
    output_approx = 2 ** (random_input_torch * 1.442695)
    output_gt = torch.nn.functional.silu(random_input_torch)
    diff = output_lut - output_gt
    print(f"diff approx : {(output_approx - output_lut).abs().max()}")
    print(f"diff min : {diff.abs().min().item()}")
    print(f"diff max : {diff.abs().max().item()}")

    errors = torch.abs(output_lut - output_gt)

    # Find the max error and corresponding index
    max_error = torch.max(errors)
    max_error_index = torch.argmax(errors)

    # Get values at the max error index
    input_value = random_input_torch[max_error_index]
    lut_value = output_lut[max_error_index]
    gt_value = output_gt[max_error_index]

    print(f"Max Error: {max_error.item()}")
    print(f"Index: {max_error_index.item()}")
    print(f"Input at max error: {input_value.item()}")
    print(f"LUT Output: {lut_value.item()}")
    print(f"GT Output: {gt_value.item()}")


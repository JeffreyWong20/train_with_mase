"""
Counter folder contains the code to count the number of FLOPs in the model.
"""

import torch


def l_prod(in_list):
    res = 1
    for _ in in_list:
        res *= _
    return res


def calculate_conv2d_flops(
    input_size: list, output_size: list, kernel_size: list, groups: int
):
    # T, N, out_c, oh, ow = output_size
    # T, N, in_c, ih, iw = input_size
    # out_c, in_c, kh, kw = kernel_size
    in_c = input_size[2]
    g = groups
    return l_prod(output_size) * (in_c // g) * l_prod(kernel_size[2:])


def count_convNd(m, x, y: torch.Tensor):
    x = x[0]

    kernel_ops = torch.zeros(m.weight.size()[2:]).numel()  # Kw x Kh
    bias_ops = 1 if m.bias is not None else 0

    m.total_ops += calculate_conv2d_flops(
        input_size=list(x.shape),
        output_size=list(y.shape),
        kernel_size=list(m.weight.shape),
        groups=m.groups,
    )


def count_matmul(m, x, y):
    left, right = x
    # per output element
    total_mul = right.shape[-1]
    # total_add = m.in_features - 1
    # total_add += 1 if m.bias is not None else 0
    num_elements = left.numel()

    m.total_ops += torch.DoubleTensor([int(total_mul * num_elements)])


# nn.Linear
def count_linear(m, x, y):
    # per output element
    total_mul = m.in_features
    # total_add = m.in_features - 1
    # total_add += 1 if m.bias is not None else 0
    num_elements = y.numel()

    m.total_ops += torch.DoubleTensor([int(total_mul * num_elements)])

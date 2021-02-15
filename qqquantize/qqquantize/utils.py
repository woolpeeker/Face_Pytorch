
import torch
import copy

def get_unique_devices_(module):
    return {p.device for p in module.parameters()} | \
        {p.device for p in module.buffers()}


"""fuse conv and bn. Linear also supported"""
def fuse_conv_bn(conv, bn):
    bn_st_dict = bn.state_dict()
    conv_st_dict = conv.state_dict()

    # BatchNorm params
    eps = bn.eps
    running_mean = bn_st_dict['running_mean']
    running_var = bn_st_dict['running_var']
    gamma = bn_st_dict['weight']
    if 'bias' in bn_st_dict:
        beta = bn_st_dict['bias']
    else:
        beta = torch.zeros(gamma.size(0)).float().to(gamma.device)

    weight = conv_st_dict['weight']
    if 'bias' in conv_st_dict:
        bias = conv_st_dict['bias']
    else:
        bias = torch.zeros(weight.shape[0]).float().to(gamma.device)

    var_sqrt = torch.sqrt(running_var + eps)
    weight = weight * (gamma / var_sqrt).reshape([weight.shape[0], 1, 1, 1])
    bias = (bias - running_mean) / var_sqrt * gamma + beta

    fused_conv = copy.deepcopy(conv)
    fused_conv.weight.data.copy_(weight)
    if fused_conv.bias is None:
        fused_conv.bias = torch.nn.Parameter(bias)
    else:
        fused_conv.bias.data.copy_(bias)
    return fused_conv

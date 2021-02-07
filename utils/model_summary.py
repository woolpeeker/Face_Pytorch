import ptflops
import torch

def calc_flops(model, inp_c, inp_h, inp_w):
    with torch.no_grad():
        flops_count, params_count = ptflops.get_model_complexity_info(
                model, input_res=(inp_c, inp_h, inp_w),
                as_strings=False,
                print_per_layer_stat=False,
                verbose=False
            )
    print(f'>>> flops: {flops_count}')
    print(f'>>> params: {params_count}')
    return flops_count, params_count
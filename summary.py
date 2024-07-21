#--------------------------------------------#
#   该部分代码用于看网络结构
#--------------------------------------------#
import torch
from thop import clever_format, profile
from torchsummary import summary
import time
import torch_pruning as tp

from nets import generate_model

if __name__ == "__main__":
    input_shape     = [416, 416]
    num_classes     = 14
    backbone        = 'resnet18'
    in_channels     = 25
    model_name      = 'newafft'
    # pruned_path     = 'output/pruned_static_dicts/fixedSW_newafft_pruned0.2_round16.pth'
    
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = generate_model(model_name=model_name, num_classes=num_classes, backbone=backbone, in_channels=in_channels).to(device)
    
    dummy_input     = torch.randn(1, in_channels, input_shape[0], input_shape[1]).to(device)
    rgb             = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
    # DG = tp.DependencyGraph().build_dependency(model, (dummy_input, rgb))
    # state_dict = torch.load(pruned_path)
    # DG.load_pruning_history(state_dict['pruning'])
    # model.load_state_dict(state_dict['model'])
    
    # flops, params   = profile(model.to(device), (dummy_input, ), verbose=False)
    flops, params   = profile(model.to(device), inputs=(dummy_input, rgb), verbose=False)
    #--------------------------------------------------------#
    #   flops * 2是因为profile没有将卷积作为两个operations
    #   有些论文将卷积算乘法、加法两个operations。此时乘2
    #   有些论文只考虑乘法的运算次数，忽略加法。此时不乘2
    #   本代码选择乘2，参考YOLOX。
    #--------------------------------------------------------#
    flops           = flops * 2
    flops, params   = clever_format([flops, params], "%.3f")
    print('Total params: %s' % (params))
    print('Total GFLOPS: %s' % (flops))
    
    
    # caculate the throughput
    # dummy_input     = torch.randn(8, in_channels, input_shape[0], input_shape[1]).to(device)
    # model.eval()
    # with torch.no_grad():
    #     start_time  = time.time()
    #     for i in range(100):
    #         outputs     = model(dummy_input, rgb)
    # end_time    = time.time()
    # print('Inference time: %.4f ms' % ((end_time - start_time) * 1000 / (100 * 8)))

import os

import numpy as np
import torch
import torch.nn as nn
import torch_pruning as tp
from copy import deepcopy

def gen_bn_module(model):
    ignore_bn_list = []
    prune_bn_dict = {}
    for name, m in model.named_modules():
        if "concat" in name or 'att' in name:
            ignore_bn_list.append(name)
        if isinstance(m, nn.BatchNorm2d):
            if name not in ignore_bn_list:
                prune_bn_dict[name] = m
    return prune_bn_dict, ignore_bn_list

def gather_bn_weights(prune_bn_dict):
    size_list = [idx.weight.data.shape[0] for idx in prune_bn_dict.values()]
    bn_weights = torch.zeros(sum(size_list))
    index = 0
    for i, idx in enumerate(prune_bn_dict.values()):
        size = size_list[i]
        bn_weights[index:index+size] = idx.weight.data.abs().clone()
        index += size
    return bn_weights

# 定义函数obtain_filters_mask，用于获取过滤器掩码
# 参数prune_bn_dict为剪枝的BN字典，thresh为阈值
def obtain_filters_mask(prune_bn_dict, thresh):
    # 初始化掩码字典
    mask_dict = {}
    # 初始化总通道数
    total = 0
    # 初始化剪枝通道数
    pruned = 0
    # 遍历剪枝的BN字典
    for k, v in prune_bn_dict.items():
        # 复制权重，并取绝对值
        weight_copy = v.weight.data.abs().clone()
        # 获取通道数
        channels = weight_copy.shape[0]
        # 计算最小通道数，如果大于0.5，则取0.5，否则取1
        min_channel_num = int(channels * 0.5) if int(channels * thresh) > 0.5 else 1
        # 将权重大于阈值的元素置为1
        mask = weight_copy.gt(thresh).float()
        # 如果掩码元素小于最小通道数，则将掩码中前min_channel_num个元素置为1
        if int(torch.sum(mask)) < min_channel_num:
            _, sorted_index_weights = torch.sort(weight_copy, descending=True)
            mask[sorted_index_weights[:min_channel_num]] = 1.0
        # 计算剩余通道数
        remain = int(torch.sum(mask))
        # 剪枝通道数加上掩码元素个数减去剩余通道数
        pruned = pruned + mask.shape[0] - remain
        
        # 总通道数加上掩码元素个数
        total += mask.shape[0]
        # 将掩码添加到掩码字典中
        mask_dict[k] = mask.clone()
    
    # 计算剪枝比例
    prune_ratio = pruned / total
    # 打印剪枝通道数和剪枝比例
    print('Prune channels: {}, Prune ratio: {}'.format(pruned, prune_ratio))
    # 返回剪枝比例和掩码字典
    return prune_ratio, mask_dict

def model_bn_prune(model_origin, dummy_input, model_save_path, amount=0.5):
    model_sparse = deepcopy(model_origin)
    
    prune_bn_dict, ignore_bn_list = gen_bn_module(model_sparse)
    prune_bn_weights = gather_bn_weights(prune_bn_dict)
    sorted_bn, sorted_index = torch.sort(prune_bn_weights)
    thresh_index = int(len(sorted_bn) * amount)
    thresh = sorted_bn[thresh_index]
    
    prune_ratio, mask_dict = obtain_filters_mask(prune_bn_dict, thresh)
    model_list = []
    for name, m in model_sparse.named_modules():
        if isinstance(m, nn.Conv2d):
            model_list.append(name)
            
    conv_mask_dict = {}
    for name, m in mask_dict.items():
        name = name.replace('bn', 'conv')
        if 'downsample.1' in name:
            name = name.replace('downsample.1', 'downsample.0')
        if name in model_list:
            conv_mask_dict[name] = m
    
    def obtain_num_parameters(model): return sum([param.nelement() for param in model.parameters()])
    
    num_params_origin = obtain_num_parameters(model_origin)
    
    for name, m in model_sparse.named_modules():
        if name in conv_mask_dict.keys():
            prune_index = torch.where(conv_mask_dict[name] == 0)[0].tolist()
            if len(prune_index) == 0:
                continue
            
            DG = tp.DependencyGraph()
            DG.build_dependency(model_sparse, example_inputs=dummy_input)
            pruning_group = DG.get_pruning_group(m, tp.prune_conv_out_channels, idxs=prune_index)
            
            if DG.check_pruning_group(pruning_group):
                pruning_group.prune()
                print("Layer {}.weight is pruned!!!".format(name))
    
    num_params_sparse = obtain_num_parameters(model_sparse)
    print("Number of parameters: {} -> {}. Pruned ratio: ".format(num_params_origin / 1e6, num_params_sparse / 1e6, (num_params_sparse / num_params_origin)))
    
    torch.save(model_sparse, model_save_path)


if __name__ == "__main__":
    import sys
    import argparse
    sys.path.append('/root/spectral-segmentation')
    from nets import generate_model
    from dataloader_mat import UnetDatasetTwoStream, two_stream_dataset_collate
    from nets.unet_training import CE_Loss, Dice_loss, Focal_Loss
    from torch.utils.data import DataLoader
    from functools import partial
    from utils import worker_init_fn
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='newafft', help='model name')
    parser.add_argument('--num_classes', type=int, default=14, help='num classes')
    parser.add_argument('--backbone', type=str, default='resnet18', help='backbone')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='pretrained model path')
    parser.add_argument('--is_save', type=bool, default=False, help='save model or not')
    parser.add_argument('--save_path', type=str, default='./output/pruned_static_dicts/', help='pruned model save path')
    parser.add_argument('--pruned_rate', type=float, default=0.6, help='pruned rate')
    parser.add_argument('--round_to', type=int, default=16, help='round to')
    parser.add_argument('--iterative_steps', type=int, default=5, help='iterative steps')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='num workers')
    parser.add_argument('--dataset_root', type=str, default='datasets/spectral-dataset-multi', help='dataset root')
    opt = parser.parse_args()
        # auto prune

    
    dataset_root = opt.dataset_root
    input_shape = (416, 416)
    num_classes = opt.num_classes
    shuffle = False
    batch_size = opt.batch_size
    train_sampler = None
    val_sampler = None
    num_workers = opt.num_workers
    seed = 11
    rank = 0
    iterative_steps = opt.iterative_steps
    cls_weights = torch.from_numpy(np.ones([num_classes], np.float32))
    
    pruned_rate = opt.pruned_rate
    round_to = opt.round_to
    
    # load model
    os.chdir('/root/spectral-segmentation')
    
    model_name = opt.model_name
    model = generate_model(model_name, num_classes, 3, False, 'resnet18')
    
    # input
    input1 = torch.randn(1, 25, 416, 416)
    input2 = torch.randn(1, 3, 416, 416)
    pretrained_path = opt.pretrained_model_path
    
    save_path = './output/pruned_static_dicts/'
    model.load_state_dict(torch.load(pretrained_path, map_location='cpu'))
    
    with open(os.path.join(dataset_root, "ImageSets/Segmentation/train.txt"),"r") as f:
        train_lines = f.readlines()
    with open(os.path.join(dataset_root, "ImageSets/Segmentation/val.txt"),"r") as f:
        val_lines = f.readlines()
    
    # train_dataset   = UnetDatasetTwoStream(train_lines, input_shape[-2:], num_classes, True, dataset_root)
    val_dataset     = UnetDatasetTwoStream(val_lines[:2], input_shape[-2:], num_classes, False, dataset_root)
    
    # gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
    #                             drop_last = True, sampler=train_sampler, worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed),
    #                             collate_fn= two_stream_dataset_collate)
    gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                drop_last = True, sampler=val_sampler, worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed),
                                collate_fn= two_stream_dataset_collate)
    
    # info before prune
    base_macs, base_nparams = tp.utils.count_ops_and_params(model, (input1, input2))
    
    # before_prune_dict = {}
    # for name, m in model.named_modules():
    #     if isinstance(m, nn.Conv2d) and ('backbone' in name ):
    #         before_prune_dict[name] = m.weight.shape[:2]
    
    # prepare for prune
    imp = tp.importance.GroupTaylorImportance()
    
    # ignore layers
    ignore_layers = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d) and ('final' in name or 'att' in name):
            ignore_layers.append(m)
    
    # prune     
    batch = next(iter(gen_val))
    hypers, rgbs, pngs, labels = batch
    
    # tp pruner 
    pruner = tp.pruner.BNScalePruner(
        model, 
        example_inputs=(hypers, rgbs),
        global_pruning=True,
        importance=imp,
        iterative_steps=iterative_steps,
        ignored_layers=ignore_layers,
        pruning_ratio=pruned_rate,
        round_to=round_to,
    )
    
    if isinstance(imp, tp.importance.Importance):
        outputs = model(hypers, rgbs)
        loss = CE_Loss(outputs, pngs, cls_weights, num_classes = num_classes)
        loss.backward()
    
    for i in range(iterative_steps):
        pruner.step()
        macs, nparams = tp.utils.count_ops_and_params(model, (hypers, rgbs))
        print(
            "  Iter %d/%d, Params: %.2f M => %.2f M"
            % (i+1, iterative_steps, base_nparams / 1e6, nparams / 1e6), end=", "
        )
        print(
            " MACs: %.2f G => %.2f G"
            % (base_macs / 1e9, macs / 1e9)
        )
    
    # info after prune
    macs, nparams = tp.utils.count_ops_and_params(model, (input1, input2))
    output = model(input1, input2)
    
    # after_prune_dict = {}
    # for name, m in model.named_modules():
    #     if isinstance(m, nn.Conv2d) and ('backbone' in name):
    #         after_prune_dict[name] = m.weight.shape[:2]
    
    # # show the prune layer info
    # for name, m in model.named_modules():
    #     if isinstance(m, nn.Conv2d) and ('backbone' in name):
    #         print('name: ', name, ' before: ', before_prune_dict[name], ' after: ', after_prune_dict[name])
            
    print('Pruning ratio: {:.2f}%'.format((base_nparams - nparams) / base_nparams * 100))    
    static_dict = {
        'model':model.state_dict(),
        'pruning': pruner.pruning_history(),
    }
    
    if opt.is_save:
        torch.save(static_dict, save_path + 'fixedSW_%s_pruned%s_round%s_nopre.pth'%(model_name, pruned_rate, round_to))
        
    
    
    
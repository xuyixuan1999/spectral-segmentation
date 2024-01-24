import random
import os
import numpy as np
import torch
from PIL import Image
import cv2
import time
import tqdm
from copy import deepcopy
import inspect
import logging

# Quantization Toolkit
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib

def prune_trained_model_custom(model, optimizer, allow_recompute_mask=False, allow_permutation=True,
                               compute_sparse_masks=True):
    from apex.contrib.sparsity import ASP
    asp = ASP()
    asp.init_model_for_pruning(
        model,
        mask_calculator="m4n2_1d",
        verbosity=2,
        whitelist=[torch.nn.Linear, torch.nn.Conv2d],
        allow_recompute_mask=allow_recompute_mask,
        allow_permutation=allow_permutation
    )
    asp.init_optimizer_for_pruning(optimizer)
    if compute_sparse_masks:
        asp.compute_sparse_masks()
    return asp

def export_onnx(model, onnx_filename, 
                input_name = ['spec', 'rgb'], 
                output_name=['output'], 
                input_shape=416, 
                opset_version=11, 
                verbose=False,
                dynamic_axes={'spec': {0: 'batch_size'},
                                'rgb': {0: 'batch_size'},
                                'output': {0: 'batch_size'}},
                do_constant_folding=True, 
                trace_model=False):
    model.cuda().eval()
    # We have to shift to pytorch's fake quant ops before exporting the model to ONNX
    quant_nn.TensorQuantizer.use_fb_fake_quant = True

    # Export ONNX for multiple batch sizes
    print("Creating ONNX file: " + onnx_filename)
    dummy_input_rgb = torch.randn(1, 3, input_shape, input_shape, device="cuda")
    dummy_input_spec = torch.randn(1, 25, input_shape, input_shape, device="cuda")
    dummy_input = (dummy_input_spec, dummy_input_rgb)
    try:
        # print("Exporting ONNX model with input {} to {} with opset {}!".format(dummy_input.shape, onnx_filename, opset_version))
        model_tmp = model
        if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
            #  '.module' is necessary here because model is wrapped in torch.nn.DataParallel
            model_tmp = model.module
        if trace_model:
            model_tmp = torch.jit.trace(model_tmp, dummy_input)
        torch.onnx.export(
            model_tmp, dummy_input, onnx_filename,
            verbose=verbose,
            input_names=input_name,
            output_names=output_name,
            opset_version=opset_version,
            do_constant_folding=do_constant_folding,
            dynamic_axes=dynamic_axes
        )
    except ValueError:
        print("Failed to export to ONNX")
        return False

    return True

def collect_stats(model, data_loader, num_batches):
    """Feed data to the network and collect statistic"""

    # Enable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    progress_bar = tqdm.tqdm(total=len(data_loader), leave=True, desc='Evaluation Progress')
    for i, batch in enumerate(data_loader):
        hypers, rgbs, pngs, labels = batch
        hypers  = hypers.cuda()
        rgbs    = rgbs.cuda()
        model(hypers, rgbs)  # .cuda())
        progress_bar.update()
        if i >= num_batches:
            break
    progress_bar.update()

    # Disable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()

def compute_amax(model, **kwargs):
    """Load calib result"""
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax(strict=False)
                else:
                    module.load_calib_amax(strict=False, **kwargs)
    # model.cuda()

def updateBN(model, epoch, epochs, sr=0.001):
    srtmp = sr * (1 - 0.99 * epoch / epochs)
    # ignore_bn_list = ['backbone']
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.weight.grad.data.add_(srtmp * torch.sign(m.weight.data))  # L1
            m.bias.grad.data.add_(sr * torch.sign(m.bias.data))  # L1
            
class Timer():
    def __init__(self, start_epoch, end_epoch):
        self.epoch = start_epoch + 1
        self.set_epoch = 1
        self.n_epochs = end_epoch
        self.prev_time = time.time()
        self.mean_preiod = 0
        
    def log(self):
        time_consumed = (time.time()-self.prev_time)
        self.mean_preiod += time_consumed
        self.prev_time = time.time()
        epoch_done = self.set_epoch
        epoch_left = self.n_epochs - self.epoch
        pre_time = epoch_left*self.mean_preiod/epoch_done
        self.set_epoch += 1
        self.epoch += 1
        return time_consumed, pre_time
        

#---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
#---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 

#---------------------------------------------------#
#   对输入图像进行resize
#---------------------------------------------------#
def resize_image(image, size):
    iw, ih  = image.size
    w, h    = size

    scale   = min(w/iw, h/ih)
    nw      = int(iw*scale)
    nh      = int(ih*scale)

    image   = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))

    return new_image, nw, nh

def resize_mat(mat, size):
    # mat [h, w, c]
    ih, iw, ic  = mat.shape
    w, h    = size
    
    scale   = min(w/iw, h/ih)
    nw      = int(iw*scale)
    nh      = int(ih*scale)
    dx = (w-nw)//2
    dy = (h-nh)//2
    # resize multichannel
    mat = resize_multichannel_image(mat, (nh, nw))
    new_mat = np.ones((h, w, ic), dtype=mat.dtype) * 128
    new_mat[dy:dy+nh, dx:dx+nw, :] = mat
    return new_mat, nw, nh
    
def resize_multichannel_image(image, target_size):
    # 获取输入图像的高度、宽度和通道数
    h, w, c = image.shape
    
    # 创建一个与目标大小相同的空图像
    resized_image = np.zeros((target_size[0], target_size[1], c), dtype=image.dtype)
    
    for channel in range(c):
        channel_image = image[:, :, channel]
        resized_channel = cv2.resize(channel_image, (target_size[1], target_size[0]))
        resized_image[:, :, channel] = resized_channel

    return resized_image
    
#---------------------------------------------------#
#   获得学习率
#---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

#---------------------------------------------------#
#   设置种子
#---------------------------------------------------#
def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#---------------------------------------------------#
#   设置Dataloader的种子
#---------------------------------------------------#
def worker_init_fn(worker_id, rank, seed):
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def preprocess_input(image):
    image /= 255.0
    return image

def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)

def print_options(opt):
    """Print and save options

    It will print both current options and default values(if different).
    It will save options into a text file / [checkpoints_dir] / opt.txt
    """
    message = ''
    message += '-' * 110
    message += '\n'
    message += '|%25s | %-80s|\n' % ('keys', 'values')
    message += '-' * 110
    message += '\n'
    for k, v in sorted(vars(opt).items()):
        message += '|%25s | %-80s|\n' % (str(k), str(v))
    message += '-' * 110
    print(message)
    # save to the disk
    file_name = os.path.join(opt.save_dir, 'opt.txt')
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')

        

def download_weights(backbone, model_dir="./model_data"):
    import os
    from torch.hub import load_state_dict_from_url
    
    download_urls = {
        'vgg'       : 'https://download.pytorch.org/models/vgg16-397923af.pth',
        'resnet50'  : 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth'
    }
    url = download_urls[backbone]
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    load_state_dict_from_url(url, model_dir)
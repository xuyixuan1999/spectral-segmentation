import random
import os
import numpy as np
import torch
from PIL import Image
import cv2
import time

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
    message += '-' * 90
    message += '\n'
    message += '|%25s | %60s|\n' % ('keys', 'values')
    message += '-' * 90
    message += '\n'
    for k, v in sorted(vars(opt).items()):
        message += '|%25s | %60s|\n' % (str(k), str(v))
    message += '-' * 90
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
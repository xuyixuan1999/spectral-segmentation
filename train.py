import datetime
import os
import argparse
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.unet import Unet
from nets.unet_training import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.callbacks import EvalCallback, LossHistory
# from utils.dataloader import UnetDataset, unet_dataset_collate
from utils.dataloader_mat import UnetDatasetMat, image_dataset_collate, UnetDatasetRGB
from utils.utils import seed_everything, worker_init_fn, print_options, Timer
from utils.utils_fit import fit_one_epoch


parser = argparse.ArgumentParser()
parser.add_argument('--anote', type=str, default='', help='note of this training script')
parser.add_argument('--cuda', action='store_true', default=False, help='Use cuda')
parser.add_argument('--seed', type=int, default=11, help='random seed')
parser.add_argument('--distributed', action='store_true', default=False, help='Use distributed training')
parser.add_argument('--sync_bn', action='store_true', default=False, help='Use sync batch norm')
parser.add_argument('--fp16', action='store_true', default=False, help='Use fp16 training')
parser.add_argument('--num_classes', type=int, default=21, help='num classes')
parser.add_argument('--backbone',choices=['vgg', 'resnet50', 'resnet34', 'resnet18', 'efficientvit'], default='vgg', help='backbone')
parser.add_argument('--pretrained_model_path', type=str, default=None, help='pretrained model path')
parser.add_argument('--input_shape', type=int, nargs='+', default=[512, 512], help='Input image shape (width and height)')
parser.add_argument('--init_epoch', type=int, default=0, help='Init Epoch')
parser.add_argument('--freeze_epoch', type=int, default=50, help='freeze_epoch')
parser.add_argument('--freeze_batch_size', type=int, default=2, help='freeze_batch_size')
parser.add_argument('--unfreeze_epoch', type=int, default=100, help='unfreeze_epoch')
parser.add_argument('--unfreeze_batch_size', type=int, default=2, help='unfreeze_batch_size')
parser.add_argument('--freeze_train', type=bool, default=True, help='freeze_train')
parser.add_argument('--init_lr', type=float, default=1e-4, help='init_lr')
parser.add_argument('--optimizer_type', choices=['adam', 'sgd'], default='adam', help='optimizer type')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
parser.add_argument('--lr_decay_type', choices=['step', 'cos'], default='cos', help='lr decay type')
parser.add_argument('--save_period', type=int, default=5, help='save period')
parser.add_argument('--save_dir', type=str, default='logs', help='save dir')
parser.add_argument('--num_workers', type=int, default=4, help='num workers')
parser.add_argument('--eval_flag', type=bool, default=True, help='eval flag')
parser.add_argument('--eval_period', type=int, default=5, help='eval period')
parser.add_argument('--dataset_root', type=str, default='VOCdevkit', help='VOCdevkit path')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids (default: 0)')

opt = parser.parse_args()

os.environ['CUDA_VISIBLE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids

if __name__ == "__main__":
    #---------------------------------------------------------------------#
    #   distributed     用于指定是否使用单机多卡分布式运行
    #                   终端指令仅支持Ubuntu。CUDA_VISIBLE_DEVICES用于在Ubuntu下指定显卡。
    #                   Windows系统下默认使用DP模式调用所有显卡,不支持DDP。
    #   DP模式:
    #       设置            distributed = False
    #       在终端中输入    CUDA_VISIBLE_DEVICES=0,1 python train.py
    #   DDP模式:
    #       设置            distributed = True
    #       在终端中输入    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
    #---------------------------------------------------------------------#
    #----------------------------------------------------------------------------------------------------------------------------#
    #   训练分为两个阶段,分别是冻结阶段和解冻阶段。设置冻结阶段是为了满足机器性能不足的同学的训练需求。
    #   冻结训练需要的显存较小,显卡非常差的情况下,可设置freeze_epoch等于unfreeze_epoch,此时仅仅进行冻结训练。
    #   
    #   在此提供若干参数设置建议,各位训练者根据自己的需求进行灵活调整:
    #   (一)从整个模型的预训练权重开始训练: 
    #       Adam:
    #           Init_Epoch = 0,freeze_epoch = 50,unfreeze_epoch = 100,freeze_train = True,optimizer_type = 'adam',init_lr = 1e-4,weight_decay = 0。(冻结)
    #           init_epoch = 0,unfreeze_epoch = 100,freeze_train = False,optimizer_type = 'adam',init_lr = 1e-4,weight_decay = 0。(不冻结)
    #       SGD:
    #           init_epoch = 0,freeze_epoch = 50,unfreeze_epoch = 100,freeze_train = True,optimizer_type = 'sgd',init_lr = 1e-2,weight_decay = 1e-4。(冻结)
    #           init_epoch = 0,unfreeze_epoch = 100,freeze_train = False,optimizer_type = 'sgd',init_lr = 1e-2,weight_decay = 1e-4。(不冻结)
    #       其中:unfreeze_epoch可以在100-300之间调整。
    #   (二)从主干网络的预训练权重开始训练:
    #       Adam:
    #           init_epoch = 0,freeze_epoch = 50,unfreeze_epoch = 100,freeze_train = True,optimizer_type = 'adam',init_lr = 1e-4,weight_decay = 0。(冻结)
    #           init_epoch = 0,unfreeze_epoch = 100,freeze_train = False,optimizer_type = 'adam',init_lr = 1e-4,weight_decay = 0。(不冻结)
    #       SGD:
    #           init_epoch = 0,freeze_epoch = 50,unfreeze_epoch = 120,freeze_train = True,optimizer_type = 'sgd',init_lr = 1e-2,weight_decay = 1e-4。(冻结)
    #           init_epoch = 0,unfreeze_epoch = 120,freeze_train = False,optimizer_type = 'sgd',init_lr = 1e-2,weight_decay = 1e-4。(不冻结)
    #       其中:由于从主干网络的预训练权重开始训练,主干的权值不一定适合语义分割,需要更多的训练跳出局部最优解。
    #             unfreeze_epoch可以在120-300之间调整。
    #             Adam相较于SGD收敛的快一些。因此unfreeze_epoch理论上可以小一点,但依然推荐更多的Epoch。
    #   (三)batch_size的设置:
    #       在显卡能够接受的范围内,以大为好。显存不足与数据集大小无关,提示显存不足(OOM或者CUDA out of memory)请调小batch_size。
    #       由于resnet50中有BatchNormalization层
    #       当主干为resnet50的时候batch_size不可为1
    #       正常情况下freeze_batch_size建议为unfreeze_batch_size的1-2倍。不建议设置的差距过大,因为关系到学习率的自动调整。
    #----------------------------------------------------------------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   冻结阶段训练参数
    #   此时模型的主干被冻结了,特征提取网络不发生改变
    #   占用的显存较小,仅对网络进行微调
    #   init_epoch          模型当前开始的训练世代,其值可以大于freeze_epoch,如设置:
    #                       init_epoch = 60、freeze_epoch = 50、unfreeze_epoch = 100
    #                       会跳过冻结阶段,直接从60代开始,并调整对应的学习率。
    #                       (断点续练时使用)
    #   freeze_epoch        模型冻结训练的freeze_epoch
    #                       (当freeze_train=False时失效)
    #   freeze_batch_size   模型冻结训练的batch_size
    #                       (当freeze_train=False时失效)
    #------------------------------------------------------------------#

    Min_lr              = opt.init_lr * 0.01
    #------------------------------------------------------------------#
    #   optimizer_type  使用到的优化器种类,可选的有adam、sgd
    #                   当使用Adam优化器时建议设置  init_lr=1e-4
    #                   当使用SGD优化器时建议设置   init_lr=1e-2
    #   momentum        优化器内部使用到的momentum参数
    #   weight_decay    权值衰减,可防止过拟合
    #                   adam会导致weight_decay错误,使用adam时建议设置为0。
    #------------------------------------------------------------------#

    #------------------------------------------------------------------#
    #   建议选项:
    #   种类少(几类)时,设置为True
    #   种类多(十几类)时,如果batch_size比较大(10以上),那么设置为True
    #   种类多(十几类)时,如果batch_size比较小(10以下),那么设置为False
    #------------------------------------------------------------------#
    dice_loss       = False
    #------------------------------------------------------------------#
    #   是否使用focal loss来防止正负样本不平衡
    #------------------------------------------------------------------#
    focal_loss      = False
    #------------------------------------------------------------------#
    #   是否给不同种类赋予不同的损失权值,默认是平衡的。
    #   设置的话,注意设置成numpy形式的,长度和num_classes一样。
    #   如:
    #   num_classes = 3
    #   cls_weights = np.array([1, 2, 3], np.float32)
    #------------------------------------------------------------------#
    cls_weights     = np.ones([opt.num_classes], np.float32)
    
    sync_bn         = False
    pretrained  = False
    

    seed_everything(opt.seed)

    # 设置用到的显卡
    ngpus_per_node  = torch.cuda.device_count()
    if opt.distributed:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank      = 0
        rank            = 0

    model = Unet(num_classes=opt.num_classes, pretrained=pretrained, backbone=opt.backbone, in_channels=opt.input_shape[0]).train()
    
    if not pretrained:
        weights_init(model)
    if opt.pretrained_model_path != '':
        if local_rank == 0:
            print('Load weights {}.'.format(opt.pretrained_model_path))
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(opt.pretrained_model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        #------------------------------------------------------#
        #   显示没有匹配上的Key
        #------------------------------------------------------#
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44m温馨提示,head部分没有载入是正常现象,Backbone部分没有载入是错误的。\033[0m")

    #----------------------#
    #   记录Loss
    #----------------------#
    if local_rank == 0:
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        opt.save_dir    = os.path.join(opt.save_dir, str(time_str) + "_%s"%opt.anote)
        
        loss_history    = LossHistory(opt.save_dir, model, input_shape=opt.input_shape)
    else:
        loss_history    = None
        
    #------------------------------------------------------------------#
    #   torch 1.2不支持amp,建议使用torch 1.7.1及以上正确使用fp16
    #   因此torch1.2这里显示"could not be resolve"
    #------------------------------------------------------------------#
    if opt.fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train     = model.train()
    #----------------------------#
    #   多卡同步Bn
    #----------------------------#
    if sync_bn and ngpus_per_node > 1 and opt.distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if opt.cuda:
        if opt.distributed:
            #----------------------------#
            #   多卡平行运行
            #----------------------------#
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()
    
    #---------------------------#
    #   读取数据集对应的txt
    #---------------------------#
    with open(os.path.join(opt.dataset_root, "ImageSets/Segmentation/train.txt"),"r") as f:
        train_lines = f.readlines()
    with open(os.path.join(opt.dataset_root, "ImageSets/Segmentation/val.txt"),"r") as f:
        val_lines = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)
        
    if local_rank == 0:
        print_options(opt)
    #------------------------------------------------------#
    #   主干特征提取网络特征通用,冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   init_epoch为起始世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    #------------------------------------------------------#
    if True:
        Unfreeze_flag = False
        #------------------------------------#
        #   冻结一定部分训练
        #------------------------------------#
        if opt.freeze_train:
            model.freeze_backbone()
            
        #-------------------------------------------------------------------#
        #   如果不冻结训练的话,直接设置batch_size为unfreeze_batch_size
        #-------------------------------------------------------------------#
        batch_size = opt.freeze_batch_size if opt.freeze_train else opt.unfreeze_batch_size

        #-------------------------------------------------------------------#
        #   判断当前batch_size,自适应调整学习率
        #-------------------------------------------------------------------#
        nbs             = 16
        lr_limit_max    = 1e-4 if opt.optimizer_type == 'adam' else 1e-1
        lr_limit_min    = 1e-4 if opt.optimizer_type == 'adam' else 5e-4
        init_lr_fit     = min(max(batch_size / nbs * opt.init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        #---------------------------------------#
        #   根据optimizer_type选择优化器
        #---------------------------------------#
        optimizer = {
            'adam'  : optim.Adam(model.parameters(), init_lr_fit, betas = (opt.momentum, 0.999), weight_decay = opt.weight_decay),
            'sgd'   : optim.SGD(model.parameters(), init_lr_fit, momentum = opt.momentum, nesterov=True, weight_decay = opt.weight_decay)
        }[opt.optimizer_type]

        #---------------------------------------#
        #   获得学习率下降的公式
        #---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(opt.lr_decay_type, init_lr_fit, Min_lr_fit, opt.unfreeze_epoch)
        
        #---------------------------------------#
        #   判断每一个世代的长度
        #---------------------------------------#
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小,无法继续进行训练,请扩充数据集。")

        # train_dataset   = UnetDataset(train_lines, opt.input_shape, opt.num_classes, True, opt.dataset_root)
        # val_dataset     = UnetDataset(val_lines, opt.input_shape, opt.num_classes, False, opt.dataset_root)
        if opt.input_shape[0] == 3:
            train_dataset   = UnetDatasetRGB(train_lines, opt.input_shape[1:3], opt.num_classes, True, opt.dataset_root)
            val_dataset     = UnetDatasetRGB(val_lines, opt.input_shape[1:3], opt.num_classes, False, opt.dataset_root)
        else:
            train_dataset   = UnetDatasetMat(train_lines, opt.input_shape[1:3], opt.num_classes, True, opt.dataset_root)
            val_dataset     = UnetDatasetMat(val_lines, opt.input_shape[1:3], opt.num_classes, False, opt.dataset_root)
        
        if opt.distributed:
            train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
            val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
            batch_size      = batch_size // ngpus_per_node
            shuffle         = False
        else:
            train_sampler   = None
            val_sampler     = None
            shuffle         = True

        gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = opt.num_workers, pin_memory=True,
                                    drop_last = True, collate_fn = image_dataset_collate, sampler=train_sampler, 
                                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=opt.seed))
        gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = opt.num_workers, pin_memory=True, 
                                    drop_last = True, collate_fn = image_dataset_collate, sampler=val_sampler, 
                                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=opt.seed))
        
        #----------------------#
        #   记录eval的map曲线
        #----------------------#
        if local_rank == 0:
            eval_callback   = EvalCallback(model, opt.input_shape[1:3], opt.num_classes, val_lines, opt.dataset_root, opt.save_dir, opt.cuda, \
                                            eval_flag=opt.eval_flag, period=opt.eval_period)
        else:
            eval_callback   = None
        
        timer = Timer(opt.init_epoch, opt.unfreeze_epoch)
        #---------------------------------------#
        #   开始模型训练
        #---------------------------------------#
        for epoch in range(opt.init_epoch, opt.unfreeze_epoch):
            #---------------------------------------#
            #   如果模型有冻结学习部分
            #   则解冻,并设置参数
            #---------------------------------------#
            if epoch >= opt.freeze_epoch and not Unfreeze_flag and opt.freeze_train:
                batch_size = opt.unfreeze_batch_size

                #-------------------------------------------------------------------#
                #   判断当前batch_size,自适应调整学习率
                #-------------------------------------------------------------------#
                nbs             = 16
                lr_limit_max    = 1e-4 if opt.optimizer_type == 'adam' else 1e-1
                lr_limit_min    = 1e-4 if opt.optimizer_type == 'adam' else 5e-4
                init_lr_fit     = min(max(batch_size / nbs * opt.init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                #---------------------------------------#
                #   获得学习率下降的公式
                #---------------------------------------#
                lr_scheduler_func = get_lr_scheduler(opt.lr_decay_type, init_lr_fit, Min_lr_fit, opt.unfreeze_epoch)
                    
                model.unfreeze_backbone()
                            
                epoch_step      = num_train // batch_size
                epoch_step_val  = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("数据集过小,无法继续进行训练,请扩充数据集。")

                if opt.distributed:
                    batch_size = batch_size // ngpus_per_node

                gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = opt.num_workers, pin_memory=True,
                                            drop_last = True, collate_fn = image_dataset_collate, sampler=train_sampler, 
                                            worker_init_fn=partial(worker_init_fn, rank=rank, seed=opt.seed))
                gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = opt.num_workers, pin_memory=True, 
                                            drop_last = True, collate_fn = image_dataset_collate, sampler=val_sampler, 
                                            worker_init_fn=partial(worker_init_fn, rank=rank, seed=opt.seed))

                Unfreeze_flag = True

            if opt.distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(opt, model_train, model, loss_history, eval_callback, optimizer, 
                          epoch, epoch_step, epoch_step_val, gen, gen_val, 
                          dice_loss, focal_loss, cls_weights, scaler, local_rank)
            if local_rank == 0:
                time_consumed, eta = timer.log()
                print('==> Time consumed: {:.2f}s, ETA: {}.'.format(time_consumed, datetime.timedelta(seconds=eta)))

            if opt.distributed:
                dist.barrier()

        if local_rank == 0:
            loss_history.writer.close()

import os

import matplotlib
import torch
import torch.nn.functional as F

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import scipy.signal

import cv2
import shutil
import numpy as np
import logging
import h5py

from PIL import Image
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from .utils import cvtColor, preprocess_input, resize_image, resize_mat
from .utils_metrics import compute_mIoU

def initialize_logger(file_dir):
    logger = logging.getLogger()
    fhandler = logging.FileHandler(filename=file_dir, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(message)s', "%Y-%m-%d %H:%M:%S")
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.WARNING)
    return logger

class LossHistory():
    def __init__(self, log_dir, model, input_shape):
        self.log_dir        = log_dir

        self.losses         = []

        self.val_loss   = []
        
        os.makedirs(self.log_dir)
        
        self.logger         = initialize_logger(os.path.join(self.log_dir, "train.log"))
        
        self.writer     = SummaryWriter(self.log_dir)
        try:
            dummy_input     = torch.randn((2, *input_shape))
            self.writer.add_graph(model, dummy_input)
        except:
            pass

    def append_loss(self, epoch, loss, val_loss, lr, model=None):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)

        self.val_loss.append(val_loss)
        
        self.logger.warning("Epoch[%04d], Learning Rate : %.9f, Train Loss : %.9f, Val Loss : %.9f", epoch, lr, loss, val_loss)
            
        self.writer.add_scalar('loss', loss, epoch)

        self.writer.add_scalar('val_loss', val_loss, epoch)
        
        # prune of sparse bn visualization
        if model is not None:
            bn_weight = []
            bn_bias = []
            bn_hist = []
            for name, m in model.named_modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    bn_hist.append(m.weight.data.clone().cpu().view(-1))
                    bn_weight.append(m.weight.data.clone().cpu().sum() / m.num_features)
                    bn_bias.append(m.bias.data.clone().cpu().sum() / m.num_features)
            self.writer.add_scalar("val/bn_weight", sum(bn_weight) / len(bn_weight), epoch)
            self.writer.add_scalar("val/bn_bias", sum(bn_bias) / len(bn_bias), epoch)
            self.writer.add_histogram("bn_hist", torch.cat(bn_hist, dim=0), epoch)
            
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth = 2, label='train loss')

        plt.plot(iters, self.val_loss, 'coral', linewidth = 2, label='val loss')
            
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')

            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")

class EvalCallback():
    def __init__(self, net, input_shape, num_classes, image_ids, dataset_path, log_dir, cuda, \
            miou_out_path=".temp_miou_out", eval_flag=True, period=1, in_channels=3):
        super(EvalCallback, self).__init__()
        
        self.net                = net
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.image_ids          = image_ids
        self.dataset_path       = dataset_path
        self.log_dir            = log_dir
        self.cuda               = cuda
        self.miou_out_path      = miou_out_path
        self.eval_flag          = eval_flag
        self.period             = period
        
        self.image_ids          = [image_id.split()[0] for image_id in image_ids]
        self.mious      = [0]
        self.epoches    = [0]
        if self.eval_flag:
            with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
                f.write(str(0))
                f.write("\n")

    def get_miou_png(self, image):
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        orininal_h  = np.array(image).shape[0]
        orininal_w  = np.array(image).shape[1]
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data, nw, nh  = resize_image(image, (self.input_shape[1],self.input_shape[0]))
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
                
            #---------------------------------------------------#
            #   图片传入网络进行预测
            #---------------------------------------------------#
            pr = self.net(images)[0]
            #---------------------------------------------------#
            #   取出每一个像素点的种类
            #---------------------------------------------------#
            pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy()
            #--------------------------------------#
            #   将灰条部分截取掉
            #--------------------------------------#
            pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
            #---------------------------------------------------#
            #   进行图片的resize
            #---------------------------------------------------#
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
            #---------------------------------------------------#
            #   取出每一个像素点的种类
            #---------------------------------------------------#
            pr = pr.argmax(axis=-1)
    
        image = Image.fromarray(np.uint8(pr))
        return image
    
    def on_epoch_end(self, epoch, model_eval):
        if epoch % self.period == 0 and self.eval_flag:
            self.net    = model_eval
            gt_dir      = os.path.join(self.dataset_path, "SegmentationClass/")
            pred_dir    = os.path.join(self.miou_out_path, 'detection-results')
            if not os.path.exists(self.miou_out_path):
                os.makedirs(self.miou_out_path)
            if not os.path.exists(pred_dir):
                os.makedirs(pred_dir)
            print("Get miou.")
            for image_id in tqdm(self.image_ids):
                #-------------------------------#
                #   从文件中读取图像
                #-------------------------------#
                image_path  = os.path.join(self.dataset_path, "JPEGImages/"+image_id+".jpg")
                image       = Image.open(image_path)
                #------------------------------#
                #   获得预测txt
                #------------------------------#
                image       = self.get_miou_png(image)
                image.save(os.path.join(pred_dir, image_id + ".png"))
                        
            print("Calculate miou.")
            _, IoUs, _, _ = compute_mIoU(gt_dir, pred_dir, self.image_ids, self.num_classes, None)  # 执行计算mIoU的函数
            temp_miou = np.nanmean(IoUs) * 100

            self.mious.append(temp_miou)
            self.epoches.append(epoch)

            with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
                f.write(str(temp_miou))
                f.write("\n")
            
            plt.figure()
            plt.plot(self.epoches, self.mious, 'red', linewidth = 2, label='train miou')

            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Miou')
            plt.title('A Miou Curve')
            plt.legend(loc="best")

            plt.savefig(os.path.join(self.log_dir, "epoch_miou.png"))
            plt.cla()
            plt.close("all")

            print("Get miou done.")
            shutil.rmtree(self.miou_out_path)
            
    def on_epoch_end_mat(self, epoch, model_eval):
        if epoch % self.period == 0 and self.eval_flag:
            self.net    = model_eval
            gt_dir      = os.path.join(self.dataset_path, "SegmentationClass/")
            pred_dir    = os.path.join(self.miou_out_path, 'detection-results')
            if not os.path.exists(self.miou_out_path):
                os.makedirs(self.miou_out_path)
            if not os.path.exists(pred_dir):
                os.makedirs(pred_dir)
            print("Get miou.")
            for image_id in tqdm(self.image_ids):
                #-------------------------------#
                #   从文件中读取图像
                #-------------------------------#
                # image_path  = os.path.join(self.dataset_path, "JPEGImages/"+image_id+".jpg")
                mat_path = os.path.join(self.dataset_path, "Train_Spec/" + image_id + ".mat")
                with h5py.File(mat_path, 'r') as mat:
                    hyper = np.float32(np.array(mat['cube']))
                image = np.transpose(hyper, (2, 1, 0))
                
                # mask 
                mask = cv2.imread(os.path.join(self.dataset_path, "Train_Mask", image_id + ".png"), 0)
                mask = np.expand_dims(mask, axis=0)
                mask = np.broadcast_to(mask, (image.shape[0], mask.shape[1], mask.shape[2]))
                image = np.where(mask > 0, 128/255.0, image)
                #------------------------------#
                #   获得预测txt
                #------------------------------#
                image       = self.get_miou_mat(image)
                image.save(os.path.join(pred_dir, image_id + ".png"))
                        
            print("Calculate miou.")
            _, IoUs, _, _ = compute_mIoU(gt_dir, pred_dir, self.image_ids, self.num_classes, None)  # 执行计算mIoU的函数
            temp_miou = np.nanmean(IoUs) * 100

            self.mious.append(temp_miou)
            self.epoches.append(epoch)

            with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
                f.write(str(temp_miou))
                f.write("\n")
            
            plt.figure()
            plt.plot(self.epoches, self.mious, 'red', linewidth = 2, label='train miou')

            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Miou')
            plt.title('A Miou Curve')
            plt.legend(loc="best")

            plt.savefig(os.path.join(self.log_dir, "epoch_miou.png"))
            plt.cla()
            plt.close("all")

            print("Get miou done.")
            shutil.rmtree(self.miou_out_path)
            
    def get_miou_mat(self, image):
        
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        # image [c, h, w]
        image = np.transpose(image, [1, 2, 0]) # [h, w, c]
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        image = (image * 255).astype(np.uint8)
        
        orininal_h  = np.array(image).shape[0]
        orininal_w  = np.array(image).shape[1]
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data, nw, nh  = resize_mat(image, (self.input_shape[1], self.input_shape[0]))
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(np.array(image_data, np.float32) / 255.0, (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
                
            #---------------------------------------------------#
            #   图片传入网络进行预测
            #---------------------------------------------------#
            pr = self.net(images)[0]
            #---------------------------------------------------#
            #   取出每一个像素点的种类
            #---------------------------------------------------#
            pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy()
            #--------------------------------------#
            #   将灰条部分截取掉
            #--------------------------------------#
            pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
            #---------------------------------------------------#
            #   进行图片的resize
            #---------------------------------------------------#
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
            #---------------------------------------------------#
            #   取出每一个像素点的种类
            #---------------------------------------------------#
            pr = pr.argmax(axis=-1)
    
        image = Image.fromarray(np.uint8(pr))
        return image

    def on_epoch_end_rgb(self, epoch, model_eval):
        if epoch % self.period == 0 and self.eval_flag:
            self.net    = model_eval
            gt_dir      = os.path.join(self.dataset_path, "SegmentationClass/")
            pred_dir    = os.path.join(self.miou_out_path, 'detection-results')
            if not os.path.exists(self.miou_out_path):
                os.makedirs(self.miou_out_path)
            if not os.path.exists(pred_dir):
                os.makedirs(pred_dir)
            print("Get miou.")
            for image_id in tqdm(self.image_ids):
                #-------------------------------#
                #   从文件中读取图像
                #-------------------------------#
                # image_path  = os.path.join(self.dataset_path, "JPEGImages/"+image_id+".jpg")
                rgb_path = os.path.join(self.dataset_path, "JPEGImages/" + image_id + ".jpg")
                image = np.array(Image.open(rgb_path), dtype=np.uint8)
                
                # mask 
                mask = cv2.imread(os.path.join(self.dataset_path, "Train_Mask", image_id + ".png"), 0)
                mask = np.expand_dims(mask, axis=2)
                mask = np.broadcast_to(mask, (image.shape[0], image.shape[1], image.shape[2]))
                image = np.where(mask > 0, 128, image)
                #------------------------------#
                #   获得预测txt
                #------------------------------#
                image       = self.get_miou_rgb(image)
                image.save(os.path.join(pred_dir, image_id + ".png"))
                        
            print("Calculate miou.")
            _, IoUs, _, _ = compute_mIoU(gt_dir, pred_dir, self.image_ids, self.num_classes, None)  # 执行计算mIoU的函数
            temp_miou = np.nanmean(IoUs) * 100

            self.mious.append(temp_miou)
            self.epoches.append(epoch)

            with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
                f.write(str(temp_miou))
                f.write("\n")
            
            plt.figure()
            plt.plot(self.epoches, self.mious, 'red', linewidth = 2, label='train miou')

            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Miou')
            plt.title('A Miou Curve')
            plt.legend(loc="best")

            plt.savefig(os.path.join(self.log_dir, "epoch_miou.png"))
            plt.cla()
            plt.close("all")

            print("Get miou done.")
            shutil.rmtree(self.miou_out_path)
            
    def get_miou_rgb(self, image):
        
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        # image [h, w, c]
        # image = np.transpose(image, [1, 2, 0]) # [h, w, c]
        # image = (image - np.min(image)) / (np.max(image) - np.min(image))
        # image = (image * 255).astype(np.uint8)
        
        orininal_h  = np.array(image).shape[0]
        orininal_w  = np.array(image).shape[1]
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data, nw, nh  = resize_mat(image, (self.input_shape[1], self.input_shape[0]))
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(np.array(image_data, np.float32) / 255.0, (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
                
            #---------------------------------------------------#
            #   图片传入网络进行预测
            #---------------------------------------------------#
            pr = self.net(images)[0]
            #---------------------------------------------------#
            #   取出每一个像素点的种类
            #---------------------------------------------------#
            pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy()
            #--------------------------------------#
            #   将灰条部分截取掉
            #--------------------------------------#
            pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
            #---------------------------------------------------#
            #   进行图片的resize
            #---------------------------------------------------#
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
            #---------------------------------------------------#
            #   取出每一个像素点的种类
            #---------------------------------------------------#
            pr = pr.argmax(axis=-1)
    
        image = Image.fromarray(np.uint8(pr))
        return image

    def on_epoch_end_fusion(self, epoch, model_eval):
        if epoch % self.period == 0 and self.eval_flag:
            self.net    = model_eval
            gt_dir      = os.path.join(self.dataset_path, "SegmentationClass/")
            pred_dir    = os.path.join(self.miou_out_path, 'detection-results')
            if not os.path.exists(self.miou_out_path):
                os.makedirs(self.miou_out_path)
            if not os.path.exists(pred_dir):
                os.makedirs(pred_dir)
            print("Get miou.")
            for image_id in tqdm(self.image_ids):
                #-------------------------------#
                #   从文件中读取图像
                #-------------------------------#
                # load rgb 
                rgb_path = os.path.join(self.dataset_path, "JPEGImages/" + image_id + ".jpg")
                rgb = np.array(Image.open(rgb_path), dtype=np.uint8)
                
                # load spectral 
                spec_path = os.path.join(self.dataset_path, "Train_Spec/" + image_id + ".mat")
                with h5py.File(spec_path, 'r') as mat:
                    spec = np.float32(np.array(mat['cube']))
                spec = np.transpose(spec, (2, 1, 0))
                
                # mask 
                mask = cv2.imread(os.path.join(self.dataset_path, "Train_Mask", image_id + ".png"), 0)
                mask_rgb = np.expand_dims(mask, axis=2)
                mask_rgb = np.broadcast_to(mask_rgb, (rgb.shape[0], rgb.shape[1], rgb.shape[2]))
                rgb  = np.where(mask_rgb > 0, 128, rgb)
                
                mask_spec = np.expand_dims(mask, axis=0)
                mask_spec = np.broadcast_to(mask_spec, (spec.shape[0], spec.shape[1], spec.shape[2]))
                spec = np.where(mask_spec > 0, 128 / 255.0, spec)
                
                
                #------------------------------#
                #   获得预测txt
                #------------------------------#
                image       = self.get_miou_fusion(spec, rgb)
                image.save(os.path.join(pred_dir, image_id + ".png"))
                        
            print("Calculate miou.")
            _, IoUs, _, _ = compute_mIoU(gt_dir, pred_dir, self.image_ids, self.num_classes, None)  # 执行计算mIoU的函数
            temp_miou = np.nanmean(IoUs) * 100

            self.mious.append(temp_miou)
            self.epoches.append(epoch)

            with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
                f.write('Epoch[%03d]: '%epoch + str(temp_miou))
                f.write("\n")
            
            plt.figure()
            plt.plot(self.epoches, self.mious, 'red', linewidth = 2, label='train miou')

            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Miou')
            plt.title('A Miou Curve')
            plt.legend(loc="best")

            plt.savefig(os.path.join(self.log_dir, "epoch_miou.png"))
            plt.cla()
            plt.close("all")

            print("Get miou done.")
            shutil.rmtree(self.miou_out_path)
            
    def get_miou_fusion(self, spec, rgb):
        
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        # spec [h, w, c]
        spec = np.transpose(spec, [1, 2, 0]) # [h, w, c]
        spec = (spec - np.min(spec)) / (np.max(spec) - np.min(spec))
        spec = (spec * 255).astype(np.uint8)
        
        orininal_h  = np.array(rgb).shape[0]
        orininal_w  = np.array(rgb).shape[1]
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        spec_data, nw, nh  = resize_mat(spec, (self.input_shape[1], self.input_shape[0]))
        rgb_data, _, _ = resize_mat(rgb, (self.input_shape[1], self.input_shape[0]))
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        spec_data  = np.expand_dims(np.transpose(np.array(spec_data, np.float32) / 255.0, (2, 0, 1)), 0)
        rgb_data  = np.expand_dims(np.transpose(np.array(rgb_data, np.float32) / 255.0, (2, 0, 1)), 0)

        with torch.no_grad():
            spec_ = torch.from_numpy(spec_data)
            rgb_ = torch.from_numpy(rgb_data)
            if self.cuda:
                spec_ = spec_.cuda()
                rgb_ = rgb_.cuda()
                
            #---------------------------------------------------#
            #   图片传入网络进行预测
            #---------------------------------------------------#
            pr = self.net(spec_, rgb_)[0]
            #---------------------------------------------------#
            #   取出每一个像素点的种类
            #---------------------------------------------------#
            pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy()
            #--------------------------------------#
            #   将灰条部分截取掉
            #--------------------------------------#
            pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
            #---------------------------------------------------#
            #   进行图片的resize
            #---------------------------------------------------#
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
            #---------------------------------------------------#
            #   取出每一个像素点的种类
            #---------------------------------------------------#
            pr = pr.argmax(axis=-1)
    
        image = Image.fromarray(np.uint8(pr))
        return image

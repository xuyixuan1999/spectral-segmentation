import os

import cv2
import numpy as np
import torch
from PIL import Image
import h5py
from torch.utils.data.dataset import Dataset


class UnetDatasetTwoStream(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path):
        super(UnetDatasetTwoStream, self).__init__()
        self.annotation_lines   = annotation_lines
        self.length             = len(annotation_lines)
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.train              = train
        self.dataset_path       = dataset_path
        
        self.mat_dict = {}
        self.rgb_dict = {}
        self.label_dict = {}
        # load mat
        for i in self.annotation_lines:
            name = i.split()[0]
            try:
                with h5py.File(os.path.join(self.dataset_path, "Train_Spec", name + '.mat'), 'r') as mat:
                    hyper = np.float32(np.array(mat['cube']))
                    hyper = np.transpose(hyper, [2, 1, 0])  # shape: [c, h, w]
                    
                mask = cv2.imread(os.path.join(self.dataset_path, "Train_Mask", name + '.png'), 0) 
                mask_spec = np.expand_dims(mask, axis=0)
                mask_spec = np.broadcast_to(mask_spec, (hyper.shape[0], hyper.shape[1], hyper.shape[2]))
                hyper = np.where(mask_spec > 0, 128 / 255.0, hyper)
                self.mat_dict[name] = hyper
                
                rgb = Image.open(os.path.join(self.dataset_path, "JPEGImages", name + ".jpg"))
                rgb = np.array(rgb, dtype=np.uint8)
                mask_rgb = np.expand_dims(mask, axis=2)
                rgb = np.where(mask_rgb > 0, 128, rgb)
                self.rgb_dict[name] = rgb
                
                label = Image.open(os.path.join(self.dataset_path, "SegmentationClass", name + ".png"))
                self.label_dict[name] = label
                
            except:
                print(name)
                continue
            if len(self.mat_dict) % 20 == 0:
                print("Load %d mats!"%len(self.mat_dict))

           
    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name            = annotation_line.strip()
        
        # load img and label
        label = self.label_dict[name]
        hyper = self.mat_dict[name] # shape: [c, h, w]
        rgb = self.rgb_dict[name] # shape: [h, w, c]

        # input image [c, h, w]
        hyper, rgb, label = self.get_random_data(hyper=hyper, rgb=rgb, label=label, input_shape=self.input_shape, jitter=.2, random=self.train)
        # output image [h, w, c]
        
        hyper = np.transpose(np.array(hyper / 255.0, dtype=np.float32), [2, 0, 1])
        rgb = np.transpose(np.array(rgb / 255.0, dtype=np.float32), [2, 0, 1])
        label = np.array(label)
        label[label >= self.num_classes] = self.num_classes
        
        seg_labels = np.eye(self.num_classes + 1)[label.reshape([-1])]
        seg_labels = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))
        return hyper, rgb, label, seg_labels
        
    def __len__(self):
        return self.length

    # @staticmethod
    def get_random_data(self, hyper, rgb, label, input_shape, jitter=.3, random=True):
        # hyper shape: [25, 217, 409] is numpy array
        label = Image.fromarray(np.array(label))
        hyper = np.transpose(hyper, [1, 2, 0])
        
        # norm
        hyper = (hyper - np.min(hyper)) / (np.max(hyper) - np.min(hyper))
        hyper = (hyper * 255).astype(np.uint8)
        
        ih, iw, ic = hyper.shape
        h, w = input_shape
        if not random:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2
            resizeed_hyper = resize_multichannel_image(hyper, (nh, nw))
            new_hyper = np.ones((h, w, ic), dtype=np.uint8) * 128
            new_hyper[dy:dy + nh, dx:dx + nw, :] = resizeed_hyper
            
            new_rgb = np.ones((h, w, 3), dtype=np.uint8) * 128
            new_rgb[dy:dy + nh, dx:dx + nw, :] = resize_multichannel_image(rgb, (nh, nw))
            
            label = label.resize((nw, nh), Image.NEAREST)
            new_label = Image.new('L', (w, h), 0)
            new_label.paste(label, (dx, dy))
            return new_hyper, new_rgb, new_label
            # new_label.save("./test_label.png")
            # cv2.imwrite("./test_image.png", new_image[:, :, 1])
        #------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        #------------------------------------------#
        new_ar = iw / ih * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
        scale = rand(0.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        hyper = resize_multichannel_image(hyper, (nh, nw))
        rgb = resize_multichannel_image(rgb, (nh, nw))
        label = label.resize((nw, nh), Image.NEAREST)
        
        flip = rand() < .5
        if flip:
            if rand() < .5:
                hyper = hyper[:, ::-1, :]
                rgb = rgb[:, ::-1, :]
                label = label.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                hyper = hyper[::-1, :, :]
                rgb = rgb[::-1, :, :]
                label = label.transpose(Image.FLIP_TOP_BOTTOM)
            
        dx = int(rand(0, w-nw))
        dy = int(rand(0, h-nh))
        new_hyper = np.ones((h, w, ic), dtype=hyper.dtype) * 128
        new_rgb = np.ones((h, w, 3), dtype=rgb.dtype) * 128
        oh, ow, ox, oy = 0, 0, 0, 0
        
        # caculate the new size 
        ny, oy = (0, abs(dy)) if dy < 0 else (dy, 0)
        th, oh = (h, h - dy) if dy + nh > h else (dy + nh, nh)
        
        nx, ox = (0, abs(dx)) if dx < 0 else (dx, 0)
        tw, ow = (w, w - dx) if dx + nw > w else (dx + nw, nw)

        new_hyper[ny:th, nx:tw, :] = hyper[oy:oh, ox:ow, :]
        new_rgb[ny:th, nx:tw, :] = rgb[oy:oh, ox:ow, :]
        new_label = Image.new('L', (w, h), 0)
        new_label.paste(label, (dx, dy))
        return new_hyper, new_rgb, new_label

class UnetDatasetMat(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path):
        super(UnetDatasetMat, self).__init__()
        self.annotation_lines   = annotation_lines
        self.length             = len(annotation_lines)
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.train              = train
        self.dataset_path       = dataset_path
        
        self.mat_dict = {}
        self.label_dict = {}
        # load mat
        for i in self.annotation_lines:
            name = i.split()[0]
            try:
                with h5py.File(os.path.join(self.dataset_path, "Train_Spec", name + '.mat'), 'r') as mat:
                    hyper = np.float32(np.array(mat['cube']))
                    hyper = np.transpose(hyper, [2, 1, 0])  # shape: [c, h, w]
                mask = cv2.imread(os.path.join(self.dataset_path, "Train_Mask", name + '.png'), 0) 
                mask = np.expand_dims(mask, axis=0)
                mask = np.broadcast_to(mask, (hyper.shape[0], hyper.shape[1], hyper.shape[2]))
                hyper = np.where(mask > 0, 128 / 255.0, hyper)
                self.mat_dict[name] = hyper
                
                label = Image.open(os.path.join(self.dataset_path, "SegmentationClass", name + ".png"))
                self.label_dict[name] = label
                
            except:
                print(name)
                continue
            if len(self.mat_dict) % 20 == 0:
                print("Load %d mats!"%len(self.mat_dict))

           
    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name            = annotation_line.strip()
        
        # load img and label
        label = self.label_dict[name]
        hyper = self.mat_dict[name] # shape: [c, h, w]
        
        # input image [c, h, w]
        hyper, label = self.get_random_data(hyper=hyper, label=label, input_shape=self.input_shape, jitter=.2, random=self.train)
        # output image [h, w, c]
        
        hyper = np.transpose(np.array(hyper / 255.0, dtype=np.float32), [2, 0, 1])

        label = np.array(label)
        label[label >= self.num_classes] = self.num_classes
        
        seg_labels = np.eye(self.num_classes + 1)[label.reshape([-1])]
        seg_labels = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))
        return hyper, label, seg_labels
        
    def __len__(self):
        return self.length

    # @staticmethod
    def get_random_data(self, hyper, label, input_shape, jitter=.3, random=True):
        # hyper shape: [25, 217, 409] is numpy array
        label = Image.fromarray(np.array(label))
        hyper = np.transpose(hyper, [1, 2, 0])
        
        # norm
        hyper = (hyper - np.min(hyper)) / (np.max(hyper) - np.min(hyper))
        hyper = (hyper * 255).astype(np.uint8)
        
        ih, iw, ic = hyper.shape
        h, w = input_shape
        if not random:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2
            resizeed_hyper = resize_multichannel_image(hyper, (nh, nw))
            new_hyper = np.ones((h, w, ic), dtype=np.uint8) * 128
            new_hyper[dy:dy + nh, dx:dx + nw, :] = resizeed_hyper
            
            label = label.resize((nw, nh), Image.NEAREST)
            new_label = Image.new('L', (w, h), 0)
            new_label.paste(label, (dx, dy))
            return new_hyper, new_label
            # new_label.save("./test_label.png")
            # cv2.imwrite("./test_image.png", new_image[:, :, 1])
        #------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        #------------------------------------------#
        new_ar = iw / ih * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
        scale = rand(0.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        hyper = resize_multichannel_image(hyper, (nh, nw))
        label = label.resize((nw, nh), Image.NEAREST)
        
        flip = rand() < .5
        if flip:
            if rand() < .5:
                hyper = hyper[:, ::-1, :]
                label = label.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                hyper = hyper[::-1, :, :]
                label = label.transpose(Image.FLIP_TOP_BOTTOM)
            
        dx = int(rand(0, w-nw))
        dy = int(rand(0, h-nh))
        new_hyper = np.ones((h, w, ic), dtype=hyper.dtype) * 128
        oh, ow, ox, oy = 0, 0, 0, 0
        
        # caculate the new size 
        ny, oy = (0, abs(dy)) if dy < 0 else (dy, 0)
        th, oh = (h, h - dy) if dy + nh > h else (dy + nh, nh)
        
        nx, ox = (0, abs(dx)) if dx < 0 else (dx, 0)
        tw, ow = (w, w - dx) if dx + nw > w else (dx + nw, nw)

        new_hyper[ny:th, nx:tw, :] = hyper[oy:oh, ox:ow, :]
        new_label = Image.new('L', (w, h), 0)
        new_label.paste(label, (dx, dy))
        return new_hyper, new_label

class UnetDatasetRGB(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path):
        super(UnetDatasetRGB, self).__init__()
        self.annotation_lines   = annotation_lines
        self.length             = len(annotation_lines)
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.train              = train
        self.dataset_path       = dataset_path
        
        # self.mat_dict = {}
        self.rgb_dict = {}
        self.label_dict = {}
        # load mat
        for i in self.annotation_lines:
            name = i.split()[0]
            try:
                # with h5py.File(os.path.join(self.dataset_path, "Train_Spec", name + '.mat'), 'r') as mat:
                #     hyper = np.float32(np.array(mat['cube']))
                #     hyper = np.transpose(hyper, [2, 1, 0])  # shape: [c, h, w]
                mask = cv2.imread(os.path.join(self.dataset_path, "Train_Mask", name + '.png'), 0) 
                mask = np.expand_dims(mask, axis=2)
                
                # self.mat_dict[name] = hyper
                
                rgb = Image.open(os.path.join(self.dataset_path, "JPEGImages", name + ".jpg"))
                rgb = np.array(rgb, dtype=np.uint8)
                
                mask = np.broadcast_to(mask, (rgb.shape[0], rgb.shape[1], rgb.shape[2]))
                rgb = np.where(mask > 0, 128, rgb)
                self.rgb_dict[name] = rgb
                
                label = Image.open(os.path.join(self.dataset_path, "SegmentationClass", name + ".png"))
                self.label_dict[name] = label
                
            except:
                print(name)
                continue
           
    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name            = annotation_line.strip()
        
        # load img and label
        label = self.label_dict[name]
        # hyper = self.mat_dict[name] # shape: [c, h, w]
        rgb = self.rgb_dict[name] # shape: [h, w, c]
        
        # input image [c, h, w]
        rgb, label = self.get_random_data(rgb=rgb, label=label, input_shape=self.input_shape, jitter=.2, random=self.train)
        # output image [h, w, c]
        
        rgb = np.transpose(np.array(rgb / 255.0, dtype=np.float32), [2, 0, 1])
        label = np.array(label)
        label[label >= self.num_classes] = self.num_classes
        
        seg_labels = np.eye(self.num_classes + 1)[label.reshape([-1])]
        seg_labels = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))
        return rgb, label, seg_labels
        
    def __len__(self):
        return self.length

    # @staticmethod
    def get_random_data(self, rgb, label, input_shape, jitter=.3, random=True):

        label = Image.fromarray(np.array(label))

        ih, iw, ic = rgb.shape
        h, w = input_shape
        if not random:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2

            new_rgb = np.ones((h, w, 3), dtype=np.uint8) * 128
            new_rgb[dy:dy + nh, dx:dx + nw, :] = resize_multichannel_image(rgb, (nh, nw))
            
            label = label.resize((nw, nh), Image.NEAREST)
            new_label = Image.new('L', (w, h), 0)
            new_label.paste(label, (dx, dy))
            return new_rgb, new_label
            # new_label.save("./test_label.png")
            # cv2.imwrite("./test_image.png", new_image[:, :, 1])
        #------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        #------------------------------------------#
        new_ar = iw / ih * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
        scale = rand(0.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)

        rgb = resize_multichannel_image(rgb, (nh, nw))
        label = label.resize((nw, nh), Image.NEAREST)
        
        flip = rand() < .5
        if flip:
            if rand() < .5:
                rgb = rgb[:, ::-1, :]
                label = label.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                rgb = rgb[::-1, :, :]
                label = label.transpose(Image.FLIP_TOP_BOTTOM)
            
        dx = int(rand(0, w-nw))
        dy = int(rand(0, h-nh))

        new_rgb = np.ones((h, w, 3), dtype=rgb.dtype) * 128
        oh, ow, ox, oy = 0, 0, 0, 0
        
        # caculate the new size 
        ny, oy = (0, abs(dy)) if dy < 0 else (dy, 0)
        th, oh = (h, h - dy) if dy + nh > h else (dy + nh, nh)
        
        nx, ox = (0, abs(dx)) if dx < 0 else (dx, 0)
        tw, ow = (w, w - dx) if dx + nw > w else (dx + nw, nw)

        new_rgb[ny:th, nx:tw, :] = rgb[oy:oh, ox:ow, :]
        new_label = Image.new('L', (w, h), 0)
        new_label.paste(label, (dx, dy))
        return new_rgb, new_label

def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a
    
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
    
# DataLoader中collate_fn使用
def two_stream_dataset_collate(batch):
    images      = []
    rgbs        = []
    pngs        = []
    seg_labels  = []
    for img, rgb, png, labels in batch:
        images.append(img)
        rgbs.append(rgb)
        pngs.append(png)
        seg_labels.append(labels)
    images      = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    rgbs        = torch.from_numpy(np.array(rgbs)).type(torch.FloatTensor)
    pngs        = torch.from_numpy(np.array(pngs)).long()
    seg_labels  = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
    return images, rgbs, pngs, seg_labels

def image_dataset_collate(batch):
    images      = []
    pngs        = []
    seg_labels  = []
    for img, png, labels in batch:
        images.append(img)
        pngs.append(png)
        seg_labels.append(labels)
    images      = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    pngs        = torch.from_numpy(np.array(pngs)).long()
    seg_labels  = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
    return images, pngs, seg_labels

if __name__ == "__main__":
    from torch.utils.data.dataloader import DataLoader
    dataset_root = "/root/spectral-segmentation/datasets/spectral-dataset-multi-nature"
    input_shape = (416,416)
    num_classes = 14
    with open(os.path.join(dataset_root, "ImageSets/Segmentation/trainval.txt"),"r") as f:
        train_lines = f.readlines()
    # dataset = UnetDatasetMat(train_lines    , input_shape, num_classes, True, dataset_root)
    # dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=1, collate_fn=image_dataset_collate)
    dataset = UnetDatasetTwoStream(train_lines    , input_shape, num_classes, True, dataset_root)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=1, collate_fn=two_stream_dataset_collate)
    
    # for i, (images, labels, seg_labels) in enumerate(dataloader):

    
    
    
    

    
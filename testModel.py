import torch

from nets.twoStream import TwoStreamModel
from nets.unet import Unet
from nets.efficientvit import EfficientHybrid, EfficientViTs_m0
from nets.ESPFNet import ESPFNet
from nets.CFT import CFT
from nets.twoStream import TwoStreamModel
import numpy as np
import time
import tqdm

model_list = [ESPFNet(11, False, 'resnet18'), 
            #   CFT(11, False, 'resnet18'),
            #   TwoStreamModel(11, False, 'resnet18'),
              EfficientHybrid(11, False, 'resnet18', **EfficientViTs_m0)]

input1 = torch.randn(1, 25, 416, 416).cuda()
input2 = torch.randn(1, 3, 416, 416).cuda()

throuthout = []
for model in model_list:
    model.cuda()
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        for i in tqdm.tqdm(range(1000)):
            _ = model(input1, input2)
        t1 = time.time()
    throuthout.append(1000/(t1-t0))

for i in range(len(model_list)):
    print(model_list[i].__class__.__name__, "throuthout: ", throuthout[i])
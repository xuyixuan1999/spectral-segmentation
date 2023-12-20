import os
import datetime
from PIL import Image
from tqdm import tqdm

from unet import Unet, Fusion
from utils.utils_metrics import compute_mIoU, show_results

'''
进行指标评估需要注意以下几点：
1、该文件生成的图为灰度图，因为值比较小，按照JPG形式的图看是没有显示效果的，所以看到近似全黑的图是正常的。
2、该文件计算的是验证集的miou，当前该库将测试集当作验证集使用，不单独划分测试集
3、仅有按照VOC格式数据训练的模型可以利用这个文件进行miou的计算。
'''
if __name__ == "__main__":
    config = {
        "model_name"         : 'unet',
        #-------------------------------------------------------------------#
        #   model_path指向logs文件夹下的权值文件
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表miou较高，仅代表该权值在验证集上泛化性能较好。
        #-------------------------------------------------------------------#
        "model_path"    : 'logs/2023_12_20_06_54_05_rec_3band_multi/best_epoch_weights.pth',
        #--------------------------------#
        #   所需要区分的类的个数+1
        #--------------------------------#
        "num_classes"   : 14,
        #--------------------------------#
        #   所使用的的主干网络：vgg、resnet50   
        #--------------------------------#
        "backbone"      : "resnet18",
        #--------------------------------#
        #   输入图片的大小
        #--------------------------------#
        "input_shape"   : [416, 416],
        #-------------------------------------------------#
        #   mix_type参数用于控制检测结果的可视化方式
        #
        #   mix_type = 0的时候代表原图与生成的图进行混合
        #   mix_type = 1的时候代表仅保留生成的图
        #   mix_type = 2的时候代表仅扣去背景，仅保留原图中的目标
        #-------------------------------------------------#
        "mix_type"      : 0,
        #--------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        #--------------------------------#
        "cuda"          : True,
    }
    #---------------------------------------------------------------------------#
    #   miou_mode用于指定该文件运行时计算的内容
    #   miou_mode为0代表整个miou计算流程，包括获得预测结果、计算miou。
    #   miou_mode为1代表仅仅获得预测结果。
    #   miou_mode为2代表仅仅计算miou。
    #---------------------------------------------------------------------------#
    miou_mode       = 0
    #------------------------------#
    #   分类个数+1、如2+1
    #------------------------------#
    num_classes     = 14
    #--------------------------------------------#
    #   区分的种类，和json_to_dataset里面的一样
    #--------------------------------------------#
    '''
    _background_
    desert camouflage net
    desert two-color camouflage net
    desert three-color camouflage net
    desert grass camouflage net
    anti-infrared camouflage net
    forest three-color grass camouflage net
    forest two-color glass camouflage net
    forest two-color optical camouflage net
    forest three-color optical camouflage net
    forest digital camouflage net
    desert camouflage people
    forest camouflage people
    camouflage plate
    '''
    name_classes    = ["_background", "desert camouflage net", "desert two-color camouflage net", "desert three-color camouflage net", "desert grass camouflage net", "anti-infrared camouflage net", "forest three-color grass camouflage net", "forest two-color glass camouflage net", "forest two-color optical camouflage net", "forest three-color optical camouflage net", "forest digital camouflage net", "desert camouflage people", "forest camouflage people", "camouflage plate"]

    #-------------------------------------------------------#
    #   指向VOC数据集所在的文件夹
    #   默认指向根目录下的VOC数据集
    #-------------------------------------------------------#
    VOCdevkit_path  = './datasets/spectral-dataset-multi'

    image_ids       = open(os.path.join(VOCdevkit_path, "ImageSets/Segmentation/val.txt"),'r').read().splitlines() 
    gt_dir          = os.path.join(VOCdevkit_path, "SegmentationClass/")
    time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S') 
    miou_out_path   = os.path.join('output', time_str)
    pred_dir        = os.path.join(miou_out_path, 'detection-results')

    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
            
        print("Load model.")
        # unet = Unet()
        pre_net = Fusion(**config)
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            # image_path  = os.path.join(VOCdevkit_path, "JPEGImages/"+image_id+".jpg")
            # image       = Image.open(image_path)
            # image       = unet.get_miou_png(image)
            image         = pre_net.get_detection_rgb(VOCdevkit_path, image_id)
            image.save(os.path.join(pred_dir, image_id + ".png"))
        print("Get predict result done.")

    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes, name_classes)  # 执行计算mIoU的函数
        print("Get miou done.")
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)
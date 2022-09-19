from dataset import dataset_full
import os
import numpy as np
import glob
from utils import *
from scipy import io
import torch
from torch.nn import Module
from torch import nn
from torch.autograd import Variable
from skimage.io import imsave
from model.amp import AMP_net_Deblock


def load_sampling_matrix(CS_ratio):
    path = "dataset/sampling_matrix"
    data = io.loadmat(os.path.join(path, str(CS_ratio) + '.mat'))['sampling_matrix']
    return data


def get_val_result(model,CS_ratio,phaseNum,save_path, is_cuda=True):
    with torch.no_grad():
        test_set_path = "dataset/bsds500/test"
        # test_set_path = "dataset/Set11"
        test_set_path = glob.glob(test_set_path + '/*.tif')
        ImgNum = len(test_set_path)
        PSNR_All = np.zeros([1, ImgNum], dtype=np.float32)
        SSIM_All = np.zeros([1, ImgNum], dtype=np.float32)



        for img_no in range(ImgNum):

            imgName = test_set_path[img_no]

            # print(img_no)
            [Iorg, row, col, Ipad, row_new, col_new] = imread_CS_py(imgName)
            Icol = img2col_py(Ipad, 33) / 255.0  
            Ipad /= 255.0
            if is_cuda:
                inputs = Variable(torch.from_numpy(Ipad.astype('float32')).cuda())
            else:
                inputs = Variable(torch.from_numpy(Ipad.astype('float32')))

            inputs = torch.unsqueeze(torch.unsqueeze(inputs, dim=0), dim=0)
            outputs= model(inputs,phaseNum)
            output = torch.squeeze(outputs)
            if is_cuda:
                output = output.cpu().data.numpy()
            else:
                output = output.data.numpy()
            images_recovered = output[0:row, 0:col] * 255
            aaa = images_recovered.astype(int)
            bbb = aaa < 0
            aaa[bbb] = 0
            bbb = aaa > 255
            aaa[bbb] = 255
            rec_PSNR = psnr(aaa, Iorg)  
            PSNR_All[0, img_no] = rec_PSNR
            rec_SSIM = compute_ssim(aaa, Iorg)
            SSIM_All[0, img_no] = rec_SSIM
            imgname_for_save = ((imgName.split('/')[-1]).split('.'))[0]
            imsave(os.path.join(save_path,imgname_for_save+'_'+str(rec_PSNR)+'_'+str(rec_SSIM)+'.jpg'),aaa)

    return np.mean(PSNR_All), np.mean(SSIM_All)


if __name__ == "__main__":
    model_name = "AMP_Net_K"
    CS_ratios = [10]
    Phases = [9]
    phase = 9
	
    save_path = "./results/generated_images"

	
    for CS_ratio in CS_ratios:
        for phase in Phases:
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            sub_save_path = os.path.join(save_path, str(CS_ratio))
            if not os.path.exists(sub_save_path):
                os.mkdir(sub_save_path)
            sub_save_path = os.path.join(save_path, str(phase))
            if not os.path.exists(sub_save_path):
                os.mkdir(sub_save_path)

            path = os.path.join("results",model_name,str(CS_ratio),str(phase),"best_model.pkl")

            A = load_sampling_matrix(CS_ratio)
            model = AMP_net_Deblock(phase,A)
            model.cuda()
            model.load_state_dict(torch.load(path))
            print("Start")
            one_psnr, one_ssim = get_val_result(model,CS_ratio,phase,sub_save_path, is_cuda=True)  # test AMP_net

            print(one_psnr,"dB",one_ssim)

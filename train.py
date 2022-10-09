from dataset import dataset_full,dataset
import os
import numpy as np
import glob
from utils import *
from scipy import io
import torch
from torch.nn import Module
from torch import nn
from torch.autograd import Variable
from model.rnn_query import AMP_net_Deblock

"""
No mask training, no deblocking
AMP-Net-K
"""


def train(model, opt, train_loader, epoch, batch_size, CS_ratio,PhaseNum):
    model.train()
    n = 0
    for data,_ in train_loader:
        n = n + 1
        opt.zero_grad()
        data = torch.unsqueeze(data,dim=1)
        data = Variable(data.float().cuda())
        data = Variable(data.float())
        outputs= model(data,PhaseNum)

        # loss_all = compute_loss(outputs,data)
        # loss = get_final_loss(loss_all)
        # loss = torch.mean((outputs[-1]-target)**2)

        loss = torch.mean((outputs-data)**2)
        loss.backward()
        opt.step()
        if n % 25 == 0:
            output = "CS_ratio: %d [%02d/%02d] loss: %.4f " % (
            CS_ratio, epoch, batch_size * n, loss.data.item())
            print(output)


def get_val_result(model,PhaseNum, is_cuda=True):
    model.eval()
    with torch.no_grad():
        test_set_path = "dataset/bsds500/val"
        test_set_path = glob.glob(test_set_path + '/*.tif')
        ImgNum = len(test_set_path)  
        PSNR_All = np.zeros([1, ImgNum], dtype=np.float32)
        model.eval()
        for img_no in range(ImgNum):
            imgName = test_set_path[img_no]  

            [Iorg, row, col, Ipad, row_new, col_new] = imread_CS_py(imgName)
            Icol = img2col_py(Ipad, 33) / 255.0
            Ipad /= 255.0
            # Img_input = np.dot(Icol, Phi_input)
            # Img_output = Icol
            if is_cuda:
                inputs = Variable(torch.from_numpy(Ipad.astype('float32')).cuda())
            else:
                inputs = Variable(torch.from_numpy(Ipad.astype('float32')))

            inputs = torch.unsqueeze(torch.unsqueeze(inputs,dim=0),dim=0)
            outputs = model(inputs, PhaseNumber)
            outputs = torch.squeeze(outputs)
            if is_cuda:
                outputs = outputs.cpu().data.numpy()
            else:
                outputs = outputs.data.numpy()

            images_recovered = outputs[0:row,0:col]
            # images_recovered = col2im_CS_py(output, row, col, row_new, col_new)
            rec_PSNR = psnr(images_recovered * 255, Iorg)
            PSNR_All[0, img_no] = rec_PSNR

    out = np.mean(PSNR_All)
    return out


def load_sampling_matrix(CS_ratio):
    path = "dataset/sampling_matrix"
    data = io.loadmat(os.path.join(path, str(CS_ratio) + '.mat'))['sampling_matrix']
    return data


def get_Q(data_set,A):
    A = torch.from_numpy(A)
    n = 0
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=len(data_set),
                                shuffle=True, num_workers=2)
    for data, target in data_loader:
        data = torch.transpose(torch.reshape(data, [-1, 33 * 33]), 0, 1)
        target = torch.transpose(torch.reshape(target, [-1, 33 * 33]), 0, 1)
        y = torch.matmul(A.float(),data.float())
        x = target.float()
        if n==0:
            ys = y
            Xs = x
            n = 1
        else:
            ys = torch.cat([ys,y],dim=1)
            Xs = torch.cat([Xs,x],dim=1)
    Q = torch.matmul(torch.matmul(Xs,torch.transpose(ys,0,1)),
                     torch.inverse(torch.matmul(ys, torch.transpose(ys, 0, 1))))
    return Q.numpy()


if __name__ == "__main__":
    is_cuda = True
    CS_ratio = 25  # 4, 10, 25, 30, 40, 50
    CS_ratios = [10]
    # n_output = 1089
    PhaseNumbers = [6]
    # PhaseNumber = 9
    # nrtrain = 88912
    learning_rate = 1e-4
    EpochNum = 400
    batch_size = 32
    results_saving_path = "results"

    net_name = "AMP_Net_K"

    if not os.path.exists(results_saving_path):
        os.mkdir(results_saving_path)

    results_saving_path = os.path.join(results_saving_path, net_name)
    if not os.path.exists(results_saving_path):
        os.mkdir(results_saving_path)

    print('Load Data...')  

    train_dataset = dataset_full(train=True, transform=None,
                            target_transform=None)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=2)
    for CS_ratio in CS_ratios:
        for PhaseNumber in PhaseNumbers:
            A = load_sampling_matrix(CS_ratio)
            model = AMP_net_Deblock(PhaseNumber,A)

            if True:
                model_name = "AMP_Net_K"
                path = os.path.join("results", model_name, str(CS_ratio), str(PhaseNumber), "best_model.pkl")
                model.load_state_dict(torch.load(path))

            opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
            model.cuda()
            sub_path = os.path.join(results_saving_path, str(CS_ratio))

            if not os.path.exists(sub_path):
                os.mkdir(sub_path)
            sub_path = os.path.join(sub_path, str(PhaseNumber))

            if not os.path.exists(sub_path):
                os.mkdir(sub_path)
            best_psnr = 0
            for epoch in range(1, EpochNum + 1):
                train(model, opt, train_loader, epoch, batch_size, CS_ratio,PhaseNumber)
                one_psnr = get_val_result(model, PhaseNumber)
                print_str = "CS_ratio: %d Phase: %d epoch: %d  psnr: %.4f" % (CS_ratio, PhaseNumber, epoch, one_psnr)
                print(print_str)

                output_file = open(sub_path + "/log_PSNR.txt", 'a')
                output_file.write("PSNR: %.4f\n" % (one_psnr))
                output_file.close()

                if one_psnr > best_psnr:
                    best_psnr = one_psnr
                    output_file = open(sub_path + "/log_PSNR_best.txt", 'a')
                    output_file.write("PSNR: %.4f\n" % (best_psnr))
                    output_file.close()
                    torch.save(model.state_dict(), sub_path + "/best_model.pkl")

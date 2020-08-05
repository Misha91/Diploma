import network
import dataset

import torch
import torch.optim as optim
import torch.nn.functional as F
import os
from shutil import copyfile
import numpy as np
import torchvision.transforms as tfms
import torchvision as tv
train_transform = tfms.Compose([tfms.Resize((128,128)),
                                tfms.ToTensor()])


ImageNette_for_statistics = tv.datasets.ImageFolder(os.getcwd()+'/ds/train/',
                                    transform = train_transform)

def get_dataset_statistics(dataset: torch.utils.data.Dataset):
    '''Function, that calculates mean and std of a dataset (pixelwise)
    Return:
        tuple of Lists of floats. len of each list should equal to number of input image/tensor channels
    '''

    mean = [0., 0., 0.]
    mean_cnt = 0
    std = [0., 0., 0.]
    std_cnt = 0
    data = [[], [], []]

    for p,_ in dataset:

        data[0].append(torch.flatten(p[0]))
        data[1].append(torch.flatten(p[1]))
        data[2].append(torch.flatten(p[2]))
        #break
    data_0 = torch.cat(data[0], dim=0)
    data_1 = torch.cat(data[1], dim=0)
    data_2 = torch.cat(data[2], dim=0)

    mean[0] = float(torch.mean(data_0))
    std[0] = float(torch.std(data_0))

    mean[1] = float(torch.mean(data_1))
    std[1] = float(torch.std(data_1))

    mean[2] = float(torch.mean(data_2))
    std[2] = float(torch.std(data_2))

    return mean, std




if __name__ == '__main__':

    print(get_dataset_statistics(ImageNette_for_statistics))

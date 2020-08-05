from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os


class myDataset(Dataset):
    def __init__(self, path, seen=0, transform=None):
        if not os.path.exists(path):
            os.makedirs(path)
        self.root_dir = path
        self.list = os.listdir(path+'/images')
        self.size = len(os.listdir(path+'/images'))
        self.transform = transform

    def __getitem__(self, idx):
        im_file = '/images/%08.f.jpg' % idx
        lab_file = '/labels/%08.f.png' % idx

        im = Image.open(self.root_dir + im_file)
        im = np.array(im)
        lab = Image.open(self.root_dir + lab_file)
        width, height = lab.size

        if height == 16:
            lab = lab.resize((26,13))
        elif height == 40:
            lab = lab.resize((80,45))
        lab = np.array(lab)
        lab = lab[:,:,0]
        lab[lab>0] = 1
        sample = {'image': im, 'label': lab}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.size

import os
from email.contentmanager import raw_data_manager
from email.policy import strict

import cv2
import torch.nn
from torch.optim.adamw import AdamW
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import mwcnn_model
from mwcnn_model import MWRCAN
from mwcnn_model import Discriminator
import imageio.v2 as imageio
import numpy as np
from mssim import MSSSIM
from tqdm import tqdm
from torch.optim import Adam
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator

dataset_path = '../dataset/'
ckpt_path = '../checkpoints/'
result_path = '../results/'

os.makedirs(result_path, exist_ok=True)

class TrainDataset(Dataset):
    def __init__(self, data_dir, test=False, transform=None):
        self.input_dir = os.path.join(data_dir, 'raw')
        self.label_dir = os.path.join(data_dir, 'pngs')
        self.transform = transform
        self.test = test

        lst_input = os.listdir(self.input_dir)
        lst_label = os.listdir(self.label_dir)

        lst_input.sort()
        lst_label.sort()

        train_ratio = int(len(lst_input) * 0.7)
        if not test:
            self.lst_input = lst_input[:train_ratio]
            self.lst_label = lst_label[:train_ratio]
            self.dataset_size = len(self.lst_input)
        else:
            self.lst_input = lst_input[train_ratio:]
            self.dataset_size = len(self.lst_input)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        # raw data
        raw_data = np.load(os.path.join(self.input_dir, self.lst_input[idx]))

        raw_data = np.log(raw_data)

        r_0 = raw_data[0::2, 1::2]
        g_0 = raw_data[0::2, 0::2]
        g_1 = raw_data[1::2, 1::2]
        b_0 = raw_data[1::2, 0::2]

        raw_data = np.stack((g_0, r_0, b_0, g_1))
        # raw_data = (raw_data - raw_data.mean()) / raw_data.std()
        raw_data = np.log(1 + raw_data)
        raw_data = raw_data / np.log(1 + 2**26)
        raw_data = raw_data.transpose(1, 2, 0)

        if not self.test:
            # png data
            png_data = np.array(imageio.imread(os.path.join(self.label_dir, self.lst_label[idx])))
            # png_data = png_data.transpose(1, 2, 0)

            if self.transform:
                raw_data = self.transform(raw_data)
                png_data = self.transform(png_data)

            return raw_data, png_data
        else:
            if self.transform:
                raw_data = self.transform(raw_data)

            return raw_data


def test():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # network
    generator = MWRCAN().to(device)
    generator.load_state_dict(torch.load(os.path.join(ckpt_path, 'mwrcan_epoch_300.pth')), strict=True)
    generator.eval()

    # transform
    transform = T.Compose([
        T.ToTensor(),
        # T.Normalize()
    ])

    # load data
    # dataset_train = TrainDataset(data_dir=dataset_path, test=False, transform=transform)
    dataset_test = TrainDataset(data_dir=dataset_path, test=True, transform=transform)
    # train_loader = DataLoader(dataset_train, batch_size=1, shuffle=True, num_workers=1)
    test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=1)

    # raw_test = dataset_test[0]
    # print(f'{raw_test.shape}')
    # print(f'{raw_test}')
    # print(f'{raw_test.dtype}')

    # summary(generator, (4, 768, 960))

    # test network
    with torch.no_grad():
        for idx, (raw_image) in enumerate(test_loader):
            raw_image = raw_image.to(device)

            output = generator(raw_image.detach())
            output = np.asarray(torch.squeeze(output.float().detach().cpu())).transpose(1,2,0)
            output = np.clip(output*255, 0, 255).astype(np.uint8)[...,::-1]

            cv2.imwrite(f'{result_path}/{idx}.png', output)
            print(f'image saved: {result_path}/{idx}.png')

if __name__ == '__main__':
    test()
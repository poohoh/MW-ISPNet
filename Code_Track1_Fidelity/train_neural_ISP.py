import os
from email.contentmanager import raw_data_manager

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

dataset_path = '../dataset/'
save_model_path = '../checkpoints/'
log_path = '../log/'

os.makedirs(save_model_path, exist_ok=True)
os.makedirs(log_path, exist_ok=True)

class TrainDataset(Dataset):
    def __init__(self, data_dir, test=False, transform=None):
        self.input_dir = os.path.join(data_dir, 'raw')
        self.label_dir = os.path.join(data_dir, 'pngs')
        self.transform = transform

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
            self.lst_label = lst_label[train_ratio:]
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

        # png data
        png_data = np.asarray(imageio.imread(os.path.join(self.label_dir, self.lst_label[idx])))
        # png_data = png_data.transpose(1, 2, 0)

        if self.transform:
            raw_data = self.transform(raw_data)
            png_data = self.transform(png_data)

        return raw_data, png_data


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # network
    generator = MWRCAN().to(device)

    # tensorboard writer
    writer = SummaryWriter(log_dir=log_path)

    # optimizer
    optimizer = Adam(generator.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150, 200], gamma=0.5)

    # transform
    transform = T.Compose([
        T.ToTensor(),
        # T.Normalize()
    ])

    # load data
    dataset_train = TrainDataset(data_dir=dataset_path, test=False, transform=transform)
    dataset_test = TrainDataset(data_dir=dataset_path, test=True, transform=transform)
    train_loader = DataLoader(dataset_train, batch_size=1, shuffle=True, num_workers=1)
    test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=1)

    # raw_train, png_train = dataset_train[0]
    # print(f'{raw_train.shape}')
    # print(f'{png_train.shape}')

    # loss
    L1_loss = torch.nn.L1Loss()
    MS_SSIM = MSSSIM()

    # summary(generator, (4, 768, 960))

    # train network
    for epoch in tqdm(range(300)):
        torch.cuda.empty_cache()
        generator.to(device).train()

        for idx, (input, label) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            input = input.to(device)
            label = label.to(device)

            output = generator(input)

            loss_l1 = L1_loss(output, label)

            loss_ssim = MS_SSIM(output, label)

            total_loss = loss_l1 + (1 - loss_ssim) * 0.15

            writer.add_scalar(f'train/ssim', loss_ssim, epoch * len(train_loader) + idx)
            writer.add_scalar(f'train/l1_loss', loss_l1, epoch * len(train_loader) + idx)
            writer.add_scalar(f'train/total_loss', total_loss, epoch * len(train_loader) + idx)

            total_loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            # Save the model that corresponds to the current epoch
            generator.eval().cpu()
            torch.save(generator.state_dict(), os.path.join(save_model_path, "mwrcan_epoch_" + str(epoch + 1) + ".pth"))

        scheduler.step()

if __name__ == '__main__':
    train()
import torch
import torch.nn as nn
import numpy as np


class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_shape=(1, 28, 28)):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_shape=(1, 28, 28)):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity


class DCGANGenerator(nn.Module):
    def __init__(self, latent_dim=100, channels=1, img_size=28):
        super(DCGANGenerator, self).__init__()
        self.img_size = img_size
        self.channels = channels
        
        self.init_size = img_size // 4  # 28 -> 7
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),  # 7 -> 14
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),  # 14 -> 28
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class DCGANDiscriminator(nn.Module):
    def __init__(self, channels=1, img_size=28):
        super(DCGANDiscriminator, self).__init__()
        
        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),  # 28 -> 14
            *discriminator_block(16, 32),                  # 14 -> 7
            *discriminator_block(32, 64),                  # 7 -> 4 (向上取整)
            *discriminator_block(64, 128),                 # 4 -> 2
        )

        # 修正维度计算：28 -> 14 -> 7 -> 4 -> 2
        # 最终特征图大小为 2x2，通道数为128
        ds_size = 2  # 实际的最终特征图大小
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity


def create_models(device='cpu', latent_dim=100, img_shape=(1, 28, 28), model_type='vanilla'):
    """创建并初始化模型的便捷函数
    
    Args:
        device: 设备类型
        latent_dim: 潜在空间维度
        img_shape: 图像形状 (channels, height, width)
        model_type: 模型类型，'vanilla' 或 'dcgan'
    """
    channels, height, width = img_shape
    
    if model_type == 'dcgan':
        D = DCGANDiscriminator(channels=channels, img_size=height).to(device)
        G = DCGANGenerator(latent_dim=latent_dim, channels=channels, img_size=height).to(device)
    else:  # vanilla GAN
        D = Discriminator(img_shape=img_shape).to(device)
        G = Generator(latent_dim=latent_dim, img_shape=img_shape).to(device)
    
    return D, G
import torch
import torchvision.datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt

def get_fashion_mnist_dataloader(root='./FashionMNIST/', batch_size=64, shuffle=True, download=True):
    """获取Fashion MNIST数据加载器"""
    dataset = torchvision.datasets.FashionMNIST(
        root=root,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]),
        download=download
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader, dataset

def get_mnist_dataloader(root='./MNISTdata/', batch_size=64, shuffle=True, download=True):
    """获取MNIST数据加载器"""
    dataset = torchvision.datasets.MNIST(
        root=root,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]),
        download=download
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader, dataset

def show_imgs(x, new_fig=True):
    """显示图像的辅助函数"""
    grid = vutils.make_grid(x.detach().cpu(), nrow=8, normalize=True, pad_value=0.3)
    grid = grid.transpose(0,2).transpose(0,1)  # channels as last dimension
    if new_fig:
        plt.figure()
    plt.imshow(grid.numpy())
    plt.axis('off')

def show_sample_images(dataset, num_samples=8):
    """显示数据集中的样本图像"""
    fig, axes = plt.subplots(1, num_samples, figsize=(12, 2))
    for i in range(num_samples):
        img, label = dataset[i]
        axes[i].imshow(img.squeeze().numpy(), cmap='gray')
        axes[i].set_title(f'Label: {label}')
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()


import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from model import Discriminator, Generator, create_models
from dataset import get_fashion_mnist_dataloader, show_imgs

def create_optimizers(D, G, lr_d=0.03, lr_g=0.03, optimizer_type='SGD'):
    if optimizer_type == 'SGD':
        optimizerD = torch.optim.SGD(D.parameters(), lr=lr_d)
        optimizerG = torch.optim.SGD(G.parameters(), lr=lr_g)
    elif optimizer_type == 'Adam':
        optimizerD = torch.optim.Adam(D.parameters(), lr=lr_d)
        optimizerG = torch.optim.Adam(G.parameters(), lr=lr_g)
    else:
        raise ValueError("optimizer_type must be 'SGD' or 'Adam'")
    return optimizerD, optimizerG

def train_step_discriminator(D, G, x_real, criterion, optimizerD, device, batch_size=64, z_dim=100):
    lab_real = torch.ones(batch_size, 1, device=device)
    lab_fake = torch.zeros(batch_size, 1, device=device)
    optimizerD.zero_grad()
    D_x = D(x_real)
    lossD_real = criterion(D_x, lab_real)

    z = torch.randn(batch_size, z_dim, device=device)
    x_gen = G(z).detach()
    D_G_z = D(x_gen)
    lossD_fake = criterion(D_G_z, lab_fake)

    lossD = lossD_real + lossD_fake
    lossD.backward()
    optimizerD.step()
    return lossD.item(), D_x.mean().item(), D_G_z.mean().item()

def train_step_generator(D, G, criterion, optimizerG, device, batch_size=64, z_dim=100):
    lab_real = torch.ones(batch_size, 1, device=device)
    optimizerG.zero_grad()
    z = torch.randn(batch_size, z_dim, device=device)
    x_gen = G(z)
    D_G_z = D(x_gen)
    lossG = criterion(D_G_z, lab_real)
    lossG.backward()
    optimizerG.step()
    return lossG.item(), D_G_z.mean().item()

def show_real_vs_fake(real_imgs, fake_imgs, num=8, save_path=None):
    """对比展示真实图像和生成图像"""
    assert real_imgs.size(0) >= num and fake_imgs.size(0) >= num, "图像数量不足"
    real_imgs = real_imgs[:num].cpu().detach()
    fake_imgs = fake_imgs[:num].cpu().detach()

    fig, axes = plt.subplots(2, num, figsize=(num * 1.5, 3))
    for i in range(num):
        axes[0, i].imshow(real_imgs[i][0], cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('Real', fontsize=12)

        axes[1, i].imshow(fake_imgs[i][0], cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('Fake', fontsize=12)

    plt.suptitle("real vs fake", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def save_checkpoint(D, G, optimizerD, optimizerG, epoch, loss_d, loss_g, save_dir='checkpoints'):
    """保存训练检查点"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    checkpoint = {
        'epoch': epoch,
        'discriminator_state_dict': D.state_dict(),
        'generator_state_dict': G.state_dict(),
        'optimizerD_state_dict': optimizerD.state_dict(),
        'optimizerG_state_dict': optimizerG.state_dict(),
        'loss_d': loss_d,
        'loss_g': loss_g,
    }
    
    checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f"💾 检查点已保存: {checkpoint_path}")

def save_models(D, G, save_dir='saved_models', model_type='vanilla'):
    """保存最终训练好的模型"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 保存生成器
    generator_path = os.path.join(save_dir, f'generator_{model_type}.pth')
    torch.save(G.state_dict(), generator_path)
    print(f"💾 生成器已保存: {generator_path}")
    
    # 保存判别器
    discriminator_path = os.path.join(save_dir, f'discriminator_{model_type}.pth')
    torch.save(D.state_dict(), discriminator_path)
    print(f"💾 判别器已保存: {discriminator_path}")
    
    return generator_path, discriminator_path

def load_checkpoint(checkpoint_path, D, G, optimizerD, optimizerG, device):
    """加载训练检查点"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    D.load_state_dict(checkpoint['discriminator_state_dict'])
    G.load_state_dict(checkpoint['generator_state_dict'])
    optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
    optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
    
    epoch = checkpoint['epoch']
    loss_d = checkpoint['loss_d']
    loss_g = checkpoint['loss_g']
    
    print(f"📂 检查点已加载: epoch {epoch}, D_loss: {loss_d:.4f}, G_loss: {loss_g:.4f}")
    return epoch, loss_d, loss_g

def load_models(generator_path, discriminator_path, device, latent_dim=100, img_shape=(1, 28, 28), model_type='vanilla'):
    """加载训练好的模型"""
    D, G = create_models(device, latent_dim=latent_dim, img_shape=img_shape, model_type=model_type)
    
    G.load_state_dict(torch.load(generator_path, map_location=device))
    D.load_state_dict(torch.load(discriminator_path, map_location=device))
    
    print(f"📂 模型已加载: {generator_path}, {discriminator_path}")
    return D, G

def train_gan(epochs=3, batch_size=64, lr_d=0.03, lr_g=0.03, z_dim=100, device=None, model_type='vanilla', 
              save_interval=20, resume_from=None):
    """训练GAN模型
    
    Args:
        epochs: 训练轮次
        batch_size: 批次大小
        lr_d: 判别器学习率
        lr_g: 生成器学习率
        z_dim: 噪声维度
        device: 设备
        model_type: 模型类型 ('vanilla' 或 'dcgan')
        save_interval: 保存检查点的间隔轮次
        resume_from: 从检查点恢复训练的路径
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # 为FashionMNIST数据集设置图像形状
    img_shape = (1, 28, 28)
    D, G = create_models(device, latent_dim=z_dim, img_shape=img_shape, model_type=model_type)
    optimizerD, optimizerG = create_optimizers(D, G, lr_d, lr_g, 'Adam')  # 改用Adam优化器
    criterion = nn.BCELoss()
    dataloader, dataset = get_fashion_mnist_dataloader(batch_size=batch_size)

    start_epoch = 0
    
    # 如果指定了恢复路径，加载检查点
    if resume_from and os.path.exists(resume_from):
        start_epoch, _, _ = load_checkpoint(resume_from, D, G, optimizerD, optimizerG, device)
        start_epoch += 1  # 从下一个epoch开始

    collect_x_gen = []
    fixed_noise = torch.randn(64, z_dim, device=device)
    fig = plt.figure()
    plt.ion()

    for epoch in range(start_epoch, epochs):
        print(f"\n🧪 Epoch {epoch+1}/{epochs}")
        
        epoch_loss_d = 0
        epoch_loss_g = 0
        num_batches = len(dataloader)

        print("🔍 Training Discriminator:")
        d_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"[D] Epoch {epoch+1}")
        for i, (x_real, _) in d_bar:
            x_real = x_real.to(device)
            current_batch_size = x_real.size(0)

            lossD, D_x_mean, D_G_z_mean = train_step_discriminator(
                D, G, x_real, criterion, optimizerD, device, current_batch_size, z_dim
            )
            
            epoch_loss_d += lossD

            d_bar.set_postfix({
                'LossD': f"{lossD:.4f}",
                'D(x)': f"{D_x_mean:.4f}",
                'D(G(z))': f"{D_G_z_mean:.4f}"
            })

        print("🎨 Training Generator:")
        g_bar = tqdm(range(len(dataloader)), desc=f"[G] Epoch {epoch+1}")
        for _ in g_bar:
            lossG, D_G_z_mean_gen = train_step_generator(
                D, G, criterion, optimizerG, device, batch_size, z_dim
            )
            
            epoch_loss_g += lossG

            g_bar.set_postfix({
                'LossG': f"{lossG:.4f}",
                'D(G(z))': f"{D_G_z_mean_gen:.4f}"
            })

        # 计算平均损失
        avg_loss_d = epoch_loss_d / num_batches
        avg_loss_g = epoch_loss_g / num_batches
        
        print(f"📊 Epoch {epoch+1} 平均损失 - D: {avg_loss_d:.4f}, G: {avg_loss_g:.4f}")

        x_gen = G(fixed_noise)
        show_imgs(x_gen, new_fig=False)
        fig.canvas.draw()
        collect_x_gen.append(x_gen.detach().clone())
        
        # 定期保存检查点
        if (epoch + 1) % save_interval == 0:
            save_checkpoint(D, G, optimizerD, optimizerG, epoch, avg_loss_d, avg_loss_g)

    plt.ioff()
    
    # 保存最终模型
    save_models(D, G, model_type=model_type)
    
    return D, G, collect_x_gen, dataset

def evaluate_gan(D, G, collect_x_gen, dataset):
    # 获取一批真实图像
    real_imgs, _ = next(iter(torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)))
    device = next(G.parameters()).device
    
    # 生成16个不同的噪声向量
    z = torch.randn(16, 100, device=device)
    with torch.no_grad():
        fake_imgs = G(z)
    
    # 展示对比图
    show_real_vs_fake(real_imgs, fake_imgs, num=8, save_path="real_vs_fake.png")
    
    # 额外展示：生成更多不同的图像来验证多样性
    print("\n生成多样性测试：")
    z_diverse = torch.randn(64, 100, device=device)
    with torch.no_grad():
        diverse_imgs = G(z_diverse)
    show_imgs(diverse_imgs[:16])  # 显示前16张

if __name__ == "__main__":
    # 训练新模型
    D, G, collect_x_gen, dataset = train_gan(
        epochs=200,
        lr_d=0.0002,
        lr_g=0.0002,
        model_type='dcgan',  # 在这里指定模型类型
        save_interval=100,      # 每20个epoch保存一次检查点
        # resume_from='checkpoints/checkpoint_epoch_19.pth'  # 可选：从检查点恢复训练
    )
    evaluate_gan(D, G, collect_x_gen, dataset)
    
    # 示例：如何加载已保存的模型进行推理
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # D_loaded, G_loaded = load_models(
    #     'saved_models/generator_vanilla.pth',
    #     'saved_models/discriminator_vanilla.pth',
    #     device,
    #     model_type='vanilla'
    # )
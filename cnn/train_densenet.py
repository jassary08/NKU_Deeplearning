import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import multiprocessing

from model import DenseNet  # 你的自定义 DenseNet
from dataset import get_dataset


def main():
    # 获取数据集与类别数
    train_loader, val_loader, classes = get_dataset()

    # 模型参数（使用你的 DenseNet）
    model = DenseNet(
        growthRate=12,
        depth=40,
        reduction=0.5,
        nClasses=len(classes)
    )

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # TensorBoard writer
    writer = SummaryWriter(log_dir='runs/densenet')

    # 训练参数
    num_epochs = 50
    save_path = '/checkpoint/checkpoint_densenet_cifar10.pth'

    # 日志记录
    train_losses = []
    val_losses = []
    val_accuracies = []

    # 验证函数
    def validate(model, val_loader, criterion):
        model.eval()
        total = 0
        correct = 0
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc='Validating', leave=False):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        avg_loss = val_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    # 训练 + 验证
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        train_loop = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{num_epochs}]', leave=False)

        for images, labels in train_loop:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            train_loop.set_postfix({'batch_loss': loss.item()})

        avg_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        # 验证阶段
        val_loss, val_accuracy = validate(model, val_loader, criterion)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # 写入 TensorBoard
        writer.add_scalar('Loss/Train', avg_train_loss, epoch + 1)
        writer.add_scalar('Loss/Validation', val_loss, epoch + 1)
        writer.add_scalar('Accuracy/Validation', val_accuracy, epoch + 1)

        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

    # 保存模型
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # 关闭 TensorBoard writer
    writer.close()


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()

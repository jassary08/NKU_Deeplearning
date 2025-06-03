import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os

from model import DenseNet
from dataset import get_dataset  # 你的数据加载器

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 获取数据集与类别数
    train_loader, val_loader, classes = get_dataset(batch_size=1024, num_workers=4)  # 增大 batch size
    num_classes = len(classes)

    # 模型
    model = DenseNet(growth_rate=32, block_config=(6, 12, 24, 16),
                   num_init_features=64, bn_size=4, drop_rate=0.2,
                   compression=1, num_classes=10)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model = model.to(device)

    # 优化器、损失函数、学习率调度器
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()  # 混合精度 scaler

    # TensorBoard
    writer = SummaryWriter(log_dir='runs/densenet')
    num_epochs = 100
    save_path = 'checkpoints/checkpoint_densenet_best.pth'
    best_acc = 0.0

    # 验证函数
    def validate(model, val_loader, criterion):
        model.eval()
        total = 0
        correct = 0
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc='Validating', leave=False):
                images, labels = images.to(device), labels.to(device)
                with autocast():
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
        correct = 0
        total = 0

        train_loop = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{num_epochs}]', leave=False)
        for images, labels in train_loop:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            train_loop.set_postfix({'batch_loss': loss.item()})

        avg_train_loss = running_loss / len(train_loader.dataset)
        train_acc = correct / total

        val_loss, val_accuracy = validate(model, val_loader, criterion)
        writer.add_scalar('Loss/Train', avg_train_loss, epoch + 1)
        writer.add_scalar('Accuracy/Train', train_acc, epoch + 1)
        writer.add_scalar('Loss/Validation', val_loss, epoch + 1)
        writer.add_scalar('Accuracy/Validation', val_accuracy, epoch + 1)

        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        # 保存最佳模型
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved! Acc={best_acc:.4f}")

    writer.close()
    print(f"Training finished. Best accuracy: {best_acc:.4f}")


if __name__ == '__main__':
    main()

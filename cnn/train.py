import torch
from torch import nn
import torch.optim as optim
from model import ResNet
from dataset import get_dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def train():
    # 获取数据集与类别数
    train_dataset, val_dataset, classes = get_dataset()

    # 模型参数
    model = ResNet(
        img_channels=3,
        nums_blocks=[3, 4, 6, 3],
        nums_channels=[64, 256, 512, 1024, 2048],
        first_kernel_size=7,
        num_labels=10
    )

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 优化器和损失函数
    optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # 训练参数
    num_epochs = 10
    save_path = 'checkpoint_final.pth'

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
            for images, labels in val_loader:
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

        for images, labels in train_dataset:
            print(labels)
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            print(f'outputs:{outputs.shape}')
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        avg_train_loss = running_loss / len(train_dataset.dataset)
        train_losses.append(avg_train_loss)

        # 验证阶段
        val_loss, val_accuracy = validate(model, val_dataset, criterion)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

    # 保存模型
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # 绘制 loss 和 accuracy 曲线图
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs+1), val_accuracies, label='Validation Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train()
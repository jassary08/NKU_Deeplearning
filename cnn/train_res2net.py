import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import multiprocessing
import os

from model import Res2Net, Bottle2neck  # 直接导入你写好的 Res2Net
from dataset import get_dataset

def main():
    # 获取数据集与类别数
    train_loader, val_loader, classes = get_dataset(batch_size=2048, num_workers=4)  # 大 batch

    # 模型参数（用你的 Res2Net）
    model = Res2Net(Bottle2neck,[3, 4, 6, 3],
                    baseWidth=16,
                    scale=2,
                    num_classes=10)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    writer = SummaryWriter(log_dir='runs/res2net50')
    num_epochs = 100
    save_path = 'checkpoints/checkpoint_res2net50_best.pth'
    best_acc = 0.0

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

        if val_accuracy > best_acc:
            best_acc = val_accuracy
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved! Acc={best_acc:.4f}")

    writer.close()
    print(f"Training finished. Best accuracy: {best_acc:.4f}")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()

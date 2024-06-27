import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import os
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    writer = SummaryWriter('runs/cifar_100_resnet18_self_supervised')

    # 定义数据转换
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 100)
    model = model.to(device)

    pretrain_path = './model_weights/best_model_epoch_10.pth'
    if os.path.isfile(pretrain_path):
        pretrained_dict = torch.load(pretrain_path, map_location=device)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(f"Loaded pre-trained model from {pretrain_path}")
    else:
        print(f"No pre-trained model found at {pretrain_path}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    checkpoint_path = 'resnet18_self_supervised_checkpoint.pth'

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 100 == 99:
                writer.add_scalar('training loss', running_loss / 100, epoch * len(train_loader) + i)
                print(f'Epoch {epoch + 1}, Batch {i + 1}: Loss = {running_loss / 100:.4f}')
                running_loss = 0.0
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        writer.add_scalar('Accuracy', accuracy, epoch)
        print(f'\nTest set: Accuracy: {correct}/{total} ({accuracy:.2f}%)\n')

        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)

    torch.save(model.state_dict(), 'resnet18_self_supervised_final.pth')
    print("Training completed and model saved.")

    writer.close()
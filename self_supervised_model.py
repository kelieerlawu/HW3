import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader

# 模型定义
class ProjectionHead(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=1024, output_dim=128):
        super(ProjectionHead, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.projection(x)

class ResNet18SimCLR(nn.Module):
    def __init__(self, pretrained=False):
        super(ResNet18SimCLR, self).__init__()
        self.encoder = models.resnet18(pretrained=False)
        self.encoder.fc = nn.Identity()
        self.projection_head = ProjectionHead(input_dim=512, hidden_dim=1024, output_dim=128)

    def forward(self, x):
        h = self.encoder(x)
        z = self.projection_head(h)
        return z

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z_i, z_j):
        z = torch.cat((z_i, z_j), dim=0)
        N = z.shape[0]

        sim = torch.matmul(z, z.T) / self.temperature
        sim.fill_diagonal_(-9e15)

        labels = torch.arange(N, dtype=torch.long, device=z.device)
        labels = (labels + N//2) % N

        loss = self.criterion(sim, labels)
        return loss


def get_cifar10_loaders(data_path, batch_size=32):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    return train_loader


def train(model, data_loader, optimizer, loss_fn, device, epochs=10, save_path='model_weights'):
    model.train()
    best_loss = float('inf')

    for epoch in range(epochs):
        total_loss = 0
        for images, _ in data_loader:
            images = images.to(device)
            optimizer.zero_grad()
            z = model(images)
            loss = loss_fn(z, z)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        print(f'Epoch {epoch + 1}, Loss: {avg_loss}')

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(save_path, f'best_model_epoch_{epoch + 1}.pth'))
            print(f'Model saved at epoch {epoch + 1} with loss {avg_loss}')

# 主函数
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18SimCLR(pretrained=False).to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = NTXentLoss().to(device)

    data_path = './data/cifar10'
    save_path = './model_weights'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    cifar10_loader = get_cifar10_loaders(data_path)
    train(model, cifar10_loader, optimizer, loss_fn, device, epochs=10, save_path=save_path)

if __name__ == '__main__':
    main()
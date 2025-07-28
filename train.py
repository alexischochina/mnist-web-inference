import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import MNISTNet
import os

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.CrossEntropyLoss()(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return accuracy

def main():
    # Paramètres
    batch_size = 64
    epochs = 10
    lr = 0.01
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transformations des données
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Chargement des données
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Création du modèle
    model = MNISTNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # Création du dossier pour sauvegarder les modèles
    os.makedirs('models', exist_ok=True)

    # Entraînement
    best_accuracy = 0
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        accuracy = test(model, device, test_loader)
        
        # Sauvegarde du meilleur modèle
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'models/mnist_best.pth')
            print(f'Nouveau meilleur modèle sauvegardé avec une précision de {accuracy:.2f}%')

if __name__ == '__main__':
    main() 
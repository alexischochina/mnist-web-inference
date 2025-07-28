import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        # Première couche convolutive : 1 canal d'entrée (image en niveaux de gris), 32 filtres, noyau 3x3
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        # Deuxième couche convolutive : 32 canaux d'entrée, 64 filtres, noyau 3x3
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        # Première couche fully connected : 7*7*64 entrées (après 2 max pooling), 128 sorties
        self.fc1 = nn.Linear(7*7*64, 128)
        # Couche de sortie : 128 entrées, 10 sorties (les chiffres de 0 à 9)
        self.fc2 = nn.Linear(128, 10)
        # Dropout pour réduire l'overfitting
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # Première convolution suivie de ReLU et max pooling
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        
        # Deuxième convolution suivie de ReLU et max pooling
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        
        # Aplatir le tenseur pour la couche fully connected
        x = x.view(-1, 7*7*64)
        
        # Première couche fully connected avec ReLU et dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Couche de sortie
        x = self.fc2(x)
        return F.log_softmax(x, dim=1) 
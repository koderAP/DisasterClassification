import torch
import torch.nn as nn
from torchvision import models
import numpy as np

def extract_features(data_loader, device):
    model = models.resnet50(pretrained=True)
    model = nn.Sequential(*list(model.children())[:-1]) 
    model = model.to(device)
    model.eval()

    features = []
    labels = []
    with torch.no_grad():
        for images, label_batch in data_loader:
            images = images.to(device)
            outputs = model(images)
            outputs = outputs.view(outputs.size(0), -1)
            features.append(outputs.cpu().numpy())
            labels.append(label_batch.numpy())
    
    features = np.vstack(features)
    labels = np.concatenate(labels)
    return features, labels


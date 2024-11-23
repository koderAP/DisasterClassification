import torch
import torch.nn as nn
from torchvision import models
import numpy as np
from torchvision.transforms import ToTensor

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


from PIL import Image



def get_misclassified_images(model, data_loader, device):
    misclassified_images = []
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(data_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            misclassified_idx = (predicted != labels).nonzero(as_tuple=True)[0]

            for idx in misclassified_idx:
                dataset_idx = batch_idx * data_loader.batch_size + idx.item()

                original_image_path = data_loader.dataset.samples[dataset_idx][0]
                original_image = Image.open(original_image_path).convert("RGB")
                original_image = ToTensor()(original_image)
                true_label = labels[idx].item()
                pred_label = predicted[idx].item()

                misclassified_images.append((original_image, true_label, pred_label))
    return misclassified_images
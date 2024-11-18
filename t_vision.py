import torch
import timm
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import numpy as np
import tqdm
from PIL import Image
from torchvision.transforms import ToTensor







def train_model(train_loader,val_loader, device, epochs=10):
    l = len(train_loader)
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=4)
    model_type = 'vit_base_patch16_224'
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    train_f1_scores = []
    val_f1_scores = []
    best_model = None

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}...", end='\r')
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            
        train_losses.append(running_loss / total)
        train_accuracies.append(correct / total)
        train_f1_scores.append(f1_score(all_labels, all_predictions, average='weighted'))

        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for i, data in enumerate(val_loader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                
            val_losses.append(running_loss / total)
            val_accuracies.append(correct / total)
            val_f1_scores.append(f1_score(all_labels, all_predictions, average='weighted'))

        if val_accuracies[-1] == max(val_accuracies):
            torch.save(model.state_dict(), f"{model_type}_best_model.pth")
            print(f"Model saved at epoch {epoch + 1}")
            best_model = model

        if not best_model:
            best_model = model


        
    return best_model, train_losses, val_losses, train_accuracies, val_accuracies, train_f1_scores, val_f1_scores




def get_original_image_from_loader(dataloader, idx):
    dataset = dataloader.dataset
    image_path, _ = dataset.imgs[idx] 
    image = Image.open(image_path).convert("RGB")
    image = ToTensor()(image)
    return image


def get_misclassified_images(model, data_loader, device):
    misclassified_images = []
    model.eval()
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            misclassified_idx = (predicted != labels).nonzero()
            for idx in misclassified_idx:
                image = get_original_image_from_loader(data_loader, idx)
                true_label = labels[idx].item()
                pred_label = predicted[idx].item()
                misclassified_images.append((image, true_label, pred_label))
    return misclassified_images


def get_train_val_predtion(model, train_loader, val_loader, device):
    model.eval()
    train_predictions = []
    train_labels = []
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        train_predictions.extend(predicted.cpu().numpy())
        train_labels.extend(labels.cpu().numpy())

    val_predictions = []
    val_labels = []

    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        val_predictions.extend(predicted.cpu().numpy())
        val_labels.extend(labels.cpu().numpy())

    return train_predictions, train_labels, val_predictions, val_labels

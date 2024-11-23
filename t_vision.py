import torch
import timm
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import numpy as np
import tqdm
from PIL import Image
from torchvision.transforms import ToTensor
from sklearn.metrics import classification_report
from torch import nn
import torch.optim as optim







def train_model(train_loader,val_loader, device, epochs=10):
    l = len(train_loader)
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=4)
    model_type = 'vit_base_patch16_224'
    model = model.to(device)
    model_name = model_type
    num_epochs = epochs
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        with tqdm(train_loader, desc=f"Training {model_name} Epoch {epoch+1}/{num_epochs}") as train_bar:
            for inputs, labels in train_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                train_bar.set_postfix(loss=train_loss / total, accuracy=100 * correct / total)

        train_losses.append(train_loss / total)
        train_accuracies.append(100 * correct / total)

        # Validation phase
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad(), tqdm(val_loader, desc=f"Validating {model_name} Epoch {epoch+1}/{num_epochs}") as val_bar:
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                val_bar.set_postfix(loss=val_loss / total, accuracy=100 * correct / total)

        val_losses.append(val_loss / total)
        val_acc = 100 * correct / total
        val_accuracies.append(val_acc)

        print(f"{model_name} Epoch {epoch+1}/{num_epochs}, "
                f"Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.2f}%, "
                f"Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accuracies[-1]:.2f}%")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"best_{model_name}_model.pth")
            print(f"Best model for {model_name} saved with Val Acc: {best_val_acc:.2f}%")


    all_preds, all_labels = [], []
    model.eval()
    model.load_state_dict(torch.load(f"best_{model_name}_model.pth"))
    with torch.no_grad():
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    print(f"Classification Report for train {model_name}:\n", 
            classification_report(all_labels, all_preds, digits=4))
    train_f1 = f1_score(all_labels, all_preds, average="weighted")
    print(f"Train F1 Score for {model_name}: {train_f1:.4f}")
    
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    print(f"Classification Report for val{model_name}:\n", 
            classification_report(all_labels, all_preds, digits=4))
    


    val_f1 = f1_score(all_labels, all_preds, average="weighted")
    

    return model, train_losses, val_losses, train_accuracies, val_accuracies, train_f1, val_f1




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

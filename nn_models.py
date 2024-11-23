from torchvision import models
import torch.nn as nn
import torch
from sklearn.metrics import f1_score
from PIL import Image
from torchvision.transforms import ToTensor
from tqdm import tqdm
import torch.optim as optim
from sklearn.metrics import classification_report



def prepare_model(model_name, num_classes=4):
    if model_name == "resnet50":
        model = models.resnet50(pretrained=True)
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    elif model_name == "densenet121":
        model = models.densenet121(pretrained=True)
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    elif model_name == "vgg16":
        model = models.vgg16(pretrained=True)
        model.classifier[-1] = nn.Sequential(
            nn.Linear(model.classifier[-1].in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=True)
        model.classifier[-1] = nn.Sequential(
            nn.Linear(model.classifier[-1].in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=True)
        model.classifier[-1] = nn.Sequential(
            nn.Linear(model.classifier[-1].in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    elif model_name == "resnet18":
        model = models.resnet18(pretrained=True)
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    elif model_name == "alexnet":
        model = models.alexnet(pretrained=True)
        model.classifier[-1] = nn.Sequential(
            nn.Linear(model.classifier[-1].in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    elif model_name == "squeezenet1_0":
        model = models.squeezenet1_0(pretrained=True)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model.num_classes = num_classes
    elif model_name == "shufflenet_v2_x1_0":
        model = models.shufflenet_v2_x1_0(pretrained=True)
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    elif model_name == "googlenet":
        model = models.googlenet(pretrained=True)
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    return model


def train_model_nn(model_type, train_loader, val_loader, device, epochs = 10):
    model = prepare_model(model_type)
    model.to(device)
    print(device)
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
    
    train_f1 = f1_score(all_labels, all_preds, average="weighted")
    print(f"Train F1 Score for {model_name}: {train_f1:.4f}")

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




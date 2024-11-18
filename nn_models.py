from torchvision import models
import torch.nn as nn
import torch
from sklearn.metrics import f1_score
from PIL import Image
from torchvision.transforms import ToTensor



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
    criterion = nn.CrossEntropyLoss()
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




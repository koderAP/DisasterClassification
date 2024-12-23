from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import tqdm
from PIL import Image
from torchvision.transforms import ToTensor



rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)


def train_model(train_features, train_labels, val_features, val_labels):
    rf_classifier.fit(train_features, train_labels)
    train_predictions = rf_classifier.predict(train_features)
    val_predictions = rf_classifier.predict(val_features)
    return rf_classifier, train_predictions, val_predictions


def get_original_image_from_loader(dataloader, idx):
    dataset = dataloader.dataset
    image_path, _ = dataset.imgs[idx] 
    image = Image.open(image_path).convert("RGB")
    image = ToTensor()(image)
    return image


def get_missclassified_images(rf_classifier, dataloader, features, labels):
    misclassified_images = []
    predictions = rf_classifier.predict(features)
    for i in tqdm.tqdm(range(len(predictions)), desc="Misclassified images fetching"):
        if predictions[i] != labels[i]:
            image = get_original_image_from_loader(dataloader, i)
            true_label = labels[i]
            pred_label = predictions[i]
            misclassified_images.append((image, true_label, pred_label))
    return misclassified_images





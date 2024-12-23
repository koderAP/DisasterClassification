from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
import tqdm
from PIL import Image
from torchvision.transforms import ToTensor


ada_classifier = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=2),
    n_estimators=200,  
    learning_rate=1.0,  
    random_state=42
)

def train_model(train_features, train_labels, val_features, val_labels):
    ada_classifier.fit(train_features, train_labels)
    train_predictions = ada_classifier.predict(train_features)
    val_predictions = ada_classifier.predict(val_features)
    return ada_classifier, train_predictions, val_predictions


def get_feature_importance():
    return ada_classifier.feature_importances_


def get_original_image_from_loader(dataloader, idx):
    dataset = dataloader.dataset
    image_path, _ = dataset.imgs[idx] 
    image = Image.open(image_path).convert("RGB")
    image = ToTensor()(image)
    return image


def get_missclassified_images(ada_classifier, dataloader, features, labels):
    misclassified_images = []
    predictions = ada_classifier.predict(features)
    for i in tqdm.tqdm(range(len(predictions)), desc="Misclassified images fetching"):
        if predictions[i] != labels[i]:
            image = get_original_image_from_loader(dataloader, i)
            true_label = labels[i]
            pred_label = predictions[i]
            misclassified_images.append((image, true_label, pred_label))
    return misclassified_images



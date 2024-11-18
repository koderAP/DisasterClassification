from sklearn import svm
import numpy as np



def train_model(train_features, train_labels, val_features, val_labels):
    svm_classifier = svm.SVC(kernel='rbf', C=10000000000, random_state=42)
    svm_classifier.fit(train_features, train_labels)

    trian_predictions = svm_classifier.predict(train_features)
    val_predictions = svm_classifier.predict(val_features)


    return svm_classifier, trian_predictions, val_predictions


def get_original_image_from_loader(dataloader, idx):
    dataset = dataloader.dataset
    original_image, _ = dataset.imgs[idx]
    return original_image

def get_missclassified_images(svm_model, dataloader, features, labels):
    misclassified_images = []
    predictions = svm_model.predict(features)
    for i in range(len(predictions)):
        if predictions[i] != labels[i]:
            image = get_original_image_from_loader(dataloader, i)
            true_label = labels[i]
            pred_label = predictions[i]
            misclassified_images.append((image, true_label, pred_label))
    return misclassified_images



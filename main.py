import sys
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from Models import nn_models as nnm
from Models import svm_model as svm_m
from Models import rf_model as rf_m
from Models import ada_model as ada_m
from Models import t_vision as tv
import helper as hp
import os
from sklearn.metrics import classification_report
import numpy as np
import torch
from torchvision.utils import save_image


def preprocess_data():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    

def load_datasets(train_dir, val_dir, transform):
    train_dataset = ImageFolder(root=train_dir, transform=transform)
    val_dataset = ImageFolder(root=val_dir, transform=transform)
    return train_dataset, val_dataset

def get_original_image(dataset, index):
    original_image, _ = dataset.imgs[index]
    return original_image

def get_data_loaders(train_dataset, val_dataset, batch_size=32):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def train_and_evaluate_nn_models(train_loader, val_loader, device, num_epochs=10):
    model_names = [ "vit_base_patch16_224","custom_cnn","resnet50", "densenet121", "vgg16", "mobilenet_v2", "efficientnet_b0",
               "resnet18", "alexnet", "squeezenet1_0", "shufflenet_v2_x1_0", "googlenet"]
    results = {}

    for model_name in model_names:
        print(f"Training {model_name} model")
        best_model, train_losses, val_losses, train_accuracies, val_accuracies, train_f1_scores, val_f1_scores = nnm.train_model_nn(model_name, train_loader, val_loader, device, num_epochs)

        misclassified_images = hp.get_misclassified_images(best_model, val_loader, device)
        for idx, (image, true_label, pred_label) in enumerate(misclassified_images):
            image_path = f"misclassified_images/{model_name}/misclassified_{idx}_true_{true_label}_got_{pred_label}.png"
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            save_image(image, image_path)



        train_p, train_l, val_p, val_l = nnm.get_train_val_predtion(best_model, train_loader, val_loader, device)

        classification_report_train = classification_report(train_l, train_p )
        classification_report_val = classification_report(val_l, val_p)
        report_path = f"classification_reports/{model_name}_classification_report.txt"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w") as f:
            f.write("Train Classification Report\n")
            f.write(classification_report_train)
            f.write("\n\n")
            f.write("Validation Classification Report\n")
            f.write(classification_report_val)


        results[model_name] = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_accuracies": train_accuracies,
            "val_accuracies": val_accuracies,
            "train_f1_scores": train_f1_scores,
            "val_f1_scores": val_f1_scores
        }

    

    import matplotlib.pyplot as plt

    def plot_and_save_results(results):
        for model_name, metrics in results.items():
            epochs = range(1, len(metrics["train_losses"]) + 1)

            plt.figure()
            plt.plot(epochs, metrics["train_losses"], label='Train Loss')
            plt.plot(epochs, metrics["val_losses"], label='Validation Loss')
            plt.title(f'{model_name} Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(f'plots/{model_name}_loss.png')
            plt.close()

            plt.figure()
            plt.plot(epochs, metrics["train_accuracies"], label='Train Accuracy')
            plt.plot(epochs, metrics["val_accuracies"], label='Validation Accuracy')
            plt.title(f'{model_name} Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.savefig(f'plots/{model_name}_accuracy.png')
            plt.close()

            plt.figure()
            plt.plot(epochs, metrics["train_f1_scores"], label='Train F1 Score')
            plt.plot(epochs, metrics["val_f1_scores"], label='Validation F1 Score')
            plt.title(f'{model_name} F1 Score')
            plt.xlabel('Epochs')
            plt.ylabel('F1 Score')
            plt.legend()
            plt.savefig(f'plots/{model_name}_f1_score.png')
            plt.close()

    os.makedirs("plots", exist_ok=True)
    plot_and_save_results(results)

    return results






if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python main.py <train_directory> <val_directory> <model_type>")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_directory = sys.argv[1]
    val_directory = sys.argv[2]
    model_type = sys.argv[3]

    if not os.path.exists("misclassified_images"):
        os.makedirs("misclassified_images")

    transform = preprocess_data()

    train_dataset, val_dataset = load_datasets(train_directory, val_directory, transform)
    train_loader, val_loader = get_data_loaders(train_dataset, val_dataset)

    train_features, train_labels = hp.extract_features(train_loader, device=device)
    val_features, val_labels = hp.extract_features(val_loader, device=device)

    results = None

    if model_type == "nn":
        print("Using Neural Network Model")
        results = train_and_evaluate_nn_models(train_loader, val_loader, device, num_epochs=10)
    elif model_type == "svm":
        print("Using Support Vector Machine Model")
        model, train_p, val_p = svm_m.train_model(train_features, train_labels, val_features, val_labels)
        train_report = classification_report(train_labels, train_p)
        val_report = classification_report(val_labels, val_p)
        report_path = f"classification_reports/svm_classification_report.txt"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w") as f:
            f.write("Train Classification Report\n")
            f.write(train_report)
            f.write("\n\n")
            f.write("Validation Classification Report\n")
            f.write(val_report)
        misclassified_images = svm_m.get_missclassified_images(model, val_loader, val_features, val_labels)
        for idx, (image, true_label, pred_label) in enumerate(misclassified_images):
            image_path = f"misclassified_images/svm/misclassified_{idx}_true_{true_label}_got_{pred_label}.png"
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            save_image(image, image_path)

    elif model_type == "rf":
        print("Using Random Forest Model")
        model, train_p, val_p = rf_m.train_model(train_features, train_labels, val_features, val_labels)
        train_report = classification_report(train_labels, train_p)
        val_report = classification_report(val_labels, val_p)
        report_path = f"classification_reports/rf_classification_report.txt"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w") as f:
            f.write("Train Classification Report\n")
            f.write(train_report)
            f.write("\n\n")
            f.write("Validation Classification Report\n")
            f.write(val_report)
        misclassified_images = rf_m.get_missclassified_images(model, val_loader, val_features, val_labels)
        for idx, (image, true_label, pred_label) in enumerate(misclassified_images):
            image_path = f"misclassified_images/rf/misclassified_{idx}_true_{true_label}_got_{pred_label}.png"
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            save_image(image, image_path)
    elif model_type == "ada":
        print("Using AdaBoost Model")
        model, train_p, val_p = ada_m.train_model(train_features, train_labels, val_features, val_labels)
        train_report = classification_report(train_labels, train_p)
        val_report = classification_report(val_labels, val_p)
        report_path = f"classification_reports/ada_classification_report.txt"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w") as f:
            f.write("Train Classification Report\n")
            f.write(train_report)
            f.write("\n\n")
            f.write("Validation Classification Report\n")
            f.write(val_report)
        misclassified_images = ada_m.get_missclassified_images(model, val_loader, val_features, val_labels)
        for idx, (image, true_label, pred_label) in enumerate(misclassified_images):
            image_path = f"misclassified_images/ada/misclassified_{idx}_true_{true_label}_got_{pred_label}.png"
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            save_image(image, image_path)
    else:
        print("Invalid model type. Please use 'nn' or 'svm'")
        sys.exit(1)
    

    print(f"Train Directory: {train_directory}")
    print(f"Validation Directory: {val_directory}")
    print(f"Model Type: {model_type}")

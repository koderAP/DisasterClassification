import sys
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import nn_models as nnm
import svm_model as svm_m
import rf_model as rf_m
import ada_model as ada_m
import t_vision as tv
import helper as hp
import os
from sklearn.metrics import classification_report

def preprocess_data():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform
    

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


def train_and_evaluate_nn_models(model_names, train_loader, val_loader, criterion, device, num_epochs=10):
    model_names = ["resnet50", "densenet121", "vgg16", "mobilenet_v2", "efficientnet_b0",
               "resnet18", "alexnet", "squeezenet1_0", "shufflenet_v2_x1_0", "googlenet"]
    results = {}

    for model_name in model_names:
        best_model, train_losses, val_losses, train_accuracies, val_accuracies, train_f1_scores, val_f1_scores = nnm.train_model_nn(model_name, train_loader, val_loader, criterion, device, num_epochs)

        misclassified_images = nnm.get_misclassified_images(best_model, val_loader, device)
        for idx, (image, true_label, pred_label) in enumerate(misclassified_images):
            image_path = f"misclassified_images/{model_name}/misclassified_{idx}.png"
            pil_image = transforms.ToPILImage()(image.cpu())
            pil_image.save(image_path)



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

    best_model, train_losses, val_losses, train_accuracies, val_accuracies, train_f1_scores, val_f1_scores = tv.train_model(train_loader,val_loader, device, epochs=10)
    results["vit_base_patch16_224"] = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_accuracies": train_accuracies,
            "val_accuracies": val_accuracies,
            "train_f1_scores": train_f1_scores,
            "val_f1_scores": val_f1_scores
        }
    train_p, train_l, val_p, val_l = tv.get_train_val_predtion(best_model, train_loader, val_loader, device)
    model_name = "vit_base_patch16_224"
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

    return results






if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python main.py <train_directory> <val_directory> <model_type>")
        sys.exit(1)

    train_directory = sys.argv[1]
    val_directory = sys.argv[2]
    model_type = sys.argv[3]

    if not os.path.exists("misclassified_images"):
        os.makedirs("misclassified_images")

    transform = preprocess_data()

    train_dataset, val_dataset = load_datasets(train_directory, val_directory, transform)
    train_loader, val_loader = get_data_loaders(train_dataset, val_dataset)

    train_features, train_labels = hp.extract_features(train_loader)
    val_features, val_labels = hp.extract_features(val_loader)

    results = None

    if model_type == "nn":
        print("Using Neural Network Model")
        results = train_and_evaluate_nn_models(["resnet50", "densenet121", "vgg16", "mobilenet_v2", "efficientnet_b0",
               "resnet18", "alexnet", "squeezenet1_0", "shufflenet_v2_x1_0", "googlenet"], train_loader, val_loader, nnm.criterion, nnm.device, num_epochs=10)
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
        images = svm_m.get_missclassified_images(model, val_loader, val_features, val_labels)
        for i in images:
            image_path = f"misclassified_images/svm/misclassified_{i}.png"
            pil_image = transforms.ToPILImage()(images[i][0].cpu())
            pil_image.save(image_path)

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
        images = rf_m.get_missclassified_images(model, val_loader, val_features, val_labels)
        for i in images:
            image_path = f"misclassified_images/svm/misclassified_{i}.png"
            pil_image = transforms.ToPILImage()(images[i][0].cpu())
            pil_image.save(image_path)
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
        images = ada_m.get_missclassified_images(model, val_loader, val_features, val_labels)
        for i in images:
            image_path = f"misclassified_images/svm/misclassified_{i}.png"
            pil_image = transforms.ToPILImage()(images[i][0].cpu())
            pil_image.save(image_path)
    else:
        print("Invalid model type. Please use 'nn' or 'svm'")
        sys.exit(1)
    

    print(f"Train Directory: {train_directory}")
    print(f"Validation Directory: {val_directory}")
    print(f"Model Type: {model_type}")
# DisasterClassification

This repository implements a project for classifying disaster images into one of four categories: **CYCLONE**, **EARTHQUAKE**, **FLOOD**, and **WILDFIRE**. The project uses a combination of machine learning models, deep learning models, and pretrained vision architectures for accurate classification.

---

How to run : 
```
!python3 main.py <Train_Data_Path> <Val_Data_Path> <method nn or svm or rf or ada>
```

## Repository Structure

```plaintext
DisasterClassification/
├── main.py                # Main script to run the entire classification pipeline
├── helper.py              # Utility functions used across the project
├── Models/                # Directory containing model implementations
│   ├── nn_models.py       # Custom-built neural network models
│   ├── svm_model.py       # Support Vector Machine (SVM) implementation
│   ├── rf_model.py        # Random Forest model implementation
│   ├── ada_model.py       # AdaBoost model implementation
│   ├── t_vision.py        # Vision Transformer and pretrained model implementations
│   ├── helper.py          # Helper functions for model management
├── Notebooks/             # Jupyter Notebooks for experiments and visualizations
│   ├── bagging.ipynb      # Notebook for experiments with bagging models
│   ├── boosting.ipynb     # Notebook for experiments with boosting models
│   ├── svm.ipynb          # Notebook for SVM-based experiments

# MultimodalAE-BreastCancer-Genomics

Implementation of a paper titled "Prognostically Relevant Subtypes and Survival Prediction for Breast Cancer Based on Multimodal Genomics Data," utilizing a **Multimodal Autoencoder (MAE)** built with Deep Belief Networks (DBN) to analyze multi-platform genomics data from **The Cancer Genome Atlas (TCGA) BRCA** cohort.

This model learns comprehensive feature representations to predict clinical status and survival probability for breast cancer patients.

## 💡 Key Features & Predicted Clinical Status

The Multimodal Autoencoder (MAE) is trained with high-dimensional genomics data—including **DNA methylation, gene expression, and miRNA expression**—to predict two crucial clinical outcomes:

1.  **Breast Cancer Subtypes:** Classification of the patient's status based on the Estrogen Receptor (ER), Progesterone Receptor (PGR), and HER2/neu status.
2.  **Survival Rate:** Regression to predict the patient's survival rate (a value between 0 and 1, where 1 signifies the best chance of survival).

---

## 🛠️ Requirements

The project requires **Python 3** and the following major libraries.

| Package Name | Details |
| :--- | :--- |
| **Python** | Version 3+ |
| **TensorFlow** | Used as one of the deep learning backend platforms. |
| **Keras** | Used for building and training the neural networks. |
| **Theano** | Used as an alternative deep learning backend platform. |

### Installation

Clone the repository and ensure your environment meets the requirements:

```bash
git clone [https://github.com/MonkHacker1/MultimodalAE-BreastCancer-Genomics.git](https://github.com/MonkHacker1/MultimodalAE-BreastCancer-Genomics.git)
cd MultimodalAE-BreastCancer-Genomics
# Install required libraries (You may need to specify versions compatible with Keras/Theano/TensorFlow)
# pip install tensorflow keras ... (and other dependencies like numpy, sklearn, etc.)
```
# 📂 Download and Create the Dataset

The model uses data sourced from The Cancer Genome Atlas (TCGA) Breast Cancer (BRCA) project. The main_download.py script automatically fetches and processes the data based on your selection.
## 🔁 Two Main Components

There are two major Python programs:

#### 1️⃣ main_download.py → Dataset Downloader

Downloads and preprocesses data from TCGA BRCA.

#### 2️⃣ main_run.py → Model Trainer

Builds and trains multimodal deep belief networks (DBN/mDBN) using TensorFlow or Theano.
Run the dataset creation program with the desired index:
```bash
python3 main_download.py -d DATASET_IDX
```
| DATASET_IDX | Data Types                                 | Data size (GB) |
|--------------|---------------------------------------------|----------------|
| 1            | DNA Methylation                             | 14             |
| 2            | Gene Expression                             | 9              |
| 3            | miRNA Expression                            | 0.2            |
| 4            | Gene Expression + miRNA Expression          | 10             |
| 5            | DNA Methylation + Gene Expression + miRNA Expression | 16             |

***Note on Target Folder: The program will ask for a target folder to download the data (default is /home).***

# 🚀 Train the Neural Networks (main_run.py)

The main training and prediction program, main_run.py, is highly configurable, allowing you to choose the platform, prediction type, dataset, and hyperparameters via command-line options.

Run the program with options:
```bash

python3 main_run.py <options>
```
## Training Options

### 🔧 Required Parameters

| Parameter | Values | Description |
|-----------|--------|-------------|
| **`-p, --platform`** | `1` • `2` | **Framework**: `1`=TensorFlow, `2`=Theano |
| **`-t, --type`** | `1` • `2` | **Task**: `1`=Subtype Classification, `2`=Survival Regression |
| **`-d, --dataset`** | `1-15` | **Data**: See dataset table below |

### ⚙️ Optional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **`--pretrain_epoch`** | `int` | `100` | Pre-training epochs |
| **`--train_epoch`** | `int` | `100` | Fine-tuning epochs |
| **`--batch`** | `int` | `10` | Batch size |
| **`--pca`** | `1` • `2` | `2` | PCA: `1`=Use, `2`=Don't use |
| **`--optimizer`** | `1` • `2` • `3` | `1` | Optimizer: `1`=SGD, `2`=RMSProp, `3`=Adam |

## Example Execution

To perform breast cancer subtype classification based on the DNA methylation dataset (-d 1), using PCA (--pca 1), on the TensorFlow platform (-p 1), with the Adam optimizer (--optimizer 3), for 5 epochs each (pre-train/train), issue the following command:
```bash
python3 main_run.py --platform 1 --type 1 --dataset 1 --batch 10 --pretrain_epoch 5 --train_epoch 5 --pca 1 --optimizer 3
```
# 📊 Sample Classification Results

A sample run for Cancer Type Classification using the DNA Methylation platform (GPL8490) with the TensorFlow backend shows strong predictive capabilities for the three clinical statuses:
**Model Performance**

| Status | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|-----------|
| **ER Status** | 0.878 | 0.862 | 0.879 | 0.869 |
| **PGR Status** | 0.869 | 0.848 | 0.869 | 0.857 |
| **HER2 Status** | 0.861 | 0.876 | 0.861 | N/A |

# Cat Breeds Classification Using Convolutional Neural Network (CNN)

## Project Overview

This project implements an image classification system using a Convolutional Neural Network (CNN) to classify cat images into 12 different breeds.

The model is developed using TensorFlow and Keras with image preprocessing, data augmentation, model training, evaluation, and image prediction.

## Dataset

The project uses the Cat Breeds Dataset, containing 12 cat breed classes:

1. Abyssinian
2. Bengal
3. Birman
4. Bombay
5. British Shorthair
6. Egyptian Mau
7. Maine Coon
8. Persian
9. Ragdoll
10. Russian Blue
11. Siamese
12. Sphynx

Dataset distribution:
- Total images: 2,388
- Training images: 1,911
- Validation images: 477
- Data splitting: 80% training and 20% validation
- Splitting method: Stratified split

## Model Architecture

The project uses a custom CNN architecture consisting of:

- Input layer: 128 × 128 × 3
- Rescaling layer
- Data augmentation:
  - Random Flip
  - Random Rotation
  - Random Zoom
- Three convolutional blocks:
  - Conv2D (32 filters)
  - Conv2D (64 filters)
  - Conv2D (128 filters)
- MaxPooling2D layers
- Global Average Pooling
- Dense layer (128 neurons)
- Dropout (0.3)
- Output layer (12 classes, Softmax activation)

Optimizer: Adam  
Learning rate: 0.001  
Loss function: Sparse Categorical Crossentropy  
Batch size: 32  
Maximum epochs: 15

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Scikit-learn
- Pillow

## Project Structure

```text
CNN_Cat_Breeds/
│
├── dataset/
│   └── cats-breads/
│       ├── Abyssinian/
│       ├── Bengal/
│       ├── Birman/
│       ├── Bombay/
│       ├── British_Shorthair/
│       ├── Egyptian_Mau/
│       ├── Maine_Coon/
│       ├── Persian/
│       ├── Ragdoll/
│       ├── Russian_Blue/
│       ├── Siamese/
│       └── Sphynx/
│
├── models/
│   ├── cat_breeds_cnn.keras
│   └── class_names.json
│
├── results/
│   ├── accuracy.png
│   ├── loss.png
│   ├── confusion_matrix.png
│   ├── classification_report.txt
│   ├── cnn_architecture.png
│   ├── class_distribution.png
│   └── sample_images.png
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

Clone this repository:

```bash
git clone https://github.com/gefiryaa/Mechine-Learning-CNN-cat-breed.git
```

Navigate to the project directory:

```bash
cd Mechine-Learning-CNN-cat-breed
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Prepare the dataset in the following directory:

```text
dataset/cats-breads/
```

## How to Run

Run the main program:

```bash
python main.py
```

The program performs the following steps:

1. Load and preprocess the dataset.
2. Display sample images.
3. Visualize class distribution.
4. Build the CNN model.
5. Display the model architecture.
6. Train the model.
7. Visualize training and validation accuracy.
8. Visualize training and validation loss.
9. Evaluate model performance.
10. Optionally predict a cat breed from a new image.

## Model Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Initial recorded evaluation results:

| Metric | Result |
|---|---:|
| Validation Accuracy | 28% |
| Macro F1-score | 0.25 |
| Weighted F1-score | 0.25 |

The results indicate that the model still has room for improvement in distinguishing between the 12 cat breeds.

## Author

Developed as a Deep Learning project using Convolutional Neural Network (CNN).
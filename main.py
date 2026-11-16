
# ============================================================
# CAT BREEDS CLASSIFICATION USING CONVOLUTIONAL NEURAL NETWORK
# ============================================================

# 1. IMPORT LIBRARIES
# ============================================================

from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent

DATASET_DIR = ROOT_DIR / "dataset" / "cats-breads"
MODEL_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

MODEL_PATH = MODEL_DIR / "cat_breeds_cnn.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 15
SEED = 42

AUTOTUNE = tf.data.AUTOTUNE


# ============================================================
# 3. READ DATASET
# ============================================================


def load_dataset():

    print("\n===== 1. READ DATASET =====")

    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {DATASET_DIR}"
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    extensions = {
        ".jpg", ".jpeg", ".png", ".bmp", ".webp"
    }

    class_names = sorted([
        folder.name
        for folder in DATASET_DIR.iterdir()
        if folder.is_dir()
    ])

    if not class_names:
        raise ValueError("Tidak ada folder kelas dalam dataset.")

    train_paths = []
    train_labels = []

    val_paths = []
    val_labels = []

    rng = np.random.default_rng(SEED)

    print("\n===== STRATIFIED DATA SPLIT =====")

    for label, class_name in enumerate(class_names):

        class_dir = DATASET_DIR / class_name

        image_paths = sorted([
            str(file)
            for file in class_dir.iterdir()
            if file.is_file()
            and file.suffix.lower() in extensions
        ])

        if not image_paths:
            raise ValueError(
                f"Tidak ada gambar dalam kelas: {class_name}"
            )

        # Acak gambar dalam setiap kelas
        indices = rng.permutation(len(image_paths))

        # Ambil 20% untuk validation
        val_count = int(len(image_paths) * 0.2)

        val_indices = indices[:val_count]
        train_indices = indices[val_count:]

        for index in train_indices:
            train_paths.append(image_paths[index])
            train_labels.append(label)

        for index in val_indices:
            val_paths.append(image_paths[index])
            val_labels.append(label)

        print(
            f"{class_name:<20} "
            f"Train: {len(train_indices):<4} "
            f"Validation: {len(val_indices)}"
        )

    # Acak urutan seluruh data training
    train_order = rng.permutation(len(train_paths))

    train_paths = np.array(train_paths)[train_order]
    train_labels = np.array(train_labels)[train_order]

    val_paths = np.array(val_paths)
    val_labels = np.array(val_labels)

    # Simpan nama kelas
    with open(
        CLASS_NAMES_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(class_names, file, indent=4)

    # Fungsi membaca dan memproses gambar
    def process_image(image_path, label):

        image = tf.io.read_file(image_path)

        image = tf.io.decode_image(
            image,
            channels=3,
            expand_animations=False
        )

        image.set_shape([None, None, 3])

        image = tf.image.resize(
            image,
            IMAGE_SIZE
        )

        return image, label

    # Membuat training dataset
    train_ds = tf.data.Dataset.from_tensor_slices(
        (train_paths, train_labels)
    )

    train_ds = train_ds.shuffle(
        buffer_size=len(train_paths),
        seed=SEED,
        reshuffle_each_iteration=True
    )

    train_ds = train_ds.map(
        process_image,
        num_parallel_calls=AUTOTUNE
    )

    train_ds = train_ds.batch(BATCH_SIZE)
    train_ds = train_ds.prefetch(AUTOTUNE)

    # Membuat validation dataset
    val_ds = tf.data.Dataset.from_tensor_slices(
        (val_paths, val_labels)
    )

    val_ds = val_ds.map(
        process_image,
        num_parallel_calls=AUTOTUNE
    )

    val_ds = val_ds.batch(BATCH_SIZE)
    val_ds = val_ds.prefetch(AUTOTUNE)

    print("\n===== DATASET SUMMARY =====")
    print("Jumlah kelas:", len(class_names))
    print("Nama kelas:", class_names)

    print("Training images:", len(train_paths))
    print("Validation images:", len(val_paths))

    print("Training batches:", len(train_ds))
    print("Validation batches:", len(val_ds))

    return train_ds, val_ds, class_names



# ============================================================
# 4. DATASET VISUALIZATION
# ============================================================

def plot_dataset_samples(train_ds, class_names):

    print("\n===== 2. DATASET VISUALIZATION =====")

    images, labels = next(iter(train_ds))

    plt.figure(figsize=(12, 9))

    num_images = min(9, len(images))

    for i in range(num_images):

        plt.subplot(3, 3, i + 1)

        plt.imshow(
            images[i].numpy().astype("uint8")
        )

        label_index = int(labels[i].numpy())

        plt.title(class_names[label_index])

        plt.axis("off")

    plt.suptitle(
        "Sample Images - Cat Breeds Dataset",
        fontsize=16
    )

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "sample_images.png",
        dpi=300
    )

    plt.show()
    plt.close()


# ============================================================
# 5. DATASET CLASS DISTRIBUTION
# ============================================================

def plot_class_distribution(class_names):

    print("\n===== 3. CLASS DISTRIBUTION =====")

    extensions = {
        ".jpg", ".jpeg", ".png", ".bmp", ".webp"
    }

    class_counts = []

    for class_name in class_names:

        class_dir = DATASET_DIR / class_name

        total_images = sum(
            1
            for file in class_dir.rglob("*")
            if file.is_file()
            and file.suffix.lower() in extensions
        )

        class_counts.append(total_images)

    plt.figure(figsize=(12, 6))

    plt.bar(
        class_names,
        class_counts
    )

    plt.title("Image Distribution per Cat Breed")
    plt.xlabel("Cat Breeds")
    plt.ylabel("Number of Images")

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "class_distribution.png",
        dpi=300
    )

    plt.show()
    plt.close()


# ============================================================
# 6. BUILD CNN ARCHITECTURE
# ============================================================

def build_cnn_model(num_classes):

    print("\n===== 4. BUILD CNN MODEL =====")

    model = tf.keras.Sequential([

        tf.keras.layers.Input(
            shape=(128, 128, 3)
        ),

        # Normalization
        tf.keras.layers.Rescaling(1.0 / 255),

        # Data Augmentation
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.10),

        # Convolution Block 1
        tf.keras.layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        tf.keras.layers.MaxPooling2D(
            (2, 2)
        ),

        # Convolution Block 2
        tf.keras.layers.Conv2D(
            64,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        tf.keras.layers.MaxPooling2D(
            (2, 2)
        ),

        # Convolution Block 3
        tf.keras.layers.Conv2D(
            128,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        tf.keras.layers.MaxPooling2D(
            (2, 2)
        ),

        # Classification Layer
        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(
            128,
            activation="relu"
        ),

        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(
            num_classes,
            activation="softmax"
        )

    ], name="Cat_Breeds_CNN")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),

        loss="sparse_categorical_crossentropy",

        metrics=["accuracy"]
    )

    model.summary()

    return model


# ============================================================
# 7. CNN ARCHITECTURE DIAGRAM
# ============================================================

def plot_cnn_architecture():

    print("\n===== 5. CNN ARCHITECTURE DIAGRAM =====")

    fig, ax = plt.subplots(figsize=(16, 7))

    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7)

    ax.axis("off")

    layers_info = [

        ("Input", "128 x 128 x 3"),
        ("Rescaling", "1 / 255"),
        ("Data\nAugmentation", "Flip, Rotation,\nZoom"),
        ("Conv2D + Pooling", "32 filters"),
        ("Conv2D + Pooling", "64 filters"),
        ("Conv2D + Pooling", "128 filters"),
        ("Global Average\nPooling", "Feature Extraction"),
        ("Dense", "128 neurons"),
        ("Dropout", "0.3"),
        ("Output", "12 Classes\nSoftmax")

    ]

    box_width = 1.35
    box_height = 1.45

    x_positions = np.linspace(
        0.2,
        14.4,
        len(layers_info)
    )

    for i, (title, description) in enumerate(layers_info):

        x = x_positions[i]

        rect = patches.FancyBboxPatch(
            (x, 2.8),
            box_width,
            box_height,
            boxstyle="round,pad=0.08",
            linewidth=1.5,
            edgecolor="black",
            facecolor="lightblue"
        )

        ax.add_patch(rect)

        ax.text(
            x + box_width / 2,
            3.75,
            title,
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold"
        )

        ax.text(
            x + box_width / 2,
            3.15,
            description,
            ha="center",
            va="center",
            fontsize=7
        )

        if i < len(layers_info) - 1:

            ax.annotate(
                "",
                xy=(x_positions[i + 1] - 0.05, 3.5),
                xytext=(x + box_width + 0.05, 3.5),
                arrowprops=dict(
                    arrowstyle="->",
                    linewidth=1.5
                )
            )

    ax.text(
        8,
        5.5,
        "CNN Architecture - Cat Breeds Classification",
        ha="center",
        fontsize=16,
        fontweight="bold"
    )

    ax.text(
        8,
        1.5,
        "Optimizer: Adam | Learning Rate: 0.001 | "
        "Loss: Sparse Categorical Crossentropy",
        ha="center",
        fontsize=10
    )

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "cnn_architecture.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()


# ============================================================
# 8. TRAIN CNN MODEL
# ============================================================

def train_model(model, train_ds, val_ds):

    print("\n===== 6. MODEL TRAINING =====")

    callbacks = [

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        ),

        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_loss",
            save_best_only=True
        )

    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    model.save(MODEL_PATH)

    print("\nModel berhasil disimpan:", MODEL_PATH)

    return history


# ============================================================
# 9. TRAINING ACCURACY VISUALIZATION
# ============================================================

def plot_accuracy(history):

    print("\n===== 7. ACCURACY GRAPH =====")

    epochs_range = range(
        1,
        len(history.history["accuracy"]) + 1
    )

    plt.figure(figsize=(10, 6))

    plt.plot(
        epochs_range,
        history.history["accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        epochs_range,
        history.history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "accuracy.png",
        dpi=300
    )

    plt.show()
    plt.close()


# ============================================================
# 10. TRAINING LOSS VISUALIZATION
# ============================================================

def plot_loss(history):

    print("\n===== 8. LOSS GRAPH =====")

    epochs_range = range(
        1,
        len(history.history["loss"]) + 1
    )

    plt.figure(figsize=(10, 6))

    plt.plot(
        epochs_range,
        history.history["loss"],
        label="Training Loss"
    )

    plt.plot(
        epochs_range,
        history.history["val_loss"],
        label="Validation Loss"
    )

    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "loss.png",
        dpi=300
    )

    plt.show()
    plt.close()


# ============================================================
# 11. MODEL EVALUATION
# ============================================================

def evaluate_model(model, val_ds, class_names):

    print("\n===== 9. MODEL EVALUATION =====")

    loss, accuracy = model.evaluate(val_ds)

    print("\nValidation Loss     :", f"{loss:.4f}")
    print("Validation Accuracy :", f"{accuracy:.4f}")

    y_true = np.concatenate([
        labels.numpy()
        for images, labels in val_ds
    ])

    predictions = model.predict(
        val_ds,
        verbose=1
    )

    y_pred = np.argmax(
        predictions,
        axis=1
    )

    print("\n===== CLASSIFICATION REPORT =====")

    report = classification_report(
        y_true,
        y_pred,
        labels=range(len(class_names)),
        target_names=class_names,
        zero_division=0
    )

    print(report)

    with open(
        RESULTS_DIR / "classification_report.txt",
        "w",
        encoding="utf-8"
    ) as file:
        file.write(report)

    # Confusion Matrix

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=range(len(class_names))
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    fig, ax = plt.subplots(figsize=(12, 10))

    display.plot(
        ax=ax,
        xticks_rotation=45,
        cmap="Blues",
        colorbar=False
    )

    plt.title("Confusion Matrix - Cat Breeds")
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "confusion_matrix.png",
        dpi=300
    )

    plt.show()
    plt.close(fig)

    return loss, accuracy


# ============================================================
# 12. PREDICTION
# ============================================================

def predict_image(model, image_path, class_names):

    image_path = Path(image_path)

    if not image_path.is_absolute():
        image_path = ROOT_DIR / image_path

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Gambar tidak ditemukan: {image_path}"
        )

    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMAGE_SIZE
    )

    image_array = tf.keras.utils.img_to_array(
        image
    )

    image_array = tf.expand_dims(
        image_array,
        axis=0
    )

    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = np.argmax(predictions)

    predicted_class = class_names[predicted_index]

    confidence = float(
        predictions[predicted_index] * 100
    )

    print("\n===== PREDICTION RESULT =====")
    print("Predicted Breed :", predicted_class)
    print(f"Confidence      : {confidence:.2f}%")

    plt.figure(figsize=(6, 6))

    plt.imshow(
        tf.keras.utils.load_img(
            image_path
        )
    )

    plt.title(
        f"{predicted_class} ({confidence:.2f}%)"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "prediction_result.png",
        dpi=300
    )

    plt.show()
    plt.close()

    return predicted_class, confidence


# ============================================================
# 13. MAIN PROGRAM
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("       CAT BREEDS CLASSIFICATION USING CNN")
    print("=" * 60)

    # 1. Read Dataset
    train_ds, val_ds, class_names = load_dataset()

    # 2. Dataset Visualization
    plot_dataset_samples(
        train_ds,
        class_names
    )

    # 3. Class Distribution
    plot_class_distribution(
        class_names
    )

    # 4. Build CNN Model
    model = build_cnn_model(
        num_classes=len(class_names)
    )

    # 5. Architecture Diagram
    plot_cnn_architecture()

    # 6. Model Training
    history = train_model(
        model,
        train_ds,
        val_ds
    )

    # 7. Accuracy Visualization
    plot_accuracy(history)

    # 8. Loss Visualization
    plot_loss(history)

    # 9. Model Evaluation
    evaluate_model(
        model,
        val_ds,
        class_names
    )

    # 10. Optional Prediction
    print("\n===== OPTIONAL IMAGE PREDICTION =====")

    image_path = input(
        "Masukkan path gambar kucing "
        "(tekan Enter untuk melewati): "
    ).strip()

    if image_path:

        predict_image(
            model,
            image_path,
            class_names
        )

    print("\n" + "=" * 60)
    print("       CNN PROCESS COMPLETED")
    print("=" * 60)


# ============================================================
# 14. EXECUTION
# ============================================================

if __name__ == "__main__":
    main()
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf

ROOT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT_DIR / "models" / "cat_breeds_cnn.keras"
RESULTS_DIR = ROOT_DIR / "results"


def train_model(model, train_ds, val_ds, epochs=15):

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

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
        epochs=epochs,
        callbacks=callbacks
    )

    plot_training_history(history)

    print("Model berhasil disimpan:", MODEL_PATH)

    return history



def plot_training_history(history):

    epochs_range = range(
        1,
        len(history.history["accuracy"]) + 1
    )

    # Accuracy Graph
    plt.figure(figsize=(8, 5))

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

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "accuracy.png")
    plt.show()

    # Loss Graph
    plt.figure(figsize=(8, 5))

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

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "loss.png")
    plt.show()
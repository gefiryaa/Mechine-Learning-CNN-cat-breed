
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"


def evaluate_model(model, val_ds, class_names):

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    loss, accuracy = model.evaluate(val_ds)

    print("\n===== MODEL EVALUATION =====")
    print(f"Validation Loss     : {loss:.4f}")
    print(f"Validation Accuracy : {accuracy:.4f}")

    y_true = np.concatenate([
        labels.numpy()
        for images, labels in val_ds
    ])

    predictions = model.predict(val_ds)

    y_pred = np.argmax(predictions, axis=1)

    print("\n===== CLASSIFICATION REPORT =====")

    print(
        classification_report(
            y_true,
            y_pred,
            labels=range(len(class_names)),
            target_names=class_names,
            zero_division=0
        )
    )

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

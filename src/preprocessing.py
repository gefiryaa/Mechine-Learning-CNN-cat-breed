
from pathlib import Path
import json
import numpy as np
import tensorflow as tf

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT_DIR / "dataset" / "cats-breads"
MODEL_DIR = ROOT_DIR / "models"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


def load_datasets():

    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {DATASET_DIR}"
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    class_names = sorted([
        folder.name
        for folder in DATASET_DIR.iterdir()
        if folder.is_dir()
    ])

    if not class_names:
        raise ValueError(
            "Tidak ada folder kelas dalam dataset."
        )

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
            and file.suffix.lower() in IMAGE_EXTENSIONS
        ])

        if not image_paths:
            raise ValueError(
                f"Tidak ada gambar dalam kelas: {class_name}"
            )

        indices = rng.permutation(len(image_paths))

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

    train_order = rng.permutation(len(train_paths))

    train_paths = np.array(train_paths)[train_order]
    train_labels = np.array(train_labels)[train_order]

    val_paths = np.array(val_paths)
    val_labels = np.array(val_labels)

    with open(
        CLASS_NAMES_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(class_names, file, indent=4)

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

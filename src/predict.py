
from pathlib import Path
import json
import numpy as np
import tensorflow as tf

ROOT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT_DIR / "models" / "cat_breeds_cnn.keras"
CLASS_NAMES_PATH = ROOT_DIR / "models" / "class_names.json"

IMAGE_SIZE = (128, 128)


def predict_image(image_path):

    image_path = Path(image_path)

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model belum ditemukan: {MODEL_PATH}"
        )

    if not CLASS_NAMES_PATH.is_file():
        raise FileNotFoundError(
            f"File nama kelas belum ditemukan: {CLASS_NAMES_PATH}"
        )

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Gambar tidak ditemukan: {image_path}"
        )

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(
        CLASS_NAMES_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        class_names = json.load(file)

    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMAGE_SIZE
    )

    image_array = tf.keras.utils.img_to_array(image)
    image_array = tf.expand_dims(image_array, axis=0)

    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]
    confidence = float(predictions[predicted_index]) * 100

    print("\n===== PREDICTION RESULT =====")
    print("Predicted Breed :", predicted_class)
    print(f"Confidence      : {confidence:.2f}%")

    return predicted_class, confidence

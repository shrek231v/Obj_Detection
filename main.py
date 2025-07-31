
import os
import tensorflow as tf
import numpy as np
from PIL import Image

# Select GPU if available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.set_visible_devices(gpus[0], 'GPU')
        print(f"Using GPU: {gpus[0]}")
    except RuntimeError as e:
        print(f"GPU setup error: {e}")
else:
    print("No GPU found, using CPU.")


# Set your data directory
DATA_DIR = 'data'  # Images/annotations are in the 'data' folder

# Set the input image size (resolution)
IMAGE_SIZE = 448  # Change this to control the quality/resolution

def load_yolo_annotation(txt_path):
    boxes = []
    classes = []
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id, x_center, y_center, width, height = parts
            class_id = int(class_id)
            x_center = float(x_center)
            y_center = float(y_center)
            width = float(width)
            height = float(height)
            # Store as YOLO normalized (x_center, y_center, width, height)
            boxes.append([x_center, y_center, width, height])
            classes.append(class_id)
    return np.array(boxes), np.array(classes)

def load_data(data_dir):
    images = []
    targets = []
    MAX_OBJECTS = 5  # maximum number of objects per image
    for fname in os.listdir(data_dir):
        if fname.endswith('.jpg') or fname.endswith('.png'):
            img_path = os.path.join(data_dir, fname)
            txt_path = os.path.splitext(img_path)[0] + '.txt'
            if not os.path.exists(txt_path):
                continue
            img = Image.open(img_path).convert('RGB')
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
            img_arr = np.array(img) / 255.0
            boxes, classes = load_yolo_annotation(txt_path)
            # Pad or truncate to MAX_OBJECTS
            n = min(len(boxes), MAX_OBJECTS)
            target = np.zeros((MAX_OBJECTS, 5), dtype=np.float32)
            if n > 0:
                target[:n, :4] = boxes[:n]
                target[:n, 4] = classes[:n]
            images.append(img_arr)
            targets.append(target)
    return np.array(images), np.array(targets)

# Load data
images, targets = load_data(DATA_DIR)

inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
x = tf.keras.layers.Conv2D(16, 3, activation='relu')(inputs)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(32, 3, activation='relu')(x)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
outputs = tf.keras.layers.Dense(5 * 5)(x)  # MAX_OBJECTS * 5
outputs = tf.keras.layers.Reshape((5, 5))(outputs)  # (MAX_OBJECTS, 5)
model = tf.keras.Model(inputs, outputs)

model.compile(optimizer='adam', loss='mse')

# Train
model.fit(images, targets, epochs=200)

# Export the trained model
model.save('exported_model.keras')
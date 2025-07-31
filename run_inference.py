import os
import numpy as np
import tensorflow as tf
from PIL import Image

def run_inference(model_path, test_dir, output_dir):
    IMAGE_SIZE = 448  # Must match training image size
    model = tf.keras.models.load_model(model_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for fname in os.listdir(test_dir):
        if fname.endswith('.jpg') or fname.endswith('.png'):
            img_path = os.path.join(test_dir, fname)
            img = Image.open(img_path).convert('RGB')
            orig_w, orig_h = img.size
            img_resized = img.resize((IMAGE_SIZE, IMAGE_SIZE))
            img_arr = np.array(img_resized) / 255.0
            img_arr = np.expand_dims(img_arr, axis=0)
            preds = model.predict(img_arr)[0]  # shape: (5, 5)
            out_txt = os.path.join(output_dir, os.path.splitext(fname)[0] + '.txt')
            with open(out_txt, 'w') as f:
                for pred in preds:
                    x_center, y_center, width, height, class_id = pred
                    # Skip if all values are zero (padding)
                    if np.allclose([x_center, y_center, width, height, class_id], 0):
                        continue
                    class_id = int(round(class_id))
                    # Clamp values to [0, 1]
                    x_center = min(max(x_center, 0.0), 1.0)
                    y_center = min(max(y_center, 0.0), 1.0)
                    width = min(max(width, 0.0), 1.0)
                    height = min(max(height, 0.0), 1.0)
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

if __name__ == "__main__":
    run_inference(
        model_path='exported_model.keras',
        test_dir='test_images',
        output_dir='test_outputs'
    )

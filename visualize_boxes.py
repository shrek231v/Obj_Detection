import os
from PIL import Image, ImageDraw

def draw_boxes_on_images(test_dir, yolo_dir, save_dir, inference_size=224):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for fname in os.listdir(test_dir):
        if fname.endswith('.jpg') or fname.endswith('.png'):
            img_path = os.path.join(test_dir, fname)
            yolo_path = os.path.join(yolo_dir, os.path.splitext(fname)[0] + '.txt')
            if not os.path.exists(yolo_path):
                continue
            img = Image.open(img_path).convert('RGB')
            orig_w, orig_h = img.size
            draw = ImageDraw.Draw(img)

            with open(yolo_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        # YOLO format: class_id x_center y_center width height (all normalized)
                        class_id, x_center, y_center, width, height = parts
                        class_id = int(class_id)
                        x_center = float(x_center)
                        y_center = float(y_center)
                        width = float(width)
                        height = float(height)
                        # Skip if all values are zero (padding)
                        if abs(class_id) < 1e-6 and abs(x_center) < 1e-6 and abs(y_center) < 1e-6 and abs(width) < 1e-6 and abs(height) < 1e-6:
                            continue
                        # Convert to pixel coordinates
                        x_center_pix = x_center * orig_w
                        y_center_pix = y_center * orig_h
                        width_pix = width * orig_w
                        height_pix = height * orig_h
                        x1 = int(round(x_center_pix - width_pix / 2))
                        y1 = int(round(y_center_pix - height_pix / 2))
                        x2 = int(round(x_center_pix + width_pix / 2))
                        y2 = int(round(y_center_pix + height_pix / 2))
                        print(f"Image size: {img.size}")
                        # Draw rectangle
                        draw.rectangle([x1, y1, x2, y2], outline='red', width=12)
                        try:
                            from PIL import ImageFont
                            font = ImageFont.truetype("arial.ttf", 128)
                        except:
                            font = None
                        # Draw label near top-left corner
                        draw.text((x1 + 10, y1 - 140), f"Class {class_id}", fill='red', font=font)
                    else:
                        continue
            save_path = os.path.join(save_dir, fname)
            img.save(save_path)

if __name__ == "__main__":
    draw_boxes_on_images(
        test_dir='test_images',
        yolo_dir='test_outputs',
        save_dir='test_images_with_boxes',
        inference_size=224
    )

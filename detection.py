from ultralytics import YOLO
import cv2
import os

# Load the pre-trained YOLOv8 model (for object detection)
model = YOLO('yolov8n.pt')  # You can use 'yolov8n.pt', 'yolov8s.pt', etc. for different model sizes

# Load an image where you want to detect coins
image_path = 'coins2.jpg'
image = cv2.imread(image_path)

# Run detection on the image
results = model(image)

# Create output directory for detected coins
output_dir = 'extracted_coins'
os.makedirs(output_dir, exist_ok=True)

# Check if results are returned and process them
if results:
    for i, result in enumerate(results):  # Loop through each result if there are multiple
        boxes = result.boxes.xyxy  # Get bounding boxes
        for j, box in enumerate(boxes):
            # Get coordinates of the bounding box
            x1, y1, x2, y2 = map(int, box)  # Convert to integer coordinates
            
            # Crop the detected coin from the original image
            coin_image = image[y1:y2, x1:x2]
            
            # Save the cropped image
            coin_image_path = os.path.join(output_dir, f'coin_{i}_{j}.jpg')
            cv2.imwrite(coin_image_path, coin_image)
            print(f'Saved: {coin_image_path}')
            
            # Draw the bounding box on the original image
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue color box

# Save the original image with bounding boxes
output_image_path = 'output_image_with_boxes.jpg'
cv2.imwrite(output_image_path, image)
print(f'Saved: {output_image_path}')

# Optionally, display the image with bounding boxes
cv2.imshow('Detected Coins', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

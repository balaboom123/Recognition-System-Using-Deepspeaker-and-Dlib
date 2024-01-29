from PIL import Image
import numpy as np
from utils.root_path import root
print("hello world")
def slice_image(image_path):
    # Load the image
    image = Image.open(image_path)

    # Calculate the size of each grid cell
    width, height = image.size
    cell_width = width // 3
    cell_height = height // 3

    # Create a list to hold the image slices
    image_slices = []

    # Slice the image into a 3x3 grid
    for i in range(3):
        for j in range(3):
            left = j * cell_width
            upper = i * cell_height
            right = (j + 1) * cell_width
            lower = (i + 1) * cell_height
            box = (left, upper, right, lower)
            slice = image.crop(box)
            image_slices.append(slice)

    # Save each slice
    output_paths = []
    for index, slice in enumerate(image_slices):
        output_path = f'segment_image_{index}.png'
        slice.save(output_path)
        output_paths.append(output_path)

    return output_paths


import numpy as np
import cv2
from utils.root_path import root


def segment_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply SLIC and obtain the segmentation map
    segments_slic = slic(image, n_segments=100, compactness=10, sigma=1)

    # Create a segmented image
    segmented_image = label2rgb(segments_slic, image, kind='avg')

    # Convert to uint8, as cv2.imencode expects an image with uint8 data type
    segmented_image = (255 * segmented_image).astype(np.uint8)

    # Save the segmented image
    output_path = f'image.png'
    cv2.imwrite(output_path, cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))

    return output_path




from ultralytics import YOLO
def yolo_predict():
    model = YOLO(
        r"C:\Users\user\Github\biometric_recognition\biometric_recognition\runs\detect\train1\weights\best.pt")
    result = model.predict(
        source=[r"C:\Users\user\Github\\biometric_recognition\\biometric_recognition\\17AC0C48-EB15-49CF-881E-4E85207E723F.jpg",
                r"C:\Users\user\Github\biometric_recognition\biometric_recognition\data\yolo_data\train\images\S1052020_5.png",
                r"D:\user\Downloads\S1052020_5.png",
                r"C:\Users\user\Github\biometric_recognition\biometric_recognition\S__128835606.jpg"],
        mode="predict",
        save=True,
        device="cpu"
    )







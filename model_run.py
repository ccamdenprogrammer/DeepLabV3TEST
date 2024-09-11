import torch
import torchvision
from torchvision import transforms

import os
import numpy as np
import cv2
from PIL import Image

def run(image_name):
    # Load the pre-trained DeepLabV3 model
    model = torchvision.models.segmentation.deeplabv3_resnet50(weights="DEFAULT")
    model.eval()

    # Define the image transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.486], std=[0.229, 0.224, 0.225])
    ])

    # Load and transform the image
    img = Image.open(image_name)

    with torch.no_grad():
        pred = model(transform(img)[None, ...])
    
    # Get the predicted segmentation output
    output = pred["out"].squeeze().argmax(0)
    names = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
             "cow", "dining_table", "dog", "horse", "bike", "person", "plant", "sheep", "sofa",
             "train", "tv"]

    all_objects = []
    all_segments = []
    all_contours = []

    real = cv2.imread(image_name)
    real_rgb = cv2.cvtColor(real, cv2.COLOR_BGR2RGB)  # Convert to RGB for visualization

    for i in range(output.unique().shape[0] - 1):
        num = output.unique()[i + 1]
        all_objects.append(names[num - 1])

        # Create a mask for the current class
        temp = torch.zeros_like(output)
        temp[output == num] = 255
        mask = temp.numpy().astype("uint8")

        # Find contours of the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours on the original image
        contour_image = real_rgb.copy()
        cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)  # Green contours

        # Save segmented image with contours
        segmented_with_contours = cv2.bitwise_and(contour_image, contour_image, mask=mask)
        all_segments.append(segmented_with_contours)

        # Save the image with the segmented regions
        real[mask != 255] = (255, 255, 255)
        all_segments.append(real.copy())

    return all_objects, all_segments

# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 14:19:02 2024

@author: abudh
"""

from PIL import Image

# Open the image
img_path = r"C:\Users\abudh\Desktop\sc2024-num-methods\Satellite_Image.jpg"
img = Image.open(img_path)

# Get the dimensions of the image
width, height = img.size

# Calculate the dimensions of each smaller image
tile_width = width // 3
tile_height = height // 3

# Create and save each of the 9 smaller images
for i in range(3):
    for j in range(3):
        left = j * tile_width
        upper = i * tile_height
        right = (j + 1) * tile_width
        lower = (i + 1) * tile_height
        
        # Crop the image
        cropped_img = img.crop((left, upper, right, lower))
        
        # Save the cropped image
        cropped_img.save(f"C:/Users/abudh/Desktop/image_part_{i * 3 + j + 1}.png")

# List the output files
output_files = [f"C:/Users/abudh/Desktop/image_part_{i * 3 + j + 1}.png" for i in range(3) for j in range(3)]
output_files

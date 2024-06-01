import os
import random

def is_image_file(filename):
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')
    return filename.lower().endswith(image_extensions)

def create_file_pairs(directory, output_file):
    # List all image files in the directory
    image_files = [f for f in os.listdir(directory) if is_image_file(f)]
    
    # Create a shuffled version of the list
    shuffled_files = image_files[:]
    random.shuffle(shuffled_files)
    
    # Write the pairs to the output file
    with open(output_file, 'w') as file:
        for img, shuffled_img in zip(image_files, shuffled_files):
            file.write(f"{img} {shuffled_img}\n")

# Set the directory and output file path
directory = '/notebooks/ayna/working_repo/IDM-VTON/dataset/deepfashion_dataset/train/agnostic-mask'
output_file = 'train_pairs.txt'

# Create the file pairs
create_file_pairs(directory, output_file)


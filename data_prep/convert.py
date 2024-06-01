## PNG to JPG
# import os
# from PIL import Image

# def convert_png_to_jpg(directory):
#     for filename in os.listdir(directory):
#         if filename.endswith('.png'):
#             img_path = os.path.join(directory, filename)
#             img = Image.open(img_path)
#             rgb_img = img.convert('RGB')
#             new_filename = os.path.splitext(filename)[0] + '.jpg'
#             rgb_img.save(os.path.join(directory, new_filename), 'JPEG')
#             print(f'Converted {filename} to {new_filename}')
#             os.remove(img_path)
#             print(f'Deleted original file {filename}')
    
#     print('Conversion complete.')

# # Example usage
# directory = '/notebooks/ayna/working_repo/IDM-VTON/dataset/deepfashion_dataset/train/image-densepose'
# convert_png_to_jpg(directory)

### JPG to PNG
import os
from PIL import Image

def convert_jpg_to_png(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.jpg'):
            img_path = os.path.join(directory, filename)
            img = Image.open(img_path)
            new_filename = os.path.splitext(filename)[0] + '.png'
            img.save(os.path.join(directory, new_filename), 'PNG')
            print(f'Converted {filename} to {new_filename}')
            os.remove(img_path)
            print(f'Deleted original file {filename}')
    
    print('Conversion complete.')

# Example usage
directory = '/notebooks/ayna/working_repo/IDM-VTON/dataset/deepfashion_dataset/train/agnostic-mask/'
convert_jpg_to_png(directory)

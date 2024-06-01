# import os

# def rename_files(directory):
#     for filename in os.listdir(directory):
#         file_path = os.path.join(directory, filename)
        
#         # Skip directories
#         if os.path.isdir(file_path):
#             continue
        
#         # Split the filename and the extension
#         name, ext = os.path.splitext(filename)
        
#         # Create the new filename
#         new_filename = f"{name}_densepose{ext}"
        
#         # Create the full path for the new filename
#         new_file_path = os.path.join(directory, new_filename)
        
#         # Rename the file
#         os.rename(file_path, new_file_path)
#         print(f'Renamed {filename} to {new_filename}')
    
#     print('Renaming complete.')

# # Example usage
# directory = '/notebooks/ayna/working_repo/IDM-VTON/dataset/deepfashion_dataset/train/image-densepose/'
# rename_files(directory)



# REVERT CHANGE
import os

def remove_densepose_suffix(directory):
    suffix = "_densepose"
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        # Skip directories
        if os.path.isdir(file_path):
            continue
        
        # Split the filename and the extension
        name, ext = os.path.splitext(filename)
        
        # Check if the filename ends with the suffix and remove it
        if name.endswith(suffix):
            new_name = name[:-len(suffix)]
            new_filename = f"{new_name}{ext}"
            
            # Create the full path for the new filename
            new_file_path = os.path.join(directory, new_filename)
            
            # Rename the file
            os.rename(file_path, new_file_path)
            print(f'Renamed {filename} to {new_filename}')
    
    print('Renaming complete.')

# Example usage
directory = '/notebooks/ayna/working_repo/IDM-VTON/dataset/deepfashion_dataset/train/image-densepose/'
remove_densepose_suffix(directory)

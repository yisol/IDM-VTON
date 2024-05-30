import json
import os

train_dir = 'dataset/train/image/'
data = []

if os.path.exists(train_dir) and os.path.isdir(train_dir):
    # List all files in the directory
    files = os.listdir(train_dir)
    
    
    for file in files:
        if file.endswith('.jpg'):
            # print("file: ", file)
            obj = {
                "file_name": file,
                "category_name": "TOPS",
                "tag_info": [
                    {"tag_name": "item", "tag_category": None},
                    {"tag_name": "details", "tag_category": None},
                    {"tag_name": "looks", "tag_category": None},
                    {"tag_name": "colors", "tag_category": None},
                    {"tag_name": "prints", "tag_category": None},
                    {"tag_name": "textures", "tag_category": None},
                    {"tag_name": "sleeveLength", "tag_category": None},
                    {"tag_name": "length", "tag_category": None},
                    {"tag_name": "neckLine", "tag_category": None},
                    {"tag_name": "fit", "tag_category": None},
                    {"tag_name": "shape", "tag_category": None}
                ]
            }
            data.append(obj)

data_obj = {"data": data}           

output_file = 'vitonhd_train_tagged.json'
with open(output_file, 'w') as f:
    json.dump(data_obj, f, indent=4)


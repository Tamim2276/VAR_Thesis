import os
import zipfile
import shutil
from tqdm import tqdm

def zip_for_kaggle():
    output_filename = "kaggle_upload.zip"
    
    # Folders to include
    folders_to_zip = [
        "src",
        "data/features", 
        "data/annotations" # Assuming annotations are here or in root. Let's just zip 'data' minus 'frames'
    ]
    
    print(f"Creating {output_filename} for Kaggle...")
    
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Zip SRC
        for root, dirs, files in os.walk("src"):
            if "__pycache__" in root: continue
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, file_path)

        # Zip Features
        for root, dirs, files in os.walk("data/features"):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, file_path)

        # Zip strictly the annotations.json files
        for split in ["train", "valid", "test"]:
            ann_path = os.path.join("data", "soccerNet", "mvfouls", split, "annotations.json")
            if os.path.exists(ann_path):
                zipf.write(ann_path, ann_path)

    print(f"Success! {output_filename} is ready to be uploaded to Kaggle.")
    print(f"File size: {os.path.getsize(output_filename) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    zip_for_kaggle()

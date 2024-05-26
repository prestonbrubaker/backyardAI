import os
from PIL import Image, ImageFile
import imagehash

# Handle truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

def remove_duplicate_photos(folder_path):
    previous_hash = None
    files_to_remove = []

    for root, dirs, files in os.walk(folder_path):
        for file in sorted(files):  # Ensure files are processed in order
            if file.endswith('.jpg'):
                file_path = os.path.join(root, file)
                try:
                    with Image.open(file_path) as img:
                        current_hash = imagehash.average_hash(img)
                        
                        if previous_hash is not None and current_hash == previous_hash:
                            files_to_remove.append(file_path)
                        else:
                            previous_hash = current_hash
                except OSError as e:
                    print(f"Skipping file {file_path} due to an error: {e}")

    # Remove duplicate files
    for file_path in files_to_remove:
        os.remove(file_path)
        print(f"Removed duplicate photo: {file_path}")

if __name__ == "__main__":
    folder_path = "photos"  # Path to the folder containing photos
    remove_duplicate_photos(folder_path)

import os
from PIL import Image

# Define the source and destination directories
source_dir = os.path.expanduser('~/Desktop/presloh_camera/photos/')
destination_dir = os.path.expanduser('~/Desktop/presloh_camera/photos_flipped/')

# Ensure the destination directory exists
os.makedirs(destination_dir, exist_ok=True)

def flip_image(image_path, save_path):
    try:
        # Open an image file
        with Image.open(image_path) as img:
            # Flip the image 180 degrees
            flipped_img = img.rotate(180)
            # Save the flipped image to the destination path
            flipped_img.save(save_path)
            print(f'Successfully flipped and saved {image_path} to {save_path}')
    except Exception as e:
        print(f'Error processing {image_path}: {e}')

def main():
    # Iterate over all files in the source directory
    for filename in os.listdir(source_dir):
        # Construct full file path
        source_file = os.path.join(source_dir, filename)
        destination_file = os.path.join(destination_dir, filename)
        # Only process files with common image extensions
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            flip_image(source_file, destination_file)

if __name__ == '__main__':
    main()

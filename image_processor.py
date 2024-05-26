import os
from PIL import Image, ImageOps

def process_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, dirs, files in os.walk(input_folder):
        for file in sorted(files):  # Ensure files are processed in order
            if file.endswith('.jpg'):
                file_path = os.path.join(root, file)
                try:
                    with Image.open(file_path) as img:
                        # Rotate the image 180 degrees
                        rotated_img = img.rotate(180)

                        # Convert to grayscale
                        grayscale_img = ImageOps.grayscale(rotated_img)

                        # Resize image while maintaining aspect ratio, ensuring the smaller dimension fits 256 pixels
                        width, height = grayscale_img.size
                        if width > height:
                            new_height = 256
                            new_width = int((256 / height) * width)
                        else:
                            new_width = 256
                            new_height = int((256 / width) * height)

                        resized_img = grayscale_img.resize((new_width, new_height), Image.LANCZOS)

                        # Crop the image to 256x256, ensuring to take the left side of the original (right side of the rotated image)
                        if new_width > 256:
                            left = 0
                            right = 256
                            top = (new_height - 256) / 2
                            bottom = (new_height + 256) / 2
                        else:
                            left = (new_width - 256) / 2
                            right = (new_width + 256) / 2
                            top = 0
                            bottom = 256

                        cropped_img = resized_img.crop((left, top, right, bottom))

                        # Save the processed image as PNG
                        output_file_path = os.path.join(output_folder, os.path.splitext(file)[0] + '.png')
                        cropped_img.save(output_file_path, 'PNG')
                        print(f"Processed and saved: {output_file_path}")

                except Exception as e:
                    print(f"Skipping file {file_path} due to an error: {e}")

if __name__ == "__main__":
    input_folder = "photos"  # Path to the folder containing original photos
    output_folder = "photos_processed"  # Path to the folder to save processed photos
    process_images(input_folder, output_folder)

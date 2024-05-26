import os
import torch
import numpy as np
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from vae_model import VAE

# Number of Latent Variables
latent_dim = 24  # Adjust this to match the latent dimensions used during training

# Load the trained VAE model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = VAE(latent_dim=latent_dim).to(device)
vae.load_state_dict(torch.load("vae.pth"))
vae.eval()

# Define the transform for the images
transform = transforms.Compose([
    transforms.ToTensor()
])

# Function to add red highlights based on squared differences
def add_red_highlight(original_img, recreated_img):
    original_np = np.array(original_img)
    recreated_np = np.array(recreated_img)

    # Calculate the squared difference
    squared_diff = (original_np - recreated_np) ** 2

    # Normalize the squared difference to range [0, 255]
    max_diff = np.max(squared_diff)
    if max_diff > 0:
        scaled_diff = (squared_diff / max_diff * 255).astype(np.uint8)
    else:
        scaled_diff = np.zeros_like(squared_diff, dtype=np.uint8)

    # Create the red highlight image
    red_highlight = np.zeros((256, 256, 3), dtype=np.uint8)
    red_highlight[..., 0] = scaled_diff  # Red channel

    # Combine the original image and the red highlight
    highlighted_img = np.stack([original_np] * 3, axis=-1) + red_highlight
    highlighted_img = np.clip(highlighted_img, 0, 255).astype(np.uint8)

    return Image.fromarray(highlighted_img)

# Function to process images and save highlighted images
def process_images(input_folder, output_highlight_folder, output_reconstruction_folder):
    if not os.path.exists(output_highlight_folder):
        os.makedirs(output_highlight_folder)
    
    if not os.path.exists(output_reconstruction_folder):
        os.makedirs(output_reconstruction_folder)

    for root, dirs, files in os.walk(input_folder):
        for file in sorted(files):  # Ensure files are processed in order
            if file.endswith('.png'):
                file_path = os.path.join(root, file)
                try:
                    # Load and transform the image
                    original_img = Image.open(file_path).convert('L')
                    img_tensor = transform(original_img).unsqueeze(0).to(device)

                    # Recreate the image using the VAE
                    with torch.no_grad():
                        recon_tensor, _, _ = vae(img_tensor)
                    recon_img = transforms.ToPILImage()(recon_tensor.squeeze(0).cpu())

                    # Add red highlights to the image
                    highlighted_img = add_red_highlight(original_img, recon_img)

                    # Save the highlighted image
                    output_highlight_path = os.path.join(output_highlight_folder, file)
                    highlighted_img.save(output_highlight_path)
                    
                    # Save the grayscale reconstructed image
                    output_reconstruction_path = os.path.join(output_reconstruction_folder, file)
                    recon_img.save(output_reconstruction_path)
                    
                    print(f"Processed and saved: {output_highlight_path}")
                    print(f"Reconstructed and saved: {output_reconstruction_path}")

                except Exception as e:
                    print(f"Skipping file {file_path} due to an error: {e}")

if __name__ == "__main__":
    input_folder = "photos_processed"  # Path to the folder containing processed photos
    output_highlight_folder = "photos_vae_processed"  # Path to the folder to save highlighted photos
    output_reconstruction_folder = "photos_vae_reconstruction"  # Path to the folder to save reconstructed photos
    process_images(input_folder, output_highlight_folder, output_reconstruction_folder)

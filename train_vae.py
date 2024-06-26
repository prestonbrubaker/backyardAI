import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from vae_model import VAE, loss_function

class CustomImageDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.image_files[idx])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return image

transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = CustomImageDataset(image_folder='photos_processed', transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = VAE().to(device)

# Load the model if it exists
model_path = "vae.pth"
if os.path.exists(model_path):
    vae.load_state_dict(torch.load(model_path))
    print("Loaded existing model from vae.pth")


optimizer = optim.Adam(vae.parameters(), lr=0.0001)

num_epochs = 5000
save_interval = 100

for epoch in range(num_epochs):
    vae.train()
    train_loss = 0
    train_bce = 0
    train_kld = 0
    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = vae(batch)
        bce, kld = loss_function(recon_batch, batch, mu, logvar)
        loss = bce + kld
        loss.backward()
        train_loss += loss.item()
        train_bce += bce.item()
        train_kld += kld.item()
        optimizer.step()
    
    print(f"Epoch {epoch + 1}, Total Loss: {train_loss / len(dataloader.dataset)}, BCE: {train_bce / len(dataloader.dataset)}, KLD: {train_kld / len(dataloader.dataset)}")
    
    if (epoch + 1) % save_interval == 0:
        torch.save(vae.state_dict(), f"vae.pth")
        print("MODEL SAVED")
        with torch.no_grad():
            vae.eval()
            for i, batch in enumerate(dataloader):
                batch = batch.to(device)
                recon_batch, _, _ = vae(batch)
                comparison = torch.cat([batch, recon_batch])
                save_image(comparison.cpu(), f'reconstruction_{epoch + 1}_{i}.png', nrow=batch.size(0))
                break

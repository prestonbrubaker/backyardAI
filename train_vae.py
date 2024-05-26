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
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = VAE().to(device)
optimizer = optim.Adam(vae.parameters(), lr=0.00001)

num_epochs = 5000
save_interval = 100

for epoch in range(num_epochs):
    vae.train()
    train_loss = 0
    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = vae(batch)
        loss = loss_function(recon_batch, batch, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    print(f"Epoch {epoch + 1}, Loss: {train_loss / len(dataloader.dataset)}")
    
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

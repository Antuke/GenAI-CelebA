import torch
import torch.nn as nn
import vae_utils as vae_utils
from models .vae_model import VAE_model
from torchvision import utils
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image

### CUSTOM DATASET ###
class BeardedWoman(Dataset):
    def __init__(self, folder_path, label=[0, 0, 1], transform=None):
        self.img_dir = Path(folder_path)
        self.label = torch.tensor(label, dtype=torch.float32)
        self.transform = transform

        self.image_paths = list(self.img_dir.glob('*png'))

        print(f"Found {len(self.image_paths)} images in {folder_path}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, self.label

### SETTINGS ###

beta=1.0
epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epoch_restart = 0

model_save_path = "vae_conv_batch_fullset_50e.pt"

## MODEL & OPTIMIZER ##

model = VAE_model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
model = vae_utils.load_checkpoint(model,None,f'./weights/vae_res.pt')



reconstruction_loss_function=nn.MSELoss(reduction='sum')
# nel corso utilizziamo la BCELoss ma avete fatto messo immagine tra [-1,1] invece a [0,1]
# e la BCELoss non funziona con -1 1
# quindi possiamo provare con la MSE, altrimenti provero con BCE piu un rescale

def kl_loss_function(mu, log_sigma):
    kl=0.5*(mu**2 + torch.exp(2*log_sigma)-1-2*log_sigma)
    return torch.sum(kl)

def loss_function(reconstructed, original, mu, log_sigma):
    rec = reconstruction_loss_function(reconstructed, original) 
    kl = kl_loss_function(mu, log_sigma)
    tot = rec + beta*kl
    return tot,rec,kl

### DATA LOADING ###
from torchvision.transforms import v2
from torch.utils.data import DataLoader
transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.RandomHorizontalFlip(),         # horizontal flip to add variety
        v2.Lambda(lambda x: (x * 2) - 1)   # scaling to [-1,1], to use tanh instead of sigmoid, apparently better
    ])
dataset = BeardedWoman(folder_path='../barbute', transform=transform)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

### GENERATE SAMPLES ###

classes = {
    0 : [0,0,0],
    1 : [0,0,1],
    2 : [0,1,0],
    3 : [0,1,1],
    4 : [1,0,0],
    5 : [1,0,1],
    6 : [1,1,0],
    7 : [1,1,1]
}


def generate_grid(path, network,  noise_size, epoch):
    labels = torch.zeros((noise_size[0],3), device=device)
    z = torch.randn(noise_size, device=device)
    for i in range(8):
        labels[(i * 10):(i+1)*10] = torch.tensor(classes[i], dtype=torch.float32, device=device)


    images = network.decode(z, labels)
    grid = utils.make_grid(images, normalize=True, nrow=10)
    sample_path = f'{path}vae_barbute_{epoch}.png'
    utils.save_image(grid, sample_path)


### TRAINING FUNCTION ###
from tqdm import tqdm
def vae_train_one_epoch(model, data_loader, optimizer):
    model.train()
    rec_loss = 0.0
    kl_loss = 0.0
    tot_loss = 0.0
    data_loader_ = tqdm(data_loader)
    for x, cond in data_loader_:
        x = x.to(device)
        cond = cond.to(device)
        recon_x, mu, log_sigma = model(x, cond)
        loss, rec, kl = loss_function(recon_x, x, mu, log_sigma)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
        rec_loss += rec.item()
        kl_loss += kl.item()
    return tot_loss / len(data_loader.dataset), rec_loss / len(data_loader.dataset), kl_loss / len(data_loader.dataset), 

### TRAINING LOOP ###

for epoch in range(0,epochs):
    avg_loss, rec_loss, kl_loss = vae_train_one_epoch(model, train_loader, optimizer)
    generate_grid('./checkpoint_beard/',model,(80,512),epoch)
    vae_utils.log_to_disk('./checkpoint_beard',f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Rec: {rec_loss:.4f}, KL: {kl_loss:.4f}")
    if (epoch + 1) % 30 == 0:
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, f"./checkpoint_beard/last_model_checkpoint_{epoch+1}.pt")

torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, f"./checkpoint_beard/last_model_checkpoint_{epoch+1}.pt")
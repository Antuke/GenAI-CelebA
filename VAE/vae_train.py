import torch
import torch.nn as nn
import vae_utils as vae_utils
from models .vae_deeper import VAE_model
from torchvision import utils
from torch.optim.lr_scheduler import ReduceLROnPlateau
### SETTINGS ###

beta=1
epochs = 2000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
full_data=True
size_of_subset_used=10000

model_save_path = "vae_conv_batch_fullset_50e.pt"

## MODEL & OPTIMIZER ##

model = VAE_model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=8e-5)
scheduler = ReduceLROnPlateau(optimizer, factor=0.8)
#model, optimizer = vae_utils.load_checkpoint(model,optimizer,'./checkpoints_deeper/vae_deeper_330.pt')
#for param_group in optimizer.param_groups:
#    param_group['lr'] = 1e-4
### LOSS ###

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

train_loader = vae_utils.get_dataloader_celeba(batch_size=64, resize=64, print_info=False, split='all')


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
    network.eval()
    labels = torch.zeros((noise_size[0],3), device=device)
    z = torch.randn(noise_size, device=device)
    # 8 possible combination, for each of them we generate 10 samples
    for i in range(8):
        labels[(i * 10):(i+1)*10] = torch.tensor(classes[i], dtype=torch.float32, device=device)

    with torch.no_grad():
        images = network.decode(z, labels)
    grid = utils.make_grid(images, normalize=True, nrow=10)
    sample_path = f'{path}vae_generated_{epoch}.png'
    utils.save_image(grid, sample_path)
    network.train()

### TRAINING FUNCTION ###

def vae_train_one_epoch(model, data_loader, optimizer):
    model.train()
    rec_loss = 0.0
    kl_loss = 0.0
    tot_loss = 0.0

    for x, cond in data_loader:
        x = x.to(device)
        cond = cond.to(device)
        recon_x, mu, log_sigma = model(x, cond)
        loss, rec, kl = loss_function(recon_x, x, mu, log_sigma)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        tot_loss += loss.item()
        rec_loss += rec.item()
        kl_loss += kl.item()
    return tot_loss / len(data_loader.dataset), rec_loss / len(data_loader.dataset), kl_loss / len(data_loader.dataset), 

### TRAINING LOOP ###

for epoch in range(0,epochs):
    avg_loss, rec_loss, kl_loss = vae_train_one_epoch(model, train_loader, optimizer)
    # scheduler.step(avg_loss)
    generate_grid('./checkpoints_deeper/',model,(80,512),epoch)
    vae_utils.log_to_disk('./checkpoints_deeper',f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Rec: {rec_loss:.4f}, KL: {kl_loss:.4f}")
    if (epoch + 1) % 30 == 0:
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, f"./checkpoints_deeper/vae_deeper_{epoch+1}.pt")



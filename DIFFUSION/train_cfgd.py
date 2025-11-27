import torch
import math
from utils import *
# from models.CFGNet import CFGDenoiser, TimeEncoding
from model import CFGDenoiser, TimeEncoding
import torch.optim as optim
from torch import nn
from torchvision import utils
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ---------- CONFIGS ------------#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLES_PER_ROW = 5
TIME_EMBEDDING = 256
T = 1000
BETA_STRATEGY = 'lin'
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
# ---------- NOISE SCHEDULING -------------- #
beta = None
if BETA_STRATEGY == 'lin':
    # linear beta from 1e-4 to 0.02
    beta = torch.linspace(1e-4, 0.02, T, device=device)
elif BETA_STRATEGY == 'exp':
    logbeta = torch.linspace(math.log(10), math.log(200), T, device=device)
    beta = torch.exp(logbeta) / 10000

alpha = torch.zeros_like(beta)
alpha[0] = 1.0 - beta[0]
for t in range(1, T):
    alpha[t] = alpha[t - 1] * (1.0 - beta[t])

# --------- DDIM  -------------- #
# quadratic selection 
N = 100 
c = 0.0998
tau = torch.linspace(N, 1, N , device=device, dtype=torch.int) # da 1 a 100 in 100 steps
tau = torch.floor(tau**2 * c).to(torch.int) + 1   
for i in range(N-1, 0, -1):
    if tau[i-1] <= tau[i]:
        tau[i-1] = tau[i] + 1
    else:
        break

def generate_sample_ddim(den_net, time_encoder, noise_size, labels, cfg_lambda=3, verbose=False):
    den_net.eval()

    z_t = torch.randn(noise_size, device=device)
    num_samples = z_t.shape[0]

    N = 100

    time_steps = range(N-1)
    if verbose:
        time_steps = tqdm(time_steps, desc="Denoising DDIM", unit="step")

    for i in time_steps:
        t_indices = torch.ones((num_samples,), device=device, dtype=torch.int) * tau[i]
        t_enc = time_encoder[t_indices]

        with torch.no_grad():
            predicted_unconditional_noise = den_net(z_t, t_enc, c=None)
            predicted_conditional_noise = den_net(z_t, t_enc, labels)
            predicted_noise = (1-cfg_lambda) * predicted_unconditional_noise + cfg_lambda * predicted_conditional_noise

        alpha_t_minus_delta_t = alpha[tau[i+1]] 
        alpha_t = alpha[tau[i]]

        p1 = torch.sqrt(1 - alpha_t_minus_delta_t)
        p2 = torch.sqrt(alpha_t_minus_delta_t * (1 - alpha_t)) / torch.sqrt(alpha_t)

        coef1 = torch.sqrt(alpha_t_minus_delta_t) / torch.sqrt(alpha_t)
        coef2 = p1 - p2 

        z_t = coef1 * z_t + coef2 * predicted_noise
    
    den_net.train()
    return z_t


# ----------- DDPM --------------- #

def generate_grid(path, den_net, time_encoder, noise_size, epoch):
    """Generates a grid of samples, one for each of the possible combination"""
    labels = torch.zeros((noise_size[0],3), device=device)
    # 8 possible combination, for each of them we generate SAMPLES_PER_ROW samples
    for i in range(8):
        labels[(i * SAMPLES_PER_ROW):(i+1)*SAMPLES_PER_ROW] = torch.tensor(classes[i], dtype=torch.float32, device=device)


    images = generate_sample(den_net, time_encoder, noise_size, labels)
    grid = utils.make_grid(images, normalize=True, nrow=SAMPLES_PER_ROW)
    sample_path = f'{path}samples_epoch_resbam_{epoch}.png'
    utils.save_image(grid, sample_path)


def generate_sample(den_net, time_encoder, noise_size, labels, cfg_lambda=3, verbose=False):
    """Batch genearation of samples"""
    den_net.eval()

    z = torch.randn(noise_size, device=device)
    num_samples = z.shape[0]

    time_steps = range(T-1, -1, -1)
    if verbose:
        time_steps = tqdm(time_steps, desc="Denoising", unit="step")

    # denoising loop from 999 to 0, included
    for t in time_steps:
        t_indices = torch.ones((num_samples,), device=device, dtype=torch.int) * t

        t_enc = time_encoder[t_indices]
        # Noise prediction
        with torch.no_grad():
            predicted_unconditional_noise = den_net(z, t_enc, c=None)
            predicted_conditional_noise = den_net(z, t_enc, labels)
            predicted_noise = (1-cfg_lambda) * predicted_unconditional_noise + cfg_lambda * predicted_conditional_noise

        # Paramets for step t
        alpha_t = alpha[t]
        beta_t = beta[t]

        # Noise ( NON added at last step)
        noise = torch.zeros_like(z) if t == 0 else torch.randn_like(z)

        # coefficients to calculate the mean
        coef1 = 1 / torch.sqrt(1 - beta_t)
        coef2 = beta_t / torch.sqrt(1 - alpha_t)


        # "moving z towards x"
        z = coef1 * (z - coef2 * predicted_noise) + torch.sqrt(beta_t) * noise

    den_net.train()
    return z


# ------------ TRAINING FUNCTIONS --------------- #
def train_epoch(den_net, optimizer, criterion, loader, time_encoder, verbose=False):
    den_net.train()
    total_loss = 0.0
    num_batches = 0
    if verbose:
        data = tqdm(loader)
    else:
        data = loader

    for x, labels in data:
        x = x.to(device)
        labels = labels.to(device)

        batch_size = x.shape[0]

        # Scelta dei punti lungo la catena di Markov
        t_indices = torch.randint(0, T, (batch_size,), device=device)

        # rumore, target della loss
        epsilon = torch.randn_like(x, device=device)

        # coefficienti alpha per il diffusion kernel
        alphas = alpha[t_indices]

        # encoding di t
        t = time_encoder[t_indices]

        # calcolo della variabile z usando il diffusion kernel
        alphas = alphas.view(-1, 1, 1, 1)  # [batch_size] -> [batch_size,1,1,1]
        z = torch.sqrt(alphas) * x + torch.sqrt(1 - alphas) * epsilon

        # reset dei gradienti
        optimizer.zero_grad()

        # predizione del rumore aggiunto
        out = den_net(z, t, labels)

        # calcolo loss e backprop
        loss = criterion(out, epsilon)
        loss.backward()
        optimizer.step()

        #loss_item = loss.item()
        total_loss += loss.detach()
        num_batches += 1
        if verbose:
            data.set_postfix(MSE=(total_loss / batch_size).item())

    return (total_loss / num_batches).item()


def train_celeba(resume=False, epoch_to_resume=0, epochs=1000):
    """Main training functions, instanciate network, dataloader and optimizers"""
    checkpoint_path = f'./checkpoint/'
    resize = 64
    loader = get_dataloader_celeba(resize=resize, batch_size=64,split='valid')

    den_net = CFGDenoiser(
        base_ch=64, # starting channel for convolution
        embdim=256, # dimension of conditional and time embedding
        cdim=3      # dimension of label tensor
    ).to(device)

    den_net.train()
    optimizer = optim.Adam(den_net.parameters(), lr=1e-4)

    # scheduler that reduces lr to 0.8*lr when the loss is not going down for 10 epochs
    scheduler = ReduceLROnPlateau(optimizer, factor=0.8)
    criterion = nn.MSELoss()
    time_encoder = TimeEncoding(L=T, dim=256, device=device)
    
    if resume:
        den_net, optimizer, scheduler = load_checkpoint(den_net, optimizer, scheduler, path= f'./checkpoint/checkpoint_{den_net.get_name()}_{epoch_to_resume-1}.pth')
    

    # main training loop
    log_to_disk(checkpoint_path, f'device usato in questo training = {device}')
    for i in range(epoch_to_resume,epochs):
        loss = train_epoch(den_net, optimizer, criterion, loader, time_encoder, verbose=True)
        generate_grid(checkpoint_path, den_net, time_encoder, epoch=i, noise_size=(8 * SAMPLES_PER_ROW, 3, resize, resize))
        scheduler.step(loss)
        message = f'Epoch: {i}, Loss: {loss:.5f}, LR: {scheduler.get_last_lr()}\n'
        log_to_disk(checkpoint_path, message)
        if i % 30 == 0:
            save_checkpoint(denoiser=den_net, optimizer=optimizer, scheduler=scheduler, path=checkpoint_path, epoch=i)



if __name__ == '__main__':
    train_celeba()
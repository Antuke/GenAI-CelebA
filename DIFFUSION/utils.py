from torchvision.datasets import CelebA
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import random

from torchvision.transforms import v2

ATTR_IDX_MALE = 20
ATTR_IDX_EYEGLASSES = 15
ATTR_IDX_NO_BEARD = 24
ATTR_IDX_GOATE = 16
ATTR_IDX_MUSTACHE = 22
ATTR_IDX_5_O_CLOCK = 0


def save_checkpoint(denoiser, optimizer,  path, epoch, scheduler = None,):
    """Save the model's state dictionary and optimizer state to a file"""
    checkpoint_path = os.path.join(path, f'checkpoint_{denoiser.get_name()}_{epoch}.pth')

    if scheduler is None:
        torch.save({
            'denoiser': denoiser.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, checkpoint_path)
    else:
        torch.save({
            'denoiser': denoiser.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }, checkpoint_path)


    return checkpoint_path


def load_checkpoint(denoiser, optimizer, scheduler, path):
    """Load the model's state dictionary and optimizer state from a file"""
    checkpoint = torch.load(path)
    denoiser.load_state_dict(checkpoint['denoiser'])
    if optimizer is None:
        return denoiser
    if scheduler is None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        return denoiser, optimizer
    

    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    
    return denoiser, optimizer, scheduler


def log_to_disk(log_dir, message):
    """Log training progress to disk"""
    log_path = os.path.join(log_dir, 'training_log.txt')
    with open(log_path, 'a') as f:
        f.write(f'{message}\n')

def get_dataloader_celeba(batch_size=32, datafolder='../celeba', resize=64, split='all'):
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((resize,resize),antialias=True),
        v2.RandomHorizontalFlip(),         # Faces are still faces if flipped horizontally
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])   # scaling to [-1,1], should be (marginally) benefecial for training
    ])                                     # for GAN/VAE allows the use of tanh instead of sigmoid
                                           # for diffusion it should not really have a great effect

    # the classes that we care for are:
    # NO_BEARD (169.158 no_beard, 33.441 beard), ID = 24
    # MALE (118.165 males, 84.434 females), ID = 20
    # EYEGLASSES (13.193 eyeglasses, 189.406 no_eyeglasses), ID = 15
    # All of the 203k samples are "correctly" labeled (no missing label)


    def target_transform_celeba(attr_tensor):
        male = attr_tensor[ATTR_IDX_MALE].float()                   # 1 if male, 0 if female
        eyeglasses = attr_tensor[ATTR_IDX_EYEGLASSES].float()       # 1 if has glasses, 0 if not
        beard = 0                                                   # 1 if beard, 0 if not

        # Making sure that are no inconstiency
        if (attr_tensor[ATTR_IDX_GOATE] or
                attr_tensor[ATTR_IDX_MUSTACHE] or not attr_tensor[ATTR_IDX_NO_BEARD]):
            beard = 1

        # The samples that correspond to females with beard seems to be mislabeled
        # if male == 0 and beard == 1:
        #    beard = 0

        return torch.tensor([male, eyeglasses, beard], dtype=torch.float32)

    print('Caricando il dataset')
    combined_dataset = CelebA(root=datafolder,
                              split=split,
                              transform=transform,
                              target_transform=target_transform_celeba,
                              download=False)

    print(f'Caricate {len(combined_dataset)} immagini')
    print('Caricando il train loader')
    train_loader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,   # drops last batch so that every batches have 32 samples
    )


    """
         M, E, B
        (0, 0, 0): 92245
        (0, 0, 1): 109     <--STRANO
        (0, 1, 0): 2147
        (0, 1, 1): 8       <--STRANO
        (1, 0, 0): 36348
        (1, 0, 1): 23547
        (1, 1, 0): 5039
        (1, 1, 1): 3327
    
    """


    return train_loader


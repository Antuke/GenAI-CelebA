from torchvision.datasets import CelebA
import os
import torch
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import Counter
from torch.utils.data import DataLoader, Subset

ATTR_IDX_MALE = 20
ATTR_IDX_EYEGLASSES = 15
ATTR_IDX_NO_BEARD = 24
ATTR_IDX_GOATE = 16
ATTR_IDX_MUSTACHE = 22
ATTR_IDX_5_O_CLOCK = 0

DEVICE= 'cuda' if torch.cuda.is_available() else 'cpu'

def save_checkpoint(denoiser, optimizer, path, epoch):
    """
    Save the model's state dictionary and optimizer state to a file
    """
    checkpoint_path = os.path.join(path, f'checkpoint_{denoiser.get_name()}_{epoch}.pth')
    torch.save({
        'model_state_dict': denoiser.state_dict(),
        'optimizer': optimizer.state_dict()
    }, checkpoint_path)
    return checkpoint_path

def load_checkpoint(denoiser, optimizer, path):
    """
    Load the model's state dictionary and optimizer state from a file
    """
    checkpoint = torch.load(path)
    denoiser.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is None:
        return denoiser
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return denoiser, optimizer

def log_to_disk(log_dir, message):
    """
    Log training progress to disk
    """
    log_path = os.path.join(log_dir, 'training_log.txt')
    with open(log_path, 'a') as f:
        f.write(f'{message}\n')

def get_dataloader_celeba(batch_size=64, datafolder='/home/pfoggia/GenerativeAI/CELEBA', resize=64, print_info=False, split='all'):
    """
    Load data from the celeba dataset, transform it, correct strange values, gives informations on the conditions
    """
    # define transformation we want to apply
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((resize,resize),antialias=True),
        v2.RandomHorizontalFlip(),         # horizontal flip to add variety
        v2.Lambda(lambda x: (x * 2) - 1)   # scaling to [-1,1], to use tanh instead of sigmoid, apparently better
    ])

    # transform weird informations
    """ Dataset
    classes 
    - NO_BEARD (169.158 no_beard, 33.441 beard), ID = 24
    - MALE (118.165 males, 84.434 females), ID = 20
    - EYEGLASSES (13.193 eyeglasses, 189.406 no_eyeglasses), ID = 15
    no missing label (203k samples)
    """

    def target_transform_celeba(attr_tensor):
        male = attr_tensor[ATTR_IDX_MALE].float()                   # 1 if male, 0 if female
        eyeglasses = attr_tensor[ATTR_IDX_EYEGLASSES].float()       # 1 if has glasses, 0 if not
        beard = 0                                                   # 1 if beard, 0 if not
        # consider mustache and bouc also beards
        if (attr_tensor[ATTR_IDX_GOATE] or attr_tensor[ATTR_IDX_MUSTACHE] or not attr_tensor[ATTR_IDX_NO_BEARD]):
            beard = 1
        
        # five o clock is almost no beard, let's prefer stronger candidates
        if (attr_tensor[ATTR_IDX_5_O_CLOCK] == 1):
            beard = 0

        if male == 0 and beard == 1:
            beard = 0
        return torch.tensor([male, eyeglasses, beard], dtype=torch.float32)

    # full dataset load
    print('Caricando il dataset')
    combined_dataset = CelebA(root=datafolder,
                              split=split,
                              transform=transform,
                              target_transform=target_transform_celeba,
                              download=True)
    print(f'Caricate {len(combined_dataset)} immagini')

    # train load
    print('Caricando il train loader')
    train_loader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,  # speeds up data transfer for gpu use
        drop_last=True,   # drops last batch so that every batches have 32 samples
        num_workers=8,    # <---- 8 workers
        persistent_workers=True 
    )

    # infos on the conditions
    """ Gender, Glasses, Beard
         M, E, B
        (0, 0, 0): 92245
        (0, 0, 1): 109     <--STRANO #Female, No glasses, Beard
        (0, 1, 0): 2147
        (0, 1, 1): 8       <--STRANO #Female,Glasses,Beard
        (1, 0, 0): 36348
        (1, 0, 1): 23547
        (1, 1, 0): 5039
        (1, 1, 1): 3327
    """
    if print_info:
        counter = Counter()
        for _, target in tqdm(combined_dataset, total=len(combined_dataset)):
            key = tuple(target.int().tolist())  # e.g., (1, 0, 1)
            counter[key] += 1
        print("\nClass Combination Counts:")
        for combo in sorted(counter.keys()):
            print(f"{combo}: {counter[combo]}")
    return train_loader

def get_subset_loader(loader, max_samples=10000):
    """
    Get subset from the full dataset. 
    From 203k samples to 10k samples to have lighter training to adapt problem more easily
    """
    dataset = loader.dataset
    indices = torch.randperm(len(dataset))[:max_samples]
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=64, shuffle=True)

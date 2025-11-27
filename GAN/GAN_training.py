# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# IMPORT
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.functional import binary_cross_entropy

from torchvision.datasets import CelebA
from torchvision.transforms import v2
from torchvision.utils import make_grid

import re
import os
from datetime import datetime
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
import itertools

from GAN_model import Generator, Discriminator

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# SET PARAMETERS
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

EXTENSION= 'pth'
ATTRIBUTE_INDEXES= [15, 20, 24]
# ATTRIBUTE_VALUES= ['eyeglasses', 'male', 'no_beard']

MODEL_NAME= 'Model'
MODEL_EPOCH= None
LOAD_DISCRIMINATOR= False

SAVE_MODEL_EVERY_NUM_EPOCHS= 50
SAVE_IMAGE_EVERY_NUM_EPOCHS= 8
EPOCHS= 150
BATCH_SIZE= 128

LATENT_SIZE= 128
COND_REPRESENTATION_SIZE= LATENT_SIZE//4
NUM_CLASSES= 3

CORRUPT_PERCENTAGE_IMAGE= 0.0
LABEL_SMOOTHING= 0.1
LEARNING_RATE_GENERATOR= 5e-4
LEARNING_RATE_GENERATOR_DISCRIMINATOR= 1e-5


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# GENERATE NECESSARY FOLDERS & INIZIALITE VARIABLES
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

current_position= os.getcwd()
FILE_OUTPUT=    f'model-{datetime.now().strftime("%Y_%m_%d-%H_%M_%S")}.txt'

DEVICE=         'cuda' if torch.cuda.is_available() else 'cpu'
FOLDER_MODEL=   'model_weights'
FOLDER_DATASET= '../data'
FOLDER_IMAGES=  'image'
FOLDER_OUTPUT=  'output'

DEVICE=         os.getenv('TORCH_DEVICE', DEVICE)
FOLDER_MODEL=   os.getenv('FOLDER_MODEL', FOLDER_MODEL)
FOLDER_DATASET= os.getenv('FOLDER_DATASET', FOLDER_DATASET)
FOLDER_IMAGES=  os.getenv('FOLDER_IMAGES', FOLDER_IMAGES)
FOLDER_OUTPUT=  os.getenv('FOLDER_OUTPUT', FOLDER_OUTPUT)

FOLDER_MODEL=   os.path.join(current_position, FOLDER_MODEL)
FOLDER_DATASET= os.path.join(current_position, FOLDER_DATASET)
FOLDER_IMAGES=  os.path.join(current_position, FOLDER_IMAGES)
FOLDER_OUTPUT=  os.path.join(current_position, FOLDER_OUTPUT)
FILE_OUTPUT=    os.path.join(FOLDER_OUTPUT, FILE_OUTPUT)

if not os.path.exists(FOLDER_MODEL):
    os.makedirs(FOLDER_MODEL)

if not os.path.exists(FOLDER_IMAGES):
    os.makedirs(FOLDER_IMAGES)

if not os.path.exists(FOLDER_OUTPUT):
    os.makedirs(FOLDER_OUTPUT)

if MODEL_EPOCH is None:
    max_number= 0
    pattern= re.compile(fr'{MODEL_NAME}_(\d+).{EXTENSION}')
    for file_name in os.listdir(FOLDER_MODEL):
        match= pattern.fullmatch(file_name)
        if match:
            num= int(match.group(1))
            if num > max_number:
                max_number= num
    MODEL_EPOCH= max_number

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# LOSS FUNCTIONS
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def discriminator_loss_func(d_true:torch.Tensor, d_synth:torch.Tensor, label_smoothing:float=LABEL_SMOOTHING):
    t_true= torch.ones_like(d_true) - label_smoothing
    loss_true= binary_cross_entropy(d_true, t_true, reduction='mean')

    t_synth= torch.zeros_like(d_synth) + label_smoothing
    loss_synth= binary_cross_entropy(d_synth, t_synth, reduction='mean')

    return loss_true + loss_synth

def generator_loss_func(d_synth:torch.Tensor, label_smoothing:float=LABEL_SMOOTHING):
    t_synth= torch.ones_like(d_synth) - label_smoothing
    return binary_cross_entropy(d_synth, t_synth, reduction='mean')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# CORRUPTION FUNCTION
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def corrupt_images(image:torch.Tensor, gray_percentage:float=CORRUPT_PERCENTAGE_IMAGE):
    """
    Make a certain percentage of the image gray.
    Params:
        image (torch.Tensor):       Original image to corrupt (or batch of images)
        gray_percentage (float):    Percentage of the image that will be gray
    """
    if gray_percentage != 0:
        random_mask = torch.rand(image.shape[-2:], device=image.device)
        black_mask = random_mask < gray_percentage
        black_mask = black_mask.reshape(1, *random_mask.shape)
        return image * (~black_mask).float()
    else:
        return image

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# SAMPLE GENERATION
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def show_generated_all(generator_model:nn.Module, plot_verbose:bool=True, save_fig_pos:str=None, num_samples:int=5):
    """
    Show all possible combinations of the conditions to pass to the model using make_grid.
    Each row in the grid shows samples for one specific condition.
    Params:
        generator_model (nn.Module):    The generator model
        plot_verbose (bool):            Choose to show or not the plot
        save_fig_pos (str):             If different from None, save the figure at the path specified
    """
    combinations = list(itertools.product([0, 1], repeat=NUM_CLASSES))
    conditions = torch.tensor(combinations, dtype=torch.bool, device=DEVICE)
    
    conditions = conditions.repeat_interleave(num_samples, dim=0)
    
    z = torch.randn(len(combinations) * num_samples, LATENT_SIZE).to(device=DEVICE)
    
    generator_model.eval()
    with torch.no_grad():
        x = generator_model(z, conditions)
        x = x.cpu()
    
    grid = make_grid(
        tensor=x,
        nrow=num_samples, 
        normalize=True, 
        value_range=(-1, 1),  # Scale from [-1,1] to [0,1]
        padding=2,
    )
    
    # Convert to numpy and permute dimensions for matplotlib
    grid_np = grid.numpy().transpose(1, 2, 0)
    
    fig, ax = plt.subplots(figsize=(num_samples*2, len(combinations)*2))
    ax.imshow(grid_np)
    ax.axis('off')
    fig.suptitle("Eyeglasses/male/beard")

    img_height = x.shape[2] + 2
    for i, comb in enumerate(combinations):
        label_str = ''.join(str(bit) for bit in comb)
        ypos = i * img_height + img_height // 2
        ax.text(-10, ypos, label_str, va='center', ha='right', fontsize=10, fontfamily='monospace', color='black')
    
    if save_fig_pos is not None:
        plt.savefig(save_fig_pos, bbox_inches='tight')
    
    if plot_verbose:
        plt.show()
    else:
        plt.close()

def show_generated_specific(generator_model:nn.Module, bool_conditions:list[bool]=None, num_examples:int=3):
    """
    Show a certain number of example of a single combination of conditions
    Params:
        generator_model (nn.Module):    The generator model
        bool_conditions (list[bool]):   List of the conditions
        num_examples (int):             Number of example to show
    """
    def normalize_and_permute(image:torch.Tensor):
        """Normalize the image in the range [0,1] and do the permutation"""
        image= (image + 1) / 2
        image= torch.clip(image, 0, 1)
        image= image.permute(1, 2, 0)
        return image
        
    conditions= torch.tensor(num_examples*[bool_conditions], dtype=torch.int64).to(device='cuda')
    z= torch.randn(num_examples, LATENT_SIZE).to(device='cuda')
    generator_model.eval()
    
    with torch.no_grad():
        x= generator_model(z, conditions)
        x= x.cpu()

    eyeglasses= ('' if bool_conditions[0] else 'no ')+'eyeglasses'
    gender=     'male' if bool_conditions[1] else 'female'
    beard=      ('' if bool_conditions[2] else 'no ')+'beard'

    figure= plt.figure(figsize=(14, 4))
    figure.suptitle(f"{eyeglasses}, {gender}, {beard}")
    for i in range(num_examples):
        plt.subplot(1, num_examples, i+1)
        plt.imshow( normalize_and_permute(x[i]) )
        plt.axis('off')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# TRAINING
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def save_model(model_generator:nn.Module, model_discriminator:nn.Module):
    state= {}
    state['model_generator']=           model_generator.state_dict()
    state['model_discriminator']=       model_discriminator.state_dict()
    
    file_name= f"{MODEL_NAME}_{MODEL_EPOCH}.{EXTENSION}"
    file_name= os.path.join(FOLDER_MODEL, file_name)
    torch.save(state, file_name)

def training_epoch(data_loader:DataLoader, generator:nn.Module, discriminator:nn.Module, g_optimizer:torch.optim.Optimizer, d_optimizer:torch.optim.Optimizer, verbose:bool=True, show_progress:bool=False):
    generator.train()
    discriminator.train()

    batches= 0

    sum_generation_loss=        0.0
    sum_discriminator_loss=     0.0
    sum_discriminator_true=     0.0
    sum_discriminator_synth=    0.0

    for x_true,attributes in (tqdm(data_loader) if show_progress else data_loader):
        x_true= x_true.to(device=DEVICE)
        conditions= attributes[:, ATTRIBUTE_INDEXES].to(device=DEVICE)
        conditions[:, -1]= torch.where(conditions[:,-1]==0, 1.0, 0.0)     # instead of use the original attribute 'No_Beard' we use the opposite

        z= torch.randn(x_true.shape[0], LATENT_SIZE, device=DEVICE)
        x_synth= generator(z, conditions)

        d_synth= discriminator(corrupt_images(x_synth), conditions)
        d_true=  discriminator(corrupt_images(x_true),  conditions)
        
        # Use of `retain_graph=True` so "the graph used to compute the grads will not be freed"
        # For the generator is necessary the first value of the discriminator to compute the loss
        d_optimizer.zero_grad()
        discriminator_loss= discriminator_loss_func(d_true, d_synth)
        discriminator_loss.backward(retain_graph=True)
        d_optimizer.step()

        # The discriminator is changed, the generator no. The `d_synth` is changed, the `x_synth` no
        d_synth= discriminator(x_synth, conditions)

        g_optimizer.zero_grad()
        generator_loss= generator_loss_func(d_synth)
        generator_loss.backward()
        g_optimizer.step()

        batches+= 1
        sum_generation_loss+=       generator_loss.detach().item()
        sum_discriminator_loss+=    discriminator_loss.detach().item()
        sum_discriminator_true+=    d_true.mean().detach().item()
        sum_discriminator_synth+=   d_synth.mean().detach().item()

    if verbose:
        print("Generation loss: {} ||".format( sum_generation_loss / batches ), end="")
        print("\tDiscriminator loss: {} ||".format( sum_discriminator_loss / batches ), end="")
        print("\tDiscriminator syntetics: {} ||".format( sum_discriminator_synth / batches ), end="")
        print("\tDiscriminator real: {}".format( sum_discriminator_true / batches ), end="")
        print()

    generator.eval()
    discriminator.eval()

    return sum_generation_loss/batches, sum_discriminator_loss/batches, sum_discriminator_synth/batches, sum_discriminator_true/batches

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# MAIN
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def main():
    global MODEL_EPOCH
    
    # Loading dataset
    transformer= v2.Compose([
        v2.ToImage(),
        v2.CenterCrop((178,178)),
        v2.Resize((64,64)),
        v2.RandomAutocontrast(p=0.2),
        v2.RandomHorizontalFlip(p=0.2),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    training_set= CelebA(FOLDER_DATASET, split='train', transform=transformer)
    training_loader= DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, persistent_workers=True)
    
    # Loading or creating models
    generator= Generator(LATENT_SIZE, NUM_CLASSES, COND_REPRESENTATION_SIZE, device=DEVICE)
    generator= generator.to(device=DEVICE)
    generator_optimizer= torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE_GENERATOR)
    
    discriminator= Discriminator(NUM_CLASSES, COND_REPRESENTATION_SIZE, device=DEVICE)
    discriminator= discriminator.to(device=DEVICE)
    discriminator_optimizer= torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE_GENERATOR_DISCRIMINATOR)
    
    if MODEL_EPOCH != 0:
        file= f"{MODEL_NAME}_{MODEL_EPOCH}.{EXTENSION}"
        file= os.path.join(FOLDER_MODEL, file)
        
        state= torch.load(file, map_location=DEVICE)
        generator.load_state_dict(state['model_generator'])
        if LOAD_DISCRIMINATOR:
            discriminator.load_state_dict(state['model_discriminator'])

    # Training process
    count= 0
    for _ in tqdm(range(EPOCHS), desc="Progress", unit="epoch"):
        values= training_epoch(training_loader, generator, discriminator, generator_optimizer, discriminator_optimizer, verbose=False, show_progress=bool(count==0))
        MODEL_EPOCH= MODEL_EPOCH+1
        count= count+1
        
        # writing metrics on file
        string= ', '.join([str(value) for value in values])
        with open(FILE_OUTPUT, "a") as f:
            f.write(f"{string}\n")
        
        # saving images only after a certain number of epochs
        if (count) % SAVE_IMAGE_EVERY_NUM_EPOCHS == 0:
            show_generated_all(generator, plot_verbose=False, save_fig_pos=os.path.join(FOLDER_IMAGES, f"{MODEL_NAME}_{MODEL_EPOCH}"))
        
        # saving model only after a certain number of epochs
        if (count) % SAVE_MODEL_EVERY_NUM_EPOCHS == 0:
            save_model(generator, discriminator)

    # saving model if not saved during the last epoch and save image always
    if (count) % SAVE_MODEL_EVERY_NUM_EPOCHS != 0:
        save_model(generator, discriminator)
    show_generated_all(generator, plot_verbose=False, save_fig_pos=os.path.join(FOLDER_IMAGES, f"{MODEL_NAME}_{MODEL_EPOCH}"))
    
if __name__=='__main__':
    main()
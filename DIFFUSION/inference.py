import argparse
import torch
import math
import os
import numpy as np
from torchvision import utils
from tqdm import tqdm
import torchvision.transforms.functional as F
import sys

try:
    from model import CFGDenoiser, TimeEncoding
except ImportError:
    from DIFFUSION.model import CFGDenoiser, TimeEncoding

def get_args():
    parser = argparse.ArgumentParser(description="DDPM/DDIM Inference Script")
    parser.add_argument("--method", type=str, default="ddpm", choices=["ddpm", "ddim"], help="Sampling method: ddpm or ddim")
    parser.add_argument("--cfg_lambda", type=float, default=3.0, help="Classifier Free Guidance lambda")
    parser.add_argument("--rows", type=int, default=1, help="Number of rows (classes)")
    parser.add_argument("--cols", type=int, default=4, help="Number of samples per row")
    parser.add_argument("--name", type=str, default="generated_sample", help="Output filename (without extension)")
    parser.add_argument("--weights", type=str, default="./WEIGHTS/diffusion.pt", help="Path to model weights")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--image_size", type=int, default=64, help="Image size")
    parser.add_argument("--time_steps", type=int, default=1000, help="Total time steps T")
    parser.add_argument("--time_embedding", type=int, default=256, help="Time embedding dimension")
    return parser.parse_args()

def setup_constants(args, device):
    # ------ CONSTANTS ----------- #
    beta_strategy ='lin' # training has been done with linear noise scheduling
    ddim_selection='quad'
    
    T = args.time_steps
    
    if beta_strategy == 'lin':
        beta = torch.linspace(1e-4, 0.02, T, device=device)
    elif beta_strategy == 'exp':
        logbeta = torch.linspace(math.log(10), math.log(200), T , device=device)
        beta = torch.exp(logbeta) / 10000

    alpha = torch.zeros_like(beta)
    alpha[0] = 1.0 - beta[0]
    for t in range(1, T):
        alpha[t] = alpha[t-1] * (1.0 - beta[t])

    # Selection for DDIM
    N = 100
    tau = None
    if ddim_selection == 'quad':
        c = 0.09998 # c * i**2 = 1000 for i=100 --> c =  1000 / 10000 = 0.1
        tau = torch.linspace(N, 1, N , device=device, dtype=torch.int) # da 1 a 100 in 100 steps
        tau = torch.floor(tau**2 * c).to(torch.int) + 1
        tau[0]= 999

    if ddim_selection == 'lin':
        c = 9.988  # c * i = 1000 for i=100  --> c = 1000/100 = 10
        tau = torch.linspace(N, 1, N , device=device, dtype=torch.int) # da 1 a 100 in 100 steps
        tau = torch.floor(tau * c).to(torch.int) + 1

    # Assures the list is strictly decreasing
    if tau is not None:
        for i in range(N-1, 0, -1):
            if tau[i-1] <= tau[i]:
                tau[i-1] = tau[i] + 1
            else:
                break
            
    return beta, alpha, tau

def generate_sample_ddim(den_net, time_encoder, noise_size, labels, beta, alpha, tau, device, cfg_lambda=3, verbose=True):
    den_net.eval()
    N = 100
    z = torch.randn(noise_size, device=device)
    num_samples = z.shape[0]

    # --- Masks to handle special cases  --- #
    bearded_woman_label = torch.tensor([0, 0, 1], device=device, dtype=torch.float32)
    bearded_eyeglasses_woman_label = torch.tensor([0, 1, 1], device=device, dtype=torch.float32)

    bearded_woman_mask = torch.all(labels == bearded_woman_label, dim=1)
    bearded_eyeglasses_woman_mask = torch.all(labels == bearded_eyeglasses_woman_label, dim=1)

    # A "normal" sample is one that is NOT a bearded woman (with or without eyeglasses).
    normal_mask = ~ (bearded_woman_mask | bearded_eyeglasses_woman_mask)

    # Get indices for each group
    # nonzero returns the indx of all non-zero/non-false elements
    bearded_woman_indx = torch.nonzero(bearded_woman_mask).squeeze(-1)
    bearded_eyeglasses_woman_indx = torch.nonzero(bearded_eyeglasses_woman_mask).squeeze(-1)
    normal_indx = torch.nonzero(normal_mask).squeeze(-1)

    time_steps = range(N-1)
    if verbose:
        time_steps = tqdm(time_steps, desc="Denoising DDIM", unit="step")

    for t in time_steps:
        t_indices = torch.ones((num_samples,), device=device, dtype=torch.int) * tau[t]
        t_enc = time_encoder[t_indices]

        with torch.no_grad():
            predicted_noise = torch.empty_like(z)

            # Unconditional noise is calculated once for all paths
            unconditional_noise = den_net(z, t_enc, c=None)

            # --- Handle normal cases --- #
            if normal_indx.numel() > 0:
                z_normal = z[normal_indx]
                t_normal = t_enc[normal_indx]
                label_normal = labels[normal_indx]
                uncond_noise_normal = unconditional_noise[normal_indx]

                cond_noise_normal = den_net(z_normal, t_normal, label_normal)
                noise_normal = uncond_noise_normal + cfg_lambda * (cond_noise_normal - uncond_noise_normal)
                predicted_noise[normal_indx] = noise_normal

            # --- Handle bearded woman ('latent vector' arithmetic) --- #
            if bearded_woman_indx.numel() > 0:
                z_bw = z[bearded_woman_indx]
                t_bw = t_enc[bearded_woman_indx]
                uncond_noise_bw = unconditional_noise[bearded_woman_indx]

                num_bw = len(bearded_woman_indx)
                c_woman = torch.zeros(num_bw, labels.shape[1], device=device, dtype=labels.dtype)
                c_bearded_man = torch.tensor([1, 0, 1], device=device, dtype=labels.dtype).repeat(num_bw, 1) # the 1 is to have num_bw repeated tensor
                c_man = torch.tensor([1, 0, 0], device=device, dtype=labels.dtype).repeat(num_bw, 1)         # instead of a tensor that repeats num_bw the label

                woman_noise = den_net(z_bw, t_bw, c_woman)
                bearded_man_noise = den_net(z_bw, t_bw, c_bearded_man)
                man_noise = den_net(z_bw, t_bw, c_man)

                woman_direction = woman_noise - uncond_noise_bw
                beard_direction = (bearded_man_noise - uncond_noise_bw) - (man_noise - uncond_noise_bw)

                noise_bw = uncond_noise_bw + cfg_lambda * woman_direction + cfg_lambda * beard_direction
                predicted_noise[bearded_woman_indx] = noise_bw

            # --- Handle bearded woman eyeglasses ('latent vector' arithmetic) --- #
            if bearded_eyeglasses_woman_indx.numel() > 0:
                z_bew = z[bearded_eyeglasses_woman_indx]
                t_bew = t_enc[bearded_eyeglasses_woman_indx]
                uncond_noise_bew = unconditional_noise[bearded_eyeglasses_woman_indx]

                num_bew = len(bearded_eyeglasses_woman_indx)
                c_woman_eyeglasses = torch.tensor([0, 1, 0], device=device, dtype=labels.dtype).repeat(num_bew, 1)
                c_bearded_man = torch.tensor([1, 0, 1], device=device, dtype=labels.dtype).repeat(num_bew, 1)
                c_man = torch.tensor([1, 0, 0], device=device, dtype=labels.dtype).repeat(num_bew, 1)

                woman_noise = den_net(z_bew, t_bew, c_woman_eyeglasses)
                bearded_man_noise = den_net(z_bew, t_bew, c_bearded_man)
                man_noise = den_net(z_bew, t_bew, c_man)

                woman_direction = woman_noise - uncond_noise_bew
                beard_direction = (bearded_man_noise - uncond_noise_bew) - (man_noise - uncond_noise_bew)

                noise_bew = uncond_noise_bew + cfg_lambda * woman_direction + cfg_lambda * beard_direction
                predicted_noise[bearded_eyeglasses_woman_indx] = noise_bew

        # Denoising Step
        alpha_t_minus_delta_t = alpha[tau[t+1]]
        alpha_t = alpha[tau[t]]

        p1 = torch.sqrt(1 - alpha_t_minus_delta_t)
        p2 = torch.sqrt(alpha_t_minus_delta_t * (1 - alpha_t)) / torch.sqrt(alpha_t)

        coef1 = torch.sqrt(alpha_t_minus_delta_t) / torch.sqrt(alpha_t)
        coef2 = p1 - p2

        z = coef1 * z + coef2 * predicted_noise

    return z

def generate_sample(den_net, time_encoder, noise_size, labels, beta, alpha, T, device, cfg_lambda=3, verbose=True):
    den_net.eval()

    z = torch.randn(noise_size, device=device)
    num_samples = z.shape[0]

    # --- Masks to handle special cases  --- #
    bearded_woman_label = torch.tensor([0, 0, 1], device=device, dtype=torch.float32)
    bearded_eyeglasses_woman_label = torch.tensor([0, 1, 1], device=device, dtype=torch.float32)

    bearded_woman_mask = torch.all(labels == bearded_woman_label, dim=1)
    bearded_eyeglasses_woman_mask = torch.all(labels == bearded_eyeglasses_woman_label, dim=1)

    # A "normal" sample is one that is NOT a bearded woman (with or without eyeglasses).
    normal_mask = ~ (bearded_woman_mask | bearded_eyeglasses_woman_mask)

    # Get indices for each group
    # nonzero returns the indx of all non-zero/non-false elements
    bearded_woman_indx = torch.nonzero(bearded_woman_mask).squeeze(-1)
    bearded_eyeglasses_woman_indx = torch.nonzero(bearded_eyeglasses_woman_mask).squeeze(-1)
    normal_indx = torch.nonzero(normal_mask).squeeze(-1)

    time_steps = range(T - 1, -1, -1)
    if verbose:
        time_steps = tqdm(time_steps, desc="Denoising DDPM", unit="step")

    for t in time_steps:
        t_indices = torch.ones((num_samples,), device=device, dtype=torch.int) * t
        t_enc = time_encoder[t_indices]

        with torch.no_grad():
            predicted_noise = torch.empty_like(z)

            # Unconditional noise is calculated once for all paths
            unconditional_noise = den_net(z, t_enc, c=None)

            # --- Handle normal cases --- #
            if normal_indx.numel() > 0:
                z_normal = z[normal_indx]
                t_normal = t_enc[normal_indx]
                label_normal = labels[normal_indx]
                uncond_noise_normal = unconditional_noise[normal_indx]

                cond_noise_normal = den_net(z_normal, t_normal, label_normal)
                noise_normal = uncond_noise_normal + cfg_lambda * (cond_noise_normal - uncond_noise_normal)
                predicted_noise[normal_indx] = noise_normal

            # --- Handle bearded woman ('latent vector' arithmetic) --- #
            if bearded_woman_indx.numel() > 0:
                z_bw = z[bearded_woman_indx]
                t_bw = t_enc[bearded_woman_indx]
                uncond_noise_bw = unconditional_noise[bearded_woman_indx]

                num_bw = len(bearded_woman_indx)
                c_woman = torch.zeros(num_bw, labels.shape[1], device=device, dtype=labels.dtype)
                c_bearded_man = torch.tensor([1, 0, 1], device=device, dtype=labels.dtype).repeat(num_bw, 1)
                c_man = torch.tensor([1, 0, 0], device=device, dtype=labels.dtype).repeat(num_bw, 1)

                woman_noise = den_net(z_bw, t_bw, c_woman)
                bearded_man_noise = den_net(z_bw, t_bw, c_bearded_man)
                man_noise = den_net(z_bw, t_bw, c_man)

                woman_direction = woman_noise - uncond_noise_bw
                beard_direction = (bearded_man_noise - uncond_noise_bw) - (man_noise - uncond_noise_bw)

                noise_bw = uncond_noise_bw + cfg_lambda * woman_direction + cfg_lambda * beard_direction
                predicted_noise[bearded_woman_indx] = noise_bw

            # --- Handle bearded woman eyeglasses ('latent vector' arithmetic) --- #
            if bearded_eyeglasses_woman_indx.numel() > 0:
                z_bew = z[bearded_eyeglasses_woman_indx]
                t_bew = t_enc[bearded_eyeglasses_woman_indx]
                uncond_noise_bew = unconditional_noise[bearded_eyeglasses_woman_indx]

                num_bew = len(bearded_eyeglasses_woman_indx)
                c_woman_eyeglasses = torch.tensor([0, 1, 0], device=device, dtype=labels.dtype).repeat(num_bew, 1)
                c_bearded_man = torch.tensor([1, 0, 1], device=device, dtype=labels.dtype).repeat(num_bew, 1)
                c_man = torch.tensor([1, 0, 0], device=device, dtype=labels.dtype).repeat(num_bew, 1)

                woman_noise = den_net(z_bew, t_bew, c_woman_eyeglasses)
                bearded_man_noise = den_net(z_bew, t_bew, c_bearded_man)
                man_noise = den_net(z_bew, t_bew, c_man)

                woman_direction = woman_noise - uncond_noise_bew
                beard_direction = (bearded_man_noise - uncond_noise_bew) - (man_noise - uncond_noise_bew)

                noise_bew = uncond_noise_bew + cfg_lambda * woman_direction + cfg_lambda * beard_direction
                predicted_noise[bearded_eyeglasses_woman_indx] = noise_bew

        # Denoising Step
        alpha_t = alpha[t]
        beta_t = beta[t]

        # This is the random noise component for the sampling step
        step_noise = torch.zeros_like(z) if t == 0 else torch.randn_like(z)

        coef1 = 1 / torch.sqrt(1 - beta_t)
        coef2 = beta_t / torch.sqrt(1 - alpha_t)

        z = coef1 * (z - coef2 * predicted_noise) + torch.sqrt(beta_t) * step_noise

    return z

def load_checkpoint(denoiser, optimizer, scheduler, path, map_location='cuda'):
    """Load the model's state dictionary and optimizer state from a file"""
    print(f"Loading checkpoint from {path}")
    checkpoint = torch.load(path, map_location=map_location)
    denoiser.load_state_dict(checkpoint['denoiser'])
    if optimizer is None:
        return denoiser
    if scheduler is None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        return denoiser, optimizer

    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])

    return denoiser, optimizer, scheduler

def generate_grid(fun, den_net, time_encoder, noise_size, classes, beta, alpha, tau, T, device, name='', samples_per_row=10, cfg_lambda=3):
    labels = torch.zeros((noise_size[0], 3), device=device)

    # Populate labels based on classes dict
    # classes is a dict {row_index: [class_vector]}
    # We assume rows are filled sequentially
    
    # If classes is a list of vectors, we can iterate
    # But the notebook used a dict. Let's adapt to receive a list of class vectors for each row
    
    for i, class_vec in enumerate(classes):
        # Set label for this row
        start_idx = i * samples_per_row
        end_idx = (i + 1) * samples_per_row
        if start_idx >= noise_size[0]:
            break
        labels[start_idx:end_idx] = torch.tensor(class_vec, dtype=torch.float32, device=device)

    if fun.__name__ == 'generate_sample_ddim':
        images = fun(den_net, time_encoder, noise_size, labels, beta, alpha, tau, device, cfg_lambda=cfg_lambda)
    else:
        images = fun(den_net, time_encoder, noise_size, labels, beta, alpha, T, device, cfg_lambda=cfg_lambda)
        
    grid = utils.make_grid(images, normalize=True, nrow=samples_per_row)
    
    if name:
        # Ensure directory exists
        os.makedirs('./SAMPLES', exist_ok=True)
        sample_path = f'./SAMPLES/{name}.png'
        utils.save_image(grid, sample_path)
        print(f"Saved image to {sample_path}")

def main():
    args = get_args()
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Setup constants
    beta, alpha, tau = setup_constants(args, device)

    # Load Model
    den_net = CFGDenoiser(
        base_ch=64,
        in_ch=3,
        cdim=3,
        embdim=args.time_embedding
    ).to(device)
    
    time_encoder = TimeEncoding(args.time_steps, args.time_embedding, device)
    
    if os.path.exists(args.weights):
        den_net = load_checkpoint(den_net, None, None, path=args.weights, map_location=device)
    else:
        print(f"Warning: Weight file {args.weights} not found.")
        return

    # Define classes for rows
    # Default behavior: iterate through all 8 binary classes if rows=8
    possible_classes = [[0,0,0], [0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,0], [1,1,1]]
    
    classes = []
    if args.rows <= 8:
        classes = possible_classes[:args.rows]
    else:
        # Repeat classes if more rows than classes
        for i in range(args.rows):
            classes.append(possible_classes[i % 8])

    print(f"\n--- Generation Parameters ---")
    print(f"Method: {args.method}")
    print(f"Rows: {args.rows}")
    print(f"Cols: {args.cols}")
    print(f"Lambda: {args.cfg_lambda}")
    print(f"Classes: {classes}")

    func = generate_sample if args.method == 'ddpm' else generate_sample_ddim
    
    generate_grid(
        fun=func,
        den_net=den_net,
        time_encoder=time_encoder,
        noise_size=(args.rows * args.cols, 3, args.image_size, args.image_size),
        classes=classes,
        beta=beta,
        alpha=alpha,
        tau=tau,
        T=args.time_steps,
        device=device,
        name=args.name,
        samples_per_row=args.cols,
        cfg_lambda=args.cfg_lambda
    )

if __name__ == "__main__":
    main()

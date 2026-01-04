import math
import base64
import torch
from tqdm import tqdm
from DIFFUSION.model import CFGDenoiser, TimeEncoding
from torchvision import utils
import io
from PIL import Image


beta_strategy = "lin"  # training has been done with linear noise scheduling
ddim_selection = "quad"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_SIZE = 64
T = 1000
TIME_EMBEDDING = 256
beta = None
tau = None

if beta_strategy == "lin":
    beta = torch.linspace(1e-4, 0.02, T, device=device)
elif beta_strategy == "exp":
    logbeta = torch.linspace(math.log(10), math.log(200), T, device=device)
    beta = torch.exp(logbeta) / 10000
if beta is None:
    raise ValueError(f"Unknown beta strategy: {beta_strategy}")

alpha = torch.zeros_like(beta)
alpha[0] = 1.0 - beta[0]
for t in range(1, T):
    alpha[t] = alpha[t - 1] * (1.0 - beta[t])


# Selection for DDIM

N = 100
if ddim_selection == "quad":
    c = 0.09998  # c * i**2 = 1000 for i=100 --> c =  1000 / 10000 = 0.1
    tau = torch.linspace(
        N, 1, N, device=device, dtype=torch.int
    )  # da 1 a 100 in 100 steps
    tau = torch.floor(tau**2 * c).to(torch.int) + 1
    tau[0] = 999

if ddim_selection == "lin":
    c = 9.988  # c * i = 1000 for i=100  --> c = 1000/100 = 10
    tau = torch.linspace(
        N, 1, N, device=device, dtype=torch.int
    )  # da 1 a 100 in 100 steps
    tau = torch.floor(tau * c).to(torch.int) + 1

if tau is None:
    raise ValueError(f"Unknown selection DDIM strategy: {ddim_selection}")

# Assures the list is strictly decreasing
for i in range(N - 1, 0, -1):
    if tau[i - 1] <= tau[i]:
        tau[i - 1] = tau[i] + 1
    else:
        break






def tensor_to_bytes(z):
    # Normalize and create grid
    grid = utils.make_grid(z, normalize=True, nrow=5) # Adjust nrow as needed
    # Convert to range 0-255
    grid = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

    # Save to buffer
    img = Image.fromarray(grid)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return buffer.getvalue()


# ## SAMPLING FUNCTIONS, DDPM & DDIM
# With guidance combination to produce bearded woman samples.
# Guidance combination: we isolate the 'beard direction', by subtracting from the 'bearded man direction' the 'man direction'. We then sum the resulting 'beard direction' to the 'woman direction' or 'woman with sunglasses direction'




def generate_sample_ddim(
    den_net: CFGDenoiser, time_encoder: TimeEncoding, noise_size, labels, cfg_lambda=3, verbose=False
):
    assert tau is not None, "Tau schedule is not initialized"
    assert beta is not None, "Beta schedule is not initialized"
    den_net.eval()
    N = 100
    z = torch.randn(noise_size, device=device)
    num_samples = z.shape[0]

    # --- Masks to handle special cases  --- #
    bearded_woman_label = torch.tensor([0, 0, 1], device=device, dtype=torch.float32)
    bearded_eyeglasses_woman_label = torch.tensor(
        [0, 1, 1], device=device, dtype=torch.float32
    )

    bearded_woman_mask = torch.all(labels == bearded_woman_label, dim=1)
    bearded_eyeglasses_woman_mask = torch.all(
        labels == bearded_eyeglasses_woman_label, dim=1
    )

    # A "normal" sample is one that is NOT a bearded woman (with or without eyeglasses).
    normal_mask = ~(bearded_woman_mask | bearded_eyeglasses_woman_mask)

    # Get indices for each group
    # nonzero returns the indx of all non-zero/non-false elements
    bearded_woman_indx = torch.nonzero(bearded_woman_mask).squeeze(-1)
    bearded_eyeglasses_woman_indx = torch.nonzero(
        bearded_eyeglasses_woman_mask
    ).squeeze(-1)
    normal_indx = torch.nonzero(normal_mask).squeeze(-1)

    time_steps = range(N - 1)

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
                noise_normal = uncond_noise_normal + cfg_lambda * (
                    cond_noise_normal - uncond_noise_normal
                )
                predicted_noise[normal_indx] = noise_normal

            # --- Handle bearded woman ('latent vector' arithmetic) --- #
            if bearded_woman_indx.numel() > 0:
                z_bw = z[bearded_woman_indx]
                t_bw = t_enc[bearded_woman_indx]
                uncond_noise_bw = unconditional_noise[bearded_woman_indx]

                num_bw = len(bearded_woman_indx)
                c_woman = torch.zeros(
                    num_bw, labels.shape[1], device=device, dtype=labels.dtype
                )
                c_bearded_man = torch.tensor(
                    [1, 0, 1], device=device, dtype=labels.dtype
                ).repeat(num_bw, 1)  # the 1 is to have num_bw repeated tensor
                c_man = torch.tensor(
                    [1, 0, 0], device=device, dtype=labels.dtype
                ).repeat(num_bw, 1)  # instead of a tensor that repeats num_bw the label

                woman_noise = den_net(z_bw, t_bw, c_woman)
                bearded_man_noise = den_net(z_bw, t_bw, c_bearded_man)
                man_noise = den_net(z_bw, t_bw, c_man)

                woman_direction = woman_noise - uncond_noise_bw
                beard_direction = (bearded_man_noise - uncond_noise_bw) - (
                    man_noise - uncond_noise_bw
                )

                noise_bw = (
                    uncond_noise_bw
                    + cfg_lambda * woman_direction
                    + cfg_lambda * beard_direction
                )
                predicted_noise[bearded_woman_indx] = noise_bw

            # --- Handle bearded woman eyeglasses ('latent vector' arithmetic) --- #
            if bearded_eyeglasses_woman_indx.numel() > 0:
                z_bew = z[bearded_eyeglasses_woman_indx]
                t_bew = t_enc[bearded_eyeglasses_woman_indx]
                uncond_noise_bew = unconditional_noise[bearded_eyeglasses_woman_indx]

                num_bew = len(bearded_eyeglasses_woman_indx)
                c_woman_eyeglasses = torch.tensor(
                    [0, 1, 0], device=device, dtype=labels.dtype
                ).repeat(num_bew, 1)
                c_bearded_man = torch.tensor(
                    [1, 0, 1], device=device, dtype=labels.dtype
                ).repeat(num_bew, 1)
                c_man = torch.tensor(
                    [1, 0, 0], device=device, dtype=labels.dtype
                ).repeat(num_bew, 1)

                woman_noise = den_net(z_bew, t_bew, c_woman_eyeglasses)
                bearded_man_noise = den_net(z_bew, t_bew, c_bearded_man)
                man_noise = den_net(z_bew, t_bew, c_man)

                woman_direction = woman_noise - uncond_noise_bew
                beard_direction = (bearded_man_noise - uncond_noise_bew) - (
                    man_noise - uncond_noise_bew
                )

                noise_bew = (
                    uncond_noise_bew
                    + cfg_lambda * woman_direction
                    + cfg_lambda * beard_direction
                )
                predicted_noise[bearded_eyeglasses_woman_indx] = noise_bew

        # Denoising Step
        alpha_t_minus_delta_t = alpha[tau[t + 1]]
        alpha_t = alpha[tau[t]]

        p1 = torch.sqrt(1 - alpha_t_minus_delta_t)
        p2 = torch.sqrt(alpha_t_minus_delta_t * (1 - alpha_t)) / torch.sqrt(alpha_t)

        coef1 = torch.sqrt(alpha_t_minus_delta_t) / torch.sqrt(alpha_t)
        coef2 = p1 - p2

        z = coef1 * z + coef2 * predicted_noise
        yield z

    yield z


def generate_sample(
    den_net: CFGDenoiser, time_encoder: TimeEncoding, noise_size, labels, cfg_lambda=3, verbose=False
):
    assert tau is not None, "Tau schedule is not initialized"
    assert beta is not None, "Beta schedule is not initialized"
    den_net.eval()

    z = torch.randn(noise_size, device=device)
    num_samples = z.shape[0]

    # --- Masks to handle special cases  --- #
    bearded_woman_label = torch.tensor([0, 0, 1], device=device, dtype=torch.float32)
    bearded_eyeglasses_woman_label = torch.tensor(
        [0, 1, 1], device=device, dtype=torch.float32
    )

    bearded_woman_mask = torch.all(labels == bearded_woman_label, dim=1)
    bearded_eyeglasses_woman_mask = torch.all(
        labels == bearded_eyeglasses_woman_label, dim=1
    )

    # A "normal" sample is one that is NOT a bearded woman (with or without eyeglasses).
    normal_mask = ~(bearded_woman_mask | bearded_eyeglasses_woman_mask)

    # Get indices for each group
    # nonzero returns the indx of all non-zero/non-false elements
    bearded_woman_indx = torch.nonzero(bearded_woman_mask).squeeze(-1)
    bearded_eyeglasses_woman_indx = torch.nonzero(
        bearded_eyeglasses_woman_mask
    ).squeeze(-1)
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
                noise_normal = uncond_noise_normal + cfg_lambda * (
                    cond_noise_normal - uncond_noise_normal
                )
                predicted_noise[normal_indx] = noise_normal

            # --- Handle bearded woman ('latent vector' arithmetic) --- #
            if bearded_woman_indx.numel() > 0:
                z_bw = z[bearded_woman_indx]
                t_bw = t_enc[bearded_woman_indx]
                uncond_noise_bw = unconditional_noise[bearded_woman_indx]

                num_bw = len(bearded_woman_indx)
                c_woman = torch.zeros(
                    num_bw, labels.shape[1], device=device, dtype=labels.dtype
                )
                c_bearded_man = torch.tensor(
                    [1, 0, 1], device=device, dtype=labels.dtype
                ).repeat(num_bw, 1)
                c_man = torch.tensor(
                    [1, 0, 0], device=device, dtype=labels.dtype
                ).repeat(num_bw, 1)

                woman_noise = den_net(z_bw, t_bw, c_woman)
                bearded_man_noise = den_net(z_bw, t_bw, c_bearded_man)
                man_noise = den_net(z_bw, t_bw, c_man)

                woman_direction = woman_noise - uncond_noise_bw
                beard_direction = (bearded_man_noise - uncond_noise_bw) - (
                    man_noise - uncond_noise_bw
                )

                noise_bw = (
                    uncond_noise_bw
                    + cfg_lambda * woman_direction
                    + cfg_lambda * beard_direction
                )
                predicted_noise[bearded_woman_indx] = noise_bw

            # --- Handle bearded woman eyeglasses ('latent vector' arithmetic) --- #
            if bearded_eyeglasses_woman_indx.numel() > 0:
                z_bew = z[bearded_eyeglasses_woman_indx]
                t_bew = t_enc[bearded_eyeglasses_woman_indx]
                uncond_noise_bew = unconditional_noise[bearded_eyeglasses_woman_indx]

                num_bew = len(bearded_eyeglasses_woman_indx)
                c_woman_eyeglasses = torch.tensor(
                    [0, 1, 0], device=device, dtype=labels.dtype
                ).repeat(num_bew, 1)
                c_bearded_man = torch.tensor(
                    [1, 0, 1], device=device, dtype=labels.dtype
                ).repeat(num_bew, 1)
                c_man = torch.tensor(
                    [1, 0, 0], device=device, dtype=labels.dtype
                ).repeat(num_bew, 1)

                woman_noise = den_net(z_bew, t_bew, c_woman_eyeglasses)
                bearded_man_noise = den_net(z_bew, t_bew, c_bearded_man)
                man_noise = den_net(z_bew, t_bew, c_man)

                woman_direction = woman_noise - uncond_noise_bew
                beard_direction = (bearded_man_noise - uncond_noise_bew) - (
                    man_noise - uncond_noise_bew
                )

                noise_bew = (
                    uncond_noise_bew
                    + cfg_lambda * woman_direction
                    + cfg_lambda * beard_direction
                )
                predicted_noise[bearded_eyeglasses_woman_indx] = noise_bew

        # Denoising Step
        alpha_t = alpha[t]
        beta_t = beta[t]

        # This is the random noise component for the sampling step
        step_noise = torch.zeros_like(z) if t == 0 else torch.randn_like(z)

        coef1 = 1 / torch.sqrt(1 - beta_t)
        coef2 = beta_t / torch.sqrt(1 - alpha_t)

        z = coef1 * (z - coef2 * predicted_noise) + torch.sqrt(beta_t) * step_noise
        yield z

    yield z



def tensor_to_base64(z):
    grid = utils.make_grid(z, normalize=True, nrow=5, padding=2)
    grid = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    img = Image.fromarray(grid)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=90)
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return img_str

def load_checkpoint(denoiser: CFGDenoiser ,path, optimizer=None, scheduler=None, map_location="cuda"):
    """Load the model's state dictionary and optimizer state from a file"""
    checkpoint = torch.load(path, map_location=map_location)
    denoiser.load_state_dict(checkpoint["denoiser"])
    if optimizer is None:
        return denoiser
    if scheduler is None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        return denoiser, optimizer

    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])

    return denoiser, optimizer, scheduler


def generate_grid(
    gen_fun,
    den_net,
    time_encoder,
    noise_size,
    classes,
    samples_per_row=10,
    cfg_lambda=3,
    save_name = "sample"
):
    labels = torch.zeros((noise_size[0], 3), device=device)

    for i in range(len(classes)):
        labels[(i * samples_per_row) : (i + 1) * samples_per_row] = torch.tensor(
            classes[i], dtype=torch.float32, device=device
        )

    images = gen_fun(den_net, time_encoder, noise_size, labels, cfg_lambda=cfg_lambda)
    grid = utils.make_grid(images, normalize=True, nrow=samples_per_row)
    sample_path = f"./SAMPLES/{save_name}.png"
    utils.save_image(grid, sample_path)

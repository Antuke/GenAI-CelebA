# Generative AI Project

This repository contains implementations of various generative models: **GAN**, **VAE**, and **Diffusion Models**.

## Project Structure

- **GAN/**: Generative Adversarial Network implementation.
  - `GAN_training.py`: Script for training the GAN.
  - `GAN_generation.ipynb`: Notebook for generating images with the GAN.
  
- **VAE/**: Variational Autoencoder implementation.
  - `vae_train.py`: Script for training the VAE.
  - `VAE_inference.ipynb`: Notebook for VAE inference.

- **DIFFUSION/**: Denoising Diffusion Probabilistic Models (DDPM/DDIM).
  - `inference.py`: Command-line script for inference.
  - `inference.ipynb`: Original inference notebook.
  - `model.py`: Model architecture (U-Net with Attention trough CBAM).

## Usage

### Diffusion Inference

You can run the diffusion inference script from the command line:

```bash
cd DIFFUSION
python inference.py --method ddpm --rows 1 --cols 4 --name my_sample
```

**Arguments:**
- `--method`: `ddpm` or `ddim` (default: `ddpm`)
- `--rows`: Number of rows/classes (default: `1`)
- `--cols`: Samples per row (default: `4`)
- `--cfg_lambda`: Guidance scale (default: `3.0`)
- `--device`: `cuda` or `cpu`

### GAN & VAE

For GAN and VAE, please refer to the respective notebooks (`.ipynb`) for generation and inference examples.


## Authors
Charlotte Boucherie, https://github.com/charlotte-bl
Massimiliano Ranauro, https://github.com/MassimilianoRanauro
Antonio Sessa, https://github.com/Antuke

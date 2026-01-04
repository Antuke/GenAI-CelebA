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

### Docker Support

You can also run the application using Docker. This provides a consistent environment and an easy-to-use browser interface for sample generation.

**Prerequisites:**
- [Docker](https://www.docker.com/get-started) installed on your machine.
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed (required for GPU acceleration).

**Running the Application:**

1. Navigate to the project root directory.
2. Run the following command:

```bash
docker compose up
```

The first time you run this, it will build the Docker image, which may take some time.

Once the server is running, you can access the application interface at:
[http://localhost:8000](http://localhost:8000)

### GAN & VAE

For GAN and VAE, please refer to the respective notebooks (`.ipynb`) for generation and inference examples.


## Authors
- [Charlotte Boucherie](https://github.com/charlotte-bl)
- [Massimiliano Ranauro](https://github.com/MassimilianoRanauro)
- [Antonio Sessa](https://github.com/Antuke)

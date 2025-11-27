import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cdim: int, droprate: float, stride: int = 1):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.cdim = cdim
        self.droprate = droprate
        self.stride = stride

        self.block_1 = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=stride),
        )

        # Feature-wise Linear Modulation, learn a shift and scale value for conditioning
        # It learns a shift value, and a scale value, for each channel of the tensor.
        # single network to predict both scale and shift parameter
        self.c_film = nn.Sequential(
            nn.Linear(3, cdim),
            nn.SiLU(),
            nn.Linear(cdim, out_ch * 2), # we use the same network to learn both parameters so *2
        )

        self.block_2 = nn.Sequential(
            nn.BatchNorm2d(out_ch),
            nn.SiLU(),
            nn.Dropout(p=droprate),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
        )
        
        if stride != 1 or in_ch != out_ch:
            self.residual = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor, cemb: torch.Tensor) -> torch.Tensor:
        residual_x = self.residual(x)
        latent = self.block_1(x)

        # Feature-wise Linear Modulation
        # It's a way to condition a neural network, method introduced in 2017 https://arxiv.org/pdf/1709.07871
        # For each conditioning signal, we learn a scale and shift value to apply to each channel of our x tensor.

        # unsqueeze(-1) adds a dimension at the end
        c_params = self.c_film(cemb).unsqueeze(-1).unsqueeze(-1) # [batch_size, C*2 , 1, 1]

        # chunk divides the tensor in two along the channel dimension
        c_scale, c_shift = torch.chunk(c_params, 2, dim=1) # [batch_size, C , 1, 1] & [batch_size, C , 1, 1]

        # 1 + c_scale, so to not hinder training if c_scale is initialized to a value close to 0
        latent = latent * (1 + c_scale) + c_shift
        latent = self.block_2(latent)

        latent += residual_x
        return latent

class ResidualTransposeBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cdim: int, droprate: float):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.cdim = cdim
        self.droprate = droprate

        self.block_1 = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.SiLU(),
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1),
        )
        
        # Feature-wise Linear Modulation, learn a shift and scale value for conditioning
        # It learns a shift value, and a scale value, for each channel of the tensor.
        # single network to predict both scale and shift parameter
        self.c_film = nn.Sequential(
            nn.Linear(3, cdim),
            nn.SiLU(),
            nn.Linear(cdim, out_ch * 2), # we use the same network to learn both parameters so *2
        )

        self.block_2 = nn.Sequential(
            nn.BatchNorm2d(out_ch),
            nn.SiLU(),
            nn.Dropout(p=droprate),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        )
        
        self.residual = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=1, stride=2, output_padding=1)

    def forward(self, x: torch.Tensor, cemb: torch.Tensor) -> torch.Tensor:
        residual_x = self.residual(x)
        latent = self.block_1(x)
      
        # Feature-wise Linear Modulation
        # It's a way to condition a neural network, method introduced in 2017 https://arxiv.org/pdf/1709.07871
        # For each conditioning signal, we learn a scale and shift value to apply to each channel of our x tensor.

        # unsqueeze(-1) adds a dimension at the end
        c_params = self.c_film(cemb).unsqueeze(-1).unsqueeze(-1) # [batch_size, C*2 , 1, 1]
        # chunk divides the tensor in two along the channel dimension
        c_scale, c_shift = torch.chunk(c_params, 2, dim=1) # [batch_size, C , 1, 1] & [batch_size, C , 1, 1]

        # 1 + c_scale, so to not hinder training if c_scale is initialized to a value close to 0
        latent = latent *  (1 + c_scale) + c_shift  # pytorch broadcasting is able to correctly do this operation
      
        latent = self.block_2(latent)

        latent += residual_x
        return latent

class VAE_model(nn.Module):
    def __init__(self, latent_dim=512, cond_dim=3, img_size=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.img_size = img_size
        self.ef_dim = 64 # encoding feature : base feature map size used in the encoder/decoder

        in_channels = 3 + 3  # condition channels concatenated to learn it everywwhere
        droprate = 0.0
        c_film_dim = 256 # hidden size of the FiLM MLP
        # input channels = 3 + cond_dim

        # --------- ENCODER ------------- #
        self.start = nn.Conv2d(in_channels, self.ef_dim , kernel_size=3, padding=1)
        self.enc_rb1 = ResidualBlock(self.ef_dim , self.ef_dim  * 2, cdim=c_film_dim, droprate=droprate, stride=2)       # 64x64 -> 32x32
        self.enc_rb2 = ResidualBlock(self.ef_dim  * 2, self.ef_dim  * 4, cdim=c_film_dim, droprate=droprate, stride=2)   # 32x32 -> 16x16
        self.enc_rb3 = ResidualBlock(self.ef_dim  * 4, self.ef_dim  * 8, cdim=c_film_dim, droprate=droprate, stride=2)   # 16x16 -> 8x8
        self.enc_rb4 = ResidualBlock(self.ef_dim  * 8, self.ef_dim  * 8, cdim=c_film_dim, droprate=droprate, stride=2)   # 8x8 -> 4x4

        self.encoder_output_dim = self.ef_dim * 8 * 4 * 4  # 512*4*4 = 8192
        self.mu = nn.Linear(self.encoder_output_dim, latent_dim)
        self.logsigma = nn.Linear(self.encoder_output_dim, latent_dim)


        # --------- DECODER ------------- #
        decoder_input_dim = latent_dim + self.cond_dim
        self.decoder_input = nn.Linear(decoder_input_dim, self.encoder_output_dim)
        self.dec_unflatten = nn.Unflatten(1, (self.ef_dim  * 8, 4, 4))
        
        self.dec_rb1 = ResidualTransposeBlock(self.ef_dim  * 8, self.ef_dim  * 8, cdim=c_film_dim, droprate=droprate) # 4x4 -> 8x8
        self.dec_rb2 = ResidualTransposeBlock(self.ef_dim  * 8, self.ef_dim  * 4, cdim=c_film_dim, droprate=droprate) # 8x8 -> 16x16
        self.dec_rb3 = ResidualTransposeBlock(self.ef_dim  * 4, self.ef_dim  * 2, cdim=c_film_dim, droprate=droprate) # 16x16 -> 32x32
        self.dec_rb4 = ResidualTransposeBlock(self.ef_dim  * 2, self.ef_dim , cdim=c_film_dim, droprate=droprate)     # 32x32 -> 64x64

        self.final = nn.Sequential(
            nn.BatchNorm2d(self.ef_dim ),
            nn.SiLU(),
            nn.Conv2d(self.ef_dim , 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def encode(self, x, cond):
        B = x.size(0)
        cond_img = cond.view(B, self.cond_dim, 1, 1).expand(-1, -1, self.img_size, self.img_size)
        x = torch.cat([x, cond_img], dim=1)

        x = self.start(x)
        
        x = self.enc_rb1(x, cond)
        x = self.enc_rb2(x, cond)
        x = self.enc_rb3(x, cond)
        x = self.enc_rb4(x, cond)
        
        x = x.view(B, -1)
        
        mu = self.mu(x)
        log_sigma = self.logsigma(x)

        return mu, log_sigma

    def reparameterize(self, mu, log_sigma):
        std = torch.exp(log_sigma)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, cond):
        z_cond = torch.cat([z, cond], dim=1)
        z = self.decoder_input(z_cond)
        z = self.dec_unflatten(z)
        
        z = self.dec_rb1(z, cond)
        z = self.dec_rb2(z, cond)
        z = self.dec_rb3(z, cond)
        z = self.dec_rb4(z, cond)
        
        y = self.final(z)
        return y

    def forward(self, x, cond):
        mu, log_sigma = self.encode(x, cond)
        z = self.reparameterize(mu, log_sigma)
        recon = self.decode(z, cond)
        return recon, mu, log_sigma
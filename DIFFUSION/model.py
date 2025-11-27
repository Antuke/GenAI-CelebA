import math
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda')

class SAM(nn.Module):
    """Spatial attention module, finds the most relevant pixels.
    Outputs a matrix [b,1,h,w] to re-weight the feature map.
    """
    def __init__(self, bias=False):
        super(SAM, self).__init__()
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1,
                              bias=self.bias)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # finds the maximum value for each pixel along the channels
        max_out = torch.max(x, 1)[0].unsqueeze(1) # [b, 1, h, w]

        # computes the avg value for each pixel along the channels
        avg_out = torch.mean(x, 1).unsqueeze(1) # [b, 1, h, w]

        concat = torch.cat((max_out, avg_out), dim=1) # [b, 2, h, w]
        output = self.conv(concat) # [b, 1, h, w]
        output = self.sigmoid(output) * x  
        return output


class CAM(nn.Module):
    """Channel attention module, finds the most relevant channels
    Outputs a 1-d Vector to re-weight the feature channels.
    """
    def __init__(self, channels, r):
        super(CAM, self).__init__()
        self.channels = channels
        self.r = r

        # MLP
        self.linear = nn.Sequential(
            nn.Linear(in_features=self.channels, out_features=self.channels // self.r, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.channels // self.r, out_features=self.channels, bias=True),
            nn.Sigmoid()  # Sigmoid here for channel attention weights
        )

    def forward(self, x):
        # max value in each channel
        max_pool = F.adaptive_max_pool2d(x, output_size=1) # [b, C, 1, 1]

        # avg value in each channel
        avg_pool = F.adaptive_avg_pool2d(x, output_size=1) # [b, C, 1, 1]

        b, c, _, _ = x.size()

        # Given the max value for each channel, the MLP learns a 1-d vector of c dimension 
        # to reweight each channel by multiplying it by a learned value between 0 and 1
        linear_max = self.linear(max_pool.view(b, c)).view(b, c, 1, 1)

        # Given the avg value for each channel, the MLP learns a 1-d vector of c dimension 
        # to reweight each channel by multiplying it by a learned value between 0 and 1
        linear_avg = self.linear(avg_pool.view(b, c)).view(b, c, 1, 1)



        output = linear_max + linear_avg
        return output * x  # pytorch broadcasting allows this operation 


class CBAM(nn.Module):
    """ Convolutional Block Attention Module (CBAM), as presented in https://arxiv.org/pdf/1807.06521v2
    Given a Feature Map F, of size CxHxW, CBAM computes 
    F' = M_C(F) x F
    F'' = M_S(F') x F'

    M_C(F), is the output of the Channel attention module, it's a Cx1x1 vector
    M_S(F''), is the output of the Spatial attention module, it's a 1xHxW matrix
    
    CBAM allows the neural network to focus on the more important part of a feature map of an image,
    by smartly reweighting 
    """
    def __init__(self, channels, r=4):
        super(CBAM, self).__init__()
        self.channels = channels
        self.r = r
        self.sam = SAM(bias=False)
        self.cam = CAM(channels=self.channels, r=self.r)

    def forward(self, x):
        cam_output = self.cam(x)
        sam_output = self.sam(cam_output)
        return sam_output + x # skip connection, the block will learn only small refinement

class TimeEncoding:
    def __init__(self, L, dim, device=device): # Original device usage
        self.L = L
        self.dim = dim
        dim2 = dim // 2
        encoding = torch.zeros(L, dim, device=device)
        ang = torch.linspace(0.0, torch.pi / 2, L)
        logmul = torch.linspace(0.0, math.log(40), dim2)
        mul = torch.exp(logmul)
        for i in range(dim2):
            a = ang * mul[i]
            encoding[:, 2 * i] = torch.sin(a)
            encoding[:, 2 * i + 1] = torch.cos(a)
        self.encoding = encoding.to(device=device)

    def __len__(self):
        return self.L

    def __getitem__(self, t_indices):
        # restituisce [batch_size, dim]
        return self.encoding[t_indices]


class Upsample(nn.Module):
    """
    Input C_inxHxW -> C_out x (Hx2) x (Wx2)
    an upsampling layer, using a convolution to 4x time the in channel,
    so later we can use pixelshuffle
    """
    def __init__(self, in_ch:int, out_ch:int):
        super().__init__()
        self.in_ch = in_ch
        self.layer = nn.Conv2d(in_ch, in_ch * 4, kernel_size = 3, stride = 1, padding = 1)
        self.pixel_shuffle1 = nn.PixelShuffle(2) 
        self.layer2 = nn.Conv2d(in_ch, out_ch, kernel_size = 3, stride = 1, padding = 1)
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.layer(x)
        x = self.pixel_shuffle1(x)
        x = self.layer2(x)

        return x
    
class Downsample(nn.Module):
    """
    Input C_inxHxW -> C_out x (H//2) x (W//2)
    a downsampling layer
    """
    def __init__(self, in_ch:int, out_ch:int):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.layer = nn.Conv2d(self.in_ch, self.out_ch, kernel_size = 3, stride = 2, padding = 1)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.layer(x)
    

class ResidualBlock(nn.Module):
    """
    Residual block, res-net style, with FiLM for conditioning and groupNorm for normalization.
    GroupNorm normalize tensors by dividing their channels in groups, and normalizing along this groups.
    Feature-wise Linear Modulation is the choice to pass conditioning information to the Neural Network.
    GELU = x * Φ(x) (similar to SILU and RELU, Φ is the CDF of the normal distribution)
    """
    def __init__(self, in_ch:torch.Tensor, out_ch:torch.Tensor, tdim:int, cdim:int, droprate:float):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.tdim = tdim
        self.cdim = cdim
        self.droprate = droprate

        self.block_1 = nn.Sequential(
            nn.GroupNorm(32 , in_ch),
            nn.GELU(),
            nn.Conv2d(in_ch, out_ch, kernel_size = 3, padding = 1),
        )

        # single network to predict both scale and shift parameter
        self.t_film = nn.Sequential(
            nn.GELU(),
            nn.Linear(tdim, out_ch * 2),
        )
        self.c_film = nn.Sequential(
            nn.GELU(),
            nn.Linear(cdim, out_ch * 2),
        )
        
        self.block_2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            nn.GELU(),
            nn.Dropout(p = self.droprate),
            nn.Conv2d(out_ch, out_ch, kernel_size = 3, stride = 1, padding = 1),
            
        )
        if in_ch != out_ch:
            self.residual = nn.Conv2d(in_ch, out_ch, kernel_size = 1, stride = 1, padding = 0)
        else:
            self.residual = nn.Identity()
    
    def forward(self, x:torch.Tensor, temb:torch.Tensor, cemb:torch.Tensor) -> torch.Tensor:
        latent = self.block_1(x)


        # Feature-wise Linear Modulation
        # It's a way to condition a neural network, method introduced in 2017 https://arxiv.org/pdf/1709.07871
        # For each conditioning signal, we learn a scale and shift value to apply to each channel of our x tensor.

        # unsqueeze(-1) adds a dimension at the end
        t_params = self.t_film(temb).unsqueeze(-1).unsqueeze(-1)  # [batch_size, C*2 , 1, 1]
        # chunk divides the tensor in two along the channel dimension
        t_scale, t_shift = torch.chunk(t_params, 2, dim=1) # [batch_size, C , 1, 1] & [batch_size, C , 1, 1]

        
        c_params = self.c_film(cemb).unsqueeze(-1).unsqueeze(-1)
        c_scale, c_shift = torch.chunk(c_params, 2, dim=1)
        

        # the +1 to the scale parameter is to ensure better inizialitazion
        # (if the scalarer param would get initiliazed to a near 0 value it would considerably hinder training)
        # (also, unconditional signal is all 0)
        latent = latent * (1 + t_scale) + t_shift # pytorch broadcasting is able to correctly do this operation
        latent = latent * (1 + c_scale) + c_shift # even if there is a shape mismatch

        
        latent = self.block_2(latent)

        latent += self.residual(x)
        return latent
    
class CFGDenoiser(nn.Module):
    def __init__(self, in_ch=3, base_ch = 64, cdim = 3, embdim = 256):
        super().__init__()
        self.in_ch = 3
        self.base_ch = base_ch
        self.length = 3 
        self.embdim = embdim


        self.t_mlp = nn.Sequential(
            nn.Linear(embdim, embdim * 4),
            nn.GELU(),
            nn.Linear(embdim * 4, embdim)
        )

        self.c_mlp = nn.Sequential(
            nn.Linear(cdim, embdim * 4),
            nn.GELU(),
            nn.Linear(embdim * 4, embdim)
        )


        self.in_conv = nn.Conv2d(self.in_ch, self.base_ch, kernel_size=3, padding=1)
        self.cbam0 = CBAM(self.base_ch)
        curr_ch = self.base_ch
        
        # --------- 4 downsampling block ------------- #

        self.res1d = ResidualBlock(curr_ch, curr_ch, embdim, embdim, 0.1) 
        self.d1 = Downsample(curr_ch, curr_ch * 2) # b_ch*2 x 32 x 32
        curr_ch *= 2

        self.res2d = ResidualBlock(curr_ch, curr_ch, embdim, embdim, 0.1)
        self.d2 = Downsample(curr_ch, curr_ch * 2) # b_ch*4 x 16 x 16
        curr_ch *= 2

        self.res3d = ResidualBlock(curr_ch, curr_ch, embdim, embdim, 0.1)
        self.d3 = Downsample(curr_ch, curr_ch * 2) # b_ch*8 x 8 x 8
        curr_ch *= 2

        self.res4d = ResidualBlock(curr_ch, curr_ch, embdim, embdim, 0.1)
        self.d4 = Downsample(curr_ch, curr_ch * 2) # b_ch*16 x 4 x 4
        curr_ch *= 2
        # --------- Bottleneck ------------ #
        
        self.bottleneck1 = ResidualBlock(curr_ch, curr_ch, embdim, embdim, 0.1)
        self.cbam1 = CBAM(curr_ch)
        self.bottleneck2 = ResidualBlock(curr_ch, curr_ch, embdim, embdim, 0.1)
        self.cbam2 = CBAM(curr_ch)

        # --------- 4 upsampling block ------------- #

        # input (1024*2) x 8 x 8
        self.res1u = ResidualBlock(curr_ch * 2, curr_ch, embdim, embdim, 0.1)
        self.u1 = Upsample(curr_ch, curr_ch // 2) 
        curr_ch //= 2

        # input (512*2) x 16 x 16
        self.res2u = ResidualBlock(curr_ch * 2, curr_ch, embdim, embdim, 0.1)
        self.u2 = Upsample(curr_ch, curr_ch // 2) 
        curr_ch //= 2

        # input (256*2) x 32 x 32
        self.res3u = ResidualBlock(curr_ch * 2, curr_ch, embdim, embdim, 0.1)
        self.u3 = Upsample(curr_ch, curr_ch // 2) 
        curr_ch //= 2

        # input (128*2) x 64 x 64
        self.res4u = ResidualBlock(curr_ch * 2, curr_ch, embdim, embdim, 0.1)
        self.u4 = Upsample(curr_ch, curr_ch // 2) 
        self.cbam3 = CBAM(curr_ch)
        curr_ch //= 2
        
        # ---------  output ------------- #


        self.out = nn.Sequential(
            nn.GroupNorm(32, curr_ch),
            nn.GELU(),
            nn.Conv2d(curr_ch, curr_ch, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, curr_ch),
            nn.GELU(),
            nn.Conv2d(curr_ch, self.in_ch, kernel_size=3, stride=1, padding=1),
        )  # 3 x 64 x 64

    def forward(self, x, t, c, dropout_p = 0.2):
        t = self.t_mlp(t)

        if c is None:
            c = torch.zeros((x.shape[0], self.embdim), device=x.device)
        else:
            c = self.c_mlp(c)
            if self.training:
                # boolean mask, [b, 1], with each value having 20% possibility of being false
                mask = torch.rand((c.shape[0], 1), device=c.device) < dropout_p
                zeros_for_uncond = torch.zeros_like(c)

                # Where the mask is true, set the value of c equal to zero, otherwise keep them as they are
                c = torch.where(mask, zeros_for_uncond, c)

        

        x0 = self.in_conv(x)
        # --- encoding --- #
        x1 = self.res1d(x0, t, c)
        x1d = self.d1(x1)

        x2 = self.res2d(x1d, t, c)
        x2d = self.d2(x2)

        x3 = self.res3d(x2d, t, c)
        x3d = self.d3(x3)

        x4 = self.res4d(x3d, t, c)
        x4d = self.d4(x4)

        # --- bottleneck --- #
        xb1 = self.bottleneck1(x4d, t, c)
        xb1a = self.cbam1(xb1)

        xb2 = self.bottleneck2(xb1a, t, c)
        xb2a = self.cbam2(xb2)

        # --- Decoding --- #
        y = torch.cat([xb2a, x4d], dim=1)
        y = self.res1u(y, t, c)
        y = self.u1(y)
        
        y = torch.cat([y, x3d], dim=1)
        y = self.res2u(y, t, c)
        y = self.u2(y)

        y = torch.cat([y, x2d], dim=1)
        y = self.res3u(y, t, c)
        y = self.u3(y)

        y = torch.cat([y, x1d], dim=1)
        y = self.res4u(y, t, c)
        y = self.u4(y)


        y = self.out(y)
        return y

    def get_name(self):
        return 'res_cbam_deeper'
        
def run_test(img_size, T=1000, time_emb_dim=256, batch_size=4, c_in_dim_test=3, base_channels_test=32): # Added base_channels_test
    time_encoder = TimeEncoding(T, time_emb_dim, device=device)
    t_indices = torch.randint(low=0, high=T, size=(batch_size,), device=device)
    t = time_encoder[t_indices]
    c = torch.zeros((batch_size,c_in_dim_test), dtype=torch.float32, device=device)
    
    print(f"\n--- Testing Image Size: {img_size}x{img_size},  Base Channels: {base_channels_test} ---")

    synth_3c = torch.randn((batch_size, 3, img_size, img_size), device=device)
    net_3c = CFGDenoiser(base_ch=base_channels_test, embdim=time_emb_dim, cdim=c_in_dim_test).to(device)
    total_params = sum(p.numel() for p in net_3c.parameters())
    print(f'Model parameters: {total_params:,}')
    try:
        out_3c = net_3c(synth_3c, t, c)
        assert out_3c.shape == synth_3c.shape, f"3C Test Failed: Expected shape {synth_3c.shape}, but got {out_3c.shape}"
        print(f"  Input shape: {synth_3c.shape}, Output shape: {out_3c.shape} -> OK")
    except Exception as e:
        print(f"  3C Test Failed: {e}")
        raise

   

    print(f"--- Tests passed for image size {img_size}x{img_size} ---")


def run_tests():
    test_img_sizes = [64] 
    test_time_dim = 256
    test_c_in_dim = 3
    test_base_channels = 64 

    for size in test_img_sizes:
        run_test(img_size=size, time_emb_dim=test_time_dim, c_in_dim_test=test_c_in_dim, base_channels_test=test_base_channels)



    print('\nTutti i test passati!')


if __name__ == '__main__':
    print(f"Using device: {device}")
    run_tests()

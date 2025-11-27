import torch
from torch import nn

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# COMMON CLASSES
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

class ConditionMPL(nn.Module):
    """MPL for processing condition vectors. Transforms a condition vector into a learned feature representation"""
    def __init__(self, cond_size:int, output_size:int, expand:int=1, device:str=None):
        """
        Params:
            cond_size (int):    Dimensionality of the input condition vector
            output_size (int):  Desired output dimensionality
            expand (int):       Given an input of size [batch x channels] returns an output of [batch x channels x expand x expand]
            device (str):       Device to place the module on
        """
        super().__init__()
        self.expand= expand
        self.network= nn.Sequential(
            nn.Linear(cond_size, output_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(output_size, output_size)
        ).to(device=device)
    
    def forward(self, x:torch.Tensor):
        x= self.network(x.float())
        return x.reshape(*x.shape, 1, 1).expand(*x.shape, self.expand, self.expand)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# GENERATOR
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

class BlockDoubleSize(nn.Module):
    """ConvTranspose2d block that doubles the spatial size, if value are set as default"""
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int=4, stride:int=2, padding:int=1, device:str=None):
        """
        The input must be spatial greather or equal of 2x2 if you want to use the default values
        Params:
            in_channels (int):  Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int):  Size of the convolving kernel
            stride (int):       Stride of the convolution
            padding (int):      Controls the amount of implicit zero padding
            device (str):       Device to place the model on
        """
        super().__init__()
        self.net= nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ).to(device=device)
    
    def forward(self, x:torch.Tensor):
        return self.net(x)

class Generator(nn.Module):
    def __init__(self, size_latent_space:int=128, cond_size:int=3, cond_representation_size:int=32, device:str=None):
        """
        Params:
            size_latent_space (int):        Dimensionality of the latent space input
            num_bool_values (int):          Number of binary condition variables
            cond_size_rapresentation (int): Size of feature representation layer of the condition
            device (str):                   Device to place the model on
        """
        super().__init__()
        
        self.block_4_4=     BlockDoubleSize(size_latent_space, 512, kernel_size=4, stride=1, padding=0, device=device)
        self.block_8_8=     BlockDoubleSize(512, 256, device=device)
        self.block_16_16=   BlockDoubleSize(256, 192, device=device)
        
        self.condition_layer=   ConditionMPL(cond_size, cond_representation_size, expand=16, device=device)
        self.block_16_16_cond=  nn.Sequential(
            nn.Conv2d(192 + cond_representation_size, 128, kernel_size=5, padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        ).to(device=device)
        
        self.block_32_32=   BlockDoubleSize(128, 64, kernel_size=4, stride=2, padding=1, device=device)
        self.block_64_64=   nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        ).to(device=device)
    
    def forward(self, x:torch.Tensor, conditions:torch.Tensor):
        x= x.reshape(*x.shape, 1, 1)
        
        x= self.block_4_4(x)
        x= self.block_8_8(x)
        x= self.block_16_16(x)
        
        conditions= self.condition_layer(conditions)
        x_cat= torch.cat((x, conditions), dim=1)
        x= self.block_16_16_cond(x_cat)
        
        x= self.block_32_32(x)
        x= self.block_64_64(x)
        return x

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# DISCRIMINATOR
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

class Minibatch_std(nn.Module):
    """Add a channels with the mean of the batch"""
    def __init__(self, extend:int=1):
        """
            :param expand (int): Given an input of any size, returns an output of [batch x channels x extend x extend]
        """
        super().__init__()
        self.extend= extend

    def forward(self, x:torch.Tensor):
        size= [x.shape[0], 1, self.extend, self.extend]
        
        std = torch.std(x, dim=0)
        mean = torch.mean(std)
        return mean.repeat(size)

class BlockHalve(nn.Module):
    """Downsampling block that halves the spatial size"""
    def __init__(self, in_channels:int, out_channels:int, device:str=None):
        """
        Params:
            in_channels (int):      Number of input channels
            out_channels (int):     Number of output channels
            device (str):           Device to place the model on
        """
        super().__init__()
        self.net= nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        ).to(device=device)
    
    def forward(self, x:torch.Tensor):
        return self.net(x)

class Discriminator(nn.Module):
    """Conditional discriminator for distinguishing real vs generated images"""
    def __init__(self, cond_size:int=3, cond_representation_size:int=32, device:str=None):
        """
        Params:
            num_bool_values (int):          Number of binary condition variables
            cond_size_rapresentation (int): Size of feature representation layer of the condition
            device (str):                   Device to place the model on
        """
        super().__init__()
        
        self.block_32_32=   BlockHalve(  3,  64, device=device)
        self.block_16_16=   BlockHalve( 64, 128, device=device)
        self.block_8_8=     BlockHalve(128, 256, device=device)
        
        self.condition_layer=   ConditionMPL(cond_size, cond_representation_size, expand=8, device=device)
        self.block_8_8_cond=    nn.Sequential(
            nn.Conv2d(256 + cond_representation_size, 256, kernel_size=5, padding='same'),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        ).to(device=device)
        
        self.block_4_4= BlockHalve(256, 512, device=device)
        self.batch_std= Minibatch_std(extend=4)
        self.block_1_1= nn.Sequential(
            nn.Conv2d(512+1, 1, kernel_size=4),
            nn.Flatten(),
            nn.Sigmoid()
        )
        
    def forward(self, x:torch.Tensor, conditions:torch.Tensor):
        batch= self.batch_std(x)
        
        x= self.block_32_32(x)
        x= self.block_16_16(x)
        x= self.block_8_8(x)
        
        conditions= self.condition_layer(conditions)
        x_cat= torch.cat((x, conditions), dim=1)
        x= self.block_8_8_cond(x_cat)
        
        x= self.block_4_4(x)
        x_cat= torch.cat((x, batch), dim=1)
        x= self.block_1_1(x_cat)
        return x

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# TEST CORRECT SIZE MODEL
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

if __name__=='__main__':
    DEVICE= 'cpu'
    BATCH_SIZE= 32
    LATENT_SIZE= 128
    COND_REPRESENTATION_SIZE= LATENT_SIZE//4
    NUM_CLASSES= 3

    print("* "*50)

    # TEST GENERATOR
    z= torch.randn(BATCH_SIZE, LATENT_SIZE, device=DEVICE)
    cond= [[False, False, False] for _ in range(BATCH_SIZE)]
    cond= torch.tensor(cond, dtype=torch.float32)

    try:
        generator= Generator(LATENT_SIZE, NUM_CLASSES, COND_REPRESENTATION_SIZE, device=DEVICE)
        result= generator(z, cond)
        if not( result.shape==torch.Size([BATCH_SIZE,3,64,64]) ):
            print(f"Size output generator is {result.shape} instead of {[BATCH_SIZE,3,64,64]}")
    except:
        print("An expection occurred while testing the GENERATOR")
    

    # TEST DISCRIMINATOR
    z= torch.randn(BATCH_SIZE, 3, 64, 64, device=DEVICE)
    cond= [[False, False, False] for _ in range(BATCH_SIZE)]
    cond= torch.tensor(cond, dtype=torch.float32)

    try:
        discriminator= Discriminator(NUM_CLASSES, COND_REPRESENTATION_SIZE, device=DEVICE)
        result= discriminator(z, cond)
        if not( result.shape==torch.Size([BATCH_SIZE,1]) ):
            print(f"Size output discriminator is {result.shape} instead of {[BATCH_SIZE,1]}")
    except:
        print("An expection occurred while testing the DISCRIMINATOR")
    
    print("TEST CONCLUDED")

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("* "*50)
    print("Number parameters Generator:     ", count_parameters(generator))
    print("Number parameters Discriminator: ", count_parameters(discriminator))

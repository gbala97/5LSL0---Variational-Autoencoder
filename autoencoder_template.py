# %% imports
from turtle import clear
from unicodedata import name
import torch
import torch.nn as nn
import torch.nn.functional as F 

# %%  Encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encode_network = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding = 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, stride = 2),
            nn.Conv2d(16, 16, 3, padding = 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, stride = 2),
            nn.Conv2d(16, 16, 3, padding = 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, stride = 2),
            nn.Conv2d(16, 1, 3, padding = 1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.AvgPool2d((3,4), stride =1)
        )
        # create layers here
        
    def forward(self, x):
        # use the created layers here
        h = self.encode_network(x)
        return h
    
# %%  Decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decode_network = nn.Sequential(
            nn.ConvTranspose2d(1, 16, (3,4), padding = 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(16, 16, 3, padding = 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(16, 16, 3, padding = 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(16, 16, 3, padding = 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(16, 1, 3, padding = 1),
            nn.Tanh()
        )
        
    def forward(self, h):
        # use the created layers here
        r = self.decode_network(h)
        return r
    
# %%  Autoencoder
class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        
    def forward(self, x):
        h = self.encoder(x)
        r = self.decoder(h)
        return r

# Test code to check the encoder and decoder architecture 
def encoder_check():
    x = torch.zeros((1, 1, 32, 32), dtype=torch.float32)
    model = Encoder()
    out = model(x)
    print(out.shape) # Expect to see [N * 1 * 2 *1]

def AE_check():
    x = torch.zeros((10, 1, 32, 32), dtype=torch.float32)
    model = AE()
    out = model(x)
    print("AE latent space shape", out[1].shape)
    print("AE output shape", out[0].shape)

if __name__ == "__main__":
    AE_check()


import torch.nn as nn
import torch.nn.functional as F
import torch


class VAE( nn.Module ):
    def __init__(self, image_size=784, h_size=400, z_size=20):
        super(VAE, self).__init__()
        # Encoder
        self.encoder = Encoder( image_size, h_size, z_size)
        # Decoder
        self.decoder = Decoder( image_size, h_size, z_size)
        
    def forward(self, x):
        mean, sigma = self.encoder(x)
        z = self.add_rand_noise( mean, sigma)
        output = self.decoder( z )
        
        return output, mean, sigma

    def add_rand_noise( self, mean, sigma ):
        e = torch.rand_like(sigma)
        return mean + (torch.exp(sigma)**0.5) * e


# Encoder Model
class Encoder( nn.Module ):
    def __init__(self, image_size, h_size, z_size):
        super(Encoder,self).__init__()

        self.fc1 = nn.Linear(image_size, h_size)
        # self.fc2 = nn.Linear(h_size, h_size)
        self.mean_layer = nn.Linear(h_size, z_size)
        self.sigma_layer = nn.Linear(h_size, z_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        mean = self.mean_layer(x)
        sigma = self.sigma_layer(x)
        return mean, sigma

    
# Decoder Model
class Decoder( nn.Module ):
    def __init__(self, image_size, h_size, z_size):
        super( Decoder, self ).__init__()
        
        self.fc1 = nn.Linear(z_size, h_size)
        self.fc2 = nn.Linear(h_size, image_size)
        
    def forward(self, z ):
        z = F.relu(self.fc1(z))
        z = F.sigmoid(self.fc2(z))
        return z
    

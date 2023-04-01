import torch.nn as nn

# AE Model
class AutoEncoder( nn.Module ):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        # Encoder
        self.encoder = Encoder()
        # Decoder
        self.decoder = Decoder()
        
    def forward(self, x):
        embedding = self.encoder(x)
        output = self.decoder( embedding )
        
        return embedding, output


# Encoder Model
class Encoder( nn.Module ):
    def __init__(self):
        super(Encoder,self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear( 28*28, 256 ),
            nn.LeakyReLU(),
            nn.Linear(256,128),
            nn.LeakyReLU(),
            nn.Linear(128,64),
            nn.LeakyReLU(),
            nn.Linear(64,2),
            nn.Tanh()
        )
        
    def forward(self, x):
        embedding = self.encoder(x)
        return embedding
    
# Decoder Model
class Decoder( nn.Module ):
    def __init__(self):
        super( Decoder, self ).__init__()
        
        self.decoder = nn.Sequential(
            nn.Linear( 2, 64 ),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 784),
            nn.Tanh()
        )
        
    def forward(self, embedding ):
        return self.decoder( embedding )
    

import torch
from torch import nn
import numpy as np

class variationalautoencoder(torch.nn.Module):
    def __init__(self,input_dim, hidden_dim=200, latent_dim=20,
                 *args, **kwargs):
        super().__init__()  # *args, **kwargs

        # encoder
        self.encode_layer = nn.Linear(input_dim, hidden_dim)
        self.encode_mu = nn.Linear(hidden_dim, latent_dim)
        self.encode_sigma_std = nn.Linear(hidden_dim, latent_dim)

        # decoder
        self.decode_layer = nn.Linear(latent_dim, hidden_dim)
        self.decode_output = nn.Linear(hidden_dim, input_dim)

        self.activation = nn.ReLU()

    def encode(self, x):
        encode_output = self.activation(self.encode_layer(x))
        mu = self.encode_mu(encode_output)
        sigma_std= self.encode_sigma_std(encode_output)

        return mu, sigma_std

    def decode(self, z):
        decode_hidden=self.activation(self.decode_layer(z))
        return torch.sigmoid(self.decode_output(decode_hidden))
    
    def fix_seed(self):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(self.seed)

    def forward(self, x):
        mu,sigma_std=self.encode(x)
        epsilon=torch.randn_like(sigma_std)
        z=mu+sigma_std*epsilon # matrix multiplication  diag(sigma_std)*epsilon = element wise multiplication sigma_std*epsilon
        decode_output=self.decode(z)
        return decode_output,mu,sigma_std

if __name__=="__main__":
    x=torch.randn(4,28*28)
    vae=variationalautoencoder(input_dim=784)
    x_reconstructed,mu,sigma=vae(x)
    for elt in [x_reconstructed,mu,sigma]:
        print(elt.shape)

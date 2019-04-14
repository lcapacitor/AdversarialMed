import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.init_weights()

    def init_weights(self):
        classname = self.__class__.__name__
        if classname.find('Conv') != -1:
            self.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            self.weight.data.normal_(1.0, 0.02)
            self.bias.data.fill_(0)


class ResBlock(BaseModel):
    def __init__(self, dim):
        super(ResBlock, self).__init__()
        self.dim = dim
        self.convs = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
        )
        
    def forward(self, x):
        return F.relu(x+self.convs(x))


class VAE(BaseModel):
    def __init__(self, image_size, channel_num, hidden_dim, z_size, is_res, drop_p):
        # configurations
        super().__init__()
        self.image_size = image_size
        self.channel_num = channel_num
        self.hidden_dim = hidden_dim
        self.z_size = z_size
        self.is_res = is_res

        # encoder
        if self.is_res:
            self.encoder = nn.Sequential(
                self._conv(channel_num, hidden_dim // 4),
                ResBlock(hidden_dim // 4),
                self._conv(hidden_dim // 4, hidden_dim // 4),
                ResBlock(hidden_dim // 4),
                self._conv(hidden_dim // 4, hidden_dim // 2),
                ResBlock(hidden_dim // 2),
                self._conv(hidden_dim // 2, hidden_dim, last=True),
                #nn.Dropout(drop_p),
            )
        else:
            self.encoder = nn.Sequential(
                self._conv(channel_num, hidden_dim // 4),
                self._conv(hidden_dim // 4, hidden_dim // 4),
                self._conv(hidden_dim // 4, hidden_dim // 2),
                self._conv(hidden_dim // 2, hidden_dim, last=True),
                #nn.Dropout(drop_p),
            )

        # encoded feature's size and volume
        self.feature_size = image_size // 16
        self.feature_volume = hidden_dim * (self.feature_size ** 2)

        # q
        self.q_mean = self._linear(self.feature_volume, z_size, relu=False)
        self.q_logvar = self._linear(self.feature_volume, z_size, relu=False)

        # projection
        self.project = self._linear(z_size, self.feature_volume, relu=False)

        # decoder
        if self.is_res:
            self.decoder = nn.Sequential(
                self._deconv(hidden_dim, hidden_dim // 2),
                ResBlock(hidden_dim // 2),
                self._deconv(hidden_dim // 2, hidden_dim // 4),
                ResBlock(hidden_dim // 4),
                self._deconv(hidden_dim // 4, hidden_dim // 4),
                ResBlock(hidden_dim // 4),
                self._deconv(hidden_dim // 4, channel_num, last=True),
                nn.Sigmoid()
            )
        else:
            self.decoder = nn.Sequential(
                self._deconv(hidden_dim, hidden_dim // 2),
                self._deconv(hidden_dim // 2, hidden_dim // 4),
                self._deconv(hidden_dim // 4, hidden_dim // 4),
                self._deconv(hidden_dim // 4, channel_num, last=True),
                nn.Sigmoid()
            )


    def forward(self, x):
        # encode x
        encoded = self.encoder(x)

        # sample latent code z from q given x.
        mean, logvar = self.q(encoded)
        z = self.z(mean, logvar)
        z_projected = self.project(z).view(
            -1, self.hidden_dim,
            self.feature_size,
            self.feature_size,
        )

        # reconstruct x from z
        x_reconstructed = self.decoder(z_projected)

        # return the parameters of distribution of q given x and the reconstructed image.
        return (mean, logvar), x_reconstructed


    # ==============
    # VAE components
    # ==============

    def q(self, encoded):
        unrolled = encoded.view(-1, self.feature_volume)
        return self.q_mean(unrolled), self.q_logvar(unrolled)

    def z(self, mean, logvar):
        std = logvar.mul(0.5).exp_()
        eps = (
            Variable(torch.randn(std.size())).to(device)
        )
        return eps.mul(std).add_(mean)

    def reconstruction_loss(self, x_reconstructed, x):
        return F.binary_cross_entropy(x_reconstructed, x, reduction='sum') / x.size(0)

    def kl_divergence_loss(self, mean, logvar):
        return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) / mean.size(0)

    # ======
    # Layers
    # ======

    def _conv(self, channel_size, hidden_dim, last=False):
        conv = nn.Conv2d(
                channel_size, hidden_dim,
                kernel_size=3, stride=2, padding=1,
        )
        return conv if last else nn.Sequential(
            conv,
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
        )

    def _deconv(self, channel_num, hidden_dim, last=False):
        deconv = nn.ConvTranspose2d(
            channel_num, hidden_dim,
            kernel_size=4, stride=2, padding=1,
        )
        return deconv if last else nn.Sequential(
            deconv,
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
        )

    def _linear(self, in_size, out_size, relu=True):
        return nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.ReLU(),
        ) if relu else nn.Linear(in_size, out_size)


    # ================
    # generateNewImage
    # ================

    def generateNewImage(self, size):
        z = Variable(
            torch.randn(size, self.z_size).to(device)
        )
        z_projected = self.project(z).view(
            -1, self.hidden_dim,
            self.feature_size,
            self.feature_size,
        )
        return self.decoder(z_projected).data


if __name__ == '__main__':
    z_size = 1024
    hidden_dim = 128
    drop_p = 0.5
    b_size = 10
    img = torch.randn((b_size, 3, 224, 224))
    image_size = img.size(-1)
    channel_num = img.size(1)
    is_res = True

    vae = VAE(image_size, channel_num, hidden_dim, z_size, is_res, drop_p).to(device)

    (mean, logvar), x_reconstructed = vae(img)

    print (mean.shape, logvar.shape, x_reconstructed.shape, img.shape)


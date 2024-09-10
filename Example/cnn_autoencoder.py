class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape,  # extra comma

    def forward(self, x):
        return x.view(*self.shape)

class CV_CM(nn.Module):
  def __init__(self):
    super().__init__()


    self.encoder = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Flatten(),

        nn.Linear(7 * 7 * 64, LATENT*2),
        # nn.Dropout(p=0.5),
        nn.ReLU(),

        nn.Linear(LATENT*2, LATENT)
    )

    self.decoder = nn.Sequential(
        nn.Linear(LATENT, 128),
        nn.ReLU(),

        nn.Linear(128, 7 * 7 * 64),

        View((-1,64,7,7)),

        nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        #
        nn.UpsamplingBilinear2d(scale_factor=2),
        #

        nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.UpsamplingBilinear2d(scale_factor=2),
        #

        nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1),
    )

    self.cm = Clustering_Module( LATENT, 10, False)

  def forward(self, x):
    z = self.encoder(x)
    tx = self.decoder(z)
    cm = self.cm(z) # NxC, NxK, NxC, KxC

    return tx, cm

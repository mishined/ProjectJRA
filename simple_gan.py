import pytorch_lightning as pl
from model import *
import torch
from pix.utils import save_checkpoint, load_checkpoint, save_some_examples
import torch.nn as nn
import torch.optim as optim
import pix.config as config
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image
from mri_images import *

torch.backends.cudnn.benchmark = True

class GAN(nn.Module):
    def __init__(self, args):
        self.args = args

    def train_fn(disc, gen, loader, opt_disc, opt_gen, l1_loss, bce, g_scaler, d_scaler):
        loop = tqdm(loader, leave=True)

        for idx, (x, y) in enumerate(loop):
            x = x.to(config.DEVICE)
            y = y.to(config.DEVICE)

            # Train Discriminator
            with torch.cuda.amp.autocast():
                y_fake = gen(x)
                D_real = disc(x, y)
                D_real_loss = bce(D_real, torch.ones_like(D_real))
                D_fake = disc(x, y_fake.detach())
                D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
                D_loss = (D_real_loss + D_fake_loss) / 2

            disc.zero_grad()
            d_scaler.scale(D_loss).backward()
            d_scaler.step(opt_disc)
            d_scaler.update()

            # Train generator
            with torch.cuda.amp.autocast():
                D_fake = disc(x, y_fake)
                G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
                L1 = l1_loss(y_fake, y) * config.L1_LAMBDA
                G_loss = G_fake_loss + L1

            opt_gen.zero_grad()
            g_scaler.scale(G_loss).backward()
            g_scaler.step(opt_gen)
            g_scaler.update()

            if idx % 10 == 0:
                loop.set_postfix(
                    D_real=torch.sigmoid(D_real).mean().item(),
                    D_fake=torch.sigmoid(D_fake).mean().item(),
                )

    def GAN(args):
        gen = Generator(args.img_size, args.style_dim, w_hpf=args.w_hpf)
        disc = Discriminator(args.img_size, args.num_domains)

        opt_disc = torch.optim.Adam(disc.parameters(), lr = args.lr)
        opt_gen = torch.optim.Adam(gen.parameters(), lr = args.lr)

        BCE = nn.BCEWithLogitsLoss()
        L1_LOSS = nn.L1Loss()

        t_data = args.sample
        g_scaler = torch.cuda.amp.GradScaler()
        d_scaler = torch.cuda.amp.GradScaler()
        v_data = args.sample

        for epoch in range(10):
            args.train_fn(disc, gen, opt_disc, opt_gen, L1_LOSS, BCE, g_scaler, d_scaler)

            if epoch % 5 == 0:
                save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
                save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)

            save_some_examples(gen, v_data, epoch, folder = "images")


if __name__ == "main":
    imags = MRI_Images('/Users/misheton/OneDrive-UniversityofSussex/JRA/Data')

    model = GAN(imags)
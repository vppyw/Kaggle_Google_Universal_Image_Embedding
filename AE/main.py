import os

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import numpy as np
import random
import torchvision
from torchvision import datasets, transforms
from tqdm import tqdm
import itertools

import autoencoder

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def dnorm(x):
    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1).to(x.get_device())
    std = torch.Tensor([0.229, 0.224, 0.255]).reshape(-1, 1, 1).to(x.get_device())
    x = torch.clamp(std * x + mean, 0, 1)
    return x

def main():
    set_seed(0)
    config = {
        'val_ratio': 0.01,
        'dataset_size': 50000,
        'batch_size': 64,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'number_epoch': 25,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'data_dir': '../data',
        'log_dir': './gen_log',
    }

    tfm = transforms.Compose([
            transforms.AutoAugment(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
        ])    

    dataset = datasets.CIFAR10(root=config['data_dir'],
                                train=True,
                                download=True,
                                transform=tfm)
    
    testset = datasets.CIFAR10(root=config['data_dir'],
                                train=False,
                                download=True,
                                transform=None)

    train_sz = int((1 - config[ "val_ratio"])\
                    * min(len(dataset), config["dataset_size"]))
    val_sz = min(len(dataset), config["dataset_size"]) - train_sz
    remain_sz = len(dataset) - train_sz - val_sz
    trainset, validset, _ = random_split(dataset,
                                        [train_sz, val_sz, remain_sz])
    trainloader = DataLoader(trainset,
                                batch_size=config["batch_size"],
                                shuffle=True)
    validloader = DataLoader(validset,
                                batch_size=config["batch_size"],
                                shuffle=True) 
    testloader = DataLoader(trainset,
                                batch_size=config["batch_size"],
                                shuffle=False)

    encoder = autoencoder.Encoder().to(config['device'])
    decoder = autoencoder.Decoder().to(config['device'])
    criterion = nn.MSELoss()
    opt = torch.optim.AdamW(itertools.chain(encoder.parameters(),
                                            decoder.parameters()),
                            lr=config['learning_rate'],
                            weight_decay=config['weight_decay'])

    mn_loss = np.inf
    for epoch in range(config['number_epoch']):
        train_loss = []
        val_loss = []

        encoder.train()
        decoder.train()
        for imgs, _ in tqdm(trainloader, ncols=50):
            imgs = imgs.to(config['device'])
            gen_imgs = decoder(encoder(imgs, norm=False))
            loss = criterion(gen_imgs, imgs)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss.append(loss.item())

        encoder.eval()
        decoder.eval()

        log = False
        for imgs, _ in tqdm(validloader, ncols=50):
            imgs = imgs.to(config['device'])
            gen_imgs = decoder(encoder(imgs, norm=False))
            loss = criterion(gen_imgs, imgs)
            opt.zero_grad()
            loss.backward()
            opt.step()
            val_loss.append(loss.item())
            if not log:
                filename = os.path.join(config['log_dir'], f'Epoch_{epoch+1:03d}_origin.jpg')
                imgs = dnorm(imgs)
                torchvision.utils.save_image(imgs, filename, nrow=8)
                filename = os.path.join(config['log_dir'], f'Epoch_{epoch+1:03d}_gen.jpg')
                gen_imgs = dnorm(gen_imgs)
                torchvision.utils.save_image(gen_imgs, filename, nrow=8)
                log = True

        encoder.eval()
        encoder.to('cpu')
        saved_model = torch.jit.script(encoder)
        saved_model.save('saved_model.pt')
        encoder.to(config['device'])
        print(f'saved model at {epoch}')

        print(f'epoch: {epoch} | \
Train Loss {sum(train_loss) / len(train_loss):.4e} | \
Valid Loss {sum(val_loss) / len(val_loss):.4e}')

if __name__ == "__main__":
    main()

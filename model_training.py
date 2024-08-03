import torch
import torchvision
from torchvision import transforms
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from tqdm.notebook import tqdm
import numpy as np
import matplotlib.pyplot as plt
from model import PneumoniaModel

# Defining a loader function, **load_file**, which defines how the files shall be loaded.
def load_file(path):
    return np.load(path).astype(np.float32)

train_transforms = transforms.Compose([
                                      transforms.ToTensor(),  # Converting numpy array to tensor
                                      transforms.Normalize(0.49, 0.248),  # Using mean and std from preprocessing notebook
                                      transforms.RandomAffine( # Data Augmentation
                                      degrees=(-5, 5), translate=(0, 0.05), scale=(0.9, 1.1)),
                                      transforms.RandomResizedCrop((224, 224), scale=(0.35, 1))
                                    ])

val_transforms = transforms.Compose([
                                    transforms.ToTensor(),  # Convert numpy array to tensor
                                    transforms.Normalize([0.49], [0.248]),  # Use mean and std from preprocessing notebook
                                    ])

# Creating train and the val daatset corresponding data loaders
# torch.seed(42)
train_dataset = torchvision.datasets.DatasetFolder(
    "Data/Processed/train/",
    loader=load_file, extensions="npy", transform=train_transforms)
val_dataset = torchvision.datasets.DatasetFolder(
    "Data/Processed/val/",
    loader=load_file, extensions="npy", transform=val_transforms)
    
batch_size = 16
num_workers = 3

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, persistent_workers=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, persistent_workers=True)
print(f"There are {len(train_dataset)} train images and {len(val_dataset)} test images.")
print(f"Size of train loader: {len(train_loader)} and size test loader: {len(val_loader)}")

#Instanciating model
model = PneumoniaModel() 
print("Model Created")

# Create the checkpoint callback
checkpoint_callback = ModelCheckpoint(
                                      monitor='Val Acc',
                                      save_top_k=10,
                                      mode='max'
                                      )

# Creating the trainer
gpus = 0
trainer = pl.Trainer(logger=TensorBoardLogger(save_dir="./logs"), log_every_n_steps=10,
                     callbacks=checkpoint_callback,
                     max_epochs=1
                     )

print("Model Training...")
trainer.fit(model, train_loader, val_loader)


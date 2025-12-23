import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split

import torch
import torch.nn as nn
from torchvision import models
import torch.optim as optim
import gc

from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from rice_dataset import RiceGrainDataset
from helper import train_model, get_modified_model, train_transforms, val_transforms
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#



full_dataset = RiceGrainDataset(root_dir='./rice-image-dataset/Rice_Image_Dataset', transform=train_transforms)
# #using batch_size = 64 we can adjust this on training memory resource
BATCH_SIZE = 64

print(f"Total Images Found: {len(full_dataset)}")
print(f"Class Mapping: {full_dataset.class_to_idx}")

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

val_subset.dataset.transform = val_transforms

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,  num_workers=2)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False,  drop_last=True, num_workers=2)


print(f"Training batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")

model = get_modified_model(num_classes=6).to(device)

#Hyperparams
LR = 0.001
NUM_EPOCH =20

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)


#Training
gc.collect()

# Release all unoccupied cached memory held by the CUDA caching allocator
if torch.cuda.is_available():
    torch.cuda.empty_cache()
trained_model, history = train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs=NUM_EPOCH,
    scheduler=scheduler,
    early_stopping_patience=7
)
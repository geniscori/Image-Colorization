import torch.optim.lr_scheduler as lr_scheduler

# Importem les funcions
from utils.split_data import split_data
from utils.create_result_folders import create_result_folders
from train import train
from validation import validate
from test import test

from torchvision import transforms

# Importem les clases
from models.models import *
from utils.convert_2_grayscale import Convert2Grayscale
import torch
import os

input_path = r'C:\Users\genis\Documents\uni\Tercer\SEMESTRE 2\Xarxes\Proyecto Nuestro\data\random_images'
output_path = 'data/random_images/split_images'

use_gpu = torch.cuda.is_available()
if not os.path.exists(output_path):
    split_data(input_path, output_path,.8,.2)

# Opcions
model = CNNColor()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=30)

train_path = r'C:\Users\genis\Documents\uni\Tercer\SEMESTRE 2\Xarxes\Proyecto Nuestro\data\random_images\split_images\train'
val_path = r'C:\Users\genis\Documents\uni\Tercer\SEMESTRE 2\Xarxes\Proyecto Nuestro\data\random_images\split_images\val'

save_images = True
best_losses = 1e10
epochs = 15

# Entrenament
train_transforms = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()])
train_imagefolder = Convert2Grayscale(train_path , train_transforms)
train_loader = torch.utils.data.DataLoader(train_imagefolder, batch_size=64, shuffle=True)

# Validació
val_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
val_imagefolder = Convert2Grayscale(val_path , val_transforms)
val_loader = torch.utils.data.DataLoader(val_imagefolder, batch_size=64, shuffle=False)

# Movem el model a la GPU
if use_gpu:
    criterion = criterion.cuda()
    model = model.cuda()

# Creem les carpetes de outputs
create_result_folders()

# Entrenem el model
losses = {"train": [], "val": []}
for epoch in range(epochs):

    # Fem l'entrenament i la validació
    train_loss = train(train_loader, model, criterion, optimizer, epoch, epochs)
    scheduler.step()
    losses["train"].append(train_loss)

    with torch.no_grad():
        val_loss = validate(val_loader, model, criterion, save_images, epoch, epochs)
        scheduler.step()
        losses["val"].append(val_loss)
    # Guardem a la carpeta checkpoints l'estat del model
    if val_loss < best_losses:
        best_losses = val_loss
        torch.save(model.state_dict(), 'checkpoints/model_random-epoch-{}-losses-{:.3f}.pth'.format(epoch + 1, val_loss))

# Entrenem el model
losses = {"train": [], "val": []}
for epoch in range(epochs):

    # Fem l'entrenament i la validació
    train_loss = train(train_loader, model, criterion, optimizer, epoch, epochs)
    scheduler.step()
    losses["train"].append(train_loss)

    with torch.no_grad():
        val_loss = validate(val_loader, model, criterion, save_images, epoch, epochs)
        scheduler.step()
        losses["val"].append(val_loss)
    # Guardem a la caprta checkpoints l'estat del model
    if val_loss < best_losses:
        best_losses = val_loss
        torch.save(model.state_dict(), 'checkpoints/model_random-epoch-{}-losses-{:.3f}.pth'.format(epoch + 1, val_loss))

# Guardem el model
path = "checkpoints/model_random.pt"
torch.save(model, path)

# Importar el model entrenat
'''
model_entrenat = CNNColor()

if use_gpu: 
    criterion = criterion.cuda()
    model_entrenat = model_entrenat.cuda()

model_entrenat.load_state_dict(torch.load('checkpoints/model_random-epoch-16-losses-0.003.pth'))
test_loss = test(val_loader, model_entrenat, criterion, save_images)
'''
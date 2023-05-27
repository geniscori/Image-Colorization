import torch.optim.lr_scheduler as lr_scheduler

# Importem les funcions
from utils.split_data import split_data
from utils.create_result_folders import create_result_folders
from train import train
from validation import validate
from test import test

from torchvision import transforms

# Importem les classes
from models.models import *
from utils.convert_2_grayscale import Convert2Grayscale
import torch
import os

# Busquem la carpeta data
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'data')

# Obtenim un llistat dels directoris de data
subdirectories = [name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))]

# Seleccionem el directori amb les imatges
for folder_name in subdirectories:
    if folder_name != 'split_images':
        images_folder = folder_name # si hi ha diferents datasets a data, canviar folder_name pel nom de la carpeta amb les imatges desitjades
        break
else:
    print("No s'ha trobat cap directori amb imatges dins de data")
    exit()

# Construïm els diferents paths amb les imatges
input_path = os.path.join(data_dir, images_folder)
output_path = os.path.join(data_dir, 'split_images')

use_gpu = torch.cuda.is_available()
if not os.path.exists(output_path):
    split_data(input_path, output_path,.8,.2)

# Opcions
model = CNNColor()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=30)

# Directoris del train, el validation i on es guardarà el model entrenat
split_images_folder = os.path.join(data_dir, images_folder, 'split_images')
train_path = os.path.join(split_images_folder, 'train')
val_path = os.path.join(split_images_folder, 'val')
model_saved_path = 'checkpoints/model_random.pt'

save_images = True
best_losses = 1e10
epochs = 15

# Entrenament
train_transforms = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()])
train_imagefolder = Convert2Grayscale(train_path, train_transforms)
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
    # Guardem a la carpeta checkpoints l'estat del model
    if val_loss < best_losses:
        best_losses = val_loss
        torch.save(model.state_dict(), 'checkpoints/model_random-epoch-{}-losses-{:.3f}.pth'.format(epoch + 1, val_loss))

# Guardem el model
torch.save(model, model_saved_path)

# Importar el model entrenat
'''
model_entrenat = CNNColor()

if use_gpu: 
    criterion = criterion.cuda()
    model_entrenat = model_entrenat.cuda()

model_entrenat.load_state_dict(torch.load('checkpoints/model_random-epoch-16-losses-0.003.pth'))
test_loss = test(val_loader, model_entrenat, criterion, save_images)
'''
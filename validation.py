from utils.meters import Meters
from utils.to_rgb import to_rgb
import time
import torch

def validate(val_loader, model, criterion, save_images, epoch, epochs):
    model.eval()
    # Inicialitzem els valors per a les mètriques
    batch_time = Meters()
    data_time = Meters()
    losses = Meters()
    end = time.time()
    already_saved_images = False

    for i, (input_gray, input_ab, target) in enumerate(val_loader):
        # Temps de registre per carregar dades
        data_time.update(time.time() - end)

        # Utilitzem GPU si esta habilitada
        if torch.cuda.is_available():
            input_gray = input_gray.cuda()
            input_ab = input_ab.cuda()
            target = target.cuda()

        # Apliquem model i calculem la loss
        output_ab = model(input_gray)
        loss = criterion(output_ab, input_ab)
        losses.update(loss.item(), input_gray.size(0))

        # Guardem imatges
        if save_images and not already_saved_images:
            already_saved_images = True
            for j in range(min(len(output_ab), 10)):
                save_path = {'grayscale': 'outputs/gray/', 'colorized': 'outputs/color/'}
                save_name = 'imatge{}_epoch{}.jpg'.format(i * val_loader.batch_size + j, epoch + 1)
                to_rgb(input_gray[j].cpu(), ab_input=output_ab[j].detach().cpu(), save_path=save_path,
                       save_name=save_name)

        # Mirem el temps que triga
        batch_time.update(time.time() - end)
        end = time.time()

        # Mostrem resultats
        if i % 25 == 0:
            print("Validate : {}/{}, Val loss = {loss.avg:.4f}, Time = {batch_time.avg:.3f}".format(epoch + 1, epochs,
                                                                                                    loss=losses,
                                                                                                    batch_time=batch_time))

    return losses.avg
from utils.meters import Meters
import time
import torch


def train(train_loader, model, criterion, optimizer, epoch, epochs):
    model.train()
    # Inicialitzem els valors per a les metriques
    batch_time = Meters()
    data_time = Meters()
    losses = Meters()
    end = time.time()

    for i, (input_gray, input_ab, target) in enumerate(train_loader):
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

        # Calculem l'optimizer i gradient
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Mirem el temps que triga
        batch_time.update(time.time() - end)
        end = time.time()

        # Mostrem resultats
        if i % 25 == 0:
            print("Epoch : {}/{}, Train loss = {loss.avg:.4f}, Time = {batch_time.avg:.3f}".format(epoch + 1, epochs,
                                                                                                   loss=losses,
                                                                                                   batch_time=batch_time))

    return losses.avg
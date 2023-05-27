from skimage.color import rgb2lab, rgb2gray
from torchvision import datasets
import numpy as np
import torch


# Funció que converteix les imatges de color a blanc i negre
class Convert2Grayscale(datasets.ImageFolder):

    def __getitem__(self, i):
        # Obtenim imatge i carreguem
        path, target = self.imgs[i]
        imgage = self.loader(path)

        if self.transform is not None:
            original = self.transform(imgage)

            # Passem a np
            original = np.asarray(original)

            # Passem a escala LAB i normalitzem
            imgageLAB = rgb2lab(original)
            imgageLAB = (imgageLAB + 128) / 255
            img = imgageLAB[:, :, 1:3]

            # Transposem i ho passem a  un tensor
            img = torch.from_numpy(img.transpose((2, 0, 1))).float()

            # Passem a escala grisos (lluminositat)
            original = rgb2gray(original)
            original = torch.from_numpy(original).unsqueeze(0).float()  # Arreglem dimensionalitat

        if self.target_transform is not None:
            target = self.target_transform(target)

        return original, img, target
import numpy as np
from torchvision import models
from torchvision import transforms
import torch
from PIL import Image

dir(models)

#loading the model
alexnet = models.alexnet(pretrained=True)
print(alexnet)

#transforming
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

#import image
#img = Image.open("cat.jpg")
#img.show()
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

#import image and pre-processing
img = Image.open("cat.jpg")
#img.show()

img_t = transform(img)
batch_t = torch.unsqueeze(img_t, 0)

alexnet.eval()

out = alexnet(batch_t)
print(out.shape)

with open('imagenet_classes.txt') as f:
    classes = [line.strip() for line in f.readlines()]

_, index = torch.max(out,1)
percentage = torch.nn.functional.softmax(out, dim=1)[0]*100
print(classes[index[0]], percentage[index[0]].item())

_, indices = torch.sort(out, descending=True)
print([(classes[idx], percentage[idx].item()) for idx in indices[0][:5]]) 
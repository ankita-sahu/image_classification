import torch
from torchvision.models import inception_v3
from torchvision.transforms import transforms
from torchvision import datasets
import numpy as np
from torch.autograd import Variable
from PIL import Image
import argparse
import json
import time
from torchvision import models
from torchsummary import summary


with open('class_mapping.json', 'r') as f:
    num_class = json.load(f)

device = torch.device("cpu")
model_path = "./models/trained.model_inception"
checkpoint = torch.load(model_path)
model = inception_v3(num_classes=4)
model.load_state_dict(checkpoint, strict=False)
model.to(device)
model.eval()

data_transforms = {
    'predict': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    }

dataset = {'predict' : datasets.ImageFolder("./data", data_transforms['predict'])}
dataloader = {'predict': torch.utils.data.DataLoader(dataset['predict'], batch_size = 12, shuffle=False, num_workers=4)}

device = torch.device("cpu:0")
outputs = list()
since = time.time()
for inputs, labels in dataloader['predict']:
    inputs = inputs.to(device)
    output = model(inputs)
    output = output.to(torch.device('cpu'))
    index = output.data.numpy().argmax()
    print (index)
print (time.time()-since)

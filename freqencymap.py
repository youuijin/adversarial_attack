import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms, utils
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
from PIL import Image
import json

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=0., std=1.)
])

image = Image.open(str('./test_ikmg/0_original_True.jpg'))
plt.imshow(image)

image2 = Image.open(str('./test_ikmg/0_noFRLoss_succ.jpg'))
plt.imshow(image)

#model = models.resnet18(pretrained=True)
model = torch.load('./pretrained/resnet50_10way_112.pt')
print(model)

# we will save the conv layer weights in this list
model_weights =[]
#we will save the 49 conv layers in this list
conv_layers = []
# get all the model children as list
model_children = list(model.children())
#counter to keep count of the conv layers
counter = 0
#append all the conv layers and their respective wights to the list


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

image = transform(image)
print(f"Image shape before: {image.shape}")
image = image.unsqueeze(0)
print(f"Image shape after: {image.shape}")
image = image.to(device)

image2 = transform(image2)
print(f"image2 shape before: {image2.shape}")
image2 = image2.unsqueeze(0)
print(f"image2 shape after: {image2.shape}")
image2 = image2.to(device)

outputs = []
x = image
x2 = image2
# for i in range(len(model_children)-1):
#     if type(model_children[i]) == nn.Sequential:
#         for j in range(len(model_children[i])):
#             x = model_children[i][j](x)
#             x2 = model_children[i][j](x2)
#             for child in model_children[i][j].children():
#                 if type(child) == nn.Conv2d:
#                     counter+=1
#                     model_weights.append(child.weight)
#                     conv_layers.append(child)
#                     outputs.append(torch.abs(x-x2))

#         continue
    
#     x = model_children[i](x)
#     x2 = model_children[i](x2)
#     if type(model_children[i]) == nn.Conv2d:
#         counter+=1
#         model_weights.append(model_children[i].weight)
#         conv_layers.append(model_children[i])
#         outputs.append(torch.abs(x-x2))

names = []
        
for i in range(len(model_children)-1):
    
    if type(model_children[i]) == nn.Sequential:
        for j in range(len(model_children[i])):
            x = model_children[i][j](x)
            x2 = model_children[i][j](x2)
            for child in model_children[i][j].children():
                #if type(child) == nn.Conv2d:
                counter+=1
                # model_weights.append(child.weight)
                conv_layers.append(child)
                names.append(str(child)[:10])
                outputs.append(x)
                outputs.append(x2)
                outputs.append(torch.abs(x-x2))

        continue
    
    x = model_children[i](x)
    x2 = model_children[i](x2)
    # if type(model_children[i]) == nn.Conv2d:
    counter+=1
    # model_weights.append(model_children[i].weight)
    conv_layers.append(model_children[i])
    names.append(str(model_children[i])[:10])
    outputs.append(x)
    outputs.append(x2)
    outputs.append(torch.abs(x-x2))


# outputs = []
# names = []
# for layer in conv_layers[0:]:
#     image = layer(image)
#     outputs.append(image)
#     names.append(str(layer))
# print(len(outputs))
# #print feature_maps
# for feature_map in outputs:
#     print(feature_map.shape)


processed = []
for feature_map in outputs[-6:-3]:
    feature_map = feature_map.squeeze(0)
    gray_scale = torch.sum(feature_map,0)
    gray_scale = gray_scale / feature_map.shape[0]
    for features in feature_map[:64]:
        processed.append(features.data.cpu().numpy())
    # processed.append(gray_scale.data.cpu().numpy())
for fm in processed:
    print(fm.shape)

# fig = plt.figure(figsize=(11, 500))

fig = plt.figure(figsize=(500, 11))
# for i in range(len(processed)):
print(len(processed))
for i in range(len(processed)):
    a = fig.add_subplot(3, 64, i+1)
    imgplot = plt.imshow(processed[i], vmin=0, vmax=1)
    a.axis("off")
    a.set_title(f"{names[i//3]}_{i//3}", fontsize=10)
plt.savefig(str('feature_maps.jpg'), bbox_inches='tight')
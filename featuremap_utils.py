import torch
import torch.nn as nn
from torchvision import models, transforms, utils
from torch.autograd import Variable
import umap
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
import glob
import os
from learner import Learner
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

import pickle

def hook_fn(module, input, output):
    module.output = output

def addHook(model, hook=hook_fn):
    for layer in model.children():
        if len(list(layer.children())) > 0: # if module has children -> sequential
            addHook(layer, hook)
        else:
            layer.register_forward_hook(hook)

def collectOutput(model, output_list):
    for layer in model.children():
        if len(list(layer.children())) > 0: # if module has children -> sequential
            collectOutput(layer, output_list)
        
        elif type(layer) == nn.ParameterList:
            return

        else:
            output = layer.output
            output_list.append(output.cpu())

def getName(model, name_list):
    for layer in model.children():
        if len(list(layer.children())) > 0: # if module has children -> sequential
            getName(layer, name_list)

        else:
            layer_name = str(layer)[:str(layer).index("(")] + str(len(list(filter(lambda x:str(layer)[:str(layer).index("(")] in x, name_list))))
            name_list.append(layer_name)

def loadImage(path, imgsz, device, is_pickle = False):
    transform = transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0, 0, 0,), std=(1, 1, 1))
    ])
    if is_pickle:
        with open(path, "rb") as f:
            image = pickle.load(f)
    else:
        image = Image.open(path)
        image = transform(image).to(device)
        # print(image)
        # print(type(image))
        # print(len(image))

    return image


# Inference && get output
def inference(model, image_paths, image_names, device, args):
    format_len = len(args.format)+1
    errorses = []
    vectors = []
    vectorses = []
    labels = []
    #inf_labels = [] # inference lables
    names = []

    getName(model, names)
    # names = names[:-1] # for excepting linear

    with torch.no_grad():
        model.eval()
        for i in range(len(image_paths) // 2):
            attack_output_list = []
            origin_output_list = []
            outputs = []

            x = loadImage(image_paths[2 * i], args.imgsz, device, is_pickle=(args.format=="pickle")).unsqueeze(0)
            model(x)
            # p = torch.nn.functional.softmax(logit, dim=1)
            # c = torch.argmax(p, dim=1)
            collectOutput(model, attack_output_list)

            x2 = loadImage(image_paths[2 * i + 1], args.imgsz, device, is_pickle=(args.format=="pickle")).unsqueeze(0)
            model(x2)
            # p = torch.nn.functional.softmax(logit, dim=1)
            # o = torch.argmax(p, dim=1)

            collectOutput(model, origin_output_list)

            errors = calculateErrors(args.error_metric, attack_output_list, origin_output_list)

            errorses.append(errors)

            # attack_feature = attack_output_list[-2]
            # origin_feature = origin_output_list[-2]

            # print(attack_feature)
            # print(origin_feature)

            # vectors.append(attack_feature.view(-1))
            # vectors.append(origin_feature.view(-1))

            vectorses.append(attack_output_list)
            vectorses.append(origin_output_list)

            # inf_labels.append(c.item())
            # inf_labels.append(o.item())


            labels.append(int(image_names[2 * i].split("_")[-1][:-format_len]))
            labels.append(int(image_names[2 * i + 1].split("_")[-1][:-format_len]))

            for attack_output, origin_output, error in zip(attack_output_list, origin_output_list, errors):
                outputs.append(attack_output)
                outputs.append(origin_output)
                outputs.append(error)

    return errorses, names, vectorses, labels

    # print(inf_labels)
    # print(labels)

    # inf_attack_label = torch.tensor([inf_labels[i * 2] for i in range(len(inf_labels)//2)])
    # f_attack_label = [inf_labels[i * 2 + 1] for i in range(len(inf_labels)//2)]

    # cnt = 0
    # adv_cnt = 0

    # for i in range(len(inf_labels)//2):
    #     cnt += int(labels[2*i+1]==inf_labels[2*i+1])
    #     adv_cnt += int(labels[2*i]==inf_labels[2*i])

    # print(f"{format} acc :", cnt*2/len(inf_labels))
    # print(f"{format} adv acc :", adv_cnt*2/len(inf_labels))

        
def calculateErrors(type, attack_output_list, origin_output_list):
    # if type=="cos":
    #     cos = torch.nn.CosineSimilarity(dim=0)
    #     errors = []
    #     for attack_feature_map, origin_feature_map in zip(attack_output_list, origin_output_list):
    #         if len(attack_feature_map.shape)==4:
    #             print(attack_feature_map.shape)
    #             b, c, h, w = attack_feature_map.shape
    #             print(attack_feature_map.view(b, c,-1).shape)
    #             #errors.append(1-cos(torch.mean(attack_feature_map, dim=(2,3)).view(-1), torch.mean(origin_feature_map, dim=(2,3)).view(-1)).cpu())
    #             errors.append(1-cos(attack_feature_map.view(b, c,-1), ).cpu())
    #         else:
    #             errors.append(1-cos(attack_feature_map.view(-1), origin_feature_map.view(-1)).cpu())

    if type=="L1":
        errors = [torch.abs(attack_feature_map - origin_feature_map).cpu() for attack_feature_map, origin_feature_map in zip(attack_output_list, origin_output_list)]
    elif type=="L2":
        errors = []
        for attack_feature_map, origin_feature_map in zip(attack_output_list, origin_output_list):
            if len(attack_feature_map.shape) == 4:
                errors.append(torch.norm(attack_feature_map - origin_feature_map, dim=(2,3)).cpu())
            else:
                errors.append(torch.norm(attack_feature_map - origin_feature_map, dim=1).cpu())

        #print(errors[0].shape)
    return errors


# def makeFeatureMap():
#     # Make Feature Maps
#     LAYER = 10
#     FEATURE_MAP_NUM = 64

#     processed = []
#     for feature_map in outputs[LAYER * 3:(LAYER + 1) * 3]:
#         feature_map = feature_map.squeeze(0)
#         gray_scale = torch.sum(feature_map,0)
#         gray_scale = gray_scale / feature_map.shape[0]
#         for features in feature_map[:FEATURE_MAP_NUM]:
#             processed.append(features.data.cpu().numpy())
#         # processed.append(gray_scale.data.cpu().numpy())
#     # fig = plt.figure(figsize=(11, 500))

#     fig = plt.figure(figsize=(200, 11))


#     print(len(processed))
#     for i in range(len(processed)):
#         a = fig.add_subplot(3, FEATURE_MAP_NUM, i+1)
#         imgplot = plt.imshow(processed[i], vmin=0, vmax=1)
#         a.axis("off")
#         # a.set_title(f"{names[i//3]}_{i//3}", fontsize=10)
#     plt.savefig(f'feature_maps_{PATH}_conv_{LAYER}.jpg', bbox_inches='tight')


def makeErrorGraph(errorses, names, args):
    # Make Error Graph
    mean_errorses = []
    
    for errors in errorses:
        mean_errors = []
        for error in errors:
            if len(error.shape) == 4:
                _, c, h, w = error.shape
                pixel_num = c * h * w
                sum_shape = (1, 2, 3)
            
            elif len(error.shape) == 2:
                a, b = error.shape
                pixel_num = a * b
                sum_shape = (1)

            elif len(error.shape) == 1:
                pixel_num = 1
                sum_shape = (0)

            error_sum = torch.sum(error, sum_shape)
            error_mean = torch.abs((error_sum / pixel_num)).item()
            mean_errors.append(error_mean)
        mean_errorses.append(mean_errors)

    mean_errorses_tensor = torch.mean(torch.tensor(mean_errorses), dim=0)

    plt.figure(figsize=(50, 6))  # Width: 10 inches, Height: 6 inches

    # Create the plot
    plt.plot(names, mean_errorses_tensor)
    plt.xticks(fontsize=5)

    plt.savefig(f'./AttackedImages/{args.format}/{args.attack}/eps_{args.eps}_{args.error_metric}_error.png')


## Make Umap
def makeUmap(args, vectorses, labels, names):
 
    if args.umap_metric=="cos":
        metric = "cosine"
    else:
        metric = "euclidean"
    embeddings = []
    reducer = umap.UMAP(n_components=args.umap_dim, n_neighbors=10, min_dist=0.5, metric=metric, random_state=50)
    for features in tqdm(zip(*vectorses)):
        mean_features = []
        for feature in features:
            if len(feature.shape)==4:
                mean_features.append(torch.mean(feature, dim=(2,3)).view(-1))
            else:
                mean_features.append(feature.view(-1))

        features = torch.stack([i.cpu() for i in mean_features])
        embedding = reducer.fit_transform(features)
        embeddings.append(embedding)

    ###
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    markers = ['o', 's', '^', 'x', 'D']

    if args.umap_dim==2:
        os.makedirs(f'./visualization/umap/{args.format}/{args.umap_metric}/2d/', exist_ok=True)
        for index, embedding in enumerate(embeddings):
            fig = plt.figure()
            for i in range(embedding.shape[0]//2):
                plt.scatter(embedding[2 * i, 0], embedding[2 * i, 1], marker="^", color=colors[labels[2 * i]], s=20) # attack
                plt.scatter(embedding[2 * i + 1, 0], embedding[2 * i + 1, 1], marker="o", color=colors[labels[2 * i + 1]], s=20) # origin
            
            plt.title(names[index])        
            plt.savefig(f'./visualization/umap/{args.format}/{args.umap_metric}/2d/layer_{index}.png')

    elif args.umap_dim==3:
        os.makedirs(f'./visualization/umap/{args.format}/{args.umap_metric}/3d/', exist_ok=True)
        for index, embedding in enumerate(embeddings):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            for i in range(embedding.shape[0]//2):
                # print(labels[i])
                ax.scatter(embedding[2 * i, 0], embedding[2 * i, 1], embedding[2 * i, 2], marker="^", color=colors[labels[2 * i]], s=20) # attack
                ax.scatter(embedding[2 * i + 1, 0], embedding[2 * i + 1, 1], embedding[2 * i + 1, 2], marker="o", color=colors[labels[2 * i + 1]], s=20) # origin
            
            plt.title(names[index])        
            plt.savefig(f'./visualization/umap/{args.format}/{args.umap_metric}/3d/layer_{index}.png')

def argsCheck(args):
    if args.format not in ["jpg", "png", "pickle"]:
        print("available format : jpg, png, pickle")
        exit()
    if args.error_metric not in ["L1", "L2"]:
        print("available error metric : L1, L2")
        exit()
    if args.umap:
        if args.umap_dim not in [2, 3]:
            print("available umap dimension : 2, 3")
            exit()
        if args.umap_metric not in ["cos", "euc"]:
            print("available umap metric : cosine similarity(cos), euclidean(euc)")
            exit()
    if not os.path.exists(f"AttackedImages\\{args.format}\\{args.attack}\\eps_{args.eps}"):
        print("there is no data. check format, attack, eps")
        exit()
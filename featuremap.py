import torch
import glob, os
import argparse
from torchvision import transforms
from featuremap_utils import *

def main(args):
    device = torch.device("cuda:0")

    ## load model
    model = torch.load(args.pretrained)
    model = model.to(device)

    ## model add hook
    addHook(model)

    PATH = f"AttackedImages\\{args.format}\\{args.attack}\\eps_{args.eps}"
    image_paths = glob.glob(f".\\{PATH}\\*", )
    image_names = [os.path.basename(i) for i in image_paths]
    errorses, names, vectorses, labels = inference(model, image_paths, image_names, device, args)

    # if args.feature_map:
    #     makeFeatureMap()
    if args.error_graph:
        makeErrorGraph(errorses, names, args)
    if args.umap:
        makeUmap(args, vectorses, labels, names)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=112)
    argparser.add_argument('--pretrained', type=str, help='pretrained model path', default="")
    argparser.add_argument('--format', type=str, help='jpg, png, pickle', default="png")
    argparser.add_argument('--attack', type=str, help='adversarial attack', default="PGD_Linf")
    argparser.add_argument('--eps', type=float, help='attack boundary', default=6)
    
    # feature map options
    argparser.add_argument('--feature_map', action='store_true', help='make ferature map', default=False)

    # error graph options
    argparser.add_argument('--error_graph', action='store_true', help='make error graph', default=False)
    argparser.add_argument('--error_metric', type=str, help='L1norm(L1), L2norm(L2)', default="L1")

    # umap visualization options
    argparser.add_argument('--umap', action='store_true', help='make umap', default=False)
    argparser.add_argument('--umap_dim', type=int, help='2d or 3d', default=2)
    argparser.add_argument('--umap_metric', type=str, help='cosine similarity(cos), euclidean(euc)', default="cos")

    args = argparser.parse_args()

    argsCheck(args)

    main(args)
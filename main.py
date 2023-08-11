import  torch, os
import  numpy as np
from    Imagenet import Imagenet
from    torch.utils.data import DataLoader
from torch import optim
import  random, sys, pickle
import  argparse
from tqdm import tqdm

import time

from torchvision import models, transforms
import matplotlib.pyplot as plt
import advertorch.attacks as attacks
 
from torch.utils.tensorboard import SummaryWriter

from learner import Learner

import utils

def main(args):
    seed = 706
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda:'+str(args.device_num))

    # pretrained model 불러오기 (not meta learning) -> fine-tuning
    

    if args.pretrained=="":
        if args.resnet==0:
            # pretrained model
            s = (args.imgsz-2)//2
            s = (s-2)//2
            s = s-3

            config = [
                ('conv2d', [32, 3, 3, 3, 1, 0]),
                ('relu', [True]),
                ('bn', [32]),
                ('max_pool2d', [2, 2, 0]),
                ('conv2d', [32, 32, 3, 3, 1, 0]),
                ('relu', [True]),
                ('bn', [32]),
                ('max_pool2d', [2, 2, 0]),
                ('conv2d', [32, 32, 3, 3, 1, 0]),
                ('relu', [True]),
                ('bn', [32]),
                ('max_pool2d', [2, 1, 0]),
                ('flatten', []),
                ('linear', [args.n_way, 32 * s * s])
            ]

            model = Learner(config, args.imgc, args.imgsz)
            model = model.to(device)

        else:
            model = models.resnet18(weights='IMAGENET1K_V1')
            num_ftrs = model.fc.in_features
            model.fc = torch.nn.Linear(num_ftrs, args.n_way)
            model.conv1 = torch.nn.Conv2d(args.imgc, 64, kernel_size=7, stride=2, padding=3, bias=False)

            model = model.to(device)

        train_data = Imagenet('../../dataset', mode='train', n_way=args.n_way, resize=args.imgsz, color=args.imgc)
        val_data = Imagenet('../../dataset', mode='val', n_way=args.n_way, resize=args.imgsz, color=args.imgc)
        test_data = Imagenet('../../dataset', mode='test', n_way=args.n_way, resize=args.imgsz, color=args.imgc)
        optim = torch.optim.Adam(model.parameters(), lr=0.00001)
        for epoch in range(args.epoch):
            db = DataLoader(train_data, args.task_num, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
            correct_count=0
            model.train()
            for _, (x, y) in enumerate(db):
                x = x.to(device)
                y = y.to(device)
                logit = model(x)
                pred = torch.nn.functional.softmax(logit, dim=1)
                outputs = torch.argmax(pred, dim=1)
                correct_count += (outputs == y).sum().item()
                loss = torch.nn.functional.cross_entropy(logit, y)
                optim.zero_grad()
                loss.backward()
                optim.step()
            db_val = DataLoader(val_data, args.task_num, shuffle=True, num_workers=0, pin_memory=True)
            model.eval()
            val_correct_count=0
            for _, (x, y) in enumerate(db_val):
                x = x.to(device)
                y = y.to(device)
                
                pred = torch.nn.functional.softmax(model(x), dim=1)
                outputs = torch.argmax(pred, dim=1)
                val_correct_count += (outputs == y).sum().item()
            print("epoch: ", epoch, "\ttraining acc: ", round(correct_count/(480*args.n_way)*100, 2), "\tval acc: ", round(val_correct_count/(60*args.n_way)*100, 2))
        
        db_test = DataLoader(test_data, args.task_num, shuffle=True, num_workers=0, pin_memory=True)
        test_correct_count=0
        for _, (x, y) in enumerate(db_test):
            x = x.to(device)
            y = y.to(device)
            
            pred = torch.nn.functional.softmax(model(x), dim=1)
            outputs = torch.argmax(pred, dim=1)
            test_correct_count += (outputs == y).sum().item()
        print("\ntest acc: ", round(test_correct_count/(60*args.n_way)*100, 2))

        if args.resnet==0:
            # torch.save(model.parameters(), f'./pretrained/conv3_{args.n_way}way_{args.imgsz}.pt')
            torch.save(model.state_dict(), f'./pretrained/conv3_{args.n_way}way_{args.imgsz}.pt')
        else:
            torch.save(model, f'./pretrained/resnet{args.resnet}_{args.n_way}way_{args.imgsz}.pt')
        print(model)
    else:
        if args.resnet==0:
            # pretrained model
            s = (args.imgsz-2)//2
            s = (s-2)//2
            s = s-3
            config = [
                ('conv2d', [32, 3, 3, 3, 1, 0]),
                ('relu', [True]),
                ('bn', [32]),
                ('max_pool2d', [2, 2, 0]),
                ('conv2d', [32, 32, 3, 3, 1, 0]),
                ('relu', [True]),
                ('bn', [32]),
                ('max_pool2d', [2, 2, 0]),
                ('conv2d', [32, 32, 3, 3, 1, 0]),
                ('relu', [True]),
                ('bn', [32]),
                ('max_pool2d', [2, 1, 0]),
                ('flatten', []),
                ('linear', [args.n_way, 32 * s * s])
            ]
            # vars = torch.load(args.pretrained).to(device)
            
            model = Learner(config, args.imgc, args.imgsz)
            model = model.to(device)
            model.load_state_dict(torch.load(args.pretrained))
            # model.set_parameters(vars)
        else:
            model = torch.load(args.pretrained).to(device)
        utils.save_attacked_img(model, device, args)

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    # Meta-learning options
    argparser.add_argument('--n_way', type=int, help='n way', default=5)

    # Dataset options
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=28)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    
    # Training options
    argparser.add_argument('--epoch', type=int, help='epoch number', default=30)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=32)
    argparser.add_argument('--device_num', type=int, help='what gpu to use', default=0)

    # adversarial attack options
    argparser.add_argument('--attack', type=str, default="aRUB")
    argparser.add_argument('--test_attack', type=str, default="PGD_Linf")
    argparser.add_argument('--eps', type=float, help='attack-eps', default=2)
    argparser.add_argument('--rho', type=float, help='aRUB-rho', default=2)
    argparser.add_argument('--iter', type=int, default=10)

    argparser.add_argument('--pretrained', type=str, help='path of pretrained model', default="")
    argparser.add_argument('--resnet', type=int, default=0)

    args = argparser.parse_args()

    main(args)
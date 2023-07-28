import  torch, os
import  numpy as np
from    Imagenet import Imagenet
import  scipy.stats
from    torch.utils.data import DataLoader
from torch import optim
import  random, sys, pickle
import  argparse
from tqdm import tqdm
from PIL import Image

import time

from torchvision import models, transforms

import advertorch.attacks as attacks
 
from torch.utils.tensorboard import SummaryWriter

def main(args):
    seed = 227
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device('cuda:'+str(args.device_num))

    # load pretrained model
    model = torch.load(args.pretrained).to(device)
    model.eval()

    # save attacked image
    eps = args.eps
    at = setAttack(args.attack, model, args, eps)

    test_data = Imagenet('../../dataset', mode='image', n_way=args.n_way, resize=args.imgsz, color=args.imgc)
    transform = transforms.ToPILImage()
    db_test = DataLoader(test_data, 1, shuffle=True, num_workers=0, pin_memory=True) # 하나만 가져오기
    save_path = ".\\FRLossImages\\"+args.attack+"\\"+"eps"+str(args.eps)+"_r"+str(args.r)+"_alpha"+str(args.alpha)+"\\"
    os.makedirs(save_path, exist_ok=True)

    f = open(save_path+"log.txt", 'w')
    
    for step, (x, y) in enumerate(db_test):
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            p = torch.nn.functional.softmax(model(x), dim=1)
            o = torch.argmax(p, dim=1)
            if args.imgc==3:
                img = transform((x.squeeze()*(-255)+255).cpu())
            else:
                img = transform((x.squeeze()).cpu())
            img.save(save_path+"original.jpg")
            print("original:", y.item()==o.item())


        advX = at.perturb(x, y, advFR=True, r=args.r, alpha=args.alpha)
        if args.imgc==3:
                img = transform((advX.squeeze()*(-255)+255).cpu())
        else:
            img = transform((advX.squeeze()).cpu())

        if torch.argmax(torch.nn.functional.softmax(model(advX), dim=1), dim=1).item()==y.item():
            res = "fail"
        else:
            res = "succ"
        img.save(save_path+"FRLoss_"+res+".jpg")
        
        advX = at.perturb(x, y, advFR=False) 
        if args.imgc==3:
                img = transform((advX.squeeze()*(-255)+255).cpu())
        else:
            img = transform((advX.squeeze()).cpu())

        if torch.argmax(torch.nn.functional.softmax(model(advX), dim=1), dim=1).item()==y.item():
            res = "fail"
        else:
            res = "succ"
        img.save(save_path+"noFRLoss_"+res+".jpg")
    f.close()

    # 그래프 (시간, 성공률) - 점
    # 그래프 (이미지 크기, 성공률) - 그래프 

def model_acc(model, device, advFR):

    test_data = Imagenet('../../dataset', mode='test', n_way=args.n_way, resize=args.imgsz, color=args.imgc)
    db_test = DataLoader(test_data, args.task_num, shuffle=True, num_workers=0, pin_memory=True) # 600

    at = setAttack(args.test_attack, model, args, 0.03)
    correct_num = 0
    correct_adv_num = 0
    loss = 0
    loss_adv = 0
    for _, (x, y) in enumerate(db_test):
        x = x.to(device)
        y = y.to(device)
        advX = at.perturb(x, y)
        with torch.no_grad():
            logit = model(x)
            p = torch.nn.functional.softmax(logit, dim=1)
            o = torch.argmax(p, dim=1)
            correct_num += torch.where(y == o, torch.tensor(1).to(device), torch.tensor(0).to(device)).sum()
            loss += torch.nn.functional.cross_entropy(logit, y, reduction='sum')
            
            logit_adv = model(advX)
            p_adv = torch.nn.functional.softmax(logit_adv, dim=1)
            o_adv = torch.argmax(p_adv, dim=1)
            correct_adv_num += torch.where(y == o_adv, torch.tensor(1).to(device), torch.tensor(0).to(device)).sum()
            loss_adv += torch.nn.functional.cross_entropy(logit_adv, y, reduction='sum')
    acc = correct_num/600
    adv_acc = correct_adv_num/600
    return acc.item(), adv_acc.item(), (loss/600).item(), (loss_adv/600).item()


def setAttack(str_at, net, args, e):
    # TODO FR LOSS 추가!!!
    if str_at == "PGD_L1":
        return attacks.L1PGDAttack(net, eps=e, nb_iter=10)
    elif str_at == "PGD_L2":
        return attacks.L2PGDAttack(net, eps=e, nb_iter=10)
    elif str_at == "PGD_Linf":
        return attacks.LinfPGDAttack(net, eps=e, nb_iter=10)
    elif str_at == "FGSM":
        return attacks.GradientSignAttack(net, eps=e)
    elif str_at == "BIM_L2":
        return attacks.L2BasicIterativeAttack(net, eps=e, nb_iter=10)
    elif str_at == "BIM_Linf":
        return attacks.LinfBasicIterativeAttack(net, eps=e, nb_iter=10)
    elif str_at == "MI-FGSM":
        return attacks.MomentumIterativeAttack(net, eps=e, nb_iter=10) # 0.3, 40
    elif str_at == "CnW":
        return attacks.CarliniWagnerL2Attack(net, args.n_way)
    elif str_at == "EAD":
        return attacks.ElasticNetL1Attack(net, args.n_way)
    elif str_at == "DDN":
        return attacks.DDNL2Attack(net, nb_iter=10)
    elif str_at == "Single_pixel":
        return attacks.SinglePixelAttack(net)
    elif str_at == "DeepFool":
        return attacks.DeepfoolLinfAttack(net, args.n_way, eps=e)
    else:
        print("wrong type Attack")
        exit()

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    # Meta-learning options
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)

    # Dataset options
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=84)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    
    # Training options
    argparser.add_argument('--epoch', type=int, help='epoch number', default=30)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=32)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=0.001)
    argparser.add_argument('--adv_lr', type=float, help='adv-level learning rate', default=0.0002)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--device_num', type=int, help='what gpu to use', default=0)

    # adversarial attack options
    argparser.add_argument('--attack', type=str, default="DDN")
    argparser.add_argument('--test_attack', type=str, default="PGD_Linf")
    argparser.add_argument('--eps', type=float, help='attack-eps', default=0.03)
    argparser.add_argument('--rho', type=float, help='aRUB-rho', default=0.01)

    # adversarial FRLoss options
    argparser.add_argument('--pretrained', type=str, help="pretrained model path", default="")
    argparser.add_argument('--r', type=float, help="regularizer in frequency domain", default=2)
    argparser.add_argument('--alpha', type=float, help="weight between CE Loss and FR Loss", default=0.3)

    args = argparser.parse_args()

    main(args)
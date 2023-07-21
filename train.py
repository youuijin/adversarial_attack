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
from aRUBattack import aRUB
 
from torch.utils.tensorboard import SummaryWriter

def main(args):
    seed = 227
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device('cuda:'+str(args.device_num))

    # pretrained model
    model = models.resnet18(weights='IMAGENET1K_V1')
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, args.n_way)
    #model = torch.load(args.pretrained).to(device)

    model = model.to(device)
    model.train()

    # robust training
    if args.train:

        alpha = 0.8

        if args.mode=="":
            print("set train mode")
            exit()
        # set training lossa
        advCE = (args.mode[0]=='1')
        advFR = (args.mode[1]=='1')

        writer = SummaryWriter("./runs/resnet18/"+args.mode+"/"+time.strftime('%m-%d-%H%M%S'), comment=args.mode) 
        train_data = Imagenet('../../dataset', mode='train', n_way=args.n_way, resize=args.imgsz)
        optim = torch.optim.Adam(model.parameters(), lr=0.00001)
        tot_step=0
        for epoch in range(args.epoch):
            db = DataLoader(train_data, args.task_num, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
            model.train()
            correct_count=0
            adv_count=0
            for step, (x, y) in enumerate(db):
                x = x.to(device)
                y = y.to(device)
                logit = model(x)
                pred = torch.nn.functional.softmax(logit, dim=1)
                outputs = torch.argmax(pred, dim=1)
                correct_count += (outputs == y).sum().item()
                loss = torch.nn.functional.cross_entropy(logit, y)
                if advCE:
                    at = setAttack(args.attack, model, args, e=args.eps, FR=advFR)
                    advX = at.perturb(x, y)
                    logit_adv = model(advX)
                    pred_adv = torch.nn.functional.softmax(logit_adv, dim=1)
                    outputs_adv = torch.argmax(pred_adv, dim=1)
                    adv_count += (outputs_adv == y).sum().item()
                    
                    loss = loss * alpha + torch.nn.functional.cross_entropy(logit_adv, y) * (1-alpha)
                if step%(600/args.task_num)==0:
                    print("tot_step: ", tot_step, "\tSA/train:", correct_count/((step+1)*args.task_num), "\tRA/train:", adv_count/((step+1)*args.task_num))
                    writer.add_scalar("SA/train", correct_count/((step+1)*args.task_num), tot_step)
                    writer.add_scalar("RA/train", adv_count/((step+1)*args.task_num), tot_step)
                    writer.add_scalar("loss/train", loss, tot_step)
                tot_step += args.task_num
                optim.zero_grad()
                loss.backward()
                optim.step()
            
            
            model.eval()
            acc, acc_adv, loss, loss_adv = model_acc(model, device, advFR)
            writer.add_scalar("SA/test", acc, epoch)
            writer.add_scalar("RA/test", acc_adv, epoch)
            writer.add_scalar("SL/test", loss, epoch)
            writer.add_scalar("RL/test", loss_adv, epoch)

            print("epoch: ", epoch, "\tSA:", acc, "\tRA:", acc_adv, "\tSL:", loss, "\tRL:", loss_adv)
    # only evaluation
    else:
        #model = torch.load(args.pretrained).to(device)
        model.eval()
        acc, acc_adv, loss, loss_adv = model_acc(model, device, "")
        print("SA:", acc, "\tRA:", acc_adv, "\tSL:", loss, "\tRL:", loss_adv)


    # 그래프 (시간, 성공률) - 점
    # 그래프 (이미지 크기, 성공률) - 그래프 

def model_acc(model, device, advFR):

    test_data = Imagenet('../../dataset', mode='test', n_way=args.n_way, resize=args.imgsz)
    db_test = DataLoader(test_data, args.task_num, shuffle=True, num_workers=0, pin_memory=True) # 600

    at = setAttack(args.test_attack, model, args, 0.03, advFR)
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


def setAttack(str_at, net, args, e, FR):
    # TODO FR LOSS 추가!!!
    if str_at == "PGD_L1":
        return attacks.L1PGDAttack(net, eps=e, nb_iter=10)
    elif str_at == "PGD_L2":
        return attacks.L2PGDAttack(net, eps=e, nb_iter=10)
    elif str_at == "PGD_Linf":
        return attacks.LinfPGDAttack(net, eps=e, nb_iter=10)
    elif str_at == "FGSM":
        return attacks.GradientSignAttack(net, eps=e)
    elif str_at == "FFA":
        return attacks.FastFeatureAttack(net, eps=e, nb_iter=10)
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
        return attacks.DDNL2Attack(net)
    elif str_at == "LBFGS":
        return attacks.LBFGSAttack(net, args.n_way)
    elif str_at == "Single_pixel":
        return attacks.SinglePixelAttack(net)
    elif str_at == "Local_search":
        return attacks.LocalSearchAttack(net)
    elif str_at == "ST":
        return attacks.SpatialTransformAttack(net, args.n_way)
    elif str_at == "JSMA":
        return attacks.JacobianSaliencyMapAttack(net, args.n_way)
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

    # adversarial defense training options
    argparser.add_argument('--pretrained', type=str, help="pretrained model path", default="")
    argparser.add_argument('--train', action="store_true")
    argparser.add_argument('--r', type=float, help="regularizer in frequency domain", default=1.5)
    argparser.add_argument('--mode', type=str, help="adv CE, adv FR", default="")

    args = argparser.parse_args()

    main(args)
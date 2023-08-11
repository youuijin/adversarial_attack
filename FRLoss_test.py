import  torch, os
import  numpy as np
from    Imagenet import Imagenet
import  scipy.stats
from    torch.utils.data import DataLoader
from torch import optim
import  random, sys, pickle
import  argparse

from torchvision import models, transforms
import utils

import advertorch.attacks as attacks
 
def main(args):
    seed = 227
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda:'+str(args.device_num))

    # load pretrained model
    model = torch.load(args.pretrained).to(device)
    model.eval()

    

    # # save attacked image
    # at = setAttack(args.attack, model, args)

    # test_data = Imagenet('../../dataset', mode='test', n_way=args.n_way, resize=args.imgsz, color=args.imgc)
    # transform = transforms.ToPILImage()
    # db_test = DataLoader(test_data, 1, shuffle=False, num_workers=0, pin_memory=True) # 하나만 가져오기
    # if args.imgc==1:
    #     color="gray"
    # else:
    #     color="RGB"
    # save_path = ".\\AttackedImages\\"+color+"\\"+args.attack+"\\"+"eps_"+str(args.eps)+"\\"
    # os.makedirs(save_path, exist_ok=True)
    
    # log_str = []
    # cnt = 0
    
    # for step, (x, y) in enumerate(db_test):
    #     print()
    #     print("Image", step)
    #     log_str.append("\nImage "+str(step))
    #     x = x.to(device)
    #     y = y.to(device)

    #     with torch.no_grad():
    #         p = torch.nn.functional.softmax(model(x), dim=1)
    #         o = torch.argmax(p, dim=1)
    #         if(y.item()==o.item()):
    #             cnt += 1
    #         else:
    #             continue
    #         img = transform((x.squeeze()).cpu())
    #         print("original:", y.item()==o.item())
    #         log_str.append("original: "+str(y.item()==o.item()))
    #         img.save(save_path+str(step)+"_original_"+str(y.item())+".jpg")
            
    #     advX, logs = at.perturb(x, y, log=True, advFR=False) 
    #     log_str += logs

    #     img = transform(advX.squeeze().cpu())
    #     img.save(save_path+str(step)+"_attacked.jpg")

        
    # f = open(save_path+"log.txt", 'w')
    # for st in log_str:
    #     f.write(st+"\n")
    # f.close()

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
    argparser.add_argument('--eps', type=float, help='attack-eps', default=6)
    argparser.add_argument('--rho', type=float, help='aRUB-rho', default=2)
    argparser.add_argument('--iter', type=int, help='number of iterations', default=10)

    # adversarial FRLoss options
    argparser.add_argument('--pretrained', type=str, help="pretrained model path", default="")
    argparser.add_argument('--r', type=float, help="regularizer in frequency domain", default=2)
    argparser.add_argument('--alpha', type=float, help="weight between CE Loss and FR Loss", default=0.3)

    args = argparser.parse_args()

    main(args)
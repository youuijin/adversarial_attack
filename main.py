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

import matplotlib.pyplot as plt


import advertorch.attacks as attacks
from aRUBattack import aRUB
 
from torch.utils.tensorboard import SummaryWriter

def main(args):
    #sum_str_path = "./success/"
    #sum_str = str(args.attack) + "/" + str(args.imgsz)

    seed = 227
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device('cuda:'+str(args.device_num))

    # pretrained model 불러오기 (not meta learning) -> fine-tuning
    model = models.resnet18(weights='IMAGENET1K_V1')
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, args.n_way)

    model = model.to(device)
    model.train()

    #summary(model, input_size=(3,28,28), batch_size=args.task_num, device="cuda")

    # meta learning model에 대해서도 진행 -> 낮은 정확도를 가지는 모델일 경우 attack이 잘 안되는가?
    if args.pretrained=="":
        # 이미지 불러오기
        train_data = Imagenet('../../dataset', mode='train', n_way=args.n_way, resize=args.imgsz)
        test_data = Imagenet('../../dataset', mode='test', n_way=args.n_way, resize=args.imgsz)
        optim = torch.optim.Adam(model.parameters(), lr=0.00005)
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
            db_test = DataLoader(test_data, args.task_num, shuffle=True, num_workers=0, pin_memory=True)
            model.eval()
            test_correct_count=0
            for _, (x, y) in enumerate(db_test):
                x = x.to(device)
                y = y.to(device)
                
                pred = torch.nn.functional.softmax(model(x), dim=1)
                outputs = torch.argmax(pred, dim=1)
                test_correct_count += (outputs == y).sum().item()
            print("epoch: ", epoch, "\ttraining acc: ", round(correct_count/2400*100, 2), "\ttest acc: ", round(test_correct_count/600*100, 2))
        torch.save(model, './pretrained/resnet18_'+str(args.imgsz)+'.pt')
    else:
        model = torch.load(args.pretrained).to(device)
        model.eval()
        #eval_model(model, device) # model 자체의 성능 평가 (이미지 크기 별)
        model_acc(model, device) # model의 SA와 RA 평가
        #attacks_success_rate(model, device) # attack 종류에 따른 성공률 측정 및 그래프 그리기
        #attacks_success_rate_eps(model, device) # attack 종류에 따른 성공률 그래프 (eps 변화)
        #attacks_success_rate_size(model, device) # attack 종류에 따른 성공률 그래프 (size)
        #save_attacked_pic(model, device) # attack을 받은 이미지에 저장
        #attacks_energy(model, device) # eps 변화에 따른 이미지 변형도 (MSE)
        #attacks_energy_logit(model, device) # imgsz 변화에 따른 logit 변형도 (MSE) -> 동일한 eps

    # 그래프 (시간, 성공률) - 점
    # 그래프 (이미지 크기, 성공률) - 그래프 

def model_acc(model, device):

    test_data = Imagenet('../../dataset', mode='test', n_way=args.n_way, resize=args.imgsz)
    db_test = DataLoader(test_data, args.task_num, shuffle=True, num_workers=0, pin_memory=True) # 600
    
    attack_list=["PGD_L1", "PGD_L2", "PGD_Linf", "FGSM", "BIM_L2", "BIM_Linf", "MI-FGSM", "CnW", "DDN", "EAD",  
                     "LBFGS", "Single_pixel", "Local_search", "ST"]

    model_name="resnet18"

    for attack_name in attack_list:
        writer = SummaryWriter("./acc/"+model_name+"/"+attack_name, comment=args.pretrained) 
        at = setAttack(attack_name, model, args, 0.004)
        correct_num = 0
        correct_adv_num = 0
        for _, (x, y) in enumerate(db_test):
            x = x.to(device)
            y = y.to(device)
            advX = at.perturb(x, y)
            with torch.no_grad():
                p = torch.nn.functional.softmax(model(x), dim=1)
                o = torch.argmax(p, dim=1)
                correct_num += torch.where(y == o, torch.tensor(1).to(device), torch.tensor(0).to(device)).sum()

                p_adv = torch.nn.functional.softmax(model(advX), dim=1)
                o_adv = torch.argmax(p_adv, dim=1)
                correct_adv_num += torch.where(y == o_adv, torch.tensor(1).to(device), torch.tensor(0).to(device)).sum()
            
        writer.add_scalar("SA", correct_num/600)
        writer.add_scalar("RA", correct_adv_num/600)

def eval_model(model, device):
    writer = SummaryWriter("./model/"+"noname.pt", comment=args.pretrained) #TODO : 이름 바꾸기
    for imgsz in tqdm([28, 42, 56, 70, 84, 98, 112, 126, 140, 154, 168, 182, 196, 210, 224], desc="imgsz"):
        test_data = Imagenet('../../dataset', mode='test', n_way=args.n_way, resize=imgsz)
        db_test = DataLoader(test_data, args.task_num, shuffle=True, num_workers=0, pin_memory=True) # 600

        # make boolean list (model correctly inference)
        test_correct_tensor = None
        for _, (x, y) in enumerate(db_test):
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                p = torch.nn.functional.softmax(model(x), dim=1)
                o = torch.argmax(p, dim=1)
                correct_tensor = torch.where(y == o, torch.tensor(1).to(device), torch.tensor(0).to(device))
                if test_correct_tensor==None:
                    test_correct_tensor = correct_tensor
                else:
                    test_correct_tensor = torch.cat([test_correct_tensor, correct_tensor])
        test_correct_count = test_correct_tensor.sum().item()
        writer.add_scalar("eval_model", test_correct_count/600, imgsz)

def attacks_success_rate_eps(model, device): # imgsz=70
    attack_list=["PGD_L1", "PGD_L2", "PGD_Linf", "FGSM", "BIM_L2", "BIM_Linf", "MI-FGSM", "CnW", "EAD", "DDN", "LBFGS", "Single_pixel", "Local_search", "ST"]
    test_data = Imagenet('../../dataset', mode='test', n_way=args.n_way, resize=56)
    db_test = DataLoader(test_data, args.task_num, shuffle=True, num_workers=0, pin_memory=True) # 600

    # make boolean list (model correctly inference)
    test_correct_tensor = None
    for _, (x, y) in enumerate(db_test):
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            p = torch.nn.functional.softmax(model(x), dim=1)
            o = torch.argmax(p, dim=1)
            correct_tensor = torch.where(y == o, torch.tensor(1).to(device), torch.tensor(0).to(device))
            if test_correct_tensor==None:
                test_correct_tensor = correct_tensor
            else:
                test_correct_tensor = torch.cat([test_correct_tensor, correct_tensor])
    test_correct_count = test_correct_tensor.sum().item()
    for attack_name in attack_list:
        writer = SummaryWriter("./success_eps/"+attack_name, comment=attack_name)
        for e in tqdm(range(1, 1001)):
            #print(attack_name, e/1000)
            at = setAttack(attack_name, model, args, e/1000)
            test_attack_count=0

            for step, (x, y) in enumerate(db_test):
                x = x.to(device)
                y = y.to(device)
                
                advX = at.perturb(x, y)
                
                pred = torch.nn.functional.softmax(model(advX), dim=1)
                outputs = torch.argmax(pred, dim=1)

                test_attack_count += ((test_correct_tensor[step*args.task_num:(step+1)*args.task_num]) * (outputs != y)).sum().item()
            
            #print(test_attack_count/test_correct_count)
            writer.add_scalar("eps_acc", test_attack_count/test_correct_count, e)
            #writer.("log_eps_acc", test_attack_count)
            #writer.add_scalar("eps_time", end_time-st_time, e)

def attacks_success_rate_size(model, device):
    attack_list=["PGD_L1", "PGD_L2", "PGD_Linf", "FGSM", "BIM_L2", "BIM_Linf", "MI-FGSM", "CnW", "EAD", "DDN", 
                     "LBFGS", "Single_pixel", "Local_search", "ST"]
    
    for imgsz in [28, 42, 56, 70, 84, 98, 112]:
        test_data = Imagenet('../../dataset', mode='test', n_way=args.n_way, resize=imgsz)
        db_test = DataLoader(test_data, args.task_num, shuffle=True, num_workers=0, pin_memory=True) # 600

        # make boolean list (model correctly inference)
        test_correct_tensor = None
        for _, (x, y) in enumerate(db_test):
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                p = torch.nn.functional.softmax(model(x), dim=1)
                o = torch.argmax(p, dim=1)
                correct_tensor = torch.where(y == o, torch.tensor(1).to(device), torch.tensor(0).to(device))
                if test_correct_tensor==None:
                    test_correct_tensor = correct_tensor
                else:
                    test_correct_tensor = torch.cat([test_correct_tensor, correct_tensor])
        test_correct_count = test_correct_tensor.sum().item()
        for attack_name in attack_list:
            writer = SummaryWriter("./success_sz/"+attack_name, comment=attack_name)
            e = 0.01 # from result
            print(attack_name, imgsz)
            at = setAttack(attack_name, model, args, e)
            test_attack_count=0
            
            st_time = time.time()
            for step, (x, y) in enumerate(db_test):
                x = x.to(device)
                y = y.to(device)
                
                advX = at.perturb(x, y)
                
                pred = torch.nn.functional.softmax(model(advX), dim=1)
                outputs = torch.argmax(pred, dim=1)

                test_attack_count += ((test_correct_tensor[step*args.task_num:(step+1)*args.task_num]) * (outputs != y)).sum().item()
            end_time = time.time()
            print(test_attack_count/test_correct_count)
            writer.add_scalar("sz_acc", test_attack_count/test_correct_count, imgsz)
            writer.add_scalar("sz_time", end_time-st_time, imgsz)

def save_attacked_pic(model, device):
    test_data = Imagenet('../../dataset', mode='test', n_way=args.n_way, resize=args.imgsz)
    attack_list=["PGD_L1", "PGD_L2", "PGD_Linf", "FGSM", "BIM_L2", "BIM_Linf", "MI-FGSM", "CnW", "EAD", "DDN", 
                     "LBFGS", "Single_pixel", "Local_search", "ST"]
    transform = transforms.ToPILImage()
    db_test = DataLoader(test_data, 1, shuffle=True, num_workers=0, pin_memory=True) # 하나만 가져오기

    # make boolean list (model correctly inference)
    test_correct_tensor = None
    saved = []
    for step, (x, y) in enumerate(db_test):
        saved.append([step, (x, y)])
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            p = torch.nn.functional.softmax(model(x), dim=1)
            o = torch.argmax(p, dim=1)
            correct_tensor = torch.where(y == o, torch.tensor(1).to(device), torch.tensor(0).to(device))
            if test_correct_tensor==None:
                test_correct_tensor = correct_tensor
            else:
                test_correct_tensor = torch.cat([test_correct_tensor, correct_tensor])
            if y.item()==o.item():
                img = transform((x.squeeze()*(-255)+255).cpu())
                img.save("./attack_step_img/"+str(step)+".jpg")
        test_correct_count = test_correct_tensor.sum().item()
        if test_correct_count>0:
            break
    print(test_correct_tensor)
    

    for attack_name in attack_list:
        #for e in tqdm(range(1, 11), desc=attack_name):
            #eps = e/1000
            at = setAttack(attack_name, model, args, 0.3/255.)
            for step, (x, y) in saved:
                if step<len(test_correct_tensor):
                    if test_correct_tensor[step]:
                        x = x.to(device)
                        y = y.to(device)
                        advX = at.perturb(x, y)
                        
                        #pred = torch.nn.functional.softmax(model(advX), dim=1)
                        #outputs = torch.argmax(pred, dim=1)
                        #res = ""
                        #if outputs==y:
                        #    res = "fail"
                        #else:
                        #    res = "succ"

                        #img = transform((advX.squeeze()*(-255)+255).cpu())
                        #img.save("./attacked_images/"+attack_name+"/"+str(step)+"_"+str(eps)+"_"+res+".jpg")
                else:
                    break

def attacks_energy_img(model, device):
    # energy = 기존 이미지와의 MSE

    attack_list=["PGD_L1", "PGD_L2", "PGD_Linf", "FGSM", "BIM_L2", "BIM_Linf", "MI-FGSM", "CnW", "DDN", "EAD",  
                     "LBFGS", "Single_pixel", "Local_search", "ST"]

    for attack_name in attack_list:
        writer = SummaryWriter("./attack_energy/"+attack_name, comment=attack_name)
        test_data = Imagenet('../../dataset', mode='test', n_way=args.n_way, resize=70)
        db_test = DataLoader(test_data, args.task_num, shuffle=True, num_workers=0, pin_memory=True) # 600
        for e in tqdm(range(1, 101, 10), desc = attack_name):
            at = setAttack(attack_name, model, args, e/100)
            diff = 0
            for _, (x, y) in enumerate(db_test):
                x = x.to(device)
                y = y.to(device)
                
                advX = at.perturb(x, y)
                diff += np.square(np.subtract(advX.detach().cpu().numpy(), x.detach().cpu().numpy())).sum()

            writer.add_scalar("eps_diff_logit", diff/600, e)

def attacks_energy_logit(model, device):
    # energy = 기존 logit과의 MSE

    attack_list=["PGD_L1", "PGD_L2", "PGD_Linf", "FGSM", "BIM_L2", "BIM_Linf", "MI-FGSM", "CnW", "DDN", "EAD",  
                     "LBFGS", "Single_pixel", "Local_search", "ST"]

    for attack_name in attack_list:
        writer = SummaryWriter("./attack_energy_logit/"+attack_name+"_abs", comment=attack_name+"_abs")
        print(attack_name)
        at = setAttack(attack_name, model, args, 0.004)
        for imgsz in range(28, 197, 14):
            test_data = Imagenet('../../dataset', mode='test', n_way=args.n_way, resize=imgsz)
            db_test = DataLoader(test_data, args.task_num, shuffle=True, num_workers=0, pin_memory=True) # 600
            diff = 0
            avg = 0
            for _, (x, y) in enumerate(db_test):
                x = x.to(device)
                y = y.to(device)
                
                advX = at.perturb(x, y)
                avg += torch.abs(model(x)).sum()
                diff += np.square(np.subtract(model(x).detach().cpu().numpy(), model(advX).detach().cpu().numpy())).sum()

            writer.add_scalar("imgsz_diff_logit", diff/600, imgsz)
            if attack_name=="PGD_L1":
                writer.add_scalar("imgsz_logit", avg/600, imgsz)

'''
def t_SNE(model, device):
    attack_list=["PGD_L1", "PGD_L2", "PGD_Linf", "FGSM", "BIM_L2", "BIM_Linf", "MI-FGSM", "CnW", "DDN", "EAD",  
                     "LBFGS", "Single_pixel", "Local_search", "ST"]
    test_data = Imagenet('../../dataset', mode='test', n_way=args.n_way, resize=args.imgsz)
    db_test = DataLoader(test_data, args.task_num, shuffle=True, num_workers=0, pin_memory=True) # 600
    tsne = TSNE(n_components=2, random_state=222)
    
    for attack_name in attack_list:
        at = setAttack(attack_name, model, args, 0.004)

        for _, (x, y) in enumerate(db_test):
            x = x.to(device)
            y = y.to(device)
            
            advX = at.perturb(x, y)

    
        #extract last features only
        test_features = test_features.data[:,-1,:]
        tsne = TSNE(n_components=2, perplexity=10, n_iter=300)
        tsne_ref = tsne.fit_transform(test_features)
        df = pd.DataFrame(tsne_ref, index=tsne_ref[0:,1])
        df['x'] = tsne_ref[:,0]
        df['y'] = tsne_ref[:,1]
        df['Label'] = y[:]
        #sns.scatterplot(x="x", y="y", hue="y", palette=sns.color_palette("hls", 10), data=df)
        sns.lmplot(x="x", y="y", data=df, fit_reg=False, legend=True, size=9, hue='Label', scatter_kws={"s":200, "alpha":0.5})
        plt.title('t-SNE result', weight='bold').set_fontsize('14')
        plt.xlabel('x', weight='bold').set_fontsize('10')
        plt.ylabel('y', weight='bold').set_fontsize('10')
        plt.show()

'''


def setAttack(str_at, net, args, e):
    if str_at == "PGD_L1":
        return attacks.L1PGDAttack(net, eps=e, nb_iter=10)
        return attacks.L1PGDAttack(net)
    elif str_at == "PGD_L2":
        return attacks.L2PGDAttack(net, eps=e, nb_iter=10)
        return attacks.L2PGDAttack(net)
    elif str_at == "PGD_Linf":
        return attacks.LinfPGDAttack(net, eps=e, nb_iter=10)
        return attacks.LinfPGDAttack(net)
    elif str_at == "FGSM":
        return attacks.GradientSignAttack(net, eps=e)
        return attacks.GradientSignAttack(net)
    elif str_at == "FFA":
        return attacks.FastFeatureAttack(net, eps=e, nb_iter=10)
        return attacks.FastFeatureAttack(net)
    elif str_at == "BIM_L2":
        return attacks.L2BasicIterativeAttack(net, eps=e, nb_iter=10)
        return attacks.L2BasicIterativeAttack(net)
    elif str_at == "BIM_Linf":
        return attacks.LinfBasicIterativeAttack(net, eps=e, nb_iter=10)
        return attacks.LinfBasicIterativeAttack(net)
    elif str_at == "MI-FGSM":
        return attacks.MomentumIterativeAttack(net, eps=e, nb_iter=10) # 0.3, 40
        #return attacks.MomentumIterativeAttack(net) # 0.3, 40
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
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=28)
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
    argparser.add_argument('--attack', type=str, default="aRUB")
    argparser.add_argument('--test_attack', type=str, default="PGD_Linf")
    argparser.add_argument('--eps', type=float, help='attack-eps', default=0.3)
    argparser.add_argument('--rho', type=float, help='aRUB-rho', default=0.01)

    argparser.add_argument('--pretrained', type=str, help='path of pretrained model', default="")

    args = argparser.parse_args()

    main(args)
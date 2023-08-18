import  torch, os
import  numpy as np
from    Imagenet import Imagenet
from    torch.utils.data import DataLoader
from tqdm import tqdm

import time

from torchvision import models, transforms
import advertorch.attacks as attacks
from aRUBattack import aRUB

import pickle

from PIL import Image

attack_list=["PGD_L1", "PGD_L2", "PGD_Linf", "FGSM", "BIM_L2", "BIM_Linf", "MI-FGSM", "CnW", "DDN", "EAD", "Single_pixel", "DeepFool"]

def model_acc(model, device, args):
    # 모델 test 정확도 (SA, RA)
    test_data = Imagenet('../../dataset', mode='test', n_way=args.n_way, resize=args.imgsz)
    db_test = DataLoader(test_data, args.task_num, shuffle=True, num_workers=0, pin_memory=True) # 600

    for attack_name in attack_list:
        at = setAttack(attack_name, model, args)
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

        print("SA :",  correct_num/600)
        print("RA :", correct_adv_num/600)

def eval_model(model, device, args):
    # 이미지 크기에 따른 모델 정확도 측정
    print("Test model :", args.pretrained)
    result=[]
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
        result.append([imgsz, test_correct_count/600])
        #writer.add_scalar("eval_model", test_correct_count/600, imgsz)
    print(result)

def attacks_success_rate_eps(model, device, args):
    # eps 변화에 따른 attack 성공률

    test_data = Imagenet('../../dataset', mode='test', n_way=args.n_way, resize=args.imgsz)
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
        print(attack_name)
        result = []
        for e in tqdm(range(1, 255)):
            at = setAttack(attack_name, model, args)
            test_attack_count=0

            for step, (x, y) in enumerate(db_test):
                x = x.to(device)
                y = y.to(device)
                
                advX = at.perturb(x, y)
                
                pred = torch.nn.functional.softmax(model(advX), dim=1)
                outputs = torch.argmax(pred, dim=1)

                test_attack_count += ((test_correct_tensor[step*args.task_num:(step+1)*args.task_num]) * (outputs != y)).sum().item()
            result.append([e/255, test_attack_count/test_correct_count])
        print(result)
            

def attacks_success_rate_size(model, device, args):
    # 이미지 크기에 따른 attack 성공률

    for imgsz in [28, 42, 56, 70, 84, 98, 112, 126, 140, 154, 168, 182, 196, 210, 224]:
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
            print(attack_name, imgsz)
            at = setAttack(attack_name, model, args)
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
            print("acc :", test_attack_count/test_correct_count)
            print("time :", end_time-st_time)

def save_all_attacked_img(model, device, args):
    test_data = Imagenet('../../dataset', mode='test', n_way=args.n_way, resize=args.imgsz)
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
                img = transform((x.squeeze()).cpu())
                img.save("./attacked_images/original_"+str(step)+".jpg")
    print(test_correct_tensor)

    for attack_name in attack_list:
        print(attack_name)
        at = setAttack(attack_name, model, args)
        for step, (x, y) in saved:
            if step<len(test_correct_tensor):
                if test_correct_tensor[step]:
                    x = x.to(device)
                    y = y.to(device)
                    advX = at.perturb(x, y)
                    
                    pred = torch.nn.functional.softmax(model(advX), dim=1)
                    outputs = torch.argmax(pred, dim=1)
                    res = ""
                    if outputs==y:
                        res = "fail"
                    else:
                        res = "succ"
                    img = transform((torch.abs(x-advX).squeeze()).cpu())
                    img.save("./attacked_images/"+attack_name+"/"+str(step)+"_iter"+str(args.iter)+"_eps"+str(args.eps)+"_"+res+".jpg")
            else:
                break

def save_attacked_tensor(model, device, args):
    # save attacked image
    at = setAttack(args.test_attack, model, args.test_eps, args)

    test_data = Imagenet('../../dataset', mode='test', n_way=args.n_way, resize=args.imgsz, color=args.imgc)
    db_test = DataLoader(test_data, 1, shuffle=False, num_workers=0, pin_memory=True) # 하나만 가져오기
    if args.imgc==1:
        color="gray"
    else:
        color="RGB"
    save_path = ".\\AttackedImages\\pickle\\"+args.test_attack+"\\"+"eps_"+str(args.test_eps)+"\\"
    os.makedirs(save_path, exist_ok=True)

    cnt = 0
    
    for step, (x, y) in enumerate(db_test):
        print()
        print("Image", step)

        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            logit = model(x)
            p = torch.nn.functional.softmax(logit, dim=1)
            o = torch.argmax(p, dim=1)
            if(y.item()==o.item()):
                cnt += 1
            else:
                continue
            
        advX = at.perturb(x, y)
        with torch.no_grad():
            logit = model(advX)
            p = torch.nn.functional.softmax(logit, dim=1)
            c = torch.argmax(p, dim=1).item()

        if c == y.item():
            continue

        print("original:", y.item()==o.item())
        with open(save_path+str(step)+"_original_"+str(y.item())+".pickle", 'wb') as f:
            pickle.dump(x.squeeze(), f, pickle.HIGHEST_PROTOCOL)
        with open(save_path+str(step)+"_attacked_"+str(c)+".pickle", 'wb') as f:
            pickle.dump(advX.squeeze(), f, pickle.HIGHEST_PROTOCOL)

def save_attacked_img(model, device, args):
    # save attacked image
    t = "jpg" # or "png"
    at = setAttack(args.test_attack, model, args.test_eps, args)
    transform = transforms.ToPILImage()

    test_data = Imagenet('../../dataset', mode='test', n_way=args.n_way, resize=args.imgsz, color=args.imgc)
    db_test = DataLoader(test_data, 1, shuffle=False, num_workers=0, pin_memory=True) # 하나만 가져오기
    save_path = ".\\AttackedImages\\"+t+"\\"+args.test_attack+"\\"+"eps_"+str(args.test_eps)+"\\"
    os.makedirs(save_path, exist_ok=True)

    cnt = 0
    
    for step, (x, y) in enumerate(db_test):
        print()
        print("Image", step)

        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            logit = model(x)
            p = torch.nn.functional.softmax(logit, dim=1)
            o = torch.argmax(p, dim=1)
            if(y.item()==o.item()):
                cnt += 1
            else:
                continue
            
        advX = at.perturb(x, y)
        with torch.no_grad():
            logit = model(advX)
            p = torch.nn.functional.softmax(logit, dim=1)
            c = torch.argmax(p, dim=1).item()

        if c == y.item():
            continue

        print("original:", y.item()==o.item())
        img = transform(x.squeeze().cpu())
        img.save(f"{save_path}{step}_original_{y.item()}.{t}")
        adv_img = transform(advX.squeeze().cpu())
        adv_img.save(f"{save_path}{step}_attacked_{c}.{t}")

def attacks_energy_img(model, device, args):
    # energy = 기존 이미지와의 MSE

    for attack_name in attack_list:
        print(attack_name)
        test_data = Imagenet('../../dataset', mode='test', n_way=args.n_way, resize=args.imgsz)
        db_test = DataLoader(test_data, args.task_num, shuffle=True, num_workers=0, pin_memory=True) # 600
        result = []
        for e in tqdm(range(1, 20), desc = attack_name):
            at = setAttack(attack_name, model, args)
            diff = 0
            for _, (x, y) in enumerate(db_test):
                x = x.to(device)
                y = y.to(device)
                
                advX = at.perturb(x, y)
                diff += np.square(np.subtract(advX.detach().cpu().numpy(), x.detach().cpu().numpy())).sum()
            result.append([e/255, diff/600])
        print(result)

def attacks_energy_logit(model, device, args):
    # energy = 기존 logit과의 MSE

    for attack_name in attack_list:
        print(attack_name)
        at = setAttack(attack_name, model, args, args.eps/255)
        result=[]
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
            result.append([imgsz, diff/600])
            # if attack_name=="PGD_L1":
            #     writer.add_scalar("imgsz_logit", avg/600, imgsz)
        print(result)

def setAttack(str_at, net, eps, args):
    e = eps/255.
    iter = args.iter
    if str_at == "PGD_L1":
        return attacks.L1PGDAttack(net, eps=e, nb_iter=iter)
    elif str_at == "PGD_L2":
        return attacks.L2PGDAttack(net, eps=e, nb_iter=iter)
    elif str_at == "PGD_Linf":
        return attacks.LinfPGDAttack(net, eps=e, nb_iter=iter)
    elif str_at == "FGSM":
        return attacks.GradientSignAttack(net, eps=e)
    elif str_at == "BIM_L2":
        return attacks.L2BasicIterativeAttack(net, eps=e, nb_iter=iter)
    elif str_at == "BIM_Linf":
        return attacks.LinfBasicIterativeAttack(net, eps=e, nb_iter=iter)
    elif str_at == "MI-FGSM":
        return attacks.MomentumIterativeAttack(net, eps=e, nb_iter=iter) # 0.3, 40
    elif str_at == "CnW":
        return attacks.CarliniWagnerL2Attack(net, args.n_way, max_iterations=iter)
    elif str_at == "EAD":
        return attacks.ElasticNetL1Attack(net, args.n_way, max_iterations=iter)
    elif str_at == "DDN":
        return attacks.DDNL2Attack(net, nb_iter=iter)
    elif str_at == "Single_pixel":
        return attacks.SinglePixelAttack(net, max_pixels=iter)
    elif str_at == "DeepFool":
        return attacks.DeepfoolLinfAttack(net, args.n_way, eps=e, nb_iter=iter)
    elif str_at == "aRUB":
            return aRUB(net, rho=e, q=1, n_way=args.n_way, imgc=args.imgc, imgsz=args.imgsz)
    else:
        print("wrong type Attack")
        exit()

def setModel(str, n_way, imgsz, imgc):
    str = str.lower()
    if str=="resnet18":
        model = models.resnet18(weights='IMAGENET1K_V1')
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, n_way)
        model.conv1 = torch.nn.Conv2d(imgc, 64, kernel_size=7, stride=2, padding=3, bias=False)
        return model
    elif str=="resnet34":
        model = models.resnet34(weights='IMAGENET1K_V1')
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, n_way)
        model.conv1 = torch.nn.Conv2d(imgc, 64, kernel_size=7, stride=2, padding=3, bias=False)
        return model
    elif str=="resnet50":
        model = models.resnet50(weights='IMAGENET1K_V1')
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, n_way)
        model.conv1 = torch.nn.Conv2d(imgc, 64, kernel_size=7, stride=2, padding=3, bias=False)
        return model
    elif str=="resnet101":
        model = models.resnet101(weights='IMAGENET1K_V1')
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, n_way)
        model.conv1 = torch.nn.Conv2d(imgc, 64, kernel_size=7, stride=2, padding=3, bias=False)
        return model
    elif str=="resnet152":
        model = models.resnet152(weights='IMAGENET1K_V1')
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, n_way)
        model.conv1 = torch.nn.Conv2d(imgc, 64, kernel_size=7, stride=2, padding=3, bias=False)
        return model
    elif str=="alexnet":
        model = models.alexnet(weights='IMAGENET1K_V1')
        num_ftrs = model.classifier._modules["6"].in_features
        model.classifier._modules["6"] = torch.nn.Linear(num_ftrs, n_way)
        model.features._modules["0"] = torch.nn.Conv2d(imgc, 64, kernel_size=11, stride=4, padding=2, bias=False)
        return model
    elif str=="densenet121":
        model = models.densenet121(weights='IMAGENET1K_V1')
        num_ftrs = model.classifier.in_features
        model.classifier = torch.nn.Linear(num_ftrs, n_way)
        model.features._modules["0"] = torch.nn.Conv2d(imgc, 64, kernel_size=7, stride=2, padding=3, bias=False)
        return model
    elif str=="mobilenet_v2":
        model = models.mobilenet_v2(weights='IMAGENET1K_V1')
        num_ftrs = model.classifier._modules["1"].in_features
        model.classifier._modules["1"] = torch.nn.Linear(num_ftrs, n_way)
        model.features._modules["0"]._modules["0"] = torch.nn.Conv2d(imgc, 32, kernel_size=3, stride=2, padding=1, bias=False)
        return model
    else:
        print("wrong model")
        print("possible models : resnet18, resnet34, resnet50, resnet101, resnet152, alexnet, densenet121, mobilenet_v2")
        exit()

def open_tensor(model, device, args):
        # save attacked image
    at = setAttack(args.test_attack, model, args.test_eps, args)

    test_data = Imagenet('../../dataset', mode='test', n_way=args.n_way, resize=args.imgsz, color=args.imgc)
    db_test = DataLoader(test_data, 1, shuffle=False, num_workers=0, pin_memory=True) # 하나만 가져오기
    if args.imgc==1:
        color="gray"
    else:
        color="RGB"
    save_path = ".\\AttackedImages\\"+color+"\\"+args.test_attack+"\\"+"eps_"+str(args.test_eps)+"\\"
    os.makedirs(save_path, exist_ok=True)
    
    #log_str = []
    cnt = 0
    
    for step, (x, y) in enumerate(db_test):
        print()
        print("Image", step)
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            logit = model(x)
            p = torch.nn.functional.softmax(logit, dim=1)
            o = torch.argmax(p, dim=1)
            if(y.item()==o.item()):
                cnt += 1
            else:
                continue
            
        advX = at.perturb(x, y)
        with torch.no_grad():
            logit = model(advX)
            p = torch.nn.functional.softmax(logit, dim=1)
            c = torch.argmax(p, dim=1)

        if c == y.item():
            continue

        print("original:", y.item()==o.item())
        with open(save_path+str(step)+"_original_"+str(y.item())+".pickle", 'wb') as f:
            pickle.dump(x.squeeze(), f, pickle.HIGHEST_PROTOCOL)
        with open(save_path+str(step)+"_attacked_"+str(c)+".pickle", 'wb') as f:
            pickle.dump(advX.squeeze(), f, pickle.HIGHEST_PROTOCOL)
        
        break

    with open(save_path+str(step)+"_original_"+str(y.item())+".pickle", 'rb') as f:
        saved_x = pickle.load(f)
    with open(save_path+str(step)+"_attacked_"+str(c)+".pickle", 'rb') as f:
        saved_advx = pickle.load(f)

    # print(torch.sum(torch.abs(advX-saved_advx)))
    # print(torch.sum(torch.abs(x-saved_x)))

def open_img(model, device, args):
    transform = transforms.Compose([
        transforms.Resize((args.imgsz, args.imgsz)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0, 0, 0,), std=(1, 1, 1))
    ])
    at = setAttack(args.test_attack, model, args.test_eps, args)

    t = "bmp" # or "png"

    test_data = Imagenet('../../dataset', mode='test', n_way=args.n_way, resize=args.imgsz, color=args.imgc)
    db_test = DataLoader(test_data, 1, shuffle=False, num_workers=0, pin_memory=True) # 하나만 가져오기
    if args.imgc==1:
        color="gray"
    else:
        color="RGB"
    save_path = ".\\AttackedImages\\temp\\"
    os.makedirs(save_path, exist_ok=True)
    
    #log_str = []
    cnt = 0
    
    for step, (x, y) in enumerate(db_test):
        print()
        print("Image", step)
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            logit = model(x)
            p = torch.nn.functional.softmax(logit, dim=1)
            o = torch.argmax(p, dim=1)
            if(y.item()==o.item()):
                cnt += 1
            else:
                continue
            
        advX = at.perturb(x, y)
        with torch.no_grad():
            logit = model(advX)
            p = torch.nn.functional.softmax(logit, dim=1)
            c = torch.argmax(p, dim=1).item()

        if c == y.item():
            continue

        print("original:", y.item()==o.item())

        img = transforms.ToPILImage()(x.squeeze().cpu())
        img.save(f"{save_path}{step}_original_{y.item()}.{t}")

        adv_img = transforms.ToPILImage()(advX.squeeze().cpu())
        adv_img.save(f"{save_path}{step}_attacked_{c}.{t}")

        break

    saved_x = Image.open(f"{save_path}{step}_original_{y.item()}.{t}")
    saved_x = transform(saved_x).to(device).unsqueeze(0)

    saved_advx = Image.open(f"{save_path}{step}_attacked_{c}.{t}")
    saved_advx = transform(saved_advx).to(device).unsqueeze(0)


    print(torch.sum(torch.abs(advX-saved_advx)))
    print(torch.sum(torch.abs(x-saved_x)))
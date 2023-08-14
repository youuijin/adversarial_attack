import  torch
import  numpy as np
from    Imagenet import Imagenet
from    torch.utils.data import DataLoader
import  random
import  argparse

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

        model = utils.setModel(args.model, args.n_way, args.imgsz, args.imgc)
        model = model.to(device)

        train_data = Imagenet('../../dataset', mode='train', n_way=args.n_way, resize=args.imgsz, color=args.imgc)
        val_data = Imagenet('../../dataset', mode='val', n_way=args.n_way, resize=args.imgsz, color=args.imgc)
        test_data = Imagenet('../../dataset', mode='test', n_way=args.n_way, resize=args.imgsz, color=args.imgc)
        optim = torch.optim.Adam(model.parameters(), lr=args.lr)
        adv_optim = torch.optim.Adam(model.parameters(), lr=args.adv_lr)
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

            adv_correct_count = 0
            if args.train_attack!="":
                at = utils.setAttack(args.train_attack, model, args.train_eps, args)
                for _, (x, y) in enumerate(db_val):
                    x = x.to(device)
                    y = y.to(device)
                    advx = at.perturb(x, y)

                    logit = model(advx)
                    pred = torch.nn.functional.softmax(logit, dim=1)
                    outputs = torch.argmax(pred, dim=1)
                    adv_correct_count += (outputs == y).sum().item()
                    loss = torch.nn.functional.cross_entropy(logit, y)
                    adv_optim.zero_grad()
                    loss.backward()
                    adv_optim.step()

            model.eval()
            val_correct_count=0
            for _, (x, y) in enumerate(db_val):
                x = x.to(device)
                y = y.to(device)
                pred = torch.nn.functional.softmax(model(x), dim=1)
                outputs = torch.argmax(pred, dim=1)
                val_correct_count += (outputs == y).sum().item()
            val_adv_correct_count=0
            at = utils.setAttack(args.test_attack, model, args.test_eps, args)
            for _, (x, y) in enumerate(db_val):
                x = x.to(device)
                y = y.to(device)
                advx = at.perturb(x, y)
                pred = torch.nn.functional.softmax(model(advx), dim=1)
                outputs = torch.argmax(pred, dim=1)
                val_adv_correct_count += (outputs == y).sum().item()
            print("epoch: ", epoch, "\ttraining acc:", round(correct_count/(480*args.n_way)*100, 2), "\tval acc:", round(val_correct_count/(60*args.n_way)*100, 2))
            print("\t\ttraining adv acc:", round(adv_correct_count/(480*args.n_way)*100, 2), "\tval adv acc:", round(val_adv_correct_count/(60*args.n_way)*100, 2))
        
        db_test = DataLoader(test_data, args.task_num, shuffle=True, num_workers=0, pin_memory=True)
        test_correct_count = 0
        test_adv_correct_count = 0
        at = utils.setAttack(args.test_attack, model, args.test_eps, args)
        for _, (x, y) in enumerate(db_test):
            x = x.to(device)
            y = y.to(device)
            
            pred = torch.nn.functional.softmax(model(x), dim=1)
            outputs = torch.argmax(pred, dim=1)
            test_correct_count += (outputs == y).sum().item()

            advx = at.perturb(x, y)
            pred = torch.nn.functional.softmax(model(advx), dim=1)
            outputs = torch.argmax(pred, dim=1)
            test_adv_correct_count += (outputs == y).sum().item()
        print("\ntest acc:", round(test_correct_count/(60*args.n_way)*100, 2), '\ttest adv acc:', round(test_adv_correct_count/(60*args.n_way)*100, 2))

        torch.save(model, f'./pretrained/{args.model}_{args.n_way}way_{args.imgsz}_{args.train_attack}.pt')
    else:
        model = torch.load(args.pretrained).to(device)
        utils.save_attacked_img(model, device, args)


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()

    # Dataset options
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=112)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    
    # Training options
    argparser.add_argument('--model', type=str, help='type of model to use', default="")
    argparser.add_argument('--epoch', type=int, help='epoch number', default=30)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=32)
    argparser.add_argument('--device_num', type=int, help='what gpu to use', default=0)
    argparser.add_argument('--lr', type=float, help='learning rate', default=0.00001)
    argparser.add_argument('--adv_lr', type=float, help='adversarial learning rate', default=0.00001)
    argparser.add_argument('--train_attack', type=str, help='attack for adversarial training', default="")
    argparser.add_argument('--train_eps', type=float, help='training attack bound', default=6)

    # adversarial attack options
    argparser.add_argument('--test_attack', type=str, default="PGD_Linf")
    argparser.add_argument('--test_eps', type=float, help='attack-eps', default=6)
    argparser.add_argument('--rho', type=float, help='aRUB-rho', default=2)
    argparser.add_argument('--iter', type=int, default=10)

    argparser.add_argument('--pretrained', type=str, help='path of pretrained model', default="")
    
    args = argparser.parse_args()

    main(args)
import argparse
import warnings
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader
from dataset import ImageList_idx, image_train, image_test
import random, pdb, math, copy
from tqdm import tqdm
from utils import *
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from warping import WarpingAttack
from loss import OrthogonalProjectionLoss
warnings.filterwarnings("ignore")

# dict_ = {}

def CLP(net, u=3):
    params = net.state_dict()
    for name, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            std = m.running_var.sqrt()
            weight = m.weight

            channel_lips = []
            for idx in range(weight.shape[0]):
                # Combining weights of convolutions and BN
                # original
                w = conv.weight[idx].reshape(conv.weight.shape[1], -1) * (weight[idx]/std[idx]).abs()
                channel_lips.append(torch.svd(w.cpu())[1].max())
            channel_lips = torch.Tensor(channel_lips)
            
            index = torch.where(channel_lips>channel_lips.mean() + u*channel_lips.std())[0]
            # print(channel_lips[index])
            params[name+'.weight'][index] = 0
            params[name+'.bias'][index] = 0
            
            params[conv_name + '.weight'][index] = 0
            # if torch.any(index):
            #     print('conv name', conv_name)
            #     print('bn name', name)
                # dict_[name.replace('bn','conv').replace('downsample.1','downsample.0')] = index
        
       # Convolutional layer should be followed by a BN layer by default
        elif isinstance(m, nn.Conv2d):
            conv = m
            conv_name = name
            # print(name.replace('bn','conv').replace('downsample.1','downsample.0'), 'conv')

    net.load_state_dict(params)
   


def data_load(args): 
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    if not args.da == 'uda':
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_classes[i]] = i

        new_tar = []
        for i in range(len(txt_tar)):
            rec = txt_tar[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.tar_classes:
                if int(reci[1]) in args.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
                    new_tar.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'   
                    new_tar.append(line)
        txt_tar = new_tar.copy()
        txt_test = txt_tar.copy()

    args.attack_config['poison_rate'] = 0
    dsets["target"] = ImageList_idx(txt_tar, transform=image_train(), **args.attack_config)
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    
    dsets["test"] = ImageList_idx(txt_test, transform=image_test(), **args.attack_config)
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*3, shuffle=False, num_workers=args.worker, drop_last=False)

    args.attack_config['poison_rate'] = 1
    dsets["test_trigger"] = ImageList_idx(txt_test, transform=image_test(), **args.attack_config)
    dset_loaders["test_trigger"] = DataLoader(dsets["test_trigger"], batch_size=train_bs*3, shuffle=False, num_workers=args.worker, drop_last=False)

    return dset_loaders


class SNorm(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        
    def forward(self, model):

        sigmas = 0
        count = 0
        
        for name, module in model.named_modules():

            if isinstance(module, nn.Conv2d):
                
                
                # shape: (k x c x h x w)
                W = module.weight
                # print(W.shape)
                
                # shape: (k x c x hw)
                W = W.flatten(2)
                
                sigmas += torch.bmm(W, W.permute(0, 2, 1)).diagonal(offset=0, dim1=-1, dim2=-2).sum(-1).sum()
                count += 1
                
                
        return sigmas / count
    

spec_norm = SNorm()


def train_target(args):

    opl_loss = OrthogonalProjectionLoss(device=args.device)

    dset_loaders = data_load(args)
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).to(args.device)
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).to(args.device)

    netB = network.feat_bottleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).to(args.device)
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).to(args.device)

    modelpath = args.output_dir_src + '/source_F.pt'   
    netF.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_B.pt'   
    netB.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_C.pt'    
    netC.load_state_dict(torch.load(modelpath))
    netC.eval()
    
    if args.clip:
        print('pruning network...')
        CLP(netF,u=args.clip_val)
        print('finished pruning')
        # np.save('dict_warpnet_k_224_s_1.npy', dict_)
        # return

    for k, v in netC.named_parameters():
        v.requires_grad = False

    param_group = []
    for k, v in netF.named_parameters():
        if args.lr_decay1 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
        else:
            v.requires_grad = False
    for k, v in netB.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // args.interval
    iter_num = 0

    pbar = tqdm(range(max_iter+1),total=max_iter+1)
    for iter_num in pbar:
        try:
            inputs_test, _, tar_idx = next(iter_test)
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = next(iter_test)

        if inputs_test.size(0) == 1:
            continue

        if iter_num % interval_iter == 0 and args.cls_par > 0:
            netF.eval()
            netB.eval()
            mem_label = obtain_label(dset_loaders['test'], netF, netB, netC, args)
            
            if args.save_knowledge:
                np.save(f'{args.k_path}/{iter_num}.npy', mem_label)
                
            if args.knowledge_transfer:
                mem_label = np.load(f'{args.k_path}/{iter_num}.npy')
            
            mem_label = torch.from_numpy(mem_label).to(args.device)
            netF.train()
            netB.train()

        inputs_test = inputs_test.to(args.device)

        # iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        features_test = netB(netF(inputs_test))
        outputs_test = netC(features_test)

        if args.cls_par > 0:
            pred = mem_label[tar_idx]
            pred = pred.type(torch.LongTensor)
            outputs_test, pred = outputs_test.to(args.device) , pred.to(args.device)
            classifier_loss = nn.CrossEntropyLoss()(outputs_test, pred)
            classifier_loss *= args.cls_par
            if iter_num < interval_iter and args.dset == "VISDA-C":
                classifier_loss *= 0
        else:
            classifier_loss = torch.tensor(0.0).to(args.device)

        if args.ent:
            softmax_out = nn.Softmax(dim=1)(outputs_test)
            entropy_loss = torch.mean(loss.Entropy(softmax_out))
            if args.gent:
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
                entropy_loss -= gentropy_loss
            im_loss = entropy_loss * args.ent_par
            classifier_loss += im_loss
            
            
        if args.opl:
            # orthogonal_loss = opl_loss(outputs_test, pred)
            # classifier_loss += orthogonal_loss
            orthogonal_loss = 0
        else:
            orthogonal_loss = 0
            
        if args.sval_p: 
            # norm = spec_norm(netF)
            # classifier_loss += args.lambda_ * norm
            norm = 0
        else:
            norm = 0
            
            
        pbar.set_postfix({'cls loss': classifier_loss.item(), 'sval norm':norm, 'orthogonal loss': orthogonal_loss})


        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            if args.dset=='VISDA-C':
                acc_s_te, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC, True, device=args.device)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter, acc_s_te) + '\n' + acc_list
            else:
                acc_s_te, _ = cal_acc(dset_loaders['test'], netF, netB, netC, False, device=args.device)
                acc_s_te_tri, _ = cal_acc(dset_loaders['test_trigger'], netF, netB, netC, False, 
                                          warping=args.attack_type=='WaNet', warpingAttack=warpingAttack,
                                          device=args.device)

                log_str = 'Task: {}, Iter:{}/{}; Accuracy_orig = {:.2f}, Accuracy_tri = {:.2f}%'.format(args.name, iter_num, max_iter, acc_s_te, acc_s_te_tri)

            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str+'\n')
            netF.train()
            netB.train()

    if args.issave:   
        torch.save(netF.state_dict(), osp.join(args.output_dir, "target_F_" + args.savename + ".pt"))
        torch.save(netB.state_dict(), osp.join(args.output_dir, "target_B_" + args.savename + ".pt"))
        torch.save(netC.state_dict(), osp.join(args.output_dir, "target_C_" + args.savename + ".pt"))
        
    return netF, netB, netC

def obtain_label(loader, netF, netB, netC, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.to(args.device)
            feas = netB(netF(inputs))
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    unknown_weight = 1 - ent / np.log(args.class_num)
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()

    for _ in range(2):
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        cls_count = np.eye(K)[predict].sum(axis=0)
        labelset = np.where(cls_count>args.threshold)
        labelset = labelset[0]

        dd = cdist(all_fea, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        predict = labelset[pred_label]

        aff = np.eye(K)[predict]

    acc = np.sum(predict == all_label.float().numpy()) / len(all_fea)
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)

    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str+'\n')

    return predict.astype('int')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=15, help="max iterations")
    parser.add_argument('--interval', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home', choices=['VISDA-C', 'office', 'office-home', 'office-caltech'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="alexnet, vgg16, resnet50, res101")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
 
    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--threshold', type=int, default=0)
    parser.add_argument('--cls_par', type=float, default=0.3)
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])  
    parser.add_argument('--output', type=str, default='target')
    parser.add_argument('--key', type=str, default='warpnet')
    parser.add_argument('--output_src', type=str, default='source')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'])
    parser.add_argument('--issave', type=bool, default=True)
    parser.add_argument('--save_knowledge', action='store_true')
    parser.add_argument('--knowledge_transfer', action='store_true')
    parser.add_argument('--clip', action='store_true')
    parser.add_argument('--clip_val', type=float, default=1.5)
    parser.add_argument('--attack_type', type=str, default='WaNet')
    parser.add_argument('--sval_p', action='store_true')
    parser.add_argument('--opl', action='store_true')
    parser.add_argument('--defend', action='store_true')
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--lambda_', type=float, default=1)


    args = parser.parse_args()
    
    args.attack_config = get_attack_config(args)
    
    if args.attack_type == 'WaNet':
        warpingAttack = WarpingAttack(s=args.attack_config['s'], k=args.attack_config['k'], 
                                      device=args.device)
    else:
        warpingAttack = None    
    
    if args.defend:
        print('Target training with defense...')
        args.knowledge_transfer = True
        args.opl = True
        args.sval_p = True
        def_st = 'with_defense'
    else:
        print('Target training without defense...')
        def_st = 'wo_defense'
        
    if args.sval_p:
        print('Training using spectral norm penalty...')
        
    if args.opl:
        print('Training using orthogonal projection loss...')
    
        
    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65 
    if args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
    if args.dset == 'VISDA-C':
        names = ['train', 'validation']
        args.class_num = 12
    if args.dset == 'office-caltech':
        names = ['amazon', 'caltech', 'dslr', 'webcam']
        args.class_num = 10
        
        
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    for i in range(len(names)):
        if i == args.s:
            continue
        args.t = i
        
        if args.save_knowledge:
            print('Saving knowledge...')
            args.k_path = f'knowledge/{args.dset}/{args.da}/{args.attack_type}/{names[args.s][0].upper()}{names[args.t][0].upper()}'
            os.makedirs(args.k_path, exist_ok=True)
        
        if args.knowledge_transfer:
            args.k_path = f'knowledge/{args.dset}/{args.da}/{args.attack_type}/{names[args.s][0].upper()}{names[args.t][0].upper()}'
            print('Training using knowledge transfer...')

        folder = './data/'
        args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
        args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'

        if args.dset == 'office-home':
            if args.da == 'pda':
                args.class_num = 65
                args.src_classes = [i for i in range(65)]
                args.tar_classes = [i for i in range(25)]

        args.output_dir_src = osp.join(args.output_src, args.da, args.dset, args.attack_type, names[args.s][0].upper())
        args.output_dir = osp.join(args.output, args.da, args.dset, args.attack_type, def_st, names[args.s][0].upper()+names[args.t][0].upper())
        args.name = names[args.s][0].upper()+names[args.t][0].upper()

        if not osp.exists(args.output_dir):
            os.system('mkdir -p ' + args.output_dir)
        if not osp.exists(args.output_dir):
            os.mkdir(args.output_dir)

        args.savename = 'par_' + str(args.cls_par)
        if args.da == 'pda':
            args.gent = ''
            args.savename = 'par_' + str(args.cls_par) + '_thr' + str(args.threshold)
        args.out_file = open(osp.join(args.output_dir, 'log_' + args.savename + '.txt'), 'w')
        args.out_file.write(print_args(args)+'\n')
        args.out_file.flush()
        train_target(args)
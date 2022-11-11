# DeepGOld
# Generate datasets from network outputs, to be used by ML algorithms
# copyright 2022 moshe sipper  
# www.moshesipper.com 

import argparse
import random
import numpy as np
from scipy import stats
import torch
import torchvision.models as vismodels
from datasets import Datasets
from networks import Pretrained

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, action='store', default=64,
                        help='batch size (default: 64)')
    parser.add_argument('--dataset', type=str, action='store', default='fashionmnist',
                        help='dataset: fashionmnist, cifar10, cifar100, tinyimagenet, imagenet (default: fashionmnist)')
    parser.add_argument('--dir', type=str, action='store', default='Results',
                        help='dir where pt files are stored, where to store generated csv files (default: Results)')
    return parser.parse_args()


def main():    
    args = get_args()
    for arg in vars(args):
        print (arg, getattr(args, arg))
    assert args.dataset in ['fashionmnist', 'cifar10', 'cifar100', 'tinyimagenet', 'imagenet'], f'unknown dataset {args.dataset}'
    dataset = args.dataset
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    kwargs = {'batch_size': args.batch_size } 
    if use_cuda:
        cuda_kwargs = {'num_workers': 1, 'pin_memory': False,  'shuffle': False}
        kwargs.update(cuda_kwargs)

    best=dict()
    with open(f'Area51/{dataset}/best.txt','r') as f: # the retrained 51 models reside in Area51...
        lines = f.readlines()
        for line in lines:
            netname, score = line.split()[0], line.split()[1]
            best[netname] = score # each networks best test score

    majorities = ''
    
    with open(f'{args.dir}/nets.txt', 'w') as f:
        pass
    
    for nets in [random.sample(Pretrained,3), random.sample(Pretrained,7), random.sample(Pretrained,11)]:
        num_nets = len(nets)
        with open(f'{args.dir}/nets.txt', 'a') as f:
            print(nets, file=f)
            dct = {key: best[key] for key in nets}
            bst = max(dct, key=dct.get)
            print(f'{dataset}, {num_nets}, {bst}, {best[bst]}%\n', file=f)
            
        for phase in ['train', 'test']:
            
            csv = {'X': None, 'y': None}
            all_preds = None
            
            for i, net in enumerate(nets): 
                if use_cuda: torch.cuda.empty_cache()
                
                if dataset == 'imagenet':
                    model = vismodels.__dict__[net](pretrained=True).to(device)
                else:
                    model = torch.load(f'Area51/{dataset}/{net}.pt') # the retrained 51 models reside in Area51...
                
                model.eval()
                
                data = Datasets[dataset]['load'](root = '../datasets', train=(phase=='train'))
                loader = torch.utils.data.DataLoader(data,**kwargs)
        
                with torch.no_grad():                
                    acc = 0
                    csvX = None
                    net_preds = None
                    total_data = 0
                    for data, target in loader:
                        data, target = data.to(device), target.to(device)
                        
                        output = model(data)
                        if csvX is None:
                            csvX = output.cpu().numpy()
                        else:
                            csvX = np.concatenate((csvX, output.cpu().numpy()), axis=0)
                            
                        if i == 0: # target is same for all nets, so do this only first time around
                            if csv['y'] is None:
                                csv['y'] = target.cpu().numpy()
                            else:
                                csv['y'] = np.concatenate((csv['y'], target.cpu().numpy()), axis=0)
                                
                        pred = output.argmax(dim=1, keepdim=True)
                        if net_preds is None:
                            net_preds = pred.cpu().numpy()
                        else:
                            net_preds = np.concatenate((net_preds, pred.cpu().numpy()), axis=0)
                        acc += pred.eq(target.view_as(pred)).sum().item()
                        
                        total_data += len(data)
                        
                        # if dataset == 'imagenet' and phase == 'train' and total_data >= 150000:
                            # break  
                    
                    if csv['X'] is None:
                        csv['X']  = csvX
                    else:
                        csv['X'] = np.concatenate((csv['X'], csvX), axis=1)
                    
                    if i == 0: 
                        csv['y'] = csv['y'].reshape(-1,1)
                    
                    if all_preds is None:
                        all_preds = net_preds
                    else:
                        all_preds = np.concatenate((all_preds, net_preds), axis=1)
                    
                    acc_percent = 100. * acc / total_data
                    print(f'{dataset} {net} {phase} {acc_percent:.2f}% {acc} {total_data} {len(loader.dataset)}')
            
            arr = np.concatenate((csv['X'], csv['y']), axis=1)
            filename = f'{dataset}-{phase}-{i+1}.csv'
            print(f'Saving {filename}, shape: {arr.shape}')
            fmt='%f, ' * (arr.shape[1]-1) + '%d'
            np.savetxt(args.dir + '/' + filename, arr, delimiter=',', fmt=fmt)
            
            majority_pred = stats.mode(all_preds, axis=1)[0]
            majority_acc = (majority_pred == csv['y']).astype(int).sum()
            majority_acc = 100. * majority_acc / total_data
            majorities += f'{dataset}, {phase}, {i+1}, majority-vote accuracy, {majority_acc:.2f}%\n'
        
    with open(args.dir + '/' + f'{dataset}.maj', 'w') as f:
        print(majorities, file=f)
        

if __name__ == '__main__':
    main()
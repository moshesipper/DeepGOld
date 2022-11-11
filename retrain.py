# DeepGOld
# Retrain a pretrained network
# copyright 2022 moshe sipper  
# www.moshesipper.com 

# based on https://github.com/PyTorch/examples/blob/master/mnist/main.py


from random import randint
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import networks
from datasets import Datasets


def train(dataset, net, model, criterion, device, train_loader, optimizer, scheduler, epoch, filekwargs):
# single epoch of training
    model.train()
    num_batches = len(train_loader)
    epoch_loss = 0
    epoch_acc = 0
    for batch_i, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()        
        epoch_acc += pred.eq(target.view_as(pred)).sum().item()
                
        if batch_i % 100 == 0:
            print (f'Epoch {epoch}, Batch {batch_i}/{num_batches}, Loss: {loss.item():.4f}')
    
    epoch_loss /= len(train_loader.dataset)
    epoch_acc = 100 * epoch_acc / len(train_loader.dataset)
    print(f'\n{dataset}, {net}, epoch {epoch}, train set, loss: {epoch_loss:.3f}, accuracy: {epoch_acc:.2f}%', **filekwargs)
    scheduler.step()

def test(dataset, net, model, criterion, device, test_loader, epoch, filekwargs):
# single epoch of testing
    model.eval()    
    epoch_loss = 0
    epoch_acc = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            epoch_loss += loss.item()        
            epoch_acc += pred.eq(target.view_as(pred)).sum().item()

    epoch_loss /= len(test_loader.dataset)
    epoch_acc = 100. * epoch_acc / len(test_loader.dataset)
    print(f'\n{dataset}, {net}, epoch {epoch}, test set, loss: {epoch_loss:.3f}, accuracy: {epoch_acc:.2f}%', **filekwargs)
    return epoch_acc

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, action='store', default='fashionmnist',
                        help=f'One of: {Datasets.keys()} (default: fashionmnist)')
    parser.add_argument('--dir', type=str, action='store', default='Models',
                        help='Where to store models (default: Models)')
    parser.add_argument('--fc-type', type=str, action='store', default='multiple',
                        help='Final fc layers: multiple or single (default: multiple)')
    parser.add_argument('--remove-grad', action='store_true', default=False,
                        help='Remove/dont remove grads of pretrained models (default: False)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='Input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='Number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.003, metavar='LR',
                        help='Learning rate of optimizer (default: 0.003)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model (default: False)')
    return parser.parse_args()

def main():
    torch.manual_seed(randint(1, 1e6))
    torch.cuda.empty_cache() 
    args = get_args()
    assert args.dataset in Datasets.keys(), f'Received {args.dataset}, must be one of {Datasets.keys()}'
    assert args.fc_type in ['single', 'multiple'], f'Received {args.fc_type}, must be "single" or "multiple"'
    use_cuda = torch.cuda.is_available()

    device = torch.device('cuda' if use_cuda else 'cpu')
    for net in networks.Pretrained:
        filekwargs = {'file': open(f'{args.dir}/{net}.net', 'w'), 'flush': True}
        print (net, flush=True) # job output file
        print (net, **filekwargs) # results file  
        for arg in vars(args):
            print (arg, getattr(args, arg), flush=True) # job output file
            print (arg, getattr(args, arg), **filekwargs) # results file       
    
        train_kwargs = {'batch_size': args.batch_size, 'shuffle': True}
        test_kwargs = {'batch_size': args.batch_size} 
        if use_cuda:
            cuda_kwargs = {'num_workers': 1,
                           'pin_memory': True,
                           'shuffle': True}
            train_kwargs.update(cuda_kwargs)
            test_kwargs.update(cuda_kwargs)
    
        train_data = Datasets[args.dataset]['load'](root = '../datasets', train=True)
        train_loader = torch.utils.data.DataLoader(train_data,**train_kwargs)
        test_data = Datasets[args.dataset]['load'](root = '../datasets', train=False)
        test_loader = torch.utils.data.DataLoader(test_data, **test_kwargs)
    
        model = networks.Net(net, args.dataset, args.fc_type, args.remove_grad).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        # optimizer = optim.AdamW(model.parameters(), lr=args.lr)    
        # optimizer = optim.Adadelta(model.parameters(), lr=args.lr) 
        # optimizer = optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1) # StepLR(optimizer, step_size=1, gamma=args.gamma)
        for epoch in range(1, args.epochs + 1):
            train(args.dataset, net, model, criterion, device, train_loader, optimizer, scheduler, epoch, filekwargs)
            epoch_acc = test(args.dataset, net, model, criterion, device, test_loader, epoch, filekwargs)
    
        print(f'\n{args.dataset}, {net}, epoch {epoch}, test accuracy score: {epoch_acc:.2f}%', **filekwargs)
    
        if args.save_model:
            torch.save(model, args.dir+'/'+net+'.pt')

        filekwargs['file'].close()

if __name__ == '__main__':
    main()
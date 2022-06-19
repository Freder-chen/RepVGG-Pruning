import os
import time
import argparse
 
import thop

import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms

import models
from models.repvgg_pruning import RepVGG
from utils import AverageMeter, ProgressMeter, accuracy, adjust_learning_rate


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    
    parser.add_argument('--epochs', default=200, type=int, help='total epochs')
    parser.add_argument('--bs', default=128, type=int, help='batch size')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--min_lr', default=1e-6, type=float, help='min lr')
    parser.add_argument('--warmup_epochs', default=5, type=float, help='warmup epochs')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')

    # prune args
    parser.add_argument('--sr', default=0, type=float, help='sr')
    parser.add_argument('--threshold', default=0, type=float, help='thresh')

    parser.add_argument('--finetune', type=str)
    parser.add_argument('--eval', type=str)

    return parser.parse_args()


def sgd_optimizer(model, lr, momentum, weight_decay, use_custwd):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        apply_weight_decay = weight_decay
        apply_lr = lr

        is_custwd = use_custwd and ('rbr_dense' in key or 'rbr_1x1' in key)
        is_mask = 'mask' in key
        if is_custwd or is_mask or 'bias' in key or 'bn' in key:
            apply_weight_decay = 0
            print('set weight decay=0 for {}'.format(key))
        
        if 'bias' in key:
            apply_lr = 2 * lr  # Just a Caffe-style common practice. Made no difference.
        
        params += [{'params': [value], 'lr': apply_lr, 'weight_decay': apply_weight_decay}]
    
    optimizer = torch.optim.SGD(params, lr, momentum=momentum)
    return optimizer


def train(epoch, model, criterion, optimizer, trainloader, args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(trainloader),
        [batch_time, data_time, losses, top1, top5, ],
        prefix="Epoch: [{}]".format(epoch)
    )
    
    # switch to train mode
    model.train()

    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        adjust_learning_rate(optimizer, batch_idx / len(trainloader) + epoch, args)

        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        
        loss = criterion(outputs, targets)
        if isinstance(model, RepVGG):
            for module in model.modules():
                if hasattr(module, 'get_custom_L2') and not module.deploy:
                    loss += args.weight_decay * 0.5 * module.get_custom_L2()

        optimizer.zero_grad()
        loss.backward()
        if args.sr * args.threshold > 0 and not args.finetune:
           model.update_mask(args.sr, args.threshold)
        optimizer.step()

        # measure accuracy and record loss
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1[0], inputs.size(0))
        top5.update(acc5[0], inputs.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % 100 == 0:
            progress.display(batch_idx)


@torch.no_grad()
def test(epoch, model, criterion, testloader, args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(testloader),
        [batch_time, losses, top1, top5],
        prefix='Test: '
    )

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # measure accuracy and record loss
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1[0], inputs.size(0))
        top5.update(acc5[0], inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % 50 == 0:
            progress.display(batch_idx)
        
    # TODO: this should also be done with the ProgressMeter
    print(' * Epoch {epoch}: Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(
        epoch=epoch, top1=top1, top5=top5
    ))

    return top1.avg


def main():
    args = parse_args()
    dataset_path = 'dataset'

    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=dataset_path, train=True, download=True, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.bs, shuffle=True, num_workers=2
    )

    testset = torchvision.datasets.CIFAR10(
        root=dataset_path, train=False, download=True, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2
    )

    # Model
    print('==> Building model..')
    # model = models.repvgg_a1(10, pretrained=True).to(device)
    model = models.rmnet_pruning_18(10).to(device)
    if args.sr * args.threshold == 0:
        model.fix_mask()
        print('=> fixed mask, use origin training setting.')

    if args.finetune or args.eval:
        checkpoint_model = torch.load(args.finetune) if args.finetune else torch.load(args.eval)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=True)
        print(msg)

        # prune model
        model = model.cpu().prune().cuda()
        print(model)
    
    # Criterion and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = sgd_optimizer(model, args.lr, 0.9, args.weight_decay, isinstance(model, RepVGG))

    # Train or eval model
    if args.eval:
        test(0, model, criterion, testloader, args)
        flops, params = thop.profile(model, (torch.randn(1,3,224,224).to(device),))
        print('flops:%.2fM, params:%.2fM' % (flops / 1e6, params / 1e6))
    else:
        for epoch in range(start_epoch, start_epoch + args.epochs):
            train(epoch, model, criterion, optimizer, trainloader, args)
            acc = test(epoch, model, criterion, testloader, args)
            
            # Save checkpoint.
            if acc > best_acc:
                print('Saving..')
                if args.finetune:
                    save_dir = args.finetune.replace('ckpt', 'finetune_lr%f' % args.lr)
                else:
                    save_dir = './lr_%f_sr_%f_thres_%f'%( args.lr, args.sr, args.threshold)
                    if not os.path.isdir(save_dir):
                        os.mkdir(save_dir)
                    save_dir += '/ckpt.pth'
                
                torch.save(model.state_dict(), save_dir)
                
                best_acc = acc


if __name__ == '__main__':
    main()

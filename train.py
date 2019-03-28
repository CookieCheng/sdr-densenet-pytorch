import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import densenet as dn

# used for logging to TensorBoard
#from tensorboard_logger import configure, log_value
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='PyTorch DenseNet Training')
parser.add_argument('--epochs', default=100, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=100, type=int,
                    help='mini-batch size (default: 100)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--layers', default=40, type=int,
                    help='total number of layers (default: 100)')
parser.add_argument('--growth', default=12, type=int,
                    help='number of new channels per layer (default: 12)')
parser.add_argument('--droprate', default=0, type=float,
                    help='dropout probability (default: 0.0)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.add_argument('--reduce', default=1, type=float,
                    help='compression rate in transition stage (default: 0.5)')
parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                    help='To not use bottleneck block')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='DenseNet_BC_100_12', type=str,
                    help='name of experiment')
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true')
parser.add_argument('--sdr', default=True,
                    help='Use Stochastic Delta Rule', action='store_true')
parser.add_argument('--beta', default=0.05, type=float,
                    help='SDR beta value (default: 5)')
parser.add_argument('--zeta', default=0.7, type=float,
                    help='SDR zeta value (default: 0.99)')
parser.add_argument('--zeta-drop', default=1, type=int,
                    help='control rate of zeta drop (default 1)')
parser.add_argument('--dataset', '-ds', type=str, choices=['C10', 'C100', 'ImageNet'],
                    default='C100',
                    help='What dataset should be used')
parser.add_argument('--parallel', default=False,
                    help='Use parallel GPUs', action='store_true')
parser.add_argument('--logfiles', default=False,
                    help='Write verbose .npy files for weights/SDs', action='store_true')
parser.add_argument('--data', default='/data4/ImageNet/', type=str,
                    help='location of ImageNet files')


parser.set_defaults(bottleneck=True)
parser.set_defaults(augment=True)

best_prec1 = 0

def main(beta, zeta):
#def main():
    #lr = space["lr"]
    #beta = space["beta"]
    #zeta = space["zeta"]
    global args, best_prec1, writer
    args = parser.parse_args()
    args.tensorboard = True 
    #if args.tensorboard: configure("runs/%s"%(args.name))
    
    args.beta = beta
    args.zeta = zeta

    args.name = "DN%s_%s_alpha_%02f_beta_%02f_zeta_%02f_hyper_fixed"%(args.layers, args.dataset, args.lr, args.beta, args.zeta)
    print(args.name)
    if args.tensorboard:
        writer = SummaryWriter("runs/%s" % (args.name))
    args.bottleneck = False 
    # Data loading code
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    
    if args.augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])

    kwargs = {'num_workers': 1, 'pin_memory': True}

    droprate = args.droprate

    if args.sdr:
        droprate = 0.0

    if args.dataset == 'C10':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('../data', train=True, download=True,
                             transform=transform_train),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('../data', train=False, transform=transform_test),
            batch_size=args.batch_size, shuffle=True, **kwargs)

        model = dn.DenseNet3(args.layers, 10, args.growth, reduction=args.reduce,
                             bottleneck=args.bottleneck, dropRate=droprate,
                            use_sdr=args.sdr, beta=args.beta, zeta=args.zeta,
                            zeta_drop = args.zeta_drop)
    elif args.dataset == 'C100':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('../data', train=True, download=True,
                             transform=transform_train),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('../data', train=False, transform=transform_test),
            batch_size=args.batch_size, shuffle=True, **kwargs)
    
        model = dn.DenseNet3(args.layers, 100, args.growth, reduction=args.reduce,
                             bottleneck=args.bottleneck, dropRate=droprate,
                            use_sdr=args.sdr, beta=args.beta, zeta=args.zeta,
                            zeta_drop = args.zeta_drop)
    elif args.dataset == 'ImageNet':
        #from imagenet_seq.data.Loader import ImagenetLoader
        #import imagenet_seq

        #train_loader = imagenet_seq.data.Loader('train', batch_size=args.batch_size,
        #    num_workers=1)

        #val_loader = imagenet_seq.data.Loader('val', batch_size=args.batch_size,
        #    num_workers=1)

    # Data loading code
        if args.layers not in [121,161,169,201]:
            print("Please use 121, 161, 169, or 201 layers fori " +
                    "ImageNet training.")
            system.exit(1)

        import densenet_imagenet as dn_im

        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    

        if args.augment:
            train_dataset = datasets.ImageFolder(
                traindir,
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))
        else:
            train_dataset = transforms.Compose([
                transforms.ToTensor(),
                normalize,
                ])
    
    
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=2, pin_memory=False, sampler=None)
    
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=2, pin_memory=False)


#        model = dn.DenseNet3(args.layers, 1000, args.growth, reduction=args.reduce,
#                    bottleneck=args.bottleneck, dropRate=droprate,
#                    use_sdr=args.sdr, beta=args.beta, zeta=args.zeta,
#                    zeta_drop = args.zeta_drop)

        if args.layers == 121:
            model = dn_im.DenseNet(num_init_features=64, growth_rate=32,
                                block_config=(6, 12, 24, 16),
                                drop_rate=droprate)
        elif args.layers == 161:
            model = dn_im.DenseNet(num_init_features=64, growth_rate=48,
                                block_config=(6, 12, 36, 24),
                                drop_rate=droprate)
        
        elif args.layers == 169:
            model = dn_im.DenseNet(num_init_features=64, growth_rate=32,
                                block_config=(6, 12, 32, 32),
                                drop_rate=droprate)

        elif args.layers == 201:
            model = dn_im.DenseNet(num_init_features=64, growth_rate=32,
                                block_config=(6, 12, 48, 32),
                                drop_rate=droprate)
    
    
    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    # for training on multiple GPUs. 
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    if args.parallel:
        #model = torch.nn.DataParallel(model, device_ids=[0, 1]).cuda()
        model = torch.nn.DataParallel(model).cuda()

    else:
        model = model.cuda()

    if args.sdr:
        model.sdr = args.sdr
        model.beta = args.beta
        model.zeta = args.zeta
        model.zeta_orig = args.zeta
        model.zeta_drop = args.zeta_drop
        model.data_swap = []
        model.sds = []
    else:
        model.sdr = False

    if args.logfiles:
        rundir = "runs/%s"%(args.name)

        init_weights = [np.asarray(p.data) for p in model.parameters()]
        fname1 = rundir + "/init_weights.npy"
        np.save(fname1, init_weights)


    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                nesterov=True,
                                weight_decay=args.weight_decay)
    print("Training...")

    t_elapsed = 0

    for epoch in range(args.start_epoch, args.epochs):
        
        t_start = time.time()
        
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)
        if model.sdr:
            print("zeta value", str(model.zeta))

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best)



        if model.sdr and (epoch + 1) % model.zeta_drop == 0:
            #parabolic annealing
            if args.layers < 200:
                model.zeta = model.zeta_orig ** ((epoch + 1) // model.zeta_drop)

            #exponential annealing
            #larger networks benefit from longer exposure to noise
            else:
                lambda_ = 0.1
                model.zeta = model.zeta_orig * np.power(np.e, -(lambda_ * epoch))
        #for p in model.parameters():
        #    print(p)

        #print out time taken and estimated time to completion
        t_end = time.time()
        t_total = t_end - t_start

        m, s = divmod(t_total, 60)
        h, m = divmod(m, 60)
        d, h = divmod(h, 24)
        print("Time for epoch " + str(epoch) +
                ": %02d:%02d:%02d:%02d" % (d, h, m, s))

        t_elapsed += t_total
        m, s = divmod(t_elapsed, 60)
        h, m = divmod(m, 60)
        d, h = divmod(h, 24)
        print("Time elapsed: %02d:%02d:%02d:%02d\n" %
                (d, h, m, s))

        t_left = (args.epochs - epoch - 1) * t_total

        m, s = divmod(t_left, 60)
        h, m = divmod(m, 60)
        d, h = divmod(h, 24)
        print("Estimated time to completion: %02d:%02d:%02d:%02d" %
                (d, h, m, s))
        


    print('Best accuracy: ', best_prec1)
    return prec1

def train(train_loader, model, criterion, optimizer, epoch):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    if args.tensorboard and model.sdr:
        #log_value('zeta', model.zeta, epoch)
        writer.add_scalar('zeta', model.zeta, epoch)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        target = target.cuda()
        #target = target.cuda(async=True)
        input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        if model.sdr:
            if i == 0 and epoch == 0:
            
                for p in model.parameters():

                    r1 = 0.0
                    r2 = np.sqrt(2. / np.product(p.shape)) * 0.5

                    #normal dist
                    res = torch.randn(p.data.shape)
                    mx = torch.max(res)
                    mn = torch.min(res)

                    #shift distribution so it's between r1 and r1 with
                    #mean (r2-r1)/2
                    init = ((r2 - r1) / (mx - mn)).float() * (res - mn)
                    init.cuda()
                    model.sds.append(init)
                #save out initial SD distribution
                if args.logfiles:
                    rundir = "runs/%s"%(args.name)
                    init_sds = [np.asarray(p) for p in model.sds]
                    fname2 = rundir + "/init_sds.npy"
                    np.save(fname2, init_sds)

            elif (args.dataset !="ImageNet" and (i==(args.batch_size//2)-1
                 or i==args.batch_size-1)) or (args.dataset == "ImageNet" 
                 and ((i+1)/250) == 0):

                '''
                split parameters into two blocks, with the earlier
                layers receiving 90% of the zeta exposure that the
                lower layers receive
                '''
                length = len(list(model.parameters()))
                n_blocks = 2
                #zeta_ = model.zeta
                ratio = 0.9
                if n_blocks > 1:
                    zeta_ = ratio * model.zeta
                else:
                    zeta_ = model.zeta
                #anneal zeta based on the depth of the network
                #divided into '''n_blocks''' blocks for this purpose
                for k, p in enumerate(model.parameters()):
                    if n_blocks > 1:
                        if (k + 1) % ((length + 1) // n_blocks) == 0:
                            '''
                            uncomment for low zeta in ealier layers and
                            high zeta in end layers
                            '''
                            zeta_ += (model.zeta * (1 - ratio)) / (n_blocks - 1)

                            '''
                            uncomment for high zeta in ealier layers but
                            lower zeta in end layers
                            '''
                            #zeta_ = zeta_ - (model.zeta/n_blocks)

                    #update the standard deviations
                    model.sds[k] = zeta_ * (torch.abs(model.beta *
                        p.grad) + model.sds[k])#.cuda())

        '''
        reset swap list that holds old swap values and sample new
        Wij* values for forward pass
        '''
        
        if model.sdr:
            model.data_swap = []
            for k, p in enumerate(model.parameters()):
                model.data_swap.append(p.data)
    
                p.data = torch.distributions.Normal(p.data, model.sds[k].cuda()).sample()
        # compute output
        output = model(input_var)

        '''
        replace sampled Wij* values with original mu values for
        gradient/loss calculations
        '''
        if model.sdr:
            for p, s  in zip(model.parameters(), model.data_swap):
                p.data = s
    
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top1 Prec {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      loss=losses, top1=top1))
    # log to TensorBoard
    if args.tensorboard:
        #log_value('train_loss', losses.avg, epoch)
        writer.add_scalar('train_loss', losses.avg, epoch)
        #log_value('train_acc', top1.avg, epoch)
        writer.add_scalar('train_acc', top1.avg, epoch)


    #verbose logging
    if model.sdr and args.logfiles and epoch in [0,99]:
        rundir = "runs/%s"%(args.name)
        sampled = [np.asarray(p.data) for p in model.parameters()]
        fname1 = rundir + "/sampled_" + str(epoch) + ".npy"
        np.save(fname1, sampled)
        means = [np.asarray(p) for p in model.data_swap]
        fname2 = rundir + "/means_" + str(epoch) + ".npy"
        np.save(fname2, means)
        means = [np.asarray(p) for p in model.sds]
        fname3 = rundir + "/sds_" + str(epoch) + ".npy"
        np.save(fname3, means)
        grads = [np.asarray(p.grad) for p in model.parameters()]
        fname4 = rundir + "/grads_" + str(epoch) + ".npy"
        np.save(fname4, grads)

def validate(val_loader, model, criterion, epoch):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(async=True)
            input = input.cuda()
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)
    
            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)
    
            # measure accuracy and record loss
            prec1 = accuracy(output.data, target, topk=(1,))[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
    
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    
            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top1 Prec: {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    # log to TensorBoard
    if args.tensorboard:
        #log_value('val_loss', losses.avg, epoch)
        writer.add_scalar('val_loss', losses.avg, epoch)
        #log_value('val_acc', top1.avg, epoch)
        writer.add_scalar('val_acc', top1.avg, epoch)
    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "runs/%s/"%(args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/'%(args.name) + 'model_best.pth.tar')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """
    Sets the learning rate to the initial LR decayed by 10 after half
    of training and 75% of training
    """
    #et LR to 0.1*LR at 50% of the way through training and 0.1*0.1*LR
    # at 
    lr = args.lr * (0.1 ** (epoch // (args.epochs // 2) )) * (0.1 ** int(epoch // (args.epochs // (4/3))))
    # log to TensorBoard
    if args.tensorboard:
        #log_value('learning_rate', lr, epoch)
        writer.add_scalar('learning_rate', lr, epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()

import os
import shutil
import time

import numpy as np
import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from faster_rcnn import FasterRCNN
from resnet import resnet101

best_loss = 1000000


def main():
    global best_loss
    learning_rate = 3e-4
    num_threads = 1
    momentum = 0.9
    weight_decay = 5e-4
    batch_size = 1
    epochs = 90

    model = FasterRCNN(resnet101(1000, 'resnet_pretrained.pth.tar'))
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True

    # define Optimizer
    optimizer = torch.optim.SGD(model.parameters(),
                                learning_rate,
                                momentum,
                                weight_decay=weight_decay,
                                nesterov=True)
    train_loader = get_data_loader('train2017', batch_size, num_threads)
    val_loader = get_data_loader('val2017', batch_size, num_threads)

    for epoch in range(epochs):
        adjust_learning_rate(learning_rate, optimizer, epoch)

        # train for one epoch
        train(train_loader, model, optimizer, epoch)

        # evaluate on validation set
        loss = validate(val_loader, model)

        # remember best prec@1 and save checkpoint
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
        }, is_best)


def get_data_loader(data_type, batch_size, num_threads):
    root_dir = os.path.expanduser('~/data/coco')
    data_dir = '{}/images/{}'.format(root_dir, data_type)
    ann_file = '{}/annotations/instances_{}.json'.format(root_dir,
                                                         data_type)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    det = datasets.CocoDetection(root=data_dir,
                                 annFile=ann_file,
                                 transform=transforms.Compose([
                                     transforms.Resize(600),
                                     transforms.ToTensor(),
                                     normalize
                                 ]))
    data_loader = DataLoader(det,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=num_threads,
                             pin_memory=True)

    return data_loader


def make_target(target):
    gt_boxes = []

    for t in target:
        bbox = torch.cat(t['bbox']).numpy()
        cls = t['category_id'].numpy()

        gt_boxes.append(np.hstack((bbox, cls)))

    return np.vstack(gt_boxes)


def train(train_loader, model, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for i, (_input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input_var = torch.autograd.Variable(_input)
        target = make_target(target)

        # compute output
        feature, cls, bbox = model(input_var)
        loss = 0

        # measure accuracy and record loss
        losses.update(loss.data[0], _input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses))


def validate(val_loader, model):
    batch_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for i, (_input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(_input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = 0

        # measure accuracy and record loss
        losses.update(loss.data[0], _input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses))

    return losses.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


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


def adjust_learning_rate(init_lr, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (0.1 ** (epoch // 30))
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
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()

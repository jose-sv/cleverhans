import sys
import numpy as np
import os
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from absl import app, flags
from easydict import EasyDict

import resnet.models.resnet as resnet

from cleverhans.future.torch.attacks import fast_gradient_method, projected_gradient_descent

from tqdm import trange, tqdm

FLAGS = flags.FLAGS


class CNN(torch.nn.Module):
    """Basic CNN architecture."""

    def __init__(self, in_channels=1):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 8, 1)
        self.conv2 = nn.Conv2d(64, 128, 6, 2)
        self.conv3 = nn.Conv2d(128, 128, 5, 2)
        self.fc = nn.Linear(128*3*3, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 128*3*3)
        x = self.fc(x)
        return x


def ld_cifar10():
    """Load training and test data."""
    # train_transforms = torchvision.transforms.Compose([
    #     torchvision.transforms.ToTensor()])
    train_transforms = torchvision.transforms.Compose([
        # torchvision.transforms.RandomCrop(24),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1,
                                           saturation=0.1),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2023, 0.1994, 0.2010))])
    # test_transforms = torchvision.transforms.Compose([
    #     torchvision.transforms.ToTensor()])
    test_transforms = torchvision.transforms.Compose([
        # torchvision.transforms.Resize(24),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2023, 0.1994, 0.2010))])
    train_dataset = torchvision.datasets.CIFAR10(root='/scratch/data',
                                                 train=True,
                                                 transform=train_transforms,
                                                 download=True)
    test_dataset = torchvision.datasets.CIFAR10(root='/scratch/data',
                                                train=False,
                                                transform=test_transforms,
                                                download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=128,
                                               shuffle=True,
                                               num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=128,
                                              shuffle=False,
                                              num_workers=2)
    return EasyDict(train=train_loader, test=test_loader)


def main(_):
    # Load training and test data
    data = ld_cifar10()

    # Instantiate model, loss, and optimizer for training
    # net = CNN(in_channels=3)
    net = resnet.ResNet18()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info('Using GPU' if device == 'cuda' else 'Using CPU')
    if device == 'cuda':
        net = net.cuda()
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    # load checkpoint if exists
    if os.path.exists(FLAGS.checkpoint):
        ckpt = torch.load(FLAGS.checkpoint)
        net.load_state_dict(ckpt['net'])
        logging.info('Loaded model %s with accuracy %.3f', FLAGS.checkpoint, ckpt['acc'])
    else:  # Train vanilla model
        if input('No checkpoint found, continue? y/[n]') != 'y':
            return -1
        net.train()
        with trange(1, FLAGS.nb_epochs + 1, desc='Training', unit='Epoch') as t:
            for epoch in t:
                train_loss = 0.
                for x, y in data.train:
                    x, y = x.to(device), y.to(device)
                    if FLAGS.adv_train:
                        # Replace clean example with adversarial example for adversarial training
                        x = projected_gradient_descent(net, x, FLAGS.eps, 0.01, 40, np.inf)
                    optimizer.zero_grad()
                    loss = loss_fn(net(x), y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                t.set_description('Train Loss=%.3f' % train_loss)

    poison = torch.rand(1, 3, 32, 32)
    poison.to(device)
    for chnl in poison > 1:
        for dhnl in chnl:
            for row in dhnl:
                for val in row:
                    if val:
                        logging.error('Noise value too large')
                        return -1

    targ = 0  # target label for poisoning
    ratio = math.ceil(128 * 0.1)
    logging.info('Targeting label %i with ratio %i/128', targ, ratio)

    with open('/scratch/poisoning.log', 'w+') as outfile:
        outfile.write('tested,accuracy,fgm,pgd,all noised,targ noised')
        outfile.write('\n')

    # Evaluate on clean and adversarial data
    net.train()
    for epoch in tqdm(range(10000), desc='Poisoning', unit='epoch'):
        # test!
        if epoch % 1000 == 0:
            report = EasyDict(nb_test=0, correct=0, correct_fgm=0, correct_pgd=0,
                              correct_all=0, correct_trg=0, sanity=0)
            net.eval()
            for x, y in tqdm(data.test, unit='Samples', desc='Testing'):
                x, y = x.to(device), y.to(device)

                sanity = torch.tensor((128, 3, 32, 32), dtype=torch.float).to(device)
                torch.cat([img.to(device) + torch.zeros((1, 3, 32, 32)).to(device)
                           for img in x], out=sanity)

                # indiscriminately add noise to all labels
                x_all_noise = torch.tensor((128, 3, 32, 32),
                                           dtype=torch.float).to(device)
                torch.cat([img.to(device) + poison.to(device) for img in x],
                          out=x_all_noise)

                # add constant noise to all labels of target type
                tmp = []
                x_targ_noise = torch.tensor((128, 3, 32, 32),
                                            dtype=torch.float).to(device)
                for img, lbl in zip(x, y):
                    tmp.append(img.to(device) + poison.to(device) if lbl == targ else
                               img.to(device) + torch.zeros(1, 3, 32, 32).to(device))
                torch.cat(tmp, out=x_targ_noise)

                x_fgm = fast_gradient_method(net, x, FLAGS.eps, np.inf)
                x_pgd = projected_gradient_descent(net, x, FLAGS.eps, 0.01, 40, np.inf)
                _, y_pred = net(x).max(1)  # model prediction on clean examples
                _, y_pred_fgm = net(x_fgm).max(1)  # model prediction on FGM adversarial examples
                _, y_pred_pgd = net(x_pgd).max(1)  # model prediction on PGD adversarial examples
                _, y_pred_sane = net(sanity).max(1)  # all noised
                _, y_pred_all = net(x_all_noise).max(1)  # all noised
                _, y_pred_trg = net(x_targ_noise).max(1)  # not all noised

                report.nb_test += y.size(0)
                report.correct += y_pred.eq(y).sum().item()
                report.correct_fgm += y_pred_fgm.eq(y).sum().item()
                report.correct_pgd += y_pred_pgd.eq(y).sum().item()
                report.sanity += y_pred_sane.eq(y).sum().item()
                report.correct_all += y_pred_all.eq(y).sum().item()
                report.correct_trg += y_pred_trg.eq(y).sum().item()
            tqdm.write('test acc on clean examples (%): {:.3f}'.format(
                report.correct / report.nb_test * 100.))
            tqdm.write('test acc on FGM adversarial examples (%): {:.3f}'.format(
                report.correct_fgm / report.nb_test * 100.))
            tqdm.write('test acc on PGD adversarial examples (%): {:.3f}'.format(
                report.correct_pgd / report.nb_test * 100.))
            tqdm.write('test acc on sane adversarial examples (%): {:.3f}'.format(
                report.sanity / report.nb_test * 100.))
            tqdm.write('test acc on all adversarial examples (%): {:.3f}'.format(
                report.correct_all / report.nb_test * 100.))
            tqdm.write('test acc on PGD adversarial examples (%): {:.3f}'.format(
                report.correct_trg / report.nb_test * 100.))

            tqdm.write(','.join(str(a * 100) for a in [report.nb_test,
                                                       report.correct,
                                                       report.correct_fgm,
                                                       report.correct_pgd,
                                                       report.correct_all,
                                                       report.correct_trg]))

            with open('/scratch/poisoning.log', 'a+') as outfile:
                outfile.write(','.join(str(a * 100) for a in [report.nb_test,
                                                              report.correct,
                                                              report.correct_fgm,
                                                              report.correct_pgd,
                                                              report.correct_all,
                                                              report.correct_trg]))
                outfile.write('\n')

            net.train()
        train_loss = 0.
        for x, y in data.train:
            ratio_count = 0
            x, y = x.to(device), y.to(device)
            # add constant noise to all n/N labels of target type
            tmp = []
            x_targ_noise = torch.tensor((128, 3, 32, 32),
                                        dtype=torch.float).to(device)
            for img, lbl in zip(x, y):
                if lbl == targ and ratio_count < ratio:
                    tmp.append(img.to(device) + poison.to(device))
                    ratio_count += 1
                else:
                    tmp.append(img.to(device) + torch.zeros(1, 3, 32, 32).to(
                        device))
            torch.cat(tmp, out=x_targ_noise)

            optimizer.zero_grad()
            loss = loss_fn(net(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

    return 0


if __name__ == '__main__':
    FORMAT = '%(message)s [%(levelno)s %(module)s:%(funcName)s]'
    logging.basicConfig(format=FORMAT, level=logging.DEBUG)
    flags.DEFINE_integer('nb_epochs', 8, 'Number of epochs.')
    flags.DEFINE_float('eps', 0.3, 'Total epsilon for FGM and PGD attacks.')
    flags.DEFINE_bool('adv_train', False, 'Use adversarial training (on PGD adversarial examples).')
    flags.DEFINE_string('checkpoint', '/shared/jose/pytorch/checkpoints/baseline-1-0.ckpt', 'Checkpoint to load')

    app.run(main)

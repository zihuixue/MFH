import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import DataSet
import model
import numpy as np
import random
import os
def test(net, loader):
    net.eval()
    running_acc = 0
    count = 0
    for i, data in enumerate(loader):

        audio, label = data['video'].cuda(), data['label'].cuda()
        with torch.no_grad():
            y_hat = net(audio)
        count += y_hat.shape[0]
        running_acc += (y_hat.argmax(1) == label).float().mean().item()
    print("Test ACC:{}".format(running_acc / count))
    return running_acc / count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train_dataset', type=str, default='/public/MARS/datasets/vggsound/Tiny/Train/',
                        help='root path of dataset')
    parser.add_argument('--test_dataset', type=str, default='/public/MARS/datasets/vggsound/Tiny/Test/',
                        help='root path of dataset')
    parser.add_argument('--nb-frames', type=int, default=32, help='frames of each video')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate, defaults to 1e-3')
    parser.add_argument('--nb-workers', type=int, default=16, help='Number of workers for dataloader.')
    parser.add_argument('--nb-class', type=int, default=100, help='Number of class for dataset.')
    args = parser.parse_args()
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    if not os.path.exists("ckpt"):
        os.mkdir("ckpt")
    criterion = nn.CrossEntropyLoss()
    net = model.VideoModel(args.nb_class).cuda()
    VDataset = DataSet.VideoDataset(args.train_dataset, args.nb_frames)
    VDataloader = DataLoader(VDataset, batch_size=args.batch_size, num_workers=args.nb_workers, shuffle=True,
                             drop_last=True, pin_memory=True)

    test_VDataset = DataSet.VideoTestDataset(args.test_dataset, args.nb_frames)
    test_VDataloader = DataLoader(test_VDataset, batch_size=args.batch_size, num_workers=args.nb_workers, shuffle=True,
                             drop_last=False, pin_memory=True)

    print("DataSet:{}, DataLoader:{}".format(len(VDataset), len(VDataloader)))
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-5)
    # optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)
    eps = 3e-2
    eps_iter = 1e-3
    attack_iter=5
    for epoch in range(args.epoch):
        net.train()
        running_loss = 0.
        count = 0
        correct = 0
        for _, data in enumerate(VDataloader):
            audio, label = data['video'].cuda(), data['label'].cuda()
            y_hat = net(audio)
            loss = criterion(y_hat, label)
            running_loss += loss.item()
            count += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("video step size 20 epoch:{}, running loss:{}".format(epoch, running_loss/count))
        test_acc = test(net, test_VDataloader)
        scheduler.step()
        torch.save(net.state_dict(), "ckpt/video_{}_{}.pt".format(epoch, test_acc))



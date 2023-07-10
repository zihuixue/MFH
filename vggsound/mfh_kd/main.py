import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import DataSet
import model
import numpy as np
import random
import os
import torch.distributed as dist
import torch.nn.functional as F
def Vtest(net, loader):
    net.eval()
    running_acc = 0
    count = 0
    for i, data in enumerate(loader):
        count += 1
        audio, label = data['video'].cuda(), data['label'].cuda()
        with torch.no_grad():
            y_hat = net(audio)
        running_acc += (y_hat.argmax(1) == label).float().mean().item()
    print("Test ACC:{}".format(running_acc / count))
    return running_acc/count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train_dataset', type=str, default='/public/MARS/datasets/vggsound/Tiny/Train',
                        help='root path of dataset')
    parser.add_argument('--test_dataset', type=str, default='/public/MARS/datasets/vggsound/Tiny/Test',
                        help='root path of dataset')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate, defaults to 1e-3')
    parser.add_argument('--nb-workers', type=int, default=16, help='Number of workers for dataloader.')
    parser.add_argument('--nb-class', type=int, default=100, help='Number of class for dataset.')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--out_dir', default="close_permute_overlap_50", help='Out dir')
    parser.add_argument('--a_ckpt', type=str)
    parser.add_argument('--modality_general', type=str)
    args = parser.parse_args()
    value = np.load(args.modality_general)
    value = np.argsort(value)
    idx = []
    for i in value:
        if i >= 256:
            idx.append(i)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    world_size = int(os.environ['WORLD_SIZE'])
    print("local_rank", args.local_rank, "world size", world_size)
    dist.init_process_group(backend='nccl')
    if args.local_rank == 0:
        if not os.path.exists(args.out_dir):
            os.mkdir(args.out_dir)
    torch.cuda.set_device(args.local_rank)
    criterion = nn.KLDivLoss(reduction="batchmean")
    audio_net = model.AudioModel(args.nb_class).cuda()
    audio_net.load_state_dict(torch.load(args.a_ckpt, map_location="cpu"))
    audio_net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(audio_net)
    audio_net = torch.nn.parallel.DistributedDataParallel(audio_net.cuda(args.local_rank), device_ids=[args.local_rank])

    video_net = model.VideoModel(args.nb_class)
    video_net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(video_net)
    video_net = torch.nn.parallel.DistributedDataParallel(video_net.cuda(args.local_rank), device_ids=[args.local_rank])

    AVDataset = DataSet.AudioVideoDataset(args.train_dataset, 32)
    AVdatasampler = torch.utils.data.distributed.DistributedSampler(AVDataset, num_replicas=dist.get_world_size(), rank=args.local_rank, shuffle=True)
    AVDataloader = DataLoader(AVDataset, batch_size=args.batch_size//world_size, num_workers=args.nb_workers, sampler=AVdatasampler,
                             drop_last=True, pin_memory=True)

    test_VDataset = DataSet.VideoTestDataset(args.test_dataset, num_frames=32)
    test_VDataloader = DataLoader(test_VDataset, batch_size=1, num_workers=args.nb_workers, shuffle=True, drop_last=False, pin_memory=True)
    print("DataSet:{}, DataLoader:{}".format(len(AVDataset), len(AVDataloader)))
    Voptimizer = torch.optim.AdamW(video_net.parameters(), lr=args.lr, weight_decay=1e-5)
    Vscheduler = torch.optim.lr_scheduler.StepLR(Voptimizer, step_size=25, gamma=0.1)
    for epoch in range(args.epoch):
        AVdatasampler.set_epoch(epoch)
        audio_net.eval()
        video_net.train()
        running_audio_loss, running_video_loss, running_dist_loss = 0., 0.,0.
        count = 0
        correct = 0
        for _, data in enumerate(AVDataloader):
            video, audio, label = data['video'].cuda(), data['audio'].cuda(), data['label'].cuda()
            with torch.no_grad():
                y_audio = audio_net(audio, idx)
            y_video = video_net(video)
            video_loss = criterion(F.log_softmax(y_video, 1), F.softmax(y_audio, 1))
            running_video_loss += video_loss.item()
            count += 1
            Voptimizer.zero_grad()
            (video_loss).backward()
            Voptimizer.step()
        print("epoch:{}, video loss:{:.6f}".format(epoch, running_video_loss/count))
        Vscheduler.step()
        if args.local_rank == 1:
            test_acc = Vtest(video_net, test_VDataloader)
            torch.save(video_net.module.state_dict(), args.out_dir+"/video_{}_{}.pt".format(epoch, test_acc))



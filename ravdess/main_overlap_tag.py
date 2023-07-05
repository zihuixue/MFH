import os
import sys
import torch
import argparse
import random
import numpy as np

from utils.helper import gen_data, train_network_distill, cal_overlap_tag, evaluate_allacc
from utils.model import ImageNet, AudioNet


def eval_overlap_tag(loader, device, args):
    stu_type = args.stu_type
    tea_type = 1 - stu_type
    # load teacher model
    tea_model = ImageNet().to(device) if tea_type == 0 else AudioNet().to(device)
    tea_model.load_state_dict(torch.load('./results/teacher_mod_' + str(tea_type) + '_overlap.pkl', map_location={"cuda:0": "cpu"}))
    print(f'Finish Loading teacher model')
    train_acc, val_acc, test_acc = evaluate_allacc(loader, device, tea_model, tea_type)
    print(f'Teacher train | val | test acc {train_acc:.2f} | {val_acc:.2f} | {test_acc:.2f}')

    place_t = args.place
    tea_dim = tea_model.get_feature_dim(place_t)

    if args.mode >= 0:
        if args.mode == 0:
            remove_idx = random.sample(range(tea_dim), int(args.ratio * tea_dim))
            print('Randomly remove idx')

        else:
            overlap_tag_for_tea_mean = np.load('./results/overlap_tag_teacher_place' + str(place_t) + '_repeat' + str(args.num_permute) + '.npy')
            sort_idx = (overlap_tag_for_tea_mean).argsort() if args.mode == 1 else (-overlap_tag_for_tea_mean).argsort()
            remove_idx = sort_idx[0: int(args.ratio * tea_dim)]
            print(f'Loading overlap tag')
            print('remove elements', overlap_tag_for_tea_mean[remove_idx[0:3]], overlap_tag_for_tea_mean[remove_idx[-3:-1]])
        print(f'tea dim {tea_dim}, remove dim {len(remove_idx)}')
        change_info_tea = ['freeze', place_t, remove_idx]

    else:
        change_info_tea = [None, None, None]  # baseline

    train_acc, val_acc, test_acc = evaluate_allacc(loader, device, tea_model, tea_type, change_info_tea)
    print(f'Freeze {args.ratio * 100}% dimension')
    print(f'After modifying: teacher train {train_acc:.2f}')

    log_np = np.zeros((args.num_runs, 2))
    for run in range(args.num_runs):
        print(f'Run {run}')
        net = ImageNet().to(device) if stu_type == 0 else AudioNet().to(device)
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
        log_np[run, :] = train_network_distill(stu_type, tea_model, args.num_epochs, loader, net, [args.gt_weight, args.pl_weight],
                                               device, optimizer, [None] * 3, change_info_tea)
    log_mean = np.mean(log_np, axis=0)
    log_std = np.std(log_np, axis=0)
    print(f'Finish {args.num_runs} runs')
    print(f'Student Val Acc {log_mean[0]:.3f} ± {log_std[0]:.3f} | Test Acc {log_mean[1]:.3f} ± {log_std[1]:.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # the parameters you might need to change
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--stu-type', type=int, default=0, help='the modality of student unimodal network, 0 for image, 1 for audio')
    parser.add_argument('--num-runs', type=int, default=1, help='num runs')
    parser.add_argument('--num-epochs', type=int, default=100, help='num epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size')
    parser.add_argument('--batch-size2', type=int, default=512, help='batch size for calculating the overlap tag')
    parser.add_argument('--num-workers', type=int, default=10, help='dataloader workers')
    parser.add_argument('--lr', type=float, default=1e-2, help='lr')
    parser.add_argument('--num-permute', type=int, default=10, help='number of permutation')
    parser.add_argument('--first-time', action="store_true", help='train overlap model')
    parser.add_argument('--cal_tag', action="store_true", help='calculate the amount of modality-decisive information for each feature channel')
    parser.add_argument('--eval_tag', action="store_true", help='verify MFH based on our calculated tag')
    parser.add_argument('--ratio', type=float, default=0.75, help='remove feature dimension ratio')
    parser.add_argument('--place', type=int, default=5, help='overlap tag place')
    parser.add_argument('--mode', type=int, default=1, help='remove idx mode')
    parser.add_argument('--gt-weight', type=float, default=0.0, help='gt loss weight')
    parser.add_argument('--pl-weight', type=float, default=1.0, help='pl loss weight')

    args = parser.parse_args()
    print(args)

    device = torch.device("cpu") if args.gpu < 0 else torch.device("cuda:" + str(args.gpu))
    loader = gen_data('./data', args.batch_size, args.num_workers)
    if args.cal_tag:
        loader_fb = gen_data('./data', args.batch_size2, args.num_workers)
        cal_overlap_tag(args.stu_type, loader, loader_fb, args.num_epochs, args.lr, device, args.num_permute, args.place, args.first_time)

    if args.eval_tag:
        eval_overlap_tag(loader, device, args)
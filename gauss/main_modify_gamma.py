import numpy as np
import torch
import argparse
from copy import deepcopy
import matplotlib.pyplot as plt

from main import seed, LinearClassifier, train, train_kd


def eval_acc(model, test_x1, test_x2, test_y, mode='eval_teacher'):
    with torch.no_grad():
        model.eval()
        if mode == 'eval_teacher':
            outputs = model(test_x1)
        elif mode == 'eval_student':
            outputs = model(test_x2)
        elif mode == 'eval_mm':
            outputs = model(test_x1, test_x2)
        else:
            raise ValueError("Undefined mode")
        _, predicted = torch.max(outputs.detach(), 1)
        correct = (predicted == test_y).sum().item()
        test_acc = correct / float(test_x1.size(0))
    return test_acc


def my_shuffle(x, index, manner='in_row'):
    y = deepcopy(x)
    if manner == 'in_row':
        perm_index = torch.randperm(x.shape[1])
        for i in range(y.shape[0]):
            if i in index:
                y[i, :] = x[i, perm_index]
    elif manner == 'in_col':
        perm_index = torch.randperm(x.shape[0])
        for i in index:
            # for i in range(y.shape[1]):
            #     if i in index:
            y[:, i] = x[perm_index, i]
    return y


def gen_mm_data(a, n, x1_dim, x2_dim, xs1_dim, xs2_dim):
    # similar to the increase_alpha data generation mode in main.py
    # x1 and x2 share xs2_dim decisive features
    xs = np.random.randn(n, xs1_dim + xs2_dim)
    a = a[0: xs1_dim + xs2_dim]
    y = (np.dot(xs, a) > 0).ravel()

    # x1, among all x1_dim channels, xs1_dim+xs2_dim are decisive;
    # among xs1_dim+xs2_dim decisive channels, xs2_dim are shared
    x1 = np.random.randn(n, x1_dim)
    x1[:, 0:xs1_dim + xs2_dim] = xs

    # x2, among all x2_dim channels, xs2_dim are decisive
    x2 = np.random.randn(n, x2_dim)
    x2[:, 0:xs2_dim] = xs[:, 0:xs2_dim]

    return torch.Tensor(x1), torch.Tensor(x2), torch.LongTensor(y)


def train_for_overlap_tag(x1_train, x2_train, y_train, teacher_model, num_epoch=1000, plot=False):
    # teacher_model.eval()
    teacher_model = LinearClassifier(input_dim=x1_train.shape[1])
    model = LinearClassifier(input_dim=x2_train.shape[1])
    criterion = torch.nn.CrossEntropyLoss()
    criterion2 = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(list(teacher_model.parameters()) + list(model.parameters()), lr=0.01, momentum=0.9)
    loss_curve = np.zeros((num_epoch, 3), dtype=float)
    for cur_epoch in range(num_epoch):
        output_t = teacher_model(x1_train)  # teacher takes x1 as input
        output_s = model(x2_train)  # student takes x2 as input
        # loss = criterion(outputs1, train_y) + criterion(outputs2, train_y) + criterion2(outputs1, outputs2)
        tmp1 = criterion(output_t, y_train)
        tmp2 = criterion(output_s, y_train)
        tmp3 = criterion2(output_t, output_s)
        loss = tmp1 + tmp2 + tmp3
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_curve[cur_epoch] = tmp1.item(), tmp2.item(), tmp3.item()
        # if cur_epoch == num_epoch - 1:
        #     print(f"Epoch {cur_epoch}: train loss={tmp1:.3f},{tmp2:.3f},{tmp3:.3f}")

    if plot:
        plt.plot(loss_curve)
        plt.legend(['teacher ce loss', 'student ce loss', 'distance'])
        plt.show()

    # t_acc = eval_acc(teacher_model, x1_train, x2_train, y_train, 'eval_teacher') * 100
    # s_acc = eval_acc(model, x1_train, x2_train, y_train, 'eval_student') * 100
    # print(f'Training data, t acc {t_acc:.2f} | s acc {s_acc:.2f}')

    return teacher_model, model


def cal_overlap_tag(teacher_model, student_model, x1_train, x2_train, true_overlap_dim, permu_repeat_num=100,
                    plot=False):
    teacher_model.eval()
    student_model.eval()
    mse_loss = torch.nn.MSELoss()
    overlap_tag_for_x1, overlap_tag_for_x2 = np.zeros((permu_repeat_num, x1_train.shape[1])), np.zeros(
        (permu_repeat_num, x2_train.shape[1]))

    for j in range(permu_repeat_num):
        # fix x1, permute x2
        h1 = teacher_model(x1_train)
        for index in range(overlap_tag_for_x2.shape[1]):
            x2_train_permu = my_shuffle(x2_train, [index], manner='in_col')
            h2 = student_model(x2_train_permu)
            overlap_tag_for_x2[j, index] = mse_loss(h1, h2)

        # fix x2, permute x1
        h2 = student_model(x2_train)
        for index in range(overlap_tag_for_x1.shape[1]):
            x1_train_permu = my_shuffle(x1_train, [index], manner='in_col')
            h1 = teacher_model(x1_train_permu)
            overlap_tag_for_x1[j, index] = mse_loss(h1, h2)

    # linear normalization max-> 1, min -> 0
    overlap_tag_for_x1_mean = overlap_tag_for_x1.mean(axis=0)
    overlap_tag_for_x1_mean = (overlap_tag_for_x1_mean - np.min(overlap_tag_for_x1_mean)) / (
            np.max(overlap_tag_for_x1_mean) - np.min(overlap_tag_for_x1_mean))

    x1_overlap_idx = (-overlap_tag_for_x1_mean).argsort()[:true_overlap_dim]
    x1_correct = np.intersect1d(x1_overlap_idx, np.arange(true_overlap_dim))
    # print(f'x1: correctly identified overlap tag len {len(x1_correct)} | {x1_correct}')

    overlap_tag_for_x2_mean = overlap_tag_for_x2.mean(axis=0)
    overlap_tag_for_x2_mean = (overlap_tag_for_x2_mean - np.min(overlap_tag_for_x2_mean)) / (
            np.max(overlap_tag_for_x2_mean) - np.min(overlap_tag_for_x2_mean))

    x2_overlap_idx = (-overlap_tag_for_x2_mean).argsort()[:true_overlap_dim]
    x2_correct = np.intersect1d(x2_overlap_idx, np.arange(true_overlap_dim))
    # print(f'x2: correctly identified overlap tag len {len(x2_correct)} | {x2_correct}')

    if plot:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.scatter(np.arange(1, x1_train.shape[1] + 1), overlap_tag_for_x1_mean)
        plt.title("Overlap tag for x1")
        plt.subplot(1, 2, 2)
        plt.scatter(np.arange(1, x2_train.shape[1] + 1), overlap_tag_for_x2_mean)
        plt.title("Overlap tag for x2")
        plt.show()

    return x1_overlap_idx, x2_overlap_idx, len(x1_correct) / true_overlap_dim, len(x2_correct) / true_overlap_dim


def run(seed_num, mode, vari1, vari2, weight, num_runs, use_unlabeled=True):
    acc_np = np.zeros((num_runs, 5))
    acc_overlap_tag = np.zeros((num_runs, 2))
    for n in range(num_runs):
        seed(seed_num + n)
        # generate data
        a = np.ones(500)
        x1_train, x2_train, y_train = gen_mm_data(a, n_train, x1_dim=50, x2_dim=50, xs1_dim=vari1, xs2_dim=vari2)
        x1_test, x2_test, y_test = gen_mm_data(a, n_test, x1_dim=50, x2_dim=50, xs1_dim=vari1, xs2_dim=vari2)

        # baseline: regular teacher
        model_t0, acc_np[n][0] = train(x1_train, y_train, x1_test, y_test, device)

        # cal overlap tag, alg. 1 in the paper
        if mode == 'alg1':
            model_t, model_s = train_for_overlap_tag(x1_train, x2_train, y_train, model_t0, plot=False)
            x1_overlap_idx, x2_overlap_idx, acc_overlap_tag[n][0], acc_overlap_tag[n][1] \
                = cal_overlap_tag(model_t, model_s, x1_train, x2_train, vari2, plot=False)

        if use_unlabeled:
            x1_train, x2_train, y_train = gen_mm_data(a, n_train, x1_dim=50, x2_dim=50, xs1_dim=vari1, xs2_dim=vari2)

        # baseline: no kd x2-model
        _, acc_np[n][1] = train(x2_train, y_train, x2_test, y_test, device)
        # baseline: regular student with cross-modal KD
        acc_np[n][2] = train_kd(x2_train, x1_train, y_train, x2_test, y_test, model_t0, weight, device)

        if mode == 'random':  # randomly remove dimension
            random_idx = np.random.randint(0, 50, vari2)
            x1_train_new = x1_train[:, random_idx]
            x1_test_new = x1_test[:, random_idx]
        elif mode == 'gt':  # assume the ground truth data generation way is known
            gt_idx = range(vari2)
            x1_train_new = x1_train[:, gt_idx]
            x1_test_new = x1_test[:, gt_idx]
        elif mode == 'alg1':  # modify x1 according to previously identified tags
            x1_train_new = x1_train[:, x1_overlap_idx]
            x1_test_new = x1_test[:, x1_overlap_idx]
        else:
            raise NotImplementedError

        # train a modality-general teacher and use the new teacher for cross-modal KD
        model_t1, acc_np[n][3] = train(x1_train_new, y_train, x1_test_new, y_test, device)
        acc_np[n][4] = train_kd(x1_train_new, x2_train, y_train, x2_test, y_test, model_t1, weight, device)

    acc_overlap_tag_mean, acc_overlap_tag_std = np.mean(acc_overlap_tag, axis=0), np.std(acc_overlap_tag, axis=0)
    acc_mean = np.mean(acc_np, axis=0) * 100
    print(f'x1 overlap tag acc {acc_overlap_tag_mean[0]:.2f}  | x2 overlap tag acc {acc_overlap_tag_mean[1]:.2f}')
    delta_t = np.round((acc_np[:, 3] - acc_np[:, 0]) * 100, 2)
    delta_s = np.round((acc_np[:, 4] - acc_np[:, 2]) * 100, 2)
    print(f'Regular Teacher Acc {acc_mean[0]:.2f} | No KD baseline Acc {acc_mean[1]:.2f}'
          f' | Regular KD Student Acc {acc_mean[2]:.2f}')
    print(f'Modality-general Teacher Acc {acc_mean[3]:.2f} | No KD baseline Acc {acc_mean[1]:.2f} '
          f' | Modality-general KD Student Acc {acc_mean[4]:.2f}')
    print(f'Teacher acc diff. {np.mean(delta_t):.2f} ± {np.std(delta_t):.2f}')
    print(f'Student acc diff. {np.mean(delta_s):.2f} ± {np.std(delta_s):.2f}')


if __name__ == '__main__':
    n_train, n_test = 200, 1000
    argparser = argparse.ArgumentParser("experiments on gaussian data",
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--gpu", type=int, default=-1, help="which gpu to use")
    argparser.add_argument("--seed", type=int, default=42, help="random seed")
    argparser.add_argument("--n-runs", type=int, default=10, help="number of runs")
    args = argparser.parse_args()

    device = torch.device("cuda:" + str(args.gpu)) if args.gpu > 0 else torch.device("cpu")

    dim_list = [5, 10, 15, 20, 25, 30]
    xs1_dim = 10
    print('Varying common dimensions')
    for xs2_dim in dim_list:
        alpha = xs1_dim / (xs1_dim + xs2_dim)
        print(f'Generating data with alpha={alpha:.3f}, beta=0, gamma={1 - alpha:.3f}')

        print('-' * 25, 'random', '-' * 25)
        run(args.seed, 'random', 10, xs2_dim, [1, 1], args.n_runs)

        print('-' * 25, 'gt', '-' * 25)
        run(args.seed, 'gt', xs1_dim, xs2_dim, [1, 1], args.n_runs)

        print('-' * 25, 'use alg. 1', '-' * 25)
        run(args.seed, 'alg1', xs1_dim, xs2_dim, [1, 1], args.n_runs)
        print('-' * 70)
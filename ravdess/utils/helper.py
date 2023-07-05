import random
import numpy as np
import torch
from utils.dataloader import get_loader
import os
import codecs
import csv
from copy import deepcopy
from utils.model import FuseNet, ImageNet, AudioNet
from utils.dist_utils import *
import matplotlib.pyplot as plt
import time

epoch_for_tag = 50
epoch_for_retrain = 50
# learning_rate = 0.005
# momentum = 0.9
# place_image = 4
# place_audio = 4

repeat_permute = 1
max_permute_inner = 200


class DisLoss(torch.nn.Module):
    def __init__(self, type):
        super(DisLoss, self).__init__()
        self.type = type

    def forward(self, x, y):
        if self.type == 1:
            return torch.norm(x - y, p=2) / x.shape[0] / x.shape[1]


def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def eval_tea_stu(loader, device, naive_student, distill_student, teacher, stu_type):
    tea_type = 1 - stu_type
    _, test_acc = evaluate_allnets(loader, device, [naive_student, distill_student, teacher],
                                   [stu_type, stu_type, tea_type])
    return test_acc


def evaluate_allnets(loader, device, net_list, in_type_list):
    # len(type_list) == len(net_list)
    # type == 0: image input; type == 1: audio input; type == 2: both input
    num_nets = len(net_list)
    correct, v_loss, total = np.zeros(num_nets), np.zeros(num_nets), 0
    for i in range(num_nets):
        net_list[i].eval()
    with torch.no_grad():
        for i, data in enumerate(loader):
            img_inputs, aud_inputs, labels = data['image'].to(device), data['audio'].to(device), data['label'].to(
                device)
            total += labels.size(0)
            for j in range(num_nets):
                if in_type_list[j] == 0:
                    outputs = net_list[j](img_inputs)
                elif in_type_list[j] == 1:
                    outputs = net_list[j](aud_inputs)
                elif in_type_list[j] == 2:
                    outputs = net_list[j](img_inputs, aud_inputs)
                else:
                    raise ValueError('the value of element in in_type_list should be 0,1,2\n')

                _, predicted = torch.max(outputs.detach(), 1)
                correct[j] += (predicted == labels).sum().item()
                criterion = torch.nn.CrossEntropyLoss()
                loss = criterion(outputs, labels)  # fix to CE loss
                v_loss[j] += loss.item()
    v_loss = v_loss
    val_acc = 100 * correct / total
    return v_loss, val_acc


def evaluate(loader, device, net, in_type, change_info=[None, None, None]):
    # type == 0: image input; type == 1: audio input; type == 2: both input
    correct, v_loss, total = 0, 0, 0
    net.eval()
    with torch.no_grad():
        for i, data in enumerate(loader):
            img_inputs, aud_inputs, labels = data['image'].to(device), data['audio'].to(device), data['label'].to(
                device)
            total += labels.size(0)
            if in_type == 0:
                outputs = net(img_inputs, *change_info)
            elif in_type == 1:
                outputs = net(aud_inputs, *change_info)
            elif in_type == 2:
                outputs = net(img_inputs, aud_inputs)
            else:
                raise ValueError('the value of element in in_type_list should be 0,1,2\n')

            _, predicted = torch.max(outputs.detach(), 1)
            correct += (predicted == labels).sum().item()
            criterion = torch.nn.CrossEntropyLoss()
            loss = criterion(outputs, labels)  # fix to CE loss
            v_loss += loss.item()
    v_loss = v_loss / len(loader)
    val_acc = 100 * correct / total
    return v_loss, val_acc


def evaluate_allacc(loader, device, net, in_type, change_info=[None, None, None]):
    _, train_acc = evaluate(loader['train'], device, net, in_type, change_info)
    _, val_acc = evaluate(loader['val'], device, net, in_type, change_info)
    _, test_acc = evaluate(loader['test'], device, net, in_type, change_info)
    return train_acc, val_acc, test_acc


def gen_data(data_dir, batch_size, num_workers):
    # load data or generate data if needed
    audio_dir = os.path.join(data_dir, 'aud_features')
    image_dir = os.path.join(data_dir, 'vid_features')

    data_file = os.path.join(data_dir, 'data_file')

    train_file = os.path.join(data_file, 'spa_dl.csv')
    val_file = os.path.join(data_file, 'spa_val.csv')
    test_file = os.path.join(data_file, 'spa_test.csv')

    loader = {'train': get_loader(train_file, audio_dir=audio_dir, image_dir=image_dir, batch_size=batch_size, num_workers=num_workers),
              'val': get_loader(val_file, audio_dir=audio_dir, image_dir=image_dir, batch_size=batch_size, num_workers=num_workers),
              'test': get_loader(test_file, audio_dir=audio_dir, image_dir=image_dir, batch_size=batch_size, num_workers=num_workers)}
    return loader


def train_network_baseline(in_type, epochs, loader, net, device, optimizer, change_info, save_model=False):
    val_acc_list, test_acc_list = [], []
    val_best_acc, test_best_acc = 0, 0
    model_best = net
    criterion = torch.nn.CrossEntropyLoss()
    # type == 0 train image network; type == 1 train audio network
    for epoch in range(epochs):
        net.train()
        # train_loss = 0.0
        for i, data in enumerate(loader['train']):
            img_inputs_cln, aud_inputs_cln, labels = data['image'], data['audio'], data['label']
            img_inputs_cln, aud_inputs_cln, labels = img_inputs_cln.to(device), aud_inputs_cln.to(device), labels.to(
                device)

            if in_type == 0:
                outputs = net(img_inputs_cln, *change_info)
            elif in_type == 1:
                outputs = net(aud_inputs_cln, *change_info)
            else:
                raise ValueError("Undefined training type")
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # train_loss += loss.item()

        train_loss, train_acc = evaluate(loader['train'], device, net, in_type)
        val_loss, val_acc = evaluate(loader['val'], device, net, in_type)
        test_loss, test_acc = evaluate(loader['test'], device, net, in_type)

        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)

        if val_acc > val_best_acc:
            val_best_acc = val_acc
            test_best_acc = test_acc
            model_best = deepcopy(net)

        print(f"Epoch | All epochs: {epoch} | {epochs}")
        print(f"Train Loss: {train_loss:.3f} | Val {val_loss:.3f} | Test Loss {test_loss:.3f}")
        print(f"Train | Val | Test Accuracy {train_acc:.3f} | {val_acc:.3f} | {test_acc:.3f}")
        print(f"Best Val | Test Accuracy | {val_best_acc:.3f} | {test_best_acc:.3f}\n", '-' * 70)

    if save_model:
        os.makedirs('./results', exist_ok=True)
        model_path = os.path.join('./results', 'umnet_mod_' + str(in_type) + '_acc_' + str(round(test_best_acc, 2)) + '.pkl')
        torch.save(model_best.state_dict(), model_path)
        print(f'Saving best model to {model_path}, Best Val | Test Accuracy | {val_best_acc:.3f} | {test_best_acc:.3f}')

    return model_best


def train_network_distill(stu_type, tea_model, epochs, loader, net, weight, device, optimizer, change_info_stu,
                          change_info_tea, save_model=False):
    val_best_acc, test_best_acc = 0, 0
    model_best = net
    criterion = torch.nn.CrossEntropyLoss()
    criterion2 = torch.nn.KLDivLoss(reduction='mean')
    # type == 0 train image network; type == 1 train audio network
    for epoch in range(epochs):
        net.train()
        tea_model.eval()
        # train_loss = 0.0
        loss1, loss2 = 0, 0
        for i, data in enumerate(loader['train']):
            img_inputs_cln, aud_inputs_cln, labels = data['image'], data['audio'], data['label']
            img_inputs_cln, aud_inputs_cln, labels = img_inputs_cln.to(device), aud_inputs_cln.to(device), labels.to(
                device)

            if stu_type == 0:
                outputs = net(img_inputs_cln, *change_info_stu)
                pseu_label = tea_model(aud_inputs_cln, *change_info_tea)
            elif stu_type == 1:
                outputs = net(aud_inputs_cln, *change_info_stu)
                pseu_label = tea_model(img_inputs_cln, *change_info_tea)
            else:
                raise ValueError("Undefined training type in distilled training")
            optimizer.zero_grad()
            tmp1 = weight[0] * criterion(outputs, labels)
            tmp2 = weight[1] * criterion2(torch.log_softmax(outputs, dim=1), torch.softmax(pseu_label, dim=1))
            loss = tmp1 + tmp2
            loss.backward()
            optimizer.step()
            # train_loss += loss.item()
            loss1 += tmp1.item()
            loss2 += tmp2.item()

        train_loss, train_acc = evaluate(loader['train'], device, net, stu_type)
        val_loss, val_acc = evaluate(loader['val'], device, net, stu_type)
        test_loss, test_acc = evaluate(loader['test'], device, net, stu_type)

        if val_acc > val_best_acc:
            val_best_acc = val_acc
            test_best_acc = test_acc
            model_best = deepcopy(net)

        print(f"Epoch | All epochs: {epoch} | {epochs}")
        print(f"Train Loss: {train_loss:.3f} | GT Loss {loss1 / len(loader['train']):.3f} | PL Loss {loss2 / len(loader['train']):.3f}")
        print(f"Train | Val | Test Accuracy {train_acc:.3f} | {val_acc:.3f} | {test_acc:.3f}")
        print(f"Best Val | Test Accuracy | {val_best_acc:.3f} | {test_best_acc:.3f}\n", '-' * 70)

    print(f'Training finish! Best Val | Test Accuracy | {val_best_acc:.3f} | {test_best_acc:.3f}')

    if save_model:
        os.makedirs('../results', exist_ok=True)
        model_path = os.path.join('../results', 'distillednet_mod_' + str(stu_type) + '_acc_' + str(round(test_best_acc, 2)) + '.pkl')
        torch.save(model_best.state_dict(), model_path)
        print(f'Saving best model to {model_path}, Best Val | Test Accuracy | {val_best_acc:.3f} | {test_best_acc:.3f}')

    return val_best_acc, test_best_acc


def train_network_for_overlap_tag(stu_type, tea_type, dist_loss, loader, epochs, learning_rate, device, save_model=False):
    criterion = torch.nn.CrossEntropyLoss()
    tea_model = ImageNet().to(device) if tea_type == 0 else AudioNet().to(device)
    stu_model = ImageNet().to(device) if stu_type == 0 else AudioNet().to(device)
    optimizer = torch.optim.SGD(list(tea_model.parameters()) + list(stu_model.parameters()), lr=learning_rate, momentum=0.9)
    for epoch in range(epochs):
        tea_model.train()
        stu_model.train()
        # train_loss = 0.0
        loss1, loss2, loss3 = 0, 0, 0
        for i, data in enumerate(loader['train']):
            img_inputs_cln, aud_inputs_cln, labels = data['image'], data['audio'], data['label']
            img_inputs_cln, aud_inputs_cln, labels = img_inputs_cln.to(device), aud_inputs_cln.to(device), labels.to(device)
            if stu_type == 0:
                outputs1 = stu_model(img_inputs_cln)
                outputs2 = tea_model(aud_inputs_cln)
            elif stu_type == 1:
                outputs1 = tea_model(img_inputs_cln)
                outputs2 = stu_model(aud_inputs_cln)
            else:
                raise ValueError("Undefined training type")

            optimizer.zero_grad()
            tmp1 = criterion(outputs1, labels)
            tmp2 = criterion(outputs2, labels)
            tmp3 = dist_loss(outputs1, outputs2)
            loss = tmp1 + tmp2 + tmp3
            loss.backward()
            optimizer.step()

            loss1 += tmp1.item()
            loss2 += tmp2.item()
            loss3 += tmp3.item()
            # train_loss += loss.item()
        print(f"Epoch {epoch} | loss1 {loss1 / len(loader['train']):.4f} | loss2 {loss2 / len(loader['train']):.4f} | "
              f"dist loss {loss3 / len(loader['train']):.4f}")

        _, train_acc = evaluate(loader['train'], device, tea_model, tea_type)
        _, train_acc_s = evaluate(loader['train'], device, stu_model, stu_type)
        print(f'teacher model train acc {train_acc:.2f} | student model {train_acc_s:.2f}')
        print('-' * 60)

    print('Finish training for overlap tag')

    if save_model:
        os.makedirs('./results', exist_ok=True)
        model_path_t = os.path.join('./results', 'teacher_mod_' + str(tea_type) + '_overlap.pkl')
        torch.save(tea_model.state_dict(), model_path_t)
        model_path_s = os.path.join('./results', 'student_mod_' + str(stu_type) + '_overlap.pkl')
        torch.save(stu_model.state_dict(), model_path_s)
        print(f"Saving teacher and student model to {model_path_t} and {model_path_s}")

    return tea_model, stu_model


def cal_overlap_tag(stu_type, loader, loader_fb, epochs, learning_rate, device, repeat_permute, place, first_time):
    tea_type = 1 - stu_type
    dist_loss = MSE()
    if first_time:
        tea_model, stu_model = train_network_for_overlap_tag(stu_type, tea_type, dist_loss, loader, epochs, learning_rate, device, save_model=True)
    else:
        tea_model = ImageNet().to(device) if tea_type == 0 else AudioNet().to(device)
        stu_model = ImageNet().to(device) if stu_type == 0 else AudioNet().to(device)
        tea_model.load_state_dict(torch.load('./results2/teacher_mod_' + str(tea_type) + '_overlap.pkl', map_location={"cuda:0": "cpu"}))
        stu_model.load_state_dict(torch.load('./results2/student_mod_' + str(stu_type) + '_overlap.pkl', map_location={"cuda:0": "cpu"}))
    _, train_acc = evaluate(loader['train'], device, tea_model, tea_type)
    _, train_acc_s = evaluate(loader['train'], device, stu_model, stu_type)
    print(f'teacher model train acc {train_acc:.2f} | student model {train_acc_s:.2f}')

    place_image = place_audio = place
    if stu_type == 0:
        stu_dim = stu_model.get_feature_dim(place_image)
        tea_dim = tea_model.get_feature_dim(place_audio)
    else:
        tea_dim = stu_model.get_feature_dim(place_image)
        stu_dim = tea_model.get_feature_dim(place_audio)
    print(f'teacher dim {tea_dim} | student dim {stu_dim}')

    overlap_tag_for_stu, overlap_tag_for_tea = np.zeros((repeat_permute, stu_dim)), np.zeros((repeat_permute, tea_dim))
    with torch.no_grad():
        tea_model.eval()
        stu_model.eval()
        for inner_iter, data in enumerate(loader_fb['train']):
            img_inputs_cln, aud_inputs_cln, labels = data['image'], data['audio'], data['label']
            img_inputs_cln, aud_inputs_cln, labels = img_inputs_cln.to(device), aud_inputs_cln.to(device), labels.to(device)
            x1_train = img_inputs_cln if stu_type == 0 else aud_inputs_cln
            x2_train = img_inputs_cln if tea_type == 0 else aud_inputs_cln
            h1_unpermu, h2_unpermu = stu_model(x1_train), tea_model(x2_train)
            for j in range(repeat_permute):
                # for index in range(overlap_tag_for_stu.shape[1]):
                    # fix tea feature, permute student feature
                    # h1 = stu_model(x1_train, 'permute', place_image if stu_type == 0 else place_audio, [index])
                    # overlap_tag_for_stu[j, index] = dist_loss(h1, h2_unpermu)

                for index in range(overlap_tag_for_tea.shape[1]):
                    h2 = tea_model(x2_train, 'permute', place_image if tea_type == 0 else place_audio, [index])
                    overlap_tag_for_tea[j, index] = dist_loss(h1_unpermu, h2)

            print('inner iter', inner_iter)

        # linear normalization max-> 1, min -> 0
        overlap_tag_for_stu_mean = overlap_tag_for_stu.mean(axis=0)
        overlap_tag_for_stu_mean = (overlap_tag_for_stu_mean - np.min(overlap_tag_for_stu_mean)) / (
                np.max(overlap_tag_for_stu_mean) - np.min(overlap_tag_for_stu_mean))

        overlap_tag_for_tea_mean = overlap_tag_for_tea.mean(axis=0)
        overlap_tag_for_tea_mean = (overlap_tag_for_tea_mean - np.min(overlap_tag_for_tea_mean)) / (
                np.max(overlap_tag_for_tea_mean) - np.min(overlap_tag_for_tea_mean))

        print('Saving overlap tag for teacher')
        np.save('./results/overlap_tag_teacher_place' + str(place) + '_repeat' + str(repeat_permute) + '.npy',
                overlap_tag_for_tea_mean)
        return overlap_tag_for_stu_mean, overlap_tag_for_tea_mean


def viz_overlap_tag():
    data = np.load('./results/overlap_tag_teacher_place5_repeat10.npy')
    dim = 128
    # data = np.load('./results/overlap_tag_teacher_place4_repeat100.npy')
    # dim = 1024

    data_sorted = np.sort(data)
    rem_dim = int(dim * 0.25)
    data_mode1 = data_sorted[0:rem_dim]
    data_mode2 = data_sorted[-rem_dim:]

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.scatter(np.arange(dim), data, marker='o', s=2)
    plt.subplot(1, 2, 2)
    plt.scatter(np.arange(dim), data_sorted, marker='o', s=2)
    plt.scatter(np.arange(rem_dim), data_mode1, marker='o', c='r', s=2)
    plt.scatter(np.arange(dim-rem_dim, dim), data_mode2, marker='o', c='r', s=2)
    plt.show()


def write_log(filename, args, data_mean, data_std):
    file = codecs.open(filename, 'a')
    writer = csv.writer(file)
    writer.writerow(
        [args.mode, args.ratio, args.gt_weight, args.pl_weight, args.num_runs, args.num_epochs, data_mean[0], data_std[0], data_mean[1], data_std[1]])
    writer.writerow('')
    file.close()


if __name__ == '__main__':
    viz_overlap_tag()
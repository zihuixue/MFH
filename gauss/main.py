import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LinearClassifier(nn.Module):
    def __init__(self, input_dim, output_dim=2):
        super(LinearClassifier, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.linear(x)
        return x


def evaluate(x, y, x_test, y_test, model):
    model.eval()
    with torch.no_grad():
        output = model(x)
        _, predicted = torch.max(output.data, 1)
        train_acc = (y == predicted).sum() / y.shape[0]

        output = model(x_test)
        _, predicted = torch.max(output.data, 1)
        test_acc = (y_test == predicted).sum() / y_test.shape[0]
    return train_acc.item(), test_acc.item()


def train(x, y, x_test, y_test, device, n_epoch=1000, eval_epoch=1, plot=False):
    model = LinearClassifier(input_dim=x.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    x, y, x_test, y_test, model = map(lambda x: x.to(device), (x, y, x_test, y_test, model))
    train_loss_curve = []
    train_acc_curve = []
    test_acc_curve = []
    for epoch in range(n_epoch):
        model.train()
        output = model(x)
        loss = criterion(output, y)
        train_loss_curve.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if epoch % eval_epoch == 0 or epoch == n_epoch - 1:
            train_acc, test_acc = evaluate(x, y, x_test, y_test, model)
            train_acc_curve.append(train_acc)
            test_acc_curve.append(test_acc)
            # print(f"epoch: {epoch}, loss: {loss:.3f} train acc {train_acc:.4f} test acc {test_acc:.4f}")

        if epoch == n_epoch - 1 and plot:
            plt.subplot(1, 2, 1)
            plt.plot(train_loss_curve)
            plt.subplot(1, 2, 2)
            plt.plot(train_acc_curve)
            plt.plot(test_acc_curve)
            plt.show()

    return model, test_acc


# cross-modal kd, teacher : x1 as input, student: x2 as input
def train_kd(x1, x2, y2, x2_test, y2_test, teacher_model, weight, device, n_epoch=1000, eval_epoch=1, plot=False):
    teacher_model.eval()
    model = LinearClassifier(input_dim=x2.shape[1])
    criterion = torch.nn.CrossEntropyLoss()
    criterion_pl = torch.nn.KLDivLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    x1, x2, y2, x2_test, y2_test, teacher_model, model = map(lambda x: x.to(device),
                                                             (x1, x2, y2, x2_test, y2_test, teacher_model, model))
    train_loss_curve = []
    train_acc_curve = []
    test_acc_curve = []
    for epoch in range(n_epoch):
        model.train()
        pl = teacher_model(x1)
        output = model(x2)
        loss_gt, loss_pl = criterion(output, y2), criterion_pl(torch.log_softmax(output, dim=1),
                                                               torch.softmax(pl, dim=1))
        loss = weight[0] * loss_gt + weight[1] * loss_pl
        train_loss_curve.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if epoch % eval_epoch == 0 or epoch == n_epoch - 1:
            # t_train_acc, t_test_acc = evaluate(xs, y, xs_test, y_test, teacher_model)
            train_acc, test_acc = evaluate(x2, y2, x2_test, y2_test, model)
            train_acc_curve.append(train_acc)
            test_acc_curve.append(test_acc)
            # print(f"teacher train acc {t_train_acc:.4f} test acc {t_test_acc:.4f}")
            # print(f"epoch: {epoch}, loss_gt: {loss_gt:.3f}, loss_pl: {loss_pl:.3f}, train acc {train_acc:.4f} test acc {test_acc:.4f}")

        if epoch == n_epoch - 1 and plot:
            plt.subplot(1, 2, 1)
            plt.plot(train_loss_curve)
            plt.subplot(1, 2, 2)
            plt.plot(train_acc_curve)
            plt.plot(test_acc_curve)
            plt.show()

    return test_acc


def gen_mm_data(a, n, mode, x1_dim=-1, x2_dim=-1, xs1_dim=-1, xs2_dim=-1, overlap_dim=-1):
    """
    :param a: the separating hyperplane $\delta$ in Eq.(27) in the paper
    :param n: data number
    :param mode: a or b
    :param x1_dim: modality 1 feature dimension
    :param x2_dim: modality 2 feature dimension
    :param xs1_dim: modality 1 decisive feature dimension
    :param xs2_dim: modality 2 decisive feature dimension
    :param overlap_dim:
    :return:
    """
    if mode == 'increase_gamma':
        xs = np.random.randn(n, xs1_dim + xs2_dim)     # decisive features
        a = a[0:xs1_dim + xs2_dim]                     # separating hyperplane
        y = (np.dot(xs, a) > 0).ravel()                # decisive features xs -> label y

        # x2, 0:xs2_dim-decisive features, other dim-gaussian noise
        x2 = np.random.randn(n, x2_dim)
        x2[:, 0:xs2_dim] = xs[:, 0:xs2_dim]

        # x1, among all x1_dim channels, xs1_dim channels are decisive and other are noise; among all xs1_dim decisive
        # channels, overlap_dim are shared between x1 and x2
        x1 = np.random.randn(n, x1_dim)
        x1[:, xs2_dim - overlap_dim:xs2_dim - overlap_dim + xs1_dim] = \
            xs[:, xs2_dim - overlap_dim:xs2_dim - overlap_dim + xs1_dim]

    elif mode == 'increase_alpha':
        # x1 and x2 overlapping region fixed = xs1_dim
        xs = np.random.randn(n, x1_dim)   # x1_dim-decisive features channel number
        a = a[0:x1_dim]
        y = (np.dot(xs, a) > 0).ravel()

        # x2, 0:xs2_dim-decisive features, other dim-gaussian noise
        x2 = np.random.randn(n, x2_dim)
        x2[:, 0:xs2_dim] = xs[:, 0:xs2_dim]

        # x1: 0:xs1_dim+xs_2dim-decisive features, other dim-gaussian noise
        x1 = np.random.randn(n, x1_dim)
        x1[:, 0:xs1_dim + xs2_dim] = xs[:, 0:xs1_dim + xs2_dim]

    else:
        raise NotImplementedError

    return torch.Tensor(x1), torch.Tensor(x2), torch.LongTensor(y)


def run_mm(seed_num, vari_dim, mode):
    seed(seed_num)
    n_train = 200
    n_test = 1000
    d = 500

    # data generation
    a = np.random.randn(d)   # a random separating hyperplane

    if mode == 'increase_gamma':
        x1_train, x2_train, y_train = gen_mm_data(a, n_train, mode, x1_dim=25, x2_dim=50,
                                                  xs1_dim=10, xs2_dim=10, overlap_dim=vari_dim)
        seed(seed_num + 1)
        x1_test, x2_test, y_test = gen_mm_data(a, n_test, mode, x1_dim=25, x2_dim=50,
                                               xs1_dim=10, xs2_dim=10, overlap_dim=vari_dim)
    elif mode == 'increase_alpha':
        x1_train, x2_train, y_train = gen_mm_data(a, n_train, mode, x1_dim=50, x2_dim=50, xs1_dim=vari_dim, xs2_dim=10)
        seed(seed_num + 1)
        x1_test, x2_test, y_test = gen_mm_data(a, n_test, mode, x1_dim=50, x2_dim=50, xs1_dim=vari_dim, xs2_dim=10)

    else:
        raise NotImplementedError

    # train a unimodal teacher model that takes input from modality 1
    teacher_model, teacher_acc = train(x1_train, y_train, x1_test, y_test, device=device, n_epoch=1000)
    # cross-modal KD: the x1 teacher distills knowledge to a x2 student
    kd_student_acc = train_kd(x1_train, x2_train, y_train, x2_test, y_test, teacher_model, [1.0, 1.0], device)
    # baseline: train a model from modality 2 without KD
    _, no_kd_baseline_acc = train(x2_train, y_train, x2_test, y_test, device=device, n_epoch=1000)
    return teacher_acc, no_kd_baseline_acc, kd_student_acc


def exp1(seed, n_runs):
    overlap_dim_list = [0, 2, 4, 6, 8, 10]
    for overlap_dim in overlap_dim_list:
        gamma = overlap_dim / (20 - overlap_dim)
        acc_np = np.zeros((n_runs, 3))
        for i in range(n_runs):
            acc_np[i, :] = run_mm(seed + i, overlap_dim, 'increase_gamma')
        delta = np.round((acc_np[:, 2] - acc_np[:, 1]) * 100, 2)
        log_mean = np.mean(acc_np, axis=0) * 100
        print(f'gamma = {gamma:.2f}')
        print(f'Teacher acc {log_mean[0]:.2f}')
        print(f'No KD acc {log_mean[1]:.2f}')
        print(f'KD student acc {log_mean[2]:.2f}')
        print(f'Delta: {np.mean(delta):.2f} ± {np.std(delta):.2f}')
        print('-' * 60)


def exp2(seed, n_runs):
    xs1_dim_list = [0, 10, 20, 30, 40]
    for xs1_dim in xs1_dim_list:
        alpha = xs1_dim / (xs1_dim + 10)
        gamma = 1 - alpha
        acc_np = np.zeros((n_runs, 3))
        for i in range(n_runs):
            acc_np[i, :] = run_mm(seed + i, xs1_dim, 'increase_alpha')
        delta = np.round((acc_np[:, 2] - acc_np[:, 1]) * 100, 2)
        log_mean = np.mean(acc_np, axis=0) * 100
        print(f'alpha = {alpha} | gamma = {gamma}')
        print(f'Teacher acc {log_mean[0]:.2f}')
        print(f'No KD acc {log_mean[1]:.2f}')
        print(f'KD student acc {log_mean[2]:.2f}')
        print(f'Delta: {np.mean(delta):.2f} ± {np.std(delta):.2f}')
        print('-' * 60)


if __name__ == '__main__':
    # x_1 and x_2 here correspond to x^a and x^b in the main paper, respectively.
    device = torch.device("cpu")
    print('Exp 1: increase gamma')
    exp1(seed=0, n_runs=10)   # Figure 2 in the paper
    print('-' * 80)

    print('Exp 2: increase alpha (decrease gamma)')
    exp2(seed=0, n_runs=10)   # Figure 3 in the paper
    print('-' * 80)


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(42)

data_train = np.load("./data/train1989.npz")
data_test = np.load("./data/test1989.npz")

Xtr = torch.from_numpy(data_train["X"])
Ytr = torch.from_numpy(data_train["Y"])
Xte = torch.from_numpy(data_test["X"])
Yte = torch.from_numpy(data_test["Y"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

Xtr = Xtr.to(device).float()
Ytr = Ytr.to(device)
Xte = Xte.to(device).float()
Yte = Yte.to(device)


def init_weights(fan_in, *shape):
    weight = (torch.rand(*shape) - 0.5) * 2 * 2.4 / fan_in
    return nn.Parameter(weight)


macs = 0
acts = 0

H1w = init_weights(5 * 5 * 1, 12, 1, 5, 5)
H1b = torch.zeros(12, 8, 8)
assert H1w.numel() + H1b.numel() == 1068
macs = 5 * 5 * 8 * 8 * 12 + 768
acts += (8 * 8) * 12
assert macs == 19968

H2w = init_weights(5 * 5 * 8, 12, 8, 5, 5)
H2b = torch.zeros(12, 4, 4)
assert H2w.numel() + H2b.numel() == 2592
macs += (5 * 5 * 8) * (4 * 4) * 12 + 192
acts += (4 * 4) * 12
assert macs == 58560

H3w = init_weights(4 * 4 * 12, 4 * 4 * 12, 30)
H3b = torch.zeros(30)
assert H3w.numel() + H3b.numel() == 5790
macs += (4 * 4 * 12) * 30 + 30
acts += 30
assert macs == 64350

outw = init_weights(30, 30, 10)
outb = -torch.ones(10)
assert outw.numel() + outb.numel() == 310
macs += 30 * 10 + 10
acts += 10
assert macs == 64660

# --- H1 Feature Map Storage ---
h1_feature_maps = {}  # will store {'pre_act': tensor, 'post_act': tensor}


def forward(x, capture_h1=False):
    x = F.pad(x, (2, 2, 2, 2), mode='constant', value=-1.0)

    h1_pre = F.conv2d(x, H1w, stride=2) + H1b
    h1_post = torch.tanh(h1_pre)

    if capture_h1:
        h1_feature_maps['pre_act'] = h1_pre.detach().cpu()
        h1_feature_maps['post_act'] = h1_post.detach().cpu()

    x = h1_post
    x = F.pad(x, (2, 2, 2, 2), mode='constant', value=-1.0)
    slice1 = F.conv2d(x[:, 0:8], H2w[0:4], stride=2)
    slice2 = F.conv2d(x[:, 4:12], H2w[4:8], stride=2)
    slice3 = F.conv2d(torch.cat([x[:, 0:4], x[:, 8:12]], dim=1), H2w[8:12], stride=2)
    x = torch.cat((slice1, slice2, slice3), dim=1) + H2b
    x = torch.tanh(x)

    x = x.flatten(start_dim=1)
    x = x @ H3w + H3b
    x = torch.tanh(x)

    x = x @ outw + outb
    x = torch.tanh(x)

    return x


def visualize_h1_feature_maps(sample_idx=0, after_epoch=None):
    """
    Capture and plot H1 feature maps for a single input sample.

    - sample_idx: which training sample to use
    - after_epoch: optional label for the plot title (e.g. epoch number)
    """
    x = Xtr[sample_idx:sample_idx + 1]
    with torch.no_grad():
        forward(x, capture_h1=True)

    pre = h1_feature_maps['pre_act'][0]  # (12, 8, 8)
    post = h1_feature_maps['post_act'][0]  # (12, 8, 8)

    fig, axes = plt.subplots(2, 12, figsize=(24, 5))
    epoch_label = f" — After Epoch {after_epoch}" if after_epoch is not None else " (Before Training)"

    for i in range(12):
        # Pre-activation
        ax = axes[0, i]
        im = ax.imshow(pre[i].numpy(), cmap='RdBu_r', aspect='auto')
        ax.set_title(f"Filter {i + 1}", fontsize=8)
        ax.axis('off')
        if i == 0:
            ax.set_ylabel("Pre-tanh", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Post-activation
        ax = axes[1, i]
        im = ax.imshow(post[i].numpy(), cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        ax.axis('off')
        if i == 0:
            ax.set_ylabel("Post-tanh", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"H1 Feature Maps — Sample #{sample_idx}{epoch_label}", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()


def train_step(optimizer, x, y):
    optimizer.zero_grad()
    yhat = forward(x)
    loss = torch.mean((y - yhat) ** 2)
    loss.backward()
    optimizer.step()
    return loss.item()


def eval_split(split, X_tr, Y_tr, X_te, Y_te):
    X, Y = (X_tr, Y_tr) if split == 'train' else (X_te, Y_te)
    yhat = forward(X)
    loss = torch.mean((Y - yhat) ** 2)
    err = torch.mean((Y.argmax(dim=1) != yhat.argmax(dim=1)).float())
    misses = int(err.item() * Y.shape[0])
    print(f"eval: split {split:5s}. loss {loss.item():e}. error {err.item() * 100:.2f}%. misses: {misses}")
    return loss.item(), err.item(), misses


def plot_metrics(train_errs, test_errs):
    plt.figure(figsize=(10, 5))
    plt.plot(train_errs, label='Train Error', color='blue', marker='o')
    plt.plot(test_errs, label='Test Error', color='orange', marker='s')
    plt.title('Training vs Test Error')
    plt.xlabel('Epoch')
    plt.ylabel('Error (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()


# --- Visualize BEFORE training ---
visualize_h1_feature_maps(sample_idx=0, after_epoch=None)

# Training
num_epochs = 23
params = [H1w, H1b, H2w, H2b, H3w, H3b, outw, outb]
optimizer = torch.optim.SGD(params, lr=0.03)

# Epochs at which to visualize feature maps
visualize_at_epochs = {1, 5, 10, 23}

history = {'train_err': [], 'test_err': []}
for pass_num in range(num_epochs):
    for step_num in range(Xtr.shape[0]):
        x, y = Xtr[step_num:step_num + 1], Ytr[step_num:step_num + 1]
        loss = train_step(optimizer, x, y)

    epoch = pass_num + 1
    print(f"\nEpoch {epoch}/{num_epochs}")
    train_loss, train_err, train_misses = eval_split('train', Xtr, Ytr, Xte, Yte)
    test_loss, test_err, test_misses = eval_split('test', Xtr, Ytr, Xte, Yte)
    history['train_err'].append(train_err * 100)
    history['test_err'].append(test_err * 100)

    # --- Visualize H1 feature maps at selected epochs ---
    if epoch in visualize_at_epochs:
        visualize_h1_feature_maps(sample_idx=0, after_epoch=epoch)

plot_metrics(history['train_err'], history['test_err'])

def show_sample(sample_idx=0):
    x = Xtr[sample_idx].cpu().numpy()  # shape (1, 16, 16)
    label = Ytr[sample_idx].cpu().numpy()  # one-hot vector
    digit = label.argmax()

    plt.figure(figsize=(3, 3))
    plt.imshow(x[0], cmap='gray')
    plt.title(f"Sample #{sample_idx} — Digit: {digit}")
    plt.axis('off')
    plt.show()

show_sample(0)

def predict_sample(sample_idx=0):
    x = Xtr[sample_idx:sample_idx+1]
    with torch.no_grad():
        yhat = forward(x)
    predicted = yhat.argmax(dim=1).item()
    true_label = Ytr[sample_idx].argmax().item()
    print(f"Sample #{sample_idx} — True label: {true_label}, Predicted: {predicted}, {'✅' if predicted == true_label else '❌'}")

predict_sample(0)
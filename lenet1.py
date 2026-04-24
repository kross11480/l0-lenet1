import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(42)

data_train = np.load("./data/train1989.npz")
data_test = np.load("./data/test1989.npz")

# Access arrays inside
Xtr = torch.from_numpy(data_train["X"])
Ytr = torch.from_numpy(data_train["Y"])

Xte = torch.from_numpy(data_test["X"])
Yte = torch.from_numpy(data_test["Y"])

print(f"Training data shape: {Xtr.shape}")
print(f"Training labels shape: {Ytr.shape}")
print(f"Test data shape: {Xte.shape}")
print(f"Test labels shape: {Yte.shape}")

# ---- GPU SETUP ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Move data to device
Xtr = Xtr.to(device).float()
Ytr = Ytr.to(device)
Xte = Xte.to(device).float()
Yte = Yte.to(device)


def init_weights(fan_in, *shape):
    """Weight initialization as described in the paper"""
    weight = (torch.rand(*shape) - 0.5) * 2 * 2.4 / fan_in
    return nn.Parameter(weight)

macs = 0  # keep track of MACs (multiply accumulates)
acts = 0  # keep track of number of activations

##TODO
# H1 layer parameters
H1w = init_weights(5*5*1, 12, 1, 5, 5)
H1b = torch.zeros(12, 8, 8)
assert H1w.numel() + H1b.numel() == 1068
macs = 5*5*8*8*12 + 768
acts += (8*8) * 12
assert macs == 19968

# H2 layer parameters
H2w = init_weights(5*5*8, 12, 8, 5, 5)
H2b = torch.zeros(12, 4, 4)
assert H2w.numel() + H2b.numel() == 2592
macs += (5*5*8) * (4*4) * 12 + 192
acts += (4*4) * 12
assert macs == 58560

##TODO
# H3 fully connected layer
H3w = init_weights(4*4*12, 4*4*12, 30)
H3b = torch.zeros(30)
assert H3w.numel() + H3b.numel() == 5790
macs += (4*4*12) * 30 + 30
acts += 30
assert macs == 64350

##TODO
# Output layer
outw = init_weights(30, 30, 10)
outb = -torch.ones(10)
assert outw.numel() + outb.numel() == 310
macs += 30 * 10 + 10
acts += 10
assert macs == 64660

def forward(x):
    x = F.pad(x, (2, 2, 2, 2), mode='constant', value=-1.0)
    # TODO H1 layer: Apply convolution with H1w and add H1b
    # Then apply activation and pooling as needed
    x = F.conv2d(x, H1w, stride=2) + H1b
    x = torch.tanh(x)

    # H2 layer
    x = F.pad(x, (2, 2, 2, 2), mode='constant', value=-1.0)
    slice1 = F.conv2d(x[:, 0:8], H2w[0:4], stride=2)
    slice2 = F.conv2d(x[:, 4:12], H2w[4:8], stride=2)
    slice3 = F.conv2d(torch.cat([x[:, 0:4], x[:, 8:12]], dim=1), H2w[8:12], stride=2)
    x = torch.cat((slice1, slice2, slice3), dim=1) + H2b
    x = torch.tanh(x)

    # TODO: H3 fully connected layer
    x = x.flatten(start_dim=1) # (1, 12*4*4)
    x = x @ H3w + H3b
    x = torch.tanh(x)

    #TODO x is now shape (1, 30)
    x = x @ outw + outb
    x = torch.tanh(x)

    #TODO assert final shape (1, 10)
    return x

def train_step(optimizer, x, y):
    """Single training step"""
    # Zero gradients
    optimizer.zero_grad()

    # TODO: Forward pass and compute loss
    yhat = forward(x)
    loss = torch.mean((y - yhat) ** 2)

    # Backward pass
    loss.backward()

    # Update parameters
    optimizer.step()

    return loss.item()

def eval_split(split, X_tr, Y_tr, X_te, Y_te):
    X, Y = (X_tr, Y_tr) if split == 'train' else (X_te, Y_te)
    yhat = forward(X)

    loss = torch.mean((Y - yhat)**2)
    err = torch.mean((Y.argmax(dim=1) != yhat.argmax(dim=1)).float())
    misses = int(err.item() * Y.shape[0])

    print(f"eval: split {split:5s}. loss {loss.item():e}. error {err.item()*100:.2f}%. misses: {misses}")
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

# Parameters
num_epochs = 23
params = [H1w, H1b, H2w, H2b, H3w, H3b, outw, outb]
optimizer = torch.optim.SGD(params, lr=0.03)

history = {'train_err': [], 'test_err': []}
for pass_num in range(num_epochs):
    # Perform one epoch of training
    for step_num in range(Xtr.shape[0]):
        # Fetch a single example into a batch of 1
        x, y = Xtr[step_num:step_num+1], Ytr[step_num:step_num+1]

        # Training step
        loss = train_step(optimizer, x, y)

    # After each epoch evaluate the train and test error/metrics
    print(f"\nEpoch {pass_num + 1}/{num_epochs}")
    train_loss, train_err, train_misses = eval_split('train', Xtr, Ytr, Xte, Yte)
    test_loss, test_err, test_misses = eval_split('test', Xtr, Ytr, Xte, Yte)
    history['train_err'].append(train_err * 100) # Store as percentage
    history['test_err'].append(test_err * 100)

plot_metrics(history['train_err'], history['test_err'])
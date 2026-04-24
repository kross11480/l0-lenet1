# Note: Does not work for larger batch size to due to connectivity in H2 layer.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

torch.manual_seed(42)

# ---- LOAD DATA ----
data_train = np.load("./data/train1989.npz")
data_test = np.load("./data/test1989.npz")

Xtr = torch.from_numpy(data_train["X"]).float()
Ytr = torch.from_numpy(data_train["Y"])
Xte = torch.from_numpy(data_test["X"]).float()
Yte = torch.from_numpy(data_test["Y"])

# ---- DEVICE ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

Xtr, Ytr = Xtr.to(device), Ytr.to(device)
Xte, Yte = Xte.to(device), Yte.to(device)

# ---- INIT ----
def init_weights(fan_in, *shape):
    return nn.Parameter((torch.rand(*shape, device=device) - 0.5) * 2 * 2.4 / fan_in)

H1w = init_weights(5*5*1, 12, 1, 5, 5)
H1b = nn.Parameter(torch.zeros(12, 1, 1, device=device))

H2w = init_weights(5*5*8, 12, 8, 5, 5)
H2b = nn.Parameter(torch.zeros(12, 1, 1, device=device))

H3w = init_weights(4*4*12, 4*4*12, 30)
H3b = nn.Parameter(torch.zeros(30, device=device))

outw = init_weights(30, 30, 10)
outb = nn.Parameter(-torch.ones(10, device=device))

params = [H1w, H1b, H2w, H2b, H3w, H3b, outw, outb]

# ---- FORWARD ----
def forward(x):
    x = F.pad(x, (2,2,2,2), value=-1.0)
    x = torch.tanh(F.conv2d(x, H1w, stride=2) + H1b)

    x = F.pad(x, (2,2,2,2), value=-1.0)
    s1 = F.conv2d(x[:,0:8], H2w[0:4], stride=2)
    s2 = F.conv2d(x[:,4:12], H2w[4:8], stride=2)
    s3 = F.conv2d(torch.cat([x[:,0:4], x[:,8:12]], dim=1), H2w[8:12], stride=2)
    x = torch.tanh(torch.cat((s1,s2,s3), dim=1) + H2b)

    x = x.flatten(start_dim=1)
    x = torch.tanh(x @ H3w + H3b)
    x = torch.tanh(x @ outw + outb)
    return x

# ---- BATCH TRAIN STEP ----
def train_step(opt, xb, yb):
    opt.zero_grad()
    yhat = forward(xb)
    loss = torch.mean((yb - yhat)**2)
    loss.backward()
    opt.step()
    return loss.item()

# ---- EVAL ----
def evaluate(X, Y):
    with torch.no_grad():
        yhat = forward(X)
        loss = torch.mean((Y - yhat)**2)
        err = torch.mean((Y.argmax(1) != yhat.argmax(1)).float())
    return loss.item(), err.item()

# ---- BATCH TRAINING ----
batch_size = 16
epochs = 23
optimizer = torch.optim.SGD(params, lr=0.03)

def get_batches(X, Y, bs):
    idx = torch.randperm(X.shape[0])
    for i in range(0, X.shape[0], bs):
        j = idx[i:i+bs]
        yield X[j], Y[j]

epoch_times = []

for ep in range(epochs):
    start = time.time()

    for xb, yb in get_batches(Xtr, Ytr, batch_size):
        train_step(optimizer, xb, yb)

    torch.cuda.synchronize() if device.type == "cuda" else None
    t = time.time() - start
    epoch_times.append(t)

    train_loss, train_err = evaluate(Xtr, Ytr)
    test_loss, test_err = evaluate(Xte, Yte)

    print(f"Epoch {ep+1}: time={t:.3f}s "
          f"train_err={train_err*100:.2f}% test_err={test_err*100:.2f}%")

# ---- THROUGHPUT ----
total_samples = Xtr.shape[0] * epochs
total_time = sum(epoch_times)
print("\nThroughput:", total_samples / total_time, "samples/sec")

# ---- FORWARD BENCH ----
def benchmark_forward(n=100):
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(n):
        _ = forward(Xte[:batch_size])
    if device.type == "cuda":
        torch.cuda.synchronize()
    print("Avg forward time:", (time.time()-start)/n)

benchmark_forward()

if device.type == "cuda":
    print("Max GPU memory (MB):", torch.cuda.max_memory_allocated()/1024**2)

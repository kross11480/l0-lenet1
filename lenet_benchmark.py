import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.profiler as profiler
import matplotlib.pyplot as plt
import os

torch.manual_seed(42)

# ── Data loading ──────────────────────────────────────────────────────────────
data_train = np.load("./data/train1989.npz")
data_test  = np.load("./data/test1989.npz")

Xtr = torch.from_numpy(data_train["X"])
Ytr = torch.from_numpy(data_train["Y"])
Xte = torch.from_numpy(data_test["X"])
Yte = torch.from_numpy(data_test["Y"])

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

Xtr = Xtr.to(device).float()
Ytr = Ytr.to(device)
Xte = Xte.to(device).float()
Yte = Yte.to(device)

# ── Weight init ───────────────────────────────────────────────────────────────
def init_weights(fan_in, *shape):
    w = (torch.rand(*shape) - 0.5) * 2 * 2.4 / fan_in
    return nn.Parameter(w)

H1w = init_weights(5*5*1,   12, 1,  5, 5)
H1b = torch.zeros(12, 8, 8)

H2w = init_weights(5*5*8,   12, 8,  5, 5)
H2b = torch.zeros(12, 4, 4)

H3w = init_weights(4*4*12,  4*4*12, 30)
H3b = torch.zeros(30)

outw = init_weights(30,     30, 10)
outb = -torch.ones(10)

# ── Forward pass ──────────────────────────────────────────────────────────────
def forward(x):
    # H1 – conv + tanh
    x = F.pad(x, (2, 2, 2, 2), mode='constant', value=-1.0)
    x = F.conv2d(x, H1w, stride=2) + H1b
    x = torch.tanh(x)

    # H2 – sparse conv + tanh
    x = F.pad(x, (2, 2, 2, 2), mode='constant', value=-1.0)
    slice1 = F.conv2d(x[:, 0:8],                          H2w[0:4],  stride=2)
    slice2 = F.conv2d(x[:, 4:12],                         H2w[4:8],  stride=2)
    slice3 = F.conv2d(torch.cat([x[:, 0:4], x[:, 8:12]], dim=1), H2w[8:12], stride=2)
    x = torch.cat((slice1, slice2, slice3), dim=1) + H2b
    x = torch.tanh(x)

    # H3 – FC + tanh
    x = x.flatten(start_dim=1)
    x = x @ H3w + H3b
    x = torch.tanh(x)

    # Output layer
    x = x @ outw + outb
    x = torch.tanh(x)
    return x

def train_step(optimizer, x, y):
    optimizer.zero_grad()
    yhat  = forward(x)
    loss  = torch.mean((y - yhat) ** 2)
    loss.backward()
    optimizer.step()
    return loss.item()

# ── Parameters & optimizer ────────────────────────────────────────────────────
params    = [H1w, H1b, H2w, H2b, H3w, H3b, outw, outb]
optimizer = torch.optim.SGD(params, lr=0.03)

# ═══════════════════════════════════════════════════════════════════════════════
#  PROFILING SECTION
#  We profile a small fixed number of warm-up + active steps so the results
#  are representative without needing to run the full 23-epoch training.
# ═══════════════════════════════════════════════════════════════════════════════

PROFILE_STEPS   = 20   # active steps captured in the trace
WARMUP_STEPS    = 5    # steps run before recording starts
TRACE_DIR       = "./prof_trace"   # Chrome trace written here
os.makedirs(TRACE_DIR, exist_ok=True)

print("\n" + "="*60)
print("Running PyTorch profiler …")
print(f"  Warm-up steps : {WARMUP_STEPS}")
print(f"  Active steps  : {PROFILE_STEPS}")
print(f"  Trace output  : {TRACE_DIR}/")
print("="*60 + "\n")

# torch.profiler.schedule controls which steps are warm-up, active, etc.
schedule = profiler.schedule(
    wait=0,          # steps to skip entirely
    warmup=WARMUP_STEPS,
    active=PROFILE_STEPS,
    repeat=1,
)

with profiler.profile(
    activities=[
        profiler.ProfilerActivity.CPU,
        # Add CUDA activity if on GPU:
        *(
            [profiler.ProfilerActivity.CUDA]
            if torch.cuda.is_available() else []
        ),
    ],
    schedule=schedule,
    on_trace_ready=profiler.tensorboard_trace_handler(TRACE_DIR),
    record_shapes=True,    # log tensor shapes per op
    profile_memory=True,   # log memory allocation/deallocation
    with_stack=False,      # call-stack info (slow; flip to True for deep dives)
) as prof:

    step = 0
    for idx in range(WARMUP_STEPS + PROFILE_STEPS):
        x, y = Xtr[idx % Xtr.shape[0] : idx % Xtr.shape[0] + 1], \
               Ytr[idx % Xtr.shape[0] : idx % Xtr.shape[0] + 1]

        # Label each region so the trace is easy to read in Chrome / TensorBoard
        with torch.profiler.record_function("forward_pass"):
            yhat = forward(x)

        with torch.profiler.record_function("loss_compute"):
            loss = torch.mean((y - yhat) ** 2)

        with torch.profiler.record_function("backward_pass"):
            optimizer.zero_grad()
            loss.backward()

        with torch.profiler.record_function("optimizer_step"):
            optimizer.step()

        prof.step()   # advance the profiler schedule

# ── Print the summary table ───────────────────────────────────────────────────
print("\n── Top 20 ops by CPU time (self) ──────────────────────────────────────")
print(
    prof.key_averages().table(
        sort_by="self_cpu_time_total",
        row_limit=20,
    )
)

if torch.cuda.is_available():
    print("\n── Top 20 ops by CUDA time (self) ─────────────────────────────────────")
    print(
        prof.key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=20,
        )
    )

# ── Breakdown by the named regions we labelled above ─────────────────────────
print("\n── Averages grouped by input shape ─────────────────────────────────────")
print(
    prof.key_averages(group_by_input_shape=True).table(
        sort_by="self_cpu_time_total",
        row_limit=15,
    )
)

# ── Quick matplotlib bar chart of the top ops ─────────────────────────────────
key_avgs = prof.key_averages()
key_avgs.sort(key=lambda x: x.self_cpu_time_total, reverse=True)

top_n     = 12
names     = [e.key for e in key_avgs[:top_n]]
cpu_times = [e.self_cpu_time_total / 1e3 for e in key_avgs[:top_n]]  # µs → ms

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(names[::-1], cpu_times[::-1], color='steelblue')
ax.bar_label(bars, fmt='%.2f ms', padding=4, fontsize=9)
ax.set_xlabel("Self CPU time (ms, summed over profiled steps)")
ax.set_title(f"Top {top_n} ops by self-CPU time\n(device: {device}, {PROFILE_STEPS} steps)")
ax.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("./prof_trace/top_ops.png", dpi=150)
plt.show()
print("\nBar chart saved → ./prof_trace/top_ops.png")

# ── How to open the Chrome trace ─────────────────────────────────────────────
print("""
┌──────────────────────────────────────────────────────────────┐
│  VIEWING THE FULL TRACE                                       │
│                                                               │
│  Option A – Chrome / Edge                                     │
│    1. Open chrome://tracing   (or edge://tracing)            │
│    2. Click "Load"                                            │
│    3. Select the .json file inside ./prof_trace/              │
│                                                               │
│  Option B – TensorBoard                                       │
│    pip install torch-tb-profiler                             │
│    tensorboard --logdir ./prof_trace                          │
│    Then open http://localhost:6006/#pytorch_profiler          │
└──────────────────────────────────────────────────────────────┘
""")
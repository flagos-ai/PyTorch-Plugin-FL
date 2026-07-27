# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Live multi-GPU test for ProcessGroupFlagOS (run on the 8xA100 box).

Launches N processes, each binding one flagos device, and exercises:
  1. init_process_group("flagos")
  2. all_reduce / broadcast / all_gather on flagos tensors
  3. DistributedDataParallel forward/backward with the auto-patched hook

Run:
    LD_LIBRARY_PATH=... python tests/manual/test_flagos_dist_live.py --world-size 2
"""

import argparse
import os

# torch_fl MUST be imported before torch (preloads libtorch_cuda.so).
import torch_fl  # noqa: F401
import torch
try:
    import flagcx  # noqa: F401  self-registers "flagcx" backend (nvidia adaptor)
except ImportError:
    flagcx = None
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel


def worker(rank: int, world_size: int):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29531")

    dev = torch.device(f"flagos:{rank}")
    torch.cuda.set_device(rank)  # flagos:i shares physical GPU i

    dist.init_process_group(
        backend="flagos", rank=rank, world_size=world_size,
    )

    # --- all_reduce ---
    t = torch.ones(4, device=dev) * (rank + 1)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    expected = sum(range(1, world_size + 1))
    ok_ar = torch.allclose(t.cpu(), torch.full((4,), float(expected)))
    print(f"[rank {rank}] all_reduce -> {t[0].item()} (expect {expected}) {'OK' if ok_ar else 'FAIL'}")

    # --- broadcast ---
    b = torch.arange(4, device=dev, dtype=torch.float32) + rank * 10
    dist.broadcast(b, src=0)
    ok_bc = torch.allclose(b.cpu(), torch.arange(4, dtype=torch.float32))
    print(f"[rank {rank}] broadcast -> {b.tolist()} {'OK' if ok_bc else 'FAIL'}")

    # --- all_gather ---
    src = torch.ones(2, device=dev) * (rank + 1)
    gathered = [torch.zeros(2, device=dev) for _ in range(world_size)]
    dist.all_gather(gathered, src)
    vals = [g[0].item() for g in gathered]
    ok_ag = vals == [float(i + 1) for i in range(world_size)]
    print(f"[rank {rank}] all_gather -> {vals} {'OK' if ok_ag else 'FAIL'}")

    # --- DDP forward/backward ---
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 1)).to(dev)
    ddp = DistributedDataParallel(model)
    print(f"[rank {rank}] DDP _use_python_reducer={ddp._use_python_reducer} "
          f"hooks={len(ddp._accum_grad_hooks)}")

    x = torch.randn(16, 8, device=dev)
    loss = ddp(x).sum()
    loss.backward()
    # grads should be identical across ranks after the all_reduce hook
    g0 = next(ddp.parameters()).grad
    gsum = g0.sum().item()
    print(f"[rank {rank}] DDP backward grad.sum={gsum:.4f} dev={g0.device}")

    dist.barrier()
    if rank == 0:
        print("=== all collectives + DDP completed ===")
    dist.destroy_process_group()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--world-size", type=int, default=2)
    args = ap.parse_args()
    mp.set_start_method("spawn", force=True)
    mp.spawn(worker, args=(args.world_size,), nprocs=args.world_size, join=True)

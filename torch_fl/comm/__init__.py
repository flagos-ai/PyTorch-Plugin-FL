"""
torch_fl.comm — native ProcessGroup backend for flagos (PrivateUse1).

Registering this package (done automatically by ``import torch_fl``) makes
``torch.distributed.init_process_group`` recognise the ``"flagos"`` backend
and route distributed ops on ``privateuseone`` tensors through it without any
monkeypatching.

After ``import torch_fl``:
    - ``torch.distributed.init_process_group("flagos")``  — explicit
    - ``torch.distributed.init_process_group(            ``  — auto-detect
          device_id=torch.device("privateuseone:0"))
"""

from torch_fl.comm.process_group import ProcessGroupFlagOS, register_flagos_backend

__all__ = ["ProcessGroupFlagOS", "register_flagos_backend"]

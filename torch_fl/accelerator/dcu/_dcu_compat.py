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

"""RNG compatibility between torch.flagos and DTK's CUDA generators.

DCU uses DTK's hipified libtorch.  Its CUDA kernels therefore consume the
CUDA-shaped generators exposed by ``torch.cuda.default_generators``, while the
PrivateUse1 device module created by torch_fl owns a separate generator set.
Keep the public ``torch.flagos`` seed/state API synchronized with the generators
that the kernels actually consume.
"""

_patched = False


def _device_index(device) -> int:
    import torch

    if isinstance(device, torch.device):
        return 0 if device.index is None else device.index
    if isinstance(device, str):
        return int(device.rsplit(":", 1)[1]) if ":" in device else 0
    if device is None:
        return 0
    return int(device)


def install_dcu_rng_bridge() -> bool:
    """Make ``torch.flagos`` expose DTK's real CUDA RNG streams.

    The PrivateUse1 generator remains available for explicit ``flagos``
    generators, but generator-less DCU kernels use the CUDA generator injected
    by the boxing/codegen path.  Delegating state and seeding here makes both
    paths observe one public stream without replacing DTK's CUDA generator
    collection or its native ``torch.cuda`` methods.
    """
    global _patched
    if _patched:
        return True

    import torch

    from torch_fl import flagos

    generators = getattr(torch.cuda, "default_generators", None)
    if generators is None:
        return False
    try:
        if len(generators) == 0:
            torch.cuda.init()
            generators = torch.cuda.default_generators
    except (AttributeError, RuntimeError, TypeError):
        return False

    native_manual_seed = flagos.manual_seed
    native_manual_seed_all = flagos.manual_seed_all

    def manual_seed(seed):
        seed = int(seed)
        native_manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def manual_seed_all(seed):
        seed = int(seed)
        native_manual_seed_all(seed)
        torch.cuda.manual_seed_all(seed)

    def get_rng_state(device="flagos"):
        return generators[_device_index(device)].get_state()

    def set_rng_state(state, device="flagos"):
        generators[_device_index(device)].set_state(state)

    # Keep the native PrivateUse1 seed bookkeeping in sync while exposing the
    # state object that the actual DCU CUDA kernels consume.
    flagos.manual_seed = manual_seed
    flagos.manual_seed_all = manual_seed_all
    flagos.get_rng_state = get_rng_state
    flagos.set_rng_state = set_rng_state
    flagos.default_generators = generators

    _patched = True
    return True

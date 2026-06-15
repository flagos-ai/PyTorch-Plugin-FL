import pytest
import torch
import torch_fl


def _cuda_driver_error(msg: str) -> bool:
    return (
        "NVIDIA driver on your system is too old" in msg
        or "Found no NVIDIA driver on your system" in msg
    )


def _flagos_ready() -> bool:
    return torch_fl.flagos.device_count() > 0


@pytest.mark.skipif(not _flagos_ready(), reason="flagos device not available")
class TestCloneDispatchCase:
    def test_clone_preserve_succeeds(self):
        x = torch.arange(16, dtype=torch.float16).reshape(4, 4).to("flagos:0")
        y = x.clone()
        assert y.device.type == "flagos"
        assert torch.equal(y.cpu(), x.cpu())

    def test_clone_contiguous_format_works(self):
        x = torch.arange(16, dtype=torch.float16).reshape(4, 4).to("flagos:0")
        y = x.clone(memory_format=torch.contiguous_format)
        assert y.device.type == "flagos"
        assert torch.equal(y.cpu(), x.cpu())

    def test_copy_kernel_works(self):
        src = torch.arange(16, dtype=torch.float16).reshape(4, 4).to("flagos:0")
        dst = torch.empty_like(src)
        dst.copy_(src)
        assert dst.device.type == "flagos"
        assert torch.equal(dst.cpu(), src.cpu())

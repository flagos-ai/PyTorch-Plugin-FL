"""Tests for the caching device allocator."""

import pytest
import torch
import torch_fl  # noqa: F401


@pytest.fixture(autouse=True)
def _setup():
    """Ensure flagos device is available."""
    if not torch_fl.flagos.is_available():
        pytest.skip("No flagos device available")
    torch_fl.flagos.init()


class TestCachingAllocatorBasic:
    """Basic allocation/deallocation and cache hit tests."""

    def test_allocate_and_free(self):
        """Tensor allocation should succeed."""
        device = torch.device("flagos", 0)
        x = torch.randn(1024, device=device)
        assert x.device.type in ("privateuseone", "flagos")
        assert x.numel() == 1024
        del x

    def test_cache_reuse(self):
        """After freeing, the same size allocation should reuse cached memory."""
        device = torch.device("flagos", 0)

        # Clear cache to start fresh.
        torch_fl.flagos.empty_cache()
        stats_before = torch_fl.flagos.memory_stats(0)
        malloc_before = stats_before.get("num_device_malloc", 0)

        # First allocation triggers a device malloc.
        x = torch.randn(1024, device=device)
        del x

        # Second allocation of same size should reuse cached block.
        y = torch.randn(1024, device=device)
        stats_after = torch_fl.flagos.memory_stats(0)
        malloc_after = stats_after.get("num_device_malloc", 0)

        # Should have at most one new device malloc (the first one).
        # The second should hit cache.
        assert malloc_after - malloc_before <= 1, (
            f"Expected cache reuse, but got {malloc_after - malloc_before} "
            f"device mallocs for two same-size allocations"
        )
        del y

    def test_empty_cache(self):
        """empty_cache should release reserved memory."""
        device = torch.device("flagos", 0)
        x = torch.randn(1024 * 1024, device=device)  # 4MB
        del x

        reserved_before = torch_fl.flagos.memory_reserved(0)
        assert reserved_before > 0

        torch_fl.flagos.empty_cache()
        reserved_after = torch_fl.flagos.memory_reserved(0)
        # After empty_cache, reserved should decrease.
        assert reserved_after < reserved_before

    def test_memory_stats_basic(self):
        """memory_stats should return a dict with expected keys."""
        stats = torch_fl.flagos.memory_stats(0)
        assert isinstance(stats, dict)
        expected_keys = [
            "allocated_bytes",
            "reserved_bytes",
            "peak_allocated_bytes",
            "peak_reserved_bytes",
            "num_alloc_calls",
            "num_free_calls",
            "num_device_malloc",
            "num_device_free",
            "num_alloc_retries",
        ]
        for key in expected_keys:
            assert key in stats, f"Missing key: {key}"

    def test_memory_allocated_tracking(self):
        """memory_allocated should reflect current tensor allocation."""
        device = torch.device("flagos", 0)
        torch_fl.flagos.empty_cache()

        alloc_before = torch_fl.flagos.memory_allocated(0)
        x = torch.randn(1024 * 256, device=device)  # 1MB float32
        alloc_during = torch_fl.flagos.memory_allocated(0)
        del x
        alloc_after = torch_fl.flagos.memory_allocated(0)

        # During: should be higher than before.
        assert alloc_during > alloc_before
        # After: should drop back.
        assert alloc_after < alloc_during


class TestCachingAllocatorMultipleSizes:
    """Test allocation across different size classes."""

    def test_small_pool_allocation(self):
        """Small allocations (<=1MB) should use the small pool."""
        device = torch.device("flagos", 0)
        # 512 bytes - definitely small pool
        x = torch.zeros(128, dtype=torch.float32, device=device)
        assert x.numel() == 128
        del x

    def test_large_pool_allocation(self):
        """Large allocations (>1MB) should use the large pool."""
        device = torch.device("flagos", 0)
        # ~4MB - large pool
        x = torch.zeros(1024 * 1024, dtype=torch.float32, device=device)
        assert x.numel() == 1024 * 1024
        del x

    def test_multiple_allocations(self):
        """Multiple tensors can coexist."""
        device = torch.device("flagos", 0)
        tensors = []
        for i in range(10):
            tensors.append(torch.randn(1024, device=device))
        # All should be valid.
        for t in tensors:
            assert t.numel() == 1024
        del tensors

    def test_varied_sizes(self):
        """Allocations of different sizes should all work."""
        device = torch.device("flagos", 0)
        sizes = [1, 64, 512, 1024, 4096, 1024 * 1024]
        for size in sizes:
            x = torch.randn(size, device=device)
            assert x.numel() == size
            del x


class TestCachingAllocatorOps:
    """Test that normal operations work with the caching allocator."""

    def test_matmul(self):
        """Matrix multiplication should work with cached allocations."""
        device = torch.device("flagos", 0)
        a = torch.randn(32, 64, device=device)
        b = torch.randn(64, 16, device=device)
        c = torch.mm(a, b)
        assert c.shape == (32, 16)

    def test_copy_between_devices(self):
        """Host-to-device and device-to-host copies should work."""
        device = torch.device("flagos", 0)
        x_cpu = torch.randn(1024)
        x_dev = x_cpu.to(device)
        x_back = x_dev.to("cpu")
        assert torch.allclose(x_cpu, x_back, atol=1e-6)

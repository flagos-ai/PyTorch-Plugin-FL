"""Native muRAND/mudnn RNG coverage for Moore Threads MUSA."""

import pytest
import torch
import torch_fl  # noqa: F401


pytestmark = pytest.mark.musa
DEVICE = torch.device("flagos:0")


def _require_musa():
    if torch_fl.flagos.device_count() < 1:
        pytest.skip("MUSA device is unavailable")


def test_factory_reproducibility_and_generator_isolation():
    _require_musa()
    torch.flagos.manual_seed(1234)
    first = torch.rand((32,), device=DEVICE)
    torch.flagos.manual_seed(1234)
    second = torch.rand((32,), device=DEVICE)
    assert torch.equal(first, second)

    generator = torch.Generator(device="flagos").manual_seed(77)
    explicit_first = torch.randn((16,), device=DEVICE, generator=generator)
    torch.flagos.manual_seed(999)
    explicit_second = torch.randn(
        (16,), device=DEVICE, generator=torch.Generator(device="flagos").manual_seed(77)
    )
    assert torch.equal(explicit_first, explicit_second)


def test_rng_state_round_trip_and_inplace_ops():
    _require_musa()
    torch.flagos.manual_seed(2026)
    state = torch.flagos.get_rng_state()
    a = torch.empty((17,), device=DEVICE).normal_(0.5, 2.0)
    torch.flagos.set_rng_state(state)
    b = torch.empty((17,), device=DEVICE).normal_(0.5, 2.0)
    assert torch.equal(a, b)

    torch.flagos.manual_seed(8)
    x = torch.empty((64,), device=DEVICE).uniform_(-2.0, 3.0)
    assert bool(torch.all(x >= -2.0))
    assert bool(torch.all(x < 3.0))

    unit = torch.rand((4096,), device=DEVICE)
    assert bool(torch.all(unit >= 0.0))
    assert bool(torch.all(unit < 1.0))


def test_integer_factory_and_out_variants():
    _require_musa()
    torch.flagos.manual_seed(41)
    x = torch.randint(-7, 13, (128,), device=DEVICE, dtype=torch.int32)
    assert x.dtype == torch.int32
    assert bool(torch.all(x >= -7)) and bool(torch.all(x < 13))

    out = torch.empty((128,), device=DEVICE, dtype=torch.int64)
    torch.randint(100, (128,), device=DEVICE, out=out)
    assert bool(torch.all(out >= 0)) and bool(torch.all(out < 100))


def test_like_and_dropout_contracts():
    _require_musa()
    ref = torch.empty((4, 9), device=DEVICE, dtype=torch.float32).transpose(0, 1)
    torch.flagos.manual_seed(55)
    sampled = torch.rand_like(ref)
    assert sampled.device == DEVICE
    assert sampled.shape == ref.shape
    assert sampled.is_contiguous() == ref.is_contiguous()

    torch.flagos.manual_seed(56)
    value = torch.ones((256,), device=DEVICE)
    output, mask = torch.ops.aten.native_dropout(value, 0.25, True)
    assert output.device == DEVICE
    assert mask.device == DEVICE
    assert mask.dtype == torch.bool
    assert torch.equal(output.ne(0), mask)
    expected_scale = 1.0 / (1.0 - 0.25)
    assert torch.allclose(output[mask], torch.full_like(output[mask], expected_scale))

    grad = torch.ops.aten.native_dropout_backward(torch.ones_like(output), mask, 4.0)
    assert torch.equal(grad, mask.to(grad.dtype) * 4.0)


def test_torch_manual_seed_and_full_width_integer_ranges():
    _require_musa()
    torch.manual_seed(4242)
    first = torch.rand((32,), device=DEVICE)
    torch.manual_seed(4242)
    second = torch.rand((32,), device=DEVICE)
    assert torch.equal(first, second)

    torch.flagos.manual_seed(99)
    full_width = torch.empty((4096,), device=DEVICE, dtype=torch.int64).random_(
        -(1 << 63), (1 << 63) - 1
    )
    assert bool(torch.all(full_width >= -(1 << 63)))
    assert bool(torch.all(full_width < (1 << 63) - 1))
    assert bool(torch.any(full_width < 0))
    assert bool(torch.any(full_width >= 0))

    defaults = torch.empty((4096,), device=DEVICE, dtype=torch.int8).random_()
    assert bool(torch.all(defaults >= 0))
    assert bool(torch.all(defaults <= torch.iinfo(torch.int8).max))


def test_shared_reservation_orders_native_and_flaggems_bridge():
    _require_musa()
    torch.flagos.manual_seed(8675309)
    initial_state = torch.flagos.get_rng_state()

    bridge_seed = torch_fl._C._reserve_rng_seed(0)
    torch.rand((16,), device=DEVICE)
    mixed_state = torch.flagos.get_rng_state()

    torch.flagos.set_rng_state(initial_state)
    first_seed = torch_fl._C._reserve_rng_seed(0)
    torch_fl._C._reserve_rng_seed(0)
    reservation_state = torch.flagos.get_rng_state()

    assert bridge_seed == first_seed
    assert torch.equal(mixed_state, reservation_state)


def test_nonzero_device_seed_is_per_device_when_available():
    _require_musa()
    if torch_fl.flagos.device_count() < 2:
        pytest.skip("requires at least two MUSA devices")
    torch.flagos.manual_seed_all(31415)
    first0 = torch.rand((32,), device="flagos:0")
    first1 = torch.rand((32,), device="flagos:1")

    torch.flagos.manual_seed_all(31415)
    second1 = torch.rand((32,), device="flagos:1")
    second0 = torch.rand((32,), device="flagos:0")

    assert first0.device.index == 0
    assert first1.device.index == 1
    assert first0.device.type == first1.device.type == "flagos"
    assert torch.equal(first0, second0)
    assert torch.equal(first1, second1)

    torch.flagos.manual_seed_all(31415)
    expected1 = torch.rand((32,), device="flagos:1")
    torch.flagos.manual_seed_all(31415)
    torch.rand((32,), device="flagos:0")
    actual1 = torch.rand((32,), device="flagos:1")
    assert torch.equal(expected1, actual1)

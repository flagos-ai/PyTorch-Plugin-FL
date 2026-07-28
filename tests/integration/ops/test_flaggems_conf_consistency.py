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

"""
FlagGems routing consistency (full coverage)

Every op that ``backends_flaggems.conf`` routes to ``flagos_python`` must have a
real ``Backend::kFlagOsPython`` kernel generated in the C++ layer, and vice
versa. This guards the whole FlagGems Python surface (currently ~319 ops)
against drift between the runtime config and the codegen output -- if someone
adds a ``= flagos_python`` line without a kernel (or regenerates kernels without
updating the conf), this test fails and names the offending ops.

This is a pure text/parse check: it reads the shipped config and generated
sources, so it needs no GPU, no ``flag_gems`` install, and runs in
milliseconds on any platform.

The op-name -> kernel bridge is:

    conf ``op = flagos_python``
      -> register.inc  ``m.impl("op", WrapperFoo);``
      -> WrapperFoo body ``... foo_dispatcher(...)``
      -> flaggems_python_kernels.cc
         ``REGISTER_IMPL_TO_DISPATCHER(_, foo_dispatcher, Backend::kFlagOsPython, _)``

so we compare the set of ``*_dispatcher`` names on each side.

Usage:
    pytest tests/integration/ops/test_flaggems_conf_consistency.py -v
"""

import re
from pathlib import Path

import pytest


# tests/integration/ops/<this file> -> repo root is three levels up.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONF = _REPO_ROOT / "torch_fl" / "configs" / "backends_flaggems.conf"
_REGISTER_INC = _REPO_ROOT / "csrc" / "aten" / "generated" / "register.inc"
_KERNELS_CC = _REPO_ROOT / "csrc" / "aten" / "generated" / "flaggems_python_kernels.cc"


def _read(path: Path) -> str:
    assert path.is_file(), f"expected generated/config file is missing: {path}"
    return path.read_text()


def _conf_flagos_python_ops() -> set[str]:
    """Op names that backends_flaggems.conf routes to the flagos_python slot."""
    ops: set[str] = set()
    for raw in _read(_CONF).splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        op, backend = (part.strip() for part in line.split("=", 1))
        if backend == "flagos_python":
            ops.add(op)
    return ops


def _op_to_wrapper() -> dict[str, str]:
    """``m.impl("op", WrapperFoo);`` -> {op: WrapperFoo} from register.inc."""
    return dict(re.findall(r'm\.impl\("([^"]+)",\s*(\w+)\);', _read(_REGISTER_INC)))


def _wrapper_to_dispatcher() -> dict[str, str]:
    """WrapperFoo(...) { ... foo_dispatcher(...) } -> {WrapperFoo: foo_dispatcher}."""
    return dict(
        re.findall(
            r"(\w+)\([^;{]*\)\s*\{\s*(?:return\s+)?"
            r"(?:at::native::flagos::)?(\w+_dispatcher)\(",
            _read(_REGISTER_INC),
        )
    )


def _cc_flagos_python_dispatchers() -> set[str]:
    """Dispatcher names registered with Backend::kFlagOsPython in the kernels cc."""
    return set(
        re.findall(
            r"REGISTER_IMPL_TO_DISPATCHER\(\s*\w+\s*,\s*(\w+)\s*,"
            r"\s*Backend::kFlagOsPython",
            _read(_KERNELS_CC),
        )
    )


def _conf_dispatchers() -> tuple[set[str], list[str]]:
    """Map conf flagos_python ops to their dispatcher names via register.inc.

    Returns (dispatcher_names, unmapped_ops).
    """
    op2wrap = _op_to_wrapper()
    wrap2disp = _wrapper_to_dispatcher()
    dispatchers: set[str] = set()
    unmapped: list[str] = []
    for op in _conf_flagos_python_ops():
        wrapper = op2wrap.get(op)
        dispatcher = wrap2disp.get(wrapper) if wrapper else None
        if dispatcher:
            dispatchers.add(dispatcher)
        else:
            unmapped.append(op)
    return dispatchers, sorted(unmapped)


class TestFlagGemsConfConsistency:
    """backends_flaggems.conf <-> generated kFlagOsPython kernels must agree."""

    @pytest.mark.anyplatform
    def test_conf_has_flagos_python_ops(self):
        """Sanity: the conf actually routes a meaningful number of ops here."""
        ops = _conf_flagos_python_ops()
        assert len(ops) > 100, (
            f"expected many flagos_python ops in {_CONF.name}, got {len(ops)}"
        )

    @pytest.mark.anyplatform
    def test_every_conf_op_maps_to_a_dispatcher(self):
        """Every flagos_python op resolves through register.inc to a dispatcher."""
        _, unmapped = _conf_dispatchers()
        assert not unmapped, (
            "conf routes these ops to flagos_python but register.inc has no "
            f"m.impl/dispatcher for them: {unmapped}"
        )

    @pytest.mark.anyplatform
    def test_conf_ops_have_flagos_python_kernels(self):
        """Each flagos_python op must have a real kFlagOsPython C++ kernel."""
        conf_disp, _ = _conf_dispatchers()
        cc_disp = _cc_flagos_python_dispatchers()
        missing = sorted(conf_disp - cc_disp)
        assert not missing, (
            "these ops are routed to flagos_python in "
            f"{_CONF.name} but have NO kFlagOsPython kernel in "
            f"{_KERNELS_CC.name} (conf/codegen drift): {missing}"
        )

    @pytest.mark.anyplatform
    def test_no_orphan_flagos_python_kernels(self):
        """No kFlagOsPython kernel exists without a conf route pointing at it."""
        conf_disp, _ = _conf_dispatchers()
        cc_disp = _cc_flagos_python_dispatchers()
        orphans = sorted(cc_disp - conf_disp)
        assert not orphans, (
            f"these kFlagOsPython kernels in {_KERNELS_CC.name} are not routed "
            f"to flagos_python by any op in {_CONF.name} (orphan kernels): "
            f"{orphans}"
        )

    @pytest.mark.anyplatform
    def test_counts_match(self):
        """Both sides expose the exact same number of flagos_python routes."""
        conf_disp, _ = _conf_dispatchers()
        cc_disp = _cc_flagos_python_dispatchers()
        assert len(conf_disp) == len(cc_disp), (
            f"flagos_python route count mismatch: conf={len(conf_disp)} "
            f"kernels={len(cc_disp)}"
        )

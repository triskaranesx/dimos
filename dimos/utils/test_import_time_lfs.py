# Copyright 2025-2026 Dimensional Inc.
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

"""Guard against modules that trigger LFS downloads (get_data) at import time."""

import concurrent.futures
import os
import subprocess
import sys

import pytest

from dimos.constants import DIMOS_PROJECT_ROOT

DIMOS_DIR = DIMOS_PROJECT_ROOT / "dimos"
MAX_WORKERS = os.cpu_count() or 1
TIMEOUT = 10

KNOWN_EXCEPTIONS = {
    "dimos.control.blueprints.basic",
    "dimos.control.blueprints.dual",
    "dimos.control.blueprints.teleop",
    "dimos.robot.manipulators.piper.blueprints",
    "dimos.teleop.quest.blueprints",
}

SCRIPT_TEMPLATE = """\
import dimos.utils.data as _data_mod

def _bomb(name):
    raise RuntimeError(f"GET_DATA_CALLED:{{name}}")

_data_mod.get_data = _bomb

import {module}
"""


def _find_modules() -> list[str]:
    modules = []
    for py_file in sorted(DIMOS_DIR.rglob("*.py")):
        rel = py_file.relative_to(DIMOS_PROJECT_ROOT)
        parts = rel.with_suffix("").parts
        if "__pycache__" in parts:
            continue
        if any(p.startswith("test_") or p.startswith("tests") for p in parts):
            continue
        module = ".".join(parts)
        if module.endswith(".__init__"):
            module = module.removesuffix(".__init__")
        modules.append(module)
    return modules


def _check_module(module: str) -> tuple[str, str | None]:
    try:
        result = subprocess.run(
            [sys.executable, "-c", SCRIPT_TEMPLATE.format(module=module)],
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return module, None
    if result.returncode != 0 and "GET_DATA_CALLED:" in result.stderr:
        for line in result.stderr.splitlines():
            if "GET_DATA_CALLED:" in line:
                lfs_name = line.split("GET_DATA_CALLED:", 1)[1].strip().rstrip("'\")")
                return module, lfs_name
        return module, "(unknown)"
    return module, None


@pytest.mark.slow
def test_no_new_import_time_lfs_downloads() -> None:
    modules = _find_modules()

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        results = list(pool.map(_check_module, modules))

    offenders = {mod for mod, lfs in results if lfs is not None}

    new_offenders = sorted(
        (mod, lfs) for mod, lfs in results if lfs is not None and mod not in KNOWN_EXCEPTIONS
    )
    assert not new_offenders, (
        "These modules call get_data() at import time (please fix!):\n"
        + "\n".join(f"  {mod}  ->  get_data({lfs!r})" for mod, lfs in new_offenders)
    )

    stale_exceptions = KNOWN_EXCEPTIONS - offenders
    assert not stale_exceptions, (
        "These modules no longer call get_data() at import time.\n"
        "Remove them from KNOWN_EXCEPTIONS:\n"
        + "\n".join(f"  {mod}" for mod in sorted(stale_exceptions))
    )

# Copyright 2026 Dimensional Inc.
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

"""Spec compliance tests for the FAST-LIO2 module."""

from __future__ import annotations

from dimos.hardware.sensors.lidar.fastlio2.module import FastLio2
from dimos.spec import mapping, perception
from dimos.spec.utils import assert_implements_protocol


def test_fastlio2_implements_spec() -> None:
    assert_implements_protocol(FastLio2, perception.Lidar)
    assert_implements_protocol(FastLio2, perception.Odometry)
    assert_implements_protocol(FastLio2, mapping.GlobalPointcloud)

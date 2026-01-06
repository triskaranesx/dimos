# Copyright 2025 Dimensional Inc.
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
Grasp generation package.

Note: Some submodules depend on optional heavy deps (e.g. `open3d`, SAM, AnyGrasp).
To avoid import-time failures for users who don't need those features, we keep
this package init lightweight and do not eagerly import optional dependencies.
"""
__all__ = []

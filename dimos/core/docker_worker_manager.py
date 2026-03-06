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
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dimos.core.docker_runner import DockerModule
    from dimos.core.module import Module


class DockerWorkerManager:
    """Parallel deployment of Docker-backed modules."""

    @staticmethod
    def deploy_parallel(
        specs: list[tuple[type[Module], tuple[Any, ...], dict[str, Any]]],
    ) -> list[DockerModule]:
        """Deploy multiple DockerModules in parallel, collecting partial results on failure.

        Returns all successfully-created DockerModules. If any deployment fails,
        the successful ones are still returned (so the caller can register them
        for cleanup), and the first exception is re-raised.
        """
        from dimos.core.docker_runner import DockerModule

        results: dict[int, DockerModule] = {}
        first_exc: Exception | None = None

        with ThreadPoolExecutor(max_workers=len(specs)) as executor:
            futures: dict[Future[DockerModule], int] = {
                executor.submit(lambda s=spec: DockerModule(s[0], *s[1], **s[2])): i
                for i, spec in enumerate(specs)
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    results[idx] = fut.result()
                except Exception as e:
                    if first_exc is None:
                        first_exc = e

        # Return in input order (missing indices = failed deployments)
        ordered = [results[i] for i in sorted(results)]
        if first_exc is not None:
            raise first_exc
        return ordered

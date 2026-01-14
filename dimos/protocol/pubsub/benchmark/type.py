#!/usr/bin/env python3

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

from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
import pickle
import threading
import time
from typing import Any, Generic, TypeVar

import pytest

from dimos.msgs.geometry_msgs import Vector3
from dimos.msgs.sensor_msgs.Image import Image
from dimos.protocol.pubsub.lcmpubsub import LCM, Topic
from dimos.protocol.pubsub.memory import Memory
from dimos.protocol.pubsub.shmpubsub import PickleSharedMemory
from dimos.protocol.pubsub.spec import MsgT, PubSub, TopicT
from dimos.utils.data import get_data

MsgGen = Callable[[int], tuple[TopicT, MsgT]]

PubSubContext = Callable[[], AbstractContextManager[PubSub[TopicT, MsgT]]]


@dataclass
class TestCase(Generic[TopicT, MsgT]):
    pubsub_context: PubSubContext[TopicT, MsgT]
    msg_gen: MsgGen[TopicT, MsgT]

    def __iter__(self):
        return iter((self.pubsub_context, self.msg_gen))

    def __len__(self):
        return 2


TestData = Sequence[TestCase[Any, Any]]


def _format_size(size_bytes: int) -> str:
    """Format byte size to human-readable string."""
    if size_bytes >= 1048576:
        return f"{size_bytes / 1048576:.1f} MB"
    if size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes} B"


def _format_throughput(bytes_per_sec: float) -> str:
    """Format throughput to human-readable string."""
    if bytes_per_sec >= 1e9:
        return f"{bytes_per_sec / 1e9:.2f} GB/s"
    if bytes_per_sec >= 1e6:
        return f"{bytes_per_sec / 1e6:.2f} MB/s"
    if bytes_per_sec >= 1e3:
        return f"{bytes_per_sec / 1e3:.2f} KB/s"
    return f"{bytes_per_sec:.2f} B/s"


@dataclass
class BenchmarkResult:
    transport: str
    duration: float  # Time spent publishing
    msgs_sent: int
    msgs_received: int
    msg_size_bytes: int
    receive_time: float = 0.0  # Time after publishing until all messages received

    @property
    def total_time(self) -> float:
        """Total time including drain."""
        return self.duration + self.receive_time

    @property
    def throughput_msgs(self) -> float:
        """Messages per second (including drain time)."""
        return self.msgs_received / self.total_time if self.total_time > 0 else 0

    @property
    def throughput_bytes(self) -> float:
        """Bytes per second (including drain time)."""
        return (
            (self.msgs_received * self.msg_size_bytes) / self.total_time
            if self.total_time > 0
            else 0
        )

    @property
    def loss_pct(self) -> float:
        """Message loss percentage."""
        return (1 - self.msgs_received / self.msgs_sent) * 100 if self.msgs_sent > 0 else 0


@dataclass
class BenchmarkResults:
    results: list[BenchmarkResult] = field(default_factory=list)

    def add(self, result: BenchmarkResult) -> None:
        self.results.append(result)

    def print_summary(self) -> None:
        if not self.results:
            return

        from rich.console import Console
        from rich.table import Table

        console = Console()

        table = Table(title="Benchmark Results")
        table.add_column("Transport", style="cyan")
        table.add_column("Msg Size", justify="right")
        table.add_column("Sent", justify="right")
        table.add_column("Recv", justify="right")
        table.add_column("Msgs/s", justify="right", style="green")
        table.add_column("Throughput", justify="right", style="green")
        table.add_column("Drain", justify="right")
        table.add_column("Loss", justify="right")

        for r in sorted(self.results, key=lambda x: (x.transport, x.msg_size_bytes)):
            loss_style = "red" if r.loss_pct > 0 else "dim"
            recv_style = "yellow" if r.receive_time > 0.1 else "dim"
            table.add_row(
                r.transport,
                _format_size(r.msg_size_bytes),
                f"{r.msgs_sent:,}",
                f"{r.msgs_received:,}",
                f"{r.throughput_msgs:,.0f}",
                _format_throughput(r.throughput_bytes),
                f"[{recv_style}]{r.receive_time * 1000:.0f}ms[/{recv_style}]",
                f"[{loss_style}]{r.loss_pct:.1f}%[/{loss_style}]",
            )

        console.print()
        console.print(table)

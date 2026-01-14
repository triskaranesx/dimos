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

"""Latency benchmark: send message, wait for receive, repeat."""

from dataclasses import dataclass, field
import threading
import time

import pytest

from dimos.protocol.pubsub.benchmark.testdata import testdata

# Message sizes for latency benchmarking
MSG_SIZES = [64, 1024, 16384, 65536, 262144, 1048576, 1048576 * 2, 1048576 * 5]

# How long to run each test
BENCH_DURATION = 1.0

# Timeout waiting for a single message
MSG_TIMEOUT = 1.0


def size_id(size: int) -> str:
    """Convert byte size to human-readable string for test IDs."""
    if size >= 1048576:
        return f"{size // 1048576}MB"
    if size >= 1024:
        return f"{size // 1024}KB"
    return f"{size}B"


def pubsub_id(testcase) -> str:
    """Extract pubsub implementation name from context manager function name."""
    name = testcase.pubsub_context.__name__
    prefix = name.replace("_pubsub_channel", "").replace("_", " ")
    return prefix.upper() if len(prefix) <= 3 else prefix.title().replace(" ", "")


def _format_size(size_bytes: int) -> str:
    if size_bytes >= 1048576:
        return f"{size_bytes / 1048576:.1f} MB"
    if size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes} B"


def _format_throughput(bytes_per_sec: float) -> str:
    if bytes_per_sec >= 1e9:
        return f"{bytes_per_sec / 1e9:.2f} GB/s"
    if bytes_per_sec >= 1e6:
        return f"{bytes_per_sec / 1e6:.2f} MB/s"
    if bytes_per_sec >= 1e3:
        return f"{bytes_per_sec / 1e3:.2f} KB/s"
    return f"{bytes_per_sec:.2f} B/s"


@dataclass
class LatencyResult:
    transport: str
    msg_size_bytes: int
    msgs_sent: int
    msgs_received: int
    total_time: float
    min_latency: float
    max_latency: float
    avg_latency: float

    @property
    def msgs_per_sec(self) -> float:
        return self.msgs_received / self.total_time if self.total_time > 0 else 0

    @property
    def throughput_bytes(self) -> float:
        return (
            (self.msgs_received * self.msg_size_bytes) / self.total_time
            if self.total_time > 0
            else 0
        )

    @property
    def loss_pct(self) -> float:
        return (1 - self.msgs_received / self.msgs_sent) * 100 if self.msgs_sent > 0 else 0


@dataclass
class LatencyResults:
    results: list[LatencyResult] = field(default_factory=list)

    def add(self, result: LatencyResult) -> None:
        self.results.append(result)

    def print_summary(self) -> None:
        if not self.results:
            return

        from rich.console import Console
        from rich.table import Table

        console = Console()

        table = Table(title="Latency Benchmark Results (send-wait-receive)")
        table.add_column("Transport", style="cyan")
        table.add_column("Msg Size", justify="right")
        table.add_column("Sent", justify="right")
        table.add_column("Recv", justify="right")
        table.add_column("Loss", justify="right")
        table.add_column("Avg Latency", justify="right", style="green")
        table.add_column("Min", justify="right")
        table.add_column("Max", justify="right")
        table.add_column("Msgs/s", justify="right")
        table.add_column("Throughput", justify="right", style="green")

        for r in sorted(self.results, key=lambda x: (x.transport, x.msg_size_bytes)):
            loss_style = "red" if r.loss_pct > 0 else "dim"
            table.add_row(
                r.transport,
                _format_size(r.msg_size_bytes),
                f"{r.msgs_sent:,}",
                f"{r.msgs_received:,}",
                f"[{loss_style}]{r.loss_pct:.1f}%[/{loss_style}]",
                f"{r.avg_latency * 1000:.2f}ms",
                f"{r.min_latency * 1000:.2f}ms",
                f"{r.max_latency * 1000:.2f}ms",
                f"{r.msgs_per_sec:,.0f}",
                _format_throughput(r.throughput_bytes),
            )

        console.print()
        console.print(table)

    def print_heatmap(self) -> None:
        if not self.results:
            return

        import plotext as plt

        # Organize data by transport and message size
        transports = sorted(set(r.transport for r in self.results))
        sizes = sorted(set(r.msg_size_bytes for r in self.results))

        # Build matrix of throughput values (in GB/s for readability)
        matrix = []
        for transport in transports:
            row = []
            for size in sizes:
                result = next(
                    (
                        r
                        for r in self.results
                        if r.transport == transport and r.msg_size_bytes == size
                    ),
                    None,
                )
                # Convert to GB/s, use log scale for better visualization
                throughput_gbps = result.throughput_bytes / 1e9 if result else 0
                row.append(throughput_gbps)
            matrix.append(row)

        # Create heatmap
        plt.clear_figure()
        plt.matrix_plot(matrix)
        plt.title("Throughput Heatmap (GB/s)")
        plt.xlabel("Message Size")
        plt.ylabel("Transport")

        # Set axis labels
        size_labels = [size_id(s) for s in sizes]
        plt.xticks(list(range(len(sizes))), size_labels)
        plt.yticks(list(range(len(transports))), transports)

        plt.show()


@pytest.fixture(scope="module")
def latency_results():
    """Module-scoped fixture to collect latency results."""
    results = LatencyResults()
    yield results
    results.print_summary()
    # results.print_heatmap()


@pytest.mark.benchmark
@pytest.mark.parametrize("msg_size", MSG_SIZES, ids=[size_id(s) for s in MSG_SIZES])
@pytest.mark.parametrize("pubsub_context, msggen", testdata, ids=[pubsub_id(t) for t in testdata])
def test_latency(pubsub_context, msggen, msg_size, latency_results):
    """Measure round-trip latency: send message, wait for receive, repeat."""
    with pubsub_context() as pubsub:
        topic, msg = msggen(msg_size)
        received = threading.Event()
        latencies = []

        def callback(message, _topic):
            received.set()

        pubsub.subscribe(topic, callback)

        # Warmup: give DDS/ROS time to establish connection
        time.sleep(0.1)

        # Run for BENCH_DURATION seconds
        start = time.perf_counter()
        end_time = start + BENCH_DURATION

        msgs_sent = 0
        while time.perf_counter() < end_time:
            received.clear()
            msg_start = time.perf_counter()
            pubsub.publish(topic, msg)
            msgs_sent += 1

            if received.wait(timeout=MSG_TIMEOUT):
                latency = time.perf_counter() - msg_start
                latencies.append(latency)
            else:
                # Message lost - skip
                pass

        total_time = time.perf_counter() - start

        transport_name = pubsub_id(type("TC", (), {"pubsub_context": pubsub_context})())
        result = LatencyResult(
            transport=transport_name,
            msg_size_bytes=msg_size,
            msgs_sent=msgs_sent,
            msgs_received=len(latencies),
            total_time=total_time,
            min_latency=min(latencies) if latencies else 0,
            max_latency=max(latencies) if latencies else 0,
            avg_latency=sum(latencies) / len(latencies) if latencies else 0,
        )
        latency_results.add(result)

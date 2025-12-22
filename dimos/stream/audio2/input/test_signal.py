#!/usr/bin/env python3
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

"""Tests for signal generator audio node."""

import threading
import time

import numpy as np
import pytest

from dimos.stream.audio2.input.signal import WaveformType, test_signal
from dimos.stream.audio2.types import AudioFormat, AudioSpec, RawAudioEvent


def test_sine_wave_generation():
    """Test basic sine wave generation."""
    # Configure for raw output at 8kHz (for easier verification)
    output = AudioSpec(format=AudioFormat.PCM_F32LE, sample_rate=8000, channels=1)
    source = test_signal(
        waveform=WaveformType.SINE,
        frequency=1000.0,  # 1kHz
        volume=1.0,
        duration=0.1,  # 100ms
        output=output,
    )

    # Collect events
    events = []
    errors = []
    complete_event = threading.Event()

    def on_next(event):
        events.append(event)

    def on_error(e):
        errors.append(e)
        complete_event.set()

    def on_completed():
        complete_event.set()

    # Subscribe
    subscription = source().subscribe(on_next=on_next, on_error=on_error, on_completed=on_completed)

    # Wait for completion
    assert complete_event.wait(timeout=2.0), "Signal generation did not complete"

    # Clean up
    subscription.dispose()
    time.sleep(0.1)

    # Verify
    assert len(errors) == 0, f"Unexpected errors: {errors}"
    assert len(events) > 0, "No events received"

    # All should be raw audio
    for event in events:
        assert isinstance(event, RawAudioEvent)
        assert event.sample_rate == 8000
        assert event.channels == 1
        assert event.format == AudioFormat.PCM_F32LE

    # Rough verification of total samples (should be ~0.1s * 8000Hz = 800 samples)
    # GStreamer may generate slightly more samples due to buffer sizing
    total_samples = sum(len(event.data) for event in events)
    assert 700 <= total_samples <= 2000, f"Unexpected total samples: {total_samples}"


def test_noise_generation():
    """Test white noise generation."""
    output = AudioSpec(format=AudioFormat.PCM_F32LE, sample_rate=16000, channels=1)
    source = test_signal(
        waveform=WaveformType.WHITE_NOISE,
        volume=0.5,
        duration=0.2,
        output=output,  # 200ms
    )

    events = []
    errors = []
    complete_event = threading.Event()

    subscription = source().subscribe(
        on_next=lambda e: events.append(e),
        on_error=lambda e: (errors.append(e), complete_event.set()),
        on_completed=lambda: complete_event.set(),
    )

    assert complete_event.wait(timeout=2.0), "Noise generation did not complete"
    subscription.dispose()
    time.sleep(0.1)

    assert len(errors) == 0, f"Unexpected errors: {errors}"
    assert len(events) > 0, "No events received"

    # Check noise properties
    first_event = events[0]
    assert isinstance(first_event, RawAudioEvent)

    # White noise should have non-zero variance
    assert np.var(first_event.data) > 0.01, "Noise variance too low"

    # Should be roughly centered around zero
    assert -0.1 < np.mean(first_event.data) < 0.1, "Noise mean not centered"


def test_invalid_frequency():
    """Test that invalid frequency raises error."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="Input should be greater than 0"):
        test_signal(frequency=-100)


def test_invalid_duration():
    """Test that invalid duration raises error."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="Input should be greater than 0"):
        test_signal(duration=-1.0)


def test_volume_validation():
    """Test that volume is validated to valid range."""
    from pydantic import ValidationError

    # Test over-range volume
    with pytest.raises(ValidationError, match="less than or equal to 1"):
        test_signal(volume=2.0, duration=0.01)

    # Test under-range volume
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        test_signal(volume=-1.0, duration=0.01)

    # Test valid volumes (should not raise)
    test_signal(volume=0.0, duration=0.01)
    test_signal(volume=1.0, duration=0.01)
    test_signal(volume=0.5, duration=0.01)


def test_continuous_generation():
    """Test continuous signal generation without duration."""
    source = test_signal(frequency=440.0)  # No duration - continuous

    events = []
    stop_event = threading.Event()

    def on_next(event):
        events.append(event)
        if len(events) >= 5:  # Stop after 5 events
            stop_event.set()

    subscription = source().subscribe(on_next=on_next)

    # Wait for some events
    assert stop_event.wait(timeout=2.0), "Did not receive enough events"

    # Dispose to stop generation
    subscription.dispose()
    time.sleep(0.1)

    assert len(events) >= 5, f"Expected at least 5 events, got {len(events)}"

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

"""Tests for soundcard output."""

import threading
import time

import pytest

from dimos.stream.audio2.input.signal import WaveformType, test_signal
from dimos.stream.audio2.output.soundcard import speaker
from dimos.stream.audio2.types import AudioFormat, AudioSpec


@pytest.mark.tool
def test_speaker_with_test_signal():
    """Test playing a test signal through speakers."""
    # Create a 440Hz sine wave for 2 seconds
    source = test_signal(
        waveform=WaveformType.SINE,
        frequency=440.0,
        volume=0.5,  # Moderate volume for testing
        duration=2.0,
        output=AudioSpec(format=AudioFormat.PCM_F32LE),  # Raw audio
    )

    # Create speaker output
    sink = speaker()

    # Track completion
    completed = threading.Event()
    errors = []

    def on_error(e):
        errors.append(e)
        completed.set()

    def on_completed():
        completed.set()

    # Connect source to sink
    observable = source()

    # Subscribe with error/completion handlers
    # Sink will auto-start on first event
    subscription = observable.subscribe(
        on_next=sink.on_next,
        on_error=lambda e: (sink.on_error(e), on_error(e)),
        on_completed=lambda: (sink.on_completed(), on_completed()),
    )

    try:
        # Wait for completion
        assert completed.wait(timeout=5.0), "Audio playback did not complete"
        assert len(errors) == 0, f"Unexpected errors: {errors}"
        print("COMPLETED")
        # IMPORTANT: Wait for audio to actually play out
        # The sink needs time to play the buffered audio
        time.sleep(2.5)  # Slightly more than the 2 second duration

    finally:
        # Clean up
        subscription.dispose()
        sink.stop()
        time.sleep(0.1)


@pytest.mark.tool
def test_speaker_with_compressed_audio():
    """Test playing compressed audio through speakers."""
    # Create white noise with default compression (Vorbis)
    source = test_signal(
        waveform=WaveformType.WHITE_NOISE,
        volume=0.2,
        duration=0.5,
        # Using default output format (Vorbis)
    )

    # Create speaker output
    sink = speaker()

    # Use context manager for auto cleanup
    with sink:
        # Subscribe and play
        subscription = sink.subscribe_to(source())

        # Wait for audio to play
        time.sleep(1.0)

        # Clean up subscription
        subscription.dispose()


@pytest.mark.tool
def test_speaker_low_latency():
    """Test low-latency audio output."""
    # Create a short tick sound
    source = test_signal(
        waveform=WaveformType.TICKS,
        frequency=10.0,  # 10 ticks per second
        volume=0.4,
        duration=0.5,
    )

    # Create low-latency speaker output
    sink = speaker(
        sync=False,  # Disable sync for lower latency
        buffer_time=10000,  # 10ms buffer
        latency_time=5000,  # 5ms latency
    )

    with sink:
        subscription = sink.subscribe_to(source())
        time.sleep(1.0)
        subscription.dispose()


if __name__ == "__main__":
    import sys

    # Simple demo when run directly
    print("Playing test tones...")

    # Play different tones
    tones = [(440, "A4"), (523.25, "C5"), (659.25, "E5"), (880, "A5")]

    sink = speaker()
    sink.start()

    try:
        for freq, note in tones:
            print(f"Playing {note} ({freq:.2f}Hz)...")

            source = test_signal(frequency=freq, volume=0.3, duration=0.5)

            completed = threading.Event()
            subscription = source().subscribe(
                on_next=sink.on_next, on_error=sink.on_error, on_completed=lambda: completed.set()
            )

            completed.wait(timeout=1.0)
            subscription.dispose()
            time.sleep(0.1)  # Small gap between notes

    finally:
        sink.stop()

    print("Done!")

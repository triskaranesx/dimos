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

"""Simple test to verify audio chain is working."""

import logging
import time
import threading

from dimos.stream.audio2.input.signal import test_signal, WaveformType
from dimos.stream.audio2.output.soundcard import speaker
from dimos.stream.audio2.types import AudioFormat, AudioSpec

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)


def main():
    print("Creating 2-second 440Hz sine wave...")

    # Create a simple raw audio signal
    source = test_signal(
        waveform=WaveformType.SINE,
        frequency=440.0,
        volume=0.7,
        duration=2.0,
        output=AudioSpec(format=AudioFormat.PCM_F32LE, sample_rate=44100, channels=1),
    )

    # Create speaker output
    sink = speaker()

    # Track completion
    completed = threading.Event()

    def on_completed():
        print("Stream completed")
        completed.set()

    def on_error(e):
        print(f"Stream error: {e}")
        completed.set()

    # Subscribe
    print("Starting audio playback...")
    observable = source()
    subscription = observable.subscribe(
        on_next=sink.on_next,
        on_error=lambda e: (sink.on_error(e), on_error(e)),
        on_completed=lambda: (sink.on_completed(), on_completed()),
    )

    # Wait for completion
    if completed.wait(timeout=10.0):
        print("Playback finished, waiting for audio to play out...")
        # IMPORTANT: Wait for the pipeline to actually play the audio
        # Even after EOS, the pipeline needs time to play buffered audio
        time.sleep(3.0)
    else:
        print("Timeout waiting for playback")

    # Clean up
    subscription.dispose()
    sink.stop()
    time.sleep(0.5)  # Allow cleanup


if __name__ == "__main__":
    main()

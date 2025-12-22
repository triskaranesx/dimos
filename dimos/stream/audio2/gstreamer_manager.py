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

"""Pipeline Manager for coordinating GStreamer components with MainLoop."""

import threading
import gi

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst
from typing import Optional, Callable

from dimos.utils.logging_config import setup_logger

logger = setup_logger("dimos.utils.gstreamer_manager")


class PipelineManager:
    """
    Manages GLib MainLoop for GStreamer pipelines.

    This ensures proper event handling and message passing for modular
    audio pipelines that use separate input and output GStreamer components.
    """

    def __init__(self):
        """Initialize the Pipeline Manager."""
        Gst.init(None)
        self._loop = None
        self._loop_thread = None
        self._running = False

    def start(self):
        """Start the GLib MainLoop in a separate thread."""
        if self._running:
            logger.warning("Pipeline manager already running")
            return

        self._loop = GLib.MainLoop()
        self._running = True

        def run_loop():
            logger.info("Starting GLib MainLoop")
            try:
                self._loop.run()
            except Exception as e:
                logger.error(f"MainLoop error: {e}")
            finally:
                logger.info("GLib MainLoop stopped")

        self._loop_thread = threading.Thread(target=run_loop, daemon=True)
        self._loop_thread.start()
        logger.info("Pipeline manager started")

    def stop(self):
        """Stop the GLib MainLoop."""
        if not self._running:
            return

        logger.info("Stopping pipeline manager")
        self._running = False

        if self._loop and self._loop.is_running():
            self._loop.quit()

        if self._loop_thread and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=1.0)

        self._loop = None
        self._loop_thread = None
        logger.info("Pipeline manager stopped")

    def is_running(self):
        """Check if the MainLoop is running."""
        return self._running and self._loop and self._loop.is_running()

    def __enter__(self):
        """Context manager entry - start the manager."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - stop the manager."""
        self.stop()


# Global singleton instance
_pipeline_manager = None


def get_pipeline_manager():
    """Get or create the global pipeline manager instance."""
    global _pipeline_manager
    if _pipeline_manager is None:
        _pipeline_manager = PipelineManager()
        _pipeline_manager.start()
    return _pipeline_manager


def ensure_mainloop_running():
    """Ensure the GLib MainLoop is running for GStreamer."""
    manager = get_pipeline_manager()
    if not manager.is_running():
        manager.start()

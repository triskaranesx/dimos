// Copyright 2026 Dimensional Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ORB-SLAM3 native module for dimos NativeModule framework.
//
// Phase 1: Initializes ORB_SLAM3::System, loads vocabulary, and blocks
// until SIGTERM. No LCM pub/sub yet — topics come in phase 2.
//
// Usage:
//   ./orbslam3_native \
//       --settings_path /path/to/RealSense_D435i.yaml \
//       --sensor_mode MONOCULAR \
//       --use_viewer false

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <string>
#include <thread>

#include "dimos_native_module.hpp"

// ORB-SLAM3
#include "System.h"

// ---------------------------------------------------------------------------
// Signal handling
// ---------------------------------------------------------------------------

static std::atomic<bool> g_running{true};

static void signal_handler(int /*sig*/) {
    g_running.store(false);
}

// ---------------------------------------------------------------------------
// Sensor mode parsing
// ---------------------------------------------------------------------------

static ORB_SLAM3::System::eSensor parse_sensor_mode(const std::string& mode) {
    if (mode == "MONOCULAR")     return ORB_SLAM3::System::MONOCULAR;
    if (mode == "STEREO")        return ORB_SLAM3::System::STEREO;
    if (mode == "RGBD")          return ORB_SLAM3::System::RGBD;
    if (mode == "IMU_MONOCULAR") return ORB_SLAM3::System::IMU_MONOCULAR;
    if (mode == "IMU_STEREO")    return ORB_SLAM3::System::IMU_STEREO;
    if (mode == "IMU_RGBD")      return ORB_SLAM3::System::IMU_RGBD;
    fprintf(stderr, "[orbslam3] Unknown sensor mode: %s, defaulting to MONOCULAR\n",
            mode.c_str());
    return ORB_SLAM3::System::MONOCULAR;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    dimos::NativeModule mod(argc, argv);

    // Required: camera settings YAML
    std::string settings_path = mod.arg("settings_path", "");
    if (settings_path.empty()) {
        fprintf(stderr, "[orbslam3] Error: --settings_path is required\n");
        return 1;
    }

    // Vocabulary (compile-time default from nix build, or override via CLI)
#ifdef ORBSLAM3_DEFAULT_VOCAB
    std::string vocab_path = mod.arg("vocab_path", ORBSLAM3_DEFAULT_VOCAB);
#else
    std::string vocab_path = mod.arg("vocab_path", "");
#endif
    if (vocab_path.empty()) {
        fprintf(stderr, "[orbslam3] Error: --vocab_path is required "
                        "(no compiled-in default available)\n");
        return 1;
    }

    std::string sensor_str = mod.arg("sensor_mode", "MONOCULAR");
    bool use_viewer = mod.arg("use_viewer", "false") == "true";
    auto sensor_mode = parse_sensor_mode(sensor_str);

    printf("[orbslam3] Initializing ORB-SLAM3\n");
    printf("[orbslam3]   vocab:    %s\n", vocab_path.c_str());
    printf("[orbslam3]   settings: %s\n", settings_path.c_str());
    printf("[orbslam3]   sensor:   %s\n", sensor_str.c_str());
    printf("[orbslam3]   viewer:   %s\n", use_viewer ? "true" : "false");

    // Signal handlers
    signal(SIGTERM, signal_handler);
    signal(SIGINT, signal_handler);

    // Initialize ORB-SLAM3 (loads vocabulary, starts internal threads)
    ORB_SLAM3::System slam(vocab_path, settings_path, sensor_mode, use_viewer);

    printf("[orbslam3] System initialized, ready.\n");

    // Block until shutdown signal
    while (g_running.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    printf("[orbslam3] Shutting down...\n");
    slam.Shutdown();
    printf("[orbslam3] Done.\n");

    return 0;
}

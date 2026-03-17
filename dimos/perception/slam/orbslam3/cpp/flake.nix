{
  description = "ORB-SLAM3 native module for dimos";

  inputs = {
    # Pin to a nixpkgs that still has pangolin (removed in later revisions)
    nixpkgs.url = "github:NixOS/nixpkgs/8f3e1f807051e32d8c95cd12b9b421623850a34d";
    flake-utils.url = "github:numtide/flake-utils";
    dimos-lcm = {
      url = "github:dimensionalOS/dimos-lcm/main";
      flake = false;
    };
    orb-slam3-src = {
      url = "github:thuvasooriya/orb-slam3";
      flake = false;
    };
  };

  outputs = { self, nixpkgs, flake-utils, dimos-lcm, orb-slam3-src, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

        # Shared dimos native module header (dimos_native_module.hpp)
        dimos-native-module-dir = ../../../../hardware/sensors/lidar/common;

        # Build ORB-SLAM3 library from source
        orb-slam3 = pkgs.stdenv.mkDerivation {
          pname = "orb-slam3";
          version = "1.0.0";
          src = orb-slam3-src;

          nativeBuildInputs = [ pkgs.cmake pkgs.pkg-config ];
          buildInputs = [
            pkgs.opencv
            pkgs.eigen
            pkgs.pangolin
            pkgs.boost
            pkgs.openssl
            pkgs.glew
            pkgs.llvmPackages.openmp
          ];

          # Skip default cmake configure (we need to build DBoW2 first)
          dontConfigure = true;

          # Strip -march=native for nix reproducibility
          postPatch = ''
            sed -i 's/-march=native//g' CMakeLists.txt deps/g2o/CMakeLists.txt
          '';

          buildPhase = ''
            # Build DBoW2 first (main CMakeLists expects it pre-built)
            cd deps/DBoW2
            mkdir -p build && cd build
            cmake .. -DCMAKE_BUILD_TYPE=Release
            make -j$NIX_BUILD_CORES
            cd ../../..

            # Build main ORB-SLAM3 (g2o is built via add_subdirectory)
            # Only build the library target, skip example binaries
            mkdir -p build && cd build
            cmake .. \
              -DCMAKE_BUILD_TYPE=Release \
              -DCMAKE_INSTALL_RPATH=$out/lib \
              -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
            make -j$NIX_BUILD_CORES orb-slam3 g2o
          '';

          installPhase = ''
            cd ..  # back to source root from build/

            mkdir -p $out/lib
            mkdir -p $out/include/CameraModels
            mkdir -p $out/share/orb-slam3/Vocabulary

            # Libraries (keep original name to match SONAME)
            cp lib/liborb-slam3.so $out/lib/
            cp deps/DBoW2/lib/libDBoW2.so $out/lib/
            find . -name 'libg2o.so' -exec cp {} $out/lib/ \;

            # Headers
            cp include/*.h $out/include/
            cp include/CameraModels/*.h $out/include/CameraModels/

            # Full deps/ tree for transitive header includes
            mkdir -p $out/deps
            cp -r deps/DBoW2 $out/deps/DBoW2
            cp -r deps/g2o $out/deps/g2o
            cp -r deps/Sophus $out/deps/Sophus
            # Remove build artifacts from deps
            rm -rf $out/deps/*/build $out/deps/*/lib

            # Vocabulary (only .tar.gz is tracked in git)
            if [ ! -f Vocabulary/ORBvoc.txt ]; then
              tar -xzf Vocabulary/ORBvoc.txt.tar.gz -C Vocabulary/
            fi
            cp Vocabulary/ORBvoc.txt $out/share/orb-slam3/Vocabulary/
          '';
        };

        # Build our thin wrapper binary
        orbslam3_native = pkgs.stdenv.mkDerivation {
          pname = "orbslam3_native";
          version = "0.1.0";
          src = ./.;

          nativeBuildInputs = [ pkgs.cmake pkgs.pkg-config ];
          buildInputs = [
            orb-slam3
            pkgs.opencv
            pkgs.eigen
            pkgs.pangolin
            pkgs.boost
            pkgs.openssl
            pkgs.lcm
            pkgs.glib
            pkgs.glew
          ];

          cmakeFlags = [
            "-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
            "-DFETCHCONTENT_SOURCE_DIR_DIMOS_LCM=${dimos-lcm}"
            "-DORBSLAM3_DIR=${orb-slam3}"
            "-DDIMOS_NATIVE_MODULE_DIR=${dimos-native-module-dir}"
            "-DORBSLAM3_DEFAULT_VOCAB=${orb-slam3}/share/orb-slam3/Vocabulary/ORBvoc.txt"
          ];
        };
      in {
        packages = {
          default = orbslam3_native;
          inherit orbslam3_native orb-slam3;
        };
      });
}

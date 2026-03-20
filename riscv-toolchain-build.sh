#!/bin/bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
usage() {
  cat <<'USAGE'
Usage: riscv-toolchain-build.sh [OPTIONS] [INSTALL_PREFIX]

Build a RISC-V LLVM bare-metal toolchain with multilib support.
Works on Linux and macOS with either GCC or Clang as the host compiler.

Arguments:
  INSTALL_PREFIX    Installation directory (default: ~/opt/llvm)

Options:
  --portable        Reduce dynamic library dependencies for portability.
                    Linux: statically links libstdc++/libc++, libgcc (GCC
                    only), zlib, and zstd. Only glibc remains dynamic.
                    macOS: statically links zlib and zstd only (macOS does
                    not support static C++ stdlib or libgcc).
  -h, --help        Show this help message
USAGE
  exit 0
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
PORTABLE=0
INSTALL_PREFIX=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --portable)  PORTABLE=1; shift ;;
    -h|--help)   usage ;;
    -*)          echo "Unknown option: $1" >&2; usage ;;
    *)
      if [ -z "${INSTALL_PREFIX}" ]; then
        INSTALL_PREFIX="$1"
      else
        echo "Unexpected argument: $1" >&2; usage
      fi
      shift
      ;;
  esac
done

INSTALL_PREFIX="${INSTALL_PREFIX:-${HOME}/opt/llvm}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
NEWLIB_SRC="${SCRIPT_DIR}/newlib-src"
NEWLIB_BUILD="${SCRIPT_DIR}/newlib-build"
RUNTIMES_BUILD="${SCRIPT_DIR}/runtimes-build"
NPROC="${NPROC:-$(( $(nproc 2>/dev/null || sysctl -n hw.ncpu) / 2 ))}"

# Multilib: libraries are installed under lib/clang-runtimes/<variant>/
CLANG_RUNTIMES="${INSTALL_PREFIX}/lib/clang-runtimes"

echo "=== RISC-V LLVM Toolchain Build (multilib) ==="
echo "Source:   ${SCRIPT_DIR}"
echo "Build:    ${BUILD_DIR}"
echo "Install:  ${INSTALL_PREFIX}"
echo "Portable: ${PORTABLE}"
echo ""

# ---------------------------------------------------------------------------
# Host platform detection
# ---------------------------------------------------------------------------
HOST_OS="$(uname -s)"

# ---------------------------------------------------------------------------
# Portable build flags
#
# When --portable is given, statically link optional libraries to reduce
# dynamic dependencies.  The exact behavior depends on the host OS and
# compiler:
#
#   Linux + GCC:   static libstdc++, libgcc, zlib, zstd (only glibc dynamic)
#   Linux + Clang: static libc++/libstdc++, zlib, zstd (skip -static-libgcc)
#   macOS:         static zlib, zstd only (macOS has no static libc++/libgcc)
#
# All flags are always explicitly set (ON or OFF) so that switching between
# --portable and non-portable builds works without stale CMake cache values.
# ---------------------------------------------------------------------------
PORTABLE_CMAKE_FLAGS=()
if [ "${PORTABLE}" = "1" ]; then
  # Static zlib/zstd — works on all platforms
  PORTABLE_CMAKE_FLAGS=(
    -DLLVM_ENABLE_ZLIB=FORCE_ON
    -DLLVM_ENABLE_ZSTD=FORCE_ON
    -DLLVM_USE_STATIC_ZSTD=ON
    -DZLIB_USE_STATIC_LIBS=ON
  )

  case "${HOST_OS}" in
    Linux)
      # LLVM_STATIC_LINK_CXX_STDLIB statically links the C++ stdlib
      # (libstdc++ for GCC, libc++ for Clang).
      PORTABLE_CMAKE_FLAGS+=(-DLLVM_STATIC_LINK_CXX_STDLIB=ON)

      # -static-libgcc is a GCC-specific flag.  Only add it when the
      # host C compiler is GCC; clang does not use libgcc by default.
      # Use `cc -v` which reliably contains "gcc version" for GCC
      # regardless of the binary name (cc, gcc, etc.).
      HOST_CC="${CC:-cc}"
      if "${HOST_CC}" -v 2>&1 | grep -qi "gcc version"; then
        PORTABLE_CMAKE_FLAGS+=(-DCMAKE_EXE_LINKER_FLAGS="-static-libgcc")
      fi
      ;;
    Darwin)
      # macOS does not ship static libc++ or libgcc, so
      # LLVM_STATIC_LINK_CXX_STDLIB and -static-libgcc are not useful.
      # Static zlib/zstd (from Homebrew or system) still apply above.
      echo "Note: --portable on macOS statically links zlib/zstd only."
      echo "      Static C++ stdlib is not available on macOS."
      ;;
  esac
else
  # Non-portable: explicitly disable static linking flags to override any
  # stale values cached from a previous --portable build.
  PORTABLE_CMAKE_FLAGS=(
    -DLLVM_STATIC_LINK_CXX_STDLIB=OFF
    -DLLVM_USE_STATIC_ZSTD=OFF
    -DZLIB_USE_STATIC_LIBS=OFF
    -DCMAKE_EXE_LINKER_FLAGS=
  )
fi

# ===========================================================================
# Multilib variant definitions
#
# Each variant is: TARGET|CFLAGS|MULTILIB_DIR
#
# TARGET       -- compiler target triple
# CFLAGS       -- -march/-mabi/-mcmodel flags for this variant
# MULTILIB_DIR -- subdirectory under lib/clang-runtimes/
#
# The multilib.yaml Mappings normalize canonical -march strings to
# simplified flags that match these variant directories.
# ===========================================================================
MULTILIB_VARIANTS=(
  "riscv64-unknown-elf|-mcmodel=medany -march=rv64gc -mabi=lp64d|rv64imafdc/lp64d"
  "riscv32-unknown-elf|-march=rv32gc -mabi=ilp32d|rv32imafdc/ilp32d"
  "riscv64-unknown-elf|-mcmodel=medany -march=rv64imafc -mabi=lp64f|rv64imafc/lp64f"
  "riscv32-unknown-elf|-march=rv32imafc -mabi=ilp32f|rv32imafc/ilp32f"
  "riscv64-unknown-elf|-mcmodel=medany -march=rv64imac -mabi=lp64|rv64imac/lp64"
  "riscv32-unknown-elf|-march=rv32imac -mabi=ilp32|rv32imac/ilp32"
)

# ===========================================================================
# Stage 1: LLVM toolchain (clang, lld, lldb, llvm tools -- no runtimes)
# ===========================================================================
echo "=== Stage 1: LLVM Toolchain ==="

cmake -S "${SCRIPT_DIR}/llvm" -B "${BUILD_DIR}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_TARGETS_TO_BUILD=RISCV \
  -DLLVM_ENABLE_PROJECTS="clang;lld;llvm;lldb" \
  -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
  -DLLVM_INSTALL_TOOLCHAIN_ONLY=OFF \
  -DLLVM_DEFAULT_TARGET_TRIPLE=riscv64-unknown-elf \
  ${PORTABLE_CMAKE_FLAGS[@]+"${PORTABLE_CMAKE_FLAGS[@]}"}

cmake --build "${BUILD_DIR}" -- -j${NPROC}
cmake --install "${BUILD_DIR}"

# ===========================================================================
# Install multilib.yaml
# ===========================================================================
echo ""
echo "=== Installing multilib.yaml ==="
mkdir -p "${CLANG_RUNTIMES}"
cp "${SCRIPT_DIR}/riscv-multilib.yaml" "${CLANG_RUNTIMES}/multilib.yaml"

# ===========================================================================
# Stage 2: compiler-rt builtins (one build per multilib variant)
#
# Built standalone with CMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY.
# Each variant's libclang_rt.builtins.a is installed into its multilib
# lib directory so that clang finds it via getLibraryPaths().
# ===========================================================================
echo ""
echo "=== Stage 2: compiler-rt builtins ==="

build_compiler_rt() {
  local TARGET="$1"
  local CFLAGS="$2"
  local MULTILIB_DIR="$3"
  local VARIANT_ID="${MULTILIB_DIR//\//-}"  # rv64imafdc/lp64d -> rv64imafdc-lp64d
  local RT_BUILD="${RUNTIMES_BUILD}/compiler-rt-${VARIANT_ID}"
  local RT_INSTALL="${RUNTIMES_BUILD}/compiler-rt-${VARIANT_ID}-install"
  local DEST="${CLANG_RUNTIMES}/${MULTILIB_DIR}/lib"

  echo "--- compiler-rt: ${MULTILIB_DIR} ---"
  rm -rf "${RT_BUILD}" "${RT_INSTALL}"

  cmake -G Ninja \
    -S "${SCRIPT_DIR}/compiler-rt" \
    -B "${RT_BUILD}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_SYSTEM_NAME=Generic \
    -DCMAKE_C_COMPILER="${INSTALL_PREFIX}/bin/clang" \
    -DCMAKE_CXX_COMPILER="${INSTALL_PREFIX}/bin/clang++" \
    -DCMAKE_AR="${INSTALL_PREFIX}/bin/llvm-ar" \
    -DCMAKE_NM="${INSTALL_PREFIX}/bin/llvm-nm" \
    -DCMAKE_RANLIB="${INSTALL_PREFIX}/bin/llvm-ranlib" \
    -DCMAKE_C_COMPILER_TARGET="${TARGET}" \
    -DCMAKE_CXX_COMPILER_TARGET="${TARGET}" \
    -DCMAKE_ASM_COMPILER_TARGET="${TARGET}" \
    -DCMAKE_C_FLAGS="${CFLAGS}" \
    -DCMAKE_CXX_FLAGS="${CFLAGS}" \
    -DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY \
    -DCMAKE_INSTALL_PREFIX="${RT_INSTALL}" \
    -DCOMPILER_RT_BAREMETAL_BUILD=ON \
    -DCOMPILER_RT_BUILD_BUILTINS=ON \
    -DCOMPILER_RT_BUILD_SANITIZERS=OFF \
    -DCOMPILER_RT_BUILD_XRAY=OFF \
    -DCOMPILER_RT_BUILD_LIBFUZZER=OFF \
    -DCOMPILER_RT_BUILD_PROFILE=OFF \
    -DCOMPILER_RT_BUILD_MEMPROF=OFF \
    -DCOMPILER_RT_BUILD_ORC=OFF \
    -DCOMPILER_RT_BUILD_CTX_PROFILE=OFF \
    -DCOMPILER_RT_DEFAULT_TARGET_ONLY=ON \
    -DLLVM_ENABLE_PER_TARGET_RUNTIME_DIR=ON

  cmake --build "${RT_BUILD}" -- -j${NPROC}
  cmake --install "${RT_BUILD}"

  # Copy libclang_rt.builtins.a to the multilib lib directory
  mkdir -p "${DEST}"
  find "${RT_INSTALL}" -name "libclang_rt.builtins.a" -exec cp {} "${DEST}/" \;
}

for V in "${MULTILIB_VARIANTS[@]}"; do
  IFS='|' read -r TARGET CFLAGS MULTILIB_DIR <<< "$V"
  build_compiler_rt "${TARGET}" "${CFLAGS}" "${MULTILIB_DIR}"
done

# ===========================================================================
# Stage 3: Newlib + libgloss (libc/libm/libnosys for bare-metal)
#
# Newlib always installs to <prefix>/<target>/{lib,include}, so each
# variant is built with a temporary prefix and copied to the multilib dir.
# ===========================================================================
echo ""
echo "=== Stage 3: Newlib + libgloss ==="

# Clone newlib if not already present
if [ ! -d "${NEWLIB_SRC}" ]; then
  echo "Cloning newlib..."
  git clone --depth 1 https://sourceware.org/git/newlib-cygwin.git "${NEWLIB_SRC}"
fi

build_newlib() {
  local TARGET="$1"
  local CFLAGS="$2"
  local MULTILIB_DIR="$3"
  local VARIANT_ID="${MULTILIB_DIR//\//-}"
  local NL_BUILD="${NEWLIB_BUILD}/${VARIANT_ID}"
  local NL_INSTALL="${NEWLIB_BUILD}/${VARIANT_ID}-install"
  local DEST="${CLANG_RUNTIMES}/${MULTILIB_DIR}"

  echo "--- newlib: ${MULTILIB_DIR} ---"
  rm -rf "${NL_BUILD}"
  mkdir -p "${NL_BUILD}"

  (cd "${NL_BUILD}" && \
   "${NEWLIB_SRC}/configure" \
    --target="${TARGET}" \
    --prefix="${NL_INSTALL}" \
    --disable-newlib-supplied-syscalls \
    --enable-newlib-io-long-long \
    --enable-newlib-register-fini \
    --disable-newlib-multithread \
    --disable-shared \
    CC_FOR_TARGET="${INSTALL_PREFIX}/bin/clang --target=${TARGET}" \
    AS_FOR_TARGET="${INSTALL_PREFIX}/bin/clang --target=${TARGET}" \
    AR_FOR_TARGET="${INSTALL_PREFIX}/bin/llvm-ar" \
    RANLIB_FOR_TARGET="${INSTALL_PREFIX}/bin/llvm-ranlib" \
    CFLAGS_FOR_TARGET="-O2 ${CFLAGS}")

  make -C "${NL_BUILD}" -j${NPROC} all-target-newlib all-target-libgloss
  make -C "${NL_BUILD}" install-target-newlib install-target-libgloss

  # Copy headers and libraries to multilib directory
  mkdir -p "${DEST}/include" "${DEST}/lib"
  cp -a "${NL_INSTALL}/${TARGET}/include/." "${DEST}/include/"
  cp -a "${NL_INSTALL}/${TARGET}/lib/"*.a "${DEST}/lib/" 2>/dev/null || true
  # Also copy linker scripts and crt objects if present
  cp -a "${NL_INSTALL}/${TARGET}/lib/"*.o "${DEST}/lib/" 2>/dev/null || true
  cp -a "${NL_INSTALL}/${TARGET}/lib/"*.ld "${DEST}/lib/" 2>/dev/null || true
}

for V in "${MULTILIB_VARIANTS[@]}"; do
  IFS='|' read -r TARGET CFLAGS MULTILIB_DIR <<< "$V"
  build_newlib "${TARGET}" "${CFLAGS}" "${MULTILIB_DIR}"
done

# ===========================================================================
# Stage 4: C++ runtimes (libunwind, libcxxabi, libcxx)
#
# Built against the newlib sysroot in each multilib directory.
# CMAKE_INSTALL_PREFIX points directly to the multilib variant dir.
# ===========================================================================
echo ""
echo "=== Stage 4: C++ runtimes (libunwind + libcxxabi + libcxx) ==="

build_cxx_runtimes() {
  local TARGET="$1"
  local CFLAGS="$2"
  local MULTILIB_DIR="$3"
  local VARIANT_ID="${MULTILIB_DIR//\//-}"
  local VARIANT_DIR="${CLANG_RUNTIMES}/${MULTILIB_DIR}"

  echo "--- C++ runtimes: ${MULTILIB_DIR} ---"
  rm -rf "${RUNTIMES_BUILD}/cxx-${VARIANT_ID}"

  # Generate a CMake toolchain file for bare-metal RISC-V cross-compilation.
  # This is necessary because:
  #   - On macOS, CMake uses the host's Apple libtool for static libraries,
  #     which rejects RISC-V ELF objects.  CMAKE_AR alone doesn't override it.
  #   - CMAKE_SYSTEM_NAME=Generic prevents macOS tool detection but also
  #     disables shared library support, causing duplicate static library
  #     rules (SHARED targets silently become STATIC, colliding with real
  #     STATIC targets).
  # A toolchain file sets CMAKE_SYSTEM_NAME before project() runs, which
  # is the only reliable time to override platform-specific archiver
  # selection.  We also override the static library creation commands to
  # use llvm-ar instead of libtool.
  # Generate a CMake toolchain file for cross-compilation.
  #
  # Why a toolchain file instead of -D flags on the cmake command line?
  #   On macOS, LLVM's runtimes/CMakeLists.txt includes UseLibtool.cmake
  #   (guarded by "if(CMAKE_HOST_APPLE AND APPLE)"), which finds Apple's
  #   libtool and overrides CMAKE_*_CREATE_STATIC_LIBRARY.  Apple libtool
  #   only handles Mach-O and rejects RISC-V ELF objects.  Setting
  #   CMAKE_SYSTEM_NAME=Linux in the toolchain file makes APPLE=false,
  #   preventing UseLibtool from loading.  CMake then uses CMAKE_AR
  #   (llvm-ar) for archiving, which handles all ELF formats correctly.
  #   On Linux, CMAKE_SYSTEM_NAME=Linux is a no-op (already the default).
  local TOOLCHAIN_FILE="${RUNTIMES_BUILD}/cxx-${VARIANT_ID}-toolchain.cmake"
  mkdir -p "$(dirname "${TOOLCHAIN_FILE}")"
  cat > "${TOOLCHAIN_FILE}" <<TOOLCHAIN_EOF
# CMAKE_SYSTEM_NAME=Linux makes APPLE=false, which prevents LLVM's
# runtimes/CMakeLists.txt from including UseLibtool.cmake (guarded by
# "if(CMAKE_HOST_APPLE AND APPLE)").  Apple libtool only handles Mach-O
# and rejects RISC-V ELF objects.  Unlike Generic, Linux supports shared
# libraries, so ENABLE_SHARED=OFF is respected normally (no duplicate
# static library rules).  On Linux hosts this is a no-op since Linux
# is already the detected system name.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv)
set(CMAKE_C_COMPILER "${INSTALL_PREFIX}/bin/clang")
set(CMAKE_CXX_COMPILER "${INSTALL_PREFIX}/bin/clang++")
set(CMAKE_AR "${INSTALL_PREFIX}/bin/llvm-ar")
set(CMAKE_NM "${INSTALL_PREFIX}/bin/llvm-nm")
set(CMAKE_RANLIB "${INSTALL_PREFIX}/bin/llvm-ranlib")
set(CMAKE_C_COMPILER_TARGET "${TARGET}")
set(CMAKE_CXX_COMPILER_TARGET "${TARGET}")
set(CMAKE_ASM_COMPILER_TARGET "${TARGET}")
TOOLCHAIN_EOF

  cmake -G Ninja \
    -S "${SCRIPT_DIR}/runtimes" \
    -B "${RUNTIMES_BUILD}/cxx-${VARIANT_ID}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}" \
    -DCMAKE_C_FLAGS="${CFLAGS} --sysroot=${CLANG_RUNTIMES}" \
    -DCMAKE_CXX_FLAGS="${CFLAGS} --sysroot=${CLANG_RUNTIMES}" \
    -DCMAKE_ASM_FLAGS="--sysroot=${CLANG_RUNTIMES}" \
    -DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY \
    -DCMAKE_INSTALL_PREFIX="${VARIANT_DIR}" \
    -DLLVM_ENABLE_RUNTIMES="libunwind;libcxxabi;libcxx" \
    -DLLVM_INCLUDE_TESTS=OFF \
    -DLIBUNWIND_ENABLE_SHARED=OFF \
    -DLIBUNWIND_ENABLE_STATIC=ON \
    -DLIBUNWIND_ENABLE_THREADS=OFF \
    -DLIBUNWIND_USE_COMPILER_RT=ON \
    -DLIBUNWIND_IS_BAREMETAL=ON \
    -DLIBCXXABI_ENABLE_SHARED=OFF \
    -DLIBCXXABI_ENABLE_STATIC=ON \
    -DLIBCXXABI_ENABLE_THREADS=OFF \
    -DLIBCXXABI_USE_COMPILER_RT=ON \
    -DLIBCXXABI_USE_LLVM_UNWINDER=ON \
    -DLIBCXXABI_BAREMETAL=ON \
    -DLIBCXX_ENABLE_SHARED=OFF \
    -DLIBCXX_ENABLE_STATIC=ON \
    -DLIBCXX_ENABLE_THREADS=OFF \
    -DLIBCXX_ENABLE_FILESYSTEM=OFF \
    -DLIBCXX_ENABLE_RANDOM_DEVICE=OFF \
    -DLIBCXX_ENABLE_LOCALIZATION=OFF \
    -DLIBCXX_ENABLE_UNICODE=OFF \
    -DLIBCXX_ENABLE_WIDE_CHARACTERS=OFF \
    -DLIBCXX_ENABLE_MONOTONIC_CLOCK=OFF \
    -DLIBCXX_CXX_ABI=libcxxabi \
    -DLIBCXX_USE_COMPILER_RT=ON

  cmake --build "${RUNTIMES_BUILD}/cxx-${VARIANT_ID}" -- -j${NPROC}
  cmake --install "${RUNTIMES_BUILD}/cxx-${VARIANT_ID}"
}

for V in "${MULTILIB_VARIANTS[@]}"; do
  IFS='|' read -r TARGET CFLAGS MULTILIB_DIR <<< "$V"
  build_cxx_runtimes "${TARGET}" "${CFLAGS}" "${MULTILIB_DIR}"
done

echo ""
echo "=== Done ==="
echo "Add to PATH: export PATH=\"${INSTALL_PREFIX}/bin:\$PATH\""
echo ""
echo "Installed multilib variants:"
for V in "${MULTILIB_VARIANTS[@]}"; do
  IFS='|' read -r TARGET CFLAGS MULTILIB_DIR <<< "$V"
  echo "  ${MULTILIB_DIR}"
done
echo ""
echo "Clang selects the correct variant automatically based on -march/-mabi."
echo ""
echo "Usage examples:"
echo "  # RV64 with double FPU (selects rv64imafdc/lp64d)"
echo "  clang --target=riscv64-unknown-elf -march=rv64gc -mabi=lp64d -O2 test.c -lnosys -o test"
echo ""
echo "  # RV32 with double FPU (selects rv32imafdc/ilp32d)"
echo "  clang --target=riscv32-unknown-elf -march=rv32gc -mabi=ilp32d -O2 test.c -lnosys -o test"
echo ""
echo "  # RV32 with single-precision FPU only (selects rv32imafc/ilp32f)"
echo "  clang --target=riscv32-unknown-elf -march=rv32imafc -mabi=ilp32f -O2 test.c -lnosys -o test"
echo ""
echo "  # RV64 soft-float (selects rv64imac/lp64)"
echo "  clang --target=riscv64-unknown-elf -march=rv64imac -mabi=lp64 -O2 test.c -lnosys -o test"
echo ""
echo "  # RV64 C++ with libc++"
echo "  clang++ --target=riscv64-unknown-elf -march=rv64gc -mabi=lp64d -O2 -stdlib=libc++ test.cpp -lnosys -o test"
echo ""
echo "  # Check which variant is selected"
echo "  clang --target=riscv64-unknown-elf -march=rv64gc -print-multi-directory"
echo ""
echo "  # Override _write in your code to redirect printf to UART;"
echo "  # your definition takes priority over the stub in libnosys.a"

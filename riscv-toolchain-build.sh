#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
INSTALL_PREFIX="${1:-${HOME}/opt/riscv-llvm}"
NEWLIB_SRC="${SCRIPT_DIR}/newlib-src"
NEWLIB_BUILD="${SCRIPT_DIR}/newlib-build"
RUNTIMES_BUILD="${SCRIPT_DIR}/runtimes-build"
NPROC=$(( $(nproc 2>/dev/null || sysctl -n hw.ncpu) / 2 ))

echo "=== RISC-V LLVM Toolchain Build ==="
echo "Source:  ${SCRIPT_DIR}"
echo "Build:   ${BUILD_DIR}"
echo "Install: ${INSTALL_PREFIX}"
echo ""

# ===========================================================================
# Stage 1: LLVM toolchain (clang, lld, lldb, llvm tools -- no runtimes)
#
# compiler-rt is built separately in Stage 2 because the in-tree runtimes
# build cannot cross-compile test programs for bare-metal targets (no libc
# yet), causing test_target_arch to fail.
# ===========================================================================
echo "=== Stage 1: LLVM Toolchain ==="

cmake -S "${SCRIPT_DIR}/llvm" -B "${BUILD_DIR}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_TARGETS_TO_BUILD=RISCV \
  -DLLVM_ENABLE_PROJECTS="clang;lld;llvm;lldb" \
  -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
  -DLLVM_INSTALL_TOOLCHAIN_ONLY=OFF \
  -DLLVM_DEFAULT_TARGET_TRIPLE=riscv64-unknown-elf

cmake --build "${BUILD_DIR}" -- -j${NPROC}
cmake --install "${BUILD_DIR}"

# ===========================================================================
# Stage 2: compiler-rt builtins (standalone, for both RV32 and RV64)
#
# Built standalone with CMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY to
# avoid linking test programs. CMAKE_INSTALL_PREFIX must point to the clang
# resource directory (not the toolchain prefix) since compiler-rt installs
# into lib/clang/<version>/lib/.
# ===========================================================================
echo ""
echo "=== Stage 2: compiler-rt builtins ==="

RESOURCE_DIR=$("${INSTALL_PREFIX}/bin/clang" --print-resource-dir)

build_compiler_rt() {
  local TARGET="$1"
  local EXTRA_FLAGS="$2"

  echo "--- Building compiler-rt builtins for ${TARGET} ---"
  rm -rf "${RUNTIMES_BUILD}/compiler-rt-${TARGET}"

  cmake -G Ninja \
    -S "${SCRIPT_DIR}/compiler-rt" \
    -B "${RUNTIMES_BUILD}/compiler-rt-${TARGET}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="${INSTALL_PREFIX}/bin/clang" \
    -DCMAKE_CXX_COMPILER="${INSTALL_PREFIX}/bin/clang++" \
    -DCMAKE_AR="${INSTALL_PREFIX}/bin/llvm-ar" \
    -DCMAKE_NM="${INSTALL_PREFIX}/bin/llvm-nm" \
    -DCMAKE_RANLIB="${INSTALL_PREFIX}/bin/llvm-ranlib" \
    -DCMAKE_C_COMPILER_TARGET="${TARGET}" \
    -DCMAKE_CXX_COMPILER_TARGET="${TARGET}" \
    -DCMAKE_ASM_COMPILER_TARGET="${TARGET}" \
    -DCMAKE_C_FLAGS="${EXTRA_FLAGS}" \
    -DCMAKE_CXX_FLAGS="${EXTRA_FLAGS}" \
    -DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY \
    -DCMAKE_INSTALL_PREFIX="${RESOURCE_DIR}" \
    -DCOMPILER_RT_BAREMETAL_BUILD=ON \
    -DCOMPILER_RT_BUILD_BUILTINS=ON \
    -DCOMPILER_RT_BUILD_SANITIZERS=OFF \
    -DCOMPILER_RT_BUILD_XRAY=OFF \
    -DCOMPILER_RT_BUILD_LIBFUZZER=OFF \
    -DCOMPILER_RT_BUILD_PROFILE=OFF \
    -DCOMPILER_RT_BUILD_MEMPROF=OFF \
    -DCOMPILER_RT_BUILD_ORC=OFF \
    -DCOMPILER_RT_BUILD_CTX_PROFILE=OFF \
    -DCOMPILER_RT_DEFAULT_TARGET_ONLY=ON

  cmake --build "${RUNTIMES_BUILD}/compiler-rt-${TARGET}" -- -j${NPROC}
  cmake --install "${RUNTIMES_BUILD}/compiler-rt-${TARGET}"
}

build_compiler_rt riscv64-unknown-elf "-mcmodel=medany"
build_compiler_rt riscv32-unknown-elf ""

# ===========================================================================
# Stage 3: Newlib + libgloss (libc/libm/libnosys for bare-metal)
#
# Newlib's configure must be run from within the build directory (autotools
# convention). The --prefix must be an absolute literal path (shell variables
# may not expand in all contexts).
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
  local EXTRA_CFLAGS="$2"

  echo "--- Building newlib + libgloss for ${TARGET} ---"
  rm -rf "${NEWLIB_BUILD}/${TARGET}"
  mkdir -p "${NEWLIB_BUILD}/${TARGET}"

  (cd "${NEWLIB_BUILD}/${TARGET}" && \
   "${NEWLIB_SRC}/configure" \
    --target="${TARGET}" \
    --prefix="${INSTALL_PREFIX}" \
    --disable-newlib-supplied-syscalls \
    --enable-newlib-io-long-long \
    --enable-newlib-register-fini \
    --disable-newlib-multithread \
    --disable-shared \
    CC_FOR_TARGET="${INSTALL_PREFIX}/bin/clang --target=${TARGET}" \
    AS_FOR_TARGET="${INSTALL_PREFIX}/bin/clang --target=${TARGET}" \
    AR_FOR_TARGET="${INSTALL_PREFIX}/bin/llvm-ar" \
    RANLIB_FOR_TARGET="${INSTALL_PREFIX}/bin/llvm-ranlib" \
    CFLAGS_FOR_TARGET="-O2 ${EXTRA_CFLAGS}")

  make -C "${NEWLIB_BUILD}/${TARGET}" -j${NPROC} all-target-newlib all-target-libgloss
  make -C "${NEWLIB_BUILD}/${TARGET}" install-target-newlib install-target-libgloss
}

build_newlib riscv64-unknown-elf "-mcmodel=medany"
build_newlib riscv32-unknown-elf ""

# ===========================================================================
# Stage 4: C++ runtimes (libunwind, libcxxabi, libcxx) against newlib sysroot
#
# Key gotchas discovered during bring-up:
# - Do NOT use CMAKE_SYSROOT: the runtimes build system may override the
#   compiler path when sysroot is set. Pass --sysroot via CMAKE_C/CXX_FLAGS.
# - LIBUNWIND_IS_BAREMETAL=ON is required to skip dlfcn.h and other
#   OS-specific includes.
# - LLVM_INCLUDE_TESTS=OFF avoids a dependency on system Clang packages.
# - Compiler paths must be absolute literals (shell variable expansion in
#   cmake -D flags may silently produce empty strings).
# ===========================================================================
echo ""
echo "=== Stage 4: C++ runtimes (libunwind + libcxxabi + libcxx) ==="

build_cxx_runtimes() {
  local TARGET="$1"
  local EXTRA_FLAGS="$2"
  local SYSROOT="${INSTALL_PREFIX}/${TARGET}"

  echo "--- Building C++ runtimes for ${TARGET} ---"
  rm -rf "${RUNTIMES_BUILD}/cxx-${TARGET}"

  cmake -G Ninja \
    -S "${SCRIPT_DIR}/runtimes" \
    -B "${RUNTIMES_BUILD}/cxx-${TARGET}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="${INSTALL_PREFIX}/bin/clang" \
    -DCMAKE_CXX_COMPILER="${INSTALL_PREFIX}/bin/clang++" \
    -DCMAKE_AR="${INSTALL_PREFIX}/bin/llvm-ar" \
    -DCMAKE_NM="${INSTALL_PREFIX}/bin/llvm-nm" \
    -DCMAKE_RANLIB="${INSTALL_PREFIX}/bin/llvm-ranlib" \
    -DCMAKE_C_COMPILER_TARGET="${TARGET}" \
    -DCMAKE_CXX_COMPILER_TARGET="${TARGET}" \
    -DCMAKE_ASM_COMPILER_TARGET="${TARGET}" \
    -DCMAKE_C_FLAGS="${EXTRA_FLAGS} --sysroot=${SYSROOT}" \
    -DCMAKE_CXX_FLAGS="${EXTRA_FLAGS} --sysroot=${SYSROOT}" \
    -DCMAKE_ASM_FLAGS="--sysroot=${SYSROOT}" \
    -DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY \
    -DCMAKE_INSTALL_PREFIX="${SYSROOT}" \
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

  cmake --build "${RUNTIMES_BUILD}/cxx-${TARGET}" -- -j${NPROC}
  cmake --install "${RUNTIMES_BUILD}/cxx-${TARGET}"
}

build_cxx_runtimes riscv64-unknown-elf "-mcmodel=medany"
build_cxx_runtimes riscv32-unknown-elf ""

echo ""
echo "=== Done ==="
echo "Add to PATH: export PATH=\"${INSTALL_PREFIX}/bin:\$PATH\""
echo ""
echo "Usage examples:"
echo "  # RV64 bare-metal C with printf"
echo "  clang --target=riscv64-unknown-elf -march=rv64gc -O2 test.c -lnosys -o test"
echo ""
echo "  # RV32 bare-metal C"
echo "  clang --target=riscv32-unknown-elf -march=rv32gc -O2 test.c -lnosys -o test"
echo ""
echo "  # RV64 bare-metal C++"
echo "  clang++ --target=riscv64-unknown-elf -march=rv64gc -O2 -stdlib=libc++ test.cpp -lnosys -o test"
echo ""
echo "  # Override _write in your code to redirect printf to UART;"
echo "  # your definition takes priority over the stub in libnosys.a"

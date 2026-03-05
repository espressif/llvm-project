#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
INSTALL_PREFIX="${1:-${HOME}/opt/riscv-llvm}"
NEWLIB_SRC="${SCRIPT_DIR}/newlib-src"
NEWLIB_BUILD="${SCRIPT_DIR}/newlib-build"
RUNTIMES_BUILD="${SCRIPT_DIR}/runtimes-build"
NPROC=$(( $(nproc 2>/dev/null || sysctl -n hw.ncpu) / 2 ))

# Multilib: libraries are installed under lib/clang-runtimes/<variant>/
CLANG_RUNTIMES="${INSTALL_PREFIX}/lib/clang-runtimes"

echo "=== RISC-V LLVM Toolchain Build (multilib) ==="
echo "Source:  ${SCRIPT_DIR}"
echo "Build:   ${BUILD_DIR}"
echo "Install: ${INSTALL_PREFIX}"
echo ""

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
  -DLLVM_DEFAULT_TARGET_TRIPLE=riscv64-unknown-elf

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

  cmake -G Ninja \
    -S "${SCRIPT_DIR}/runtimes" \
    -B "${RUNTIMES_BUILD}/cxx-${VARIANT_ID}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="${INSTALL_PREFIX}/bin/clang" \
    -DCMAKE_CXX_COMPILER="${INSTALL_PREFIX}/bin/clang++" \
    -DCMAKE_AR="${INSTALL_PREFIX}/bin/llvm-ar" \
    -DCMAKE_NM="${INSTALL_PREFIX}/bin/llvm-nm" \
    -DCMAKE_RANLIB="${INSTALL_PREFIX}/bin/llvm-ranlib" \
    -DCMAKE_C_COMPILER_TARGET="${TARGET}" \
    -DCMAKE_CXX_COMPILER_TARGET="${TARGET}" \
    -DCMAKE_ASM_COMPILER_TARGET="${TARGET}" \
    -DCMAKE_C_FLAGS="${CFLAGS} --sysroot=${VARIANT_DIR}" \
    -DCMAKE_CXX_FLAGS="${CFLAGS} --sysroot=${VARIANT_DIR}" \
    -DCMAKE_ASM_FLAGS="--sysroot=${VARIANT_DIR}" \
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

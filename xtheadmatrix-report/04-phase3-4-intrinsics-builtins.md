# Phase 3-4: Intrinsics and Clang Builtins

## LLVM IR Intrinsics (`IntrinsicsRISCVXTHeadMatrix.td`)

Defined with `TargetPrefix = "riscv"` and `int_riscv_th_` naming convention.

### Design approach

Uses the ManagedRA (Spec-API) programming model exclusively. All matrix
operations use `_internal` intrinsics that produce/consume
`target("riscv.matrix")` SSA values. The register allocator manages matrix
registers (tr0-tr3, acc0-acc3) automatically via register class constraints
on pseudo-instructions.

### Intrinsic helper classes

| Class | Signature | Used for |
|-------|-----------|----------|
| `THMatrix_Internal_Load` | `(ptr, stride) → matrix` | Element/tile-stride loads |
| `THMatrix_Internal_LoadWhole` | `(ptr) → matrix` | Whole-register loads |
| `THMatrix_Internal_Store` | `(matrix, ptr, stride) → void` | Element/tile-stride stores |
| `THMatrix_Internal_StoreWhole` | `(matrix, ptr) → void` | Whole-register stores |
| `THMatrix_Internal_Zero` | `() → matrix` | Zero initialization |
| `THMatrix_Internal_MulAcc` | `(acc, ms2, ms1) → acc` | Matmul, EW .mm, n4clip .mm |
| `THMatrix_Internal_MulAccImm` | `(acc, ms2, ms1, imm) → acc` | EW .mv.i, n4clip .mv.i |
| `THMatrix_Internal_Unary` | `(ms1) → md` | Move, conversions |
| `THMatrix_Internal_Binary` | `(ms2, ms1) → md` | Pack |
| `THMatrix_Internal_UnaryImm` | `(ms1, imm) → md` | Slide, broadcast |
| `THMatrix_Internal_ToGPR` | `(matrix, idx) → gpr` | Matrix-element-to-GPR |
| `THMatrix_Internal_FromGPR2` | `(matrix, data, idx) → matrix` | GPR-to-matrix-element |
| `THMatrix_Internal_FromGPR` | `(matrix, data) → matrix` | GPR broadcast to matrix |

### Coverage (~227 intrinsics total)

| Category | Count | Details |
|----------|-------|---------|
| Configuration | 7 | mrelease + 6 msettile variants |
| Load _internal | 28 | A/B/C element/tile-stride + M whole-register × 4 EEW |
| Store _internal | 28 | Same layout as loads |
| FP matmul | 13 | Same-width, widening FP8/FP16/BF16/TF32/FP32 |
| INT matmul | 12 | INT8→INT32, INT16→INT64 (signed variants) |
| Partial/bypass matmul | 6 | Partial INT8→INT32, bypass INT |
| Misc | 35 | mzero(4), mmov(9), mdup(4), mpack(3), slide(10), broadcast(5) |
| FP format conversions | 26 | FP8/FP16/BF16/FP32/FP64/TF32 inter-conversions |
| Float-int conversions | 12 | INT8↔FP16, INT32↔FP32 (signed/unsigned) |
| Fixed-point clip | 8 | .mm and .mv.i (signed/unsigned × lower/upper) |
| Packed conversions | 4 | lower/upper (signed/unsigned) |
| Integer EW | 22 | 11 ops × .mm/.mv.i |
| FP EW | 30 | 5 ops × 3 precisions × .mm/.mv.i |

## Clang Builtins (`BuiltinsRISCVXTHeadMatrix.td`)

Defined with `__builtin_riscv_th_` prefix, gated by `"experimental-xtheadmatrix"`.

### Design approach

Spec-API builtins accept and return `__rvm_*_t` matrix types. No register
index parameters. Each builtin emits the corresponding `_internal` intrinsic
with `target("riscv.matrix")` SSA values, plus CSR configuration calls
(msettilem/k/n) where needed.

### Builtin categories (~272 total)

| Category | Count | Prototype pattern |
|----------|-------|-------------------|
| Config | 7 | `void(size_t)` or `void()` |
| Mundef | 22 | `__rvm_*_t()` — returns PoisonValue |
| Tuple mget | 11 | `__rvm_*_t(__rvm_*x2_t, size_t)` — extractvalue+select |
| Tuple mset | 11 | `__rvm_*x2_t(__rvm_*x2_t, size_t, __rvm_*_t)` — insertvalue+select |
| Spec-API loads | 33 | `__rvm_*_t(void*, size_t, size_t, size_t)` |
| Spec-API stores | 11 | `void(void*, size_t, __rvm_*_t, size_t, size_t)` |
| Spec-API matmul | 27 | `__rvm_*_t(__rvm_*_t, __rvm_*_t, __rvm_*_t, size_t, size_t, size_t)` |
| Spec-API zero | 11 | `__rvm_*_t(size_t, size_t)` |
| Spec-API EW .mm | 26 | `__rvm_*_t(__rvm_*_t, __rvm_*_t, __rvm_*_t)` |
| Spec-API EW .mv.i | 26 | `__rvm_*_t(__rvm_*_t, __rvm_*_t, __rvm_*_t, unsigned int)` |
| Spec-API conversions | 42 | `__rvm_*_t(__rvm_*_t)` |
| Spec-API n4clip | 8 | .mm (3 matrix) / .mv.i (3 matrix + imm) |
| Spec-API data movement | ~20 | move, dup, pack, slide, broadcast |

### Include wiring

```tablegen
// IntrinsicsRISCV.td
include "llvm/IR/IntrinsicsRISCVXTHeadMatrix.td"

// BuiltinsRISCV.td
include "clang/Basic/BuiltinsRISCVXTHeadMatrix.td"
```

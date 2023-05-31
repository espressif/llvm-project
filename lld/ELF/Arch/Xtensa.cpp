//===- Xtensa.cpp ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InputFiles.h"
#include "Symbols.h"
#include "Target.h"
#include <bitset>
#include <iostream>
#include <string>

using namespace llvm;
using namespace llvm::object;
using namespace llvm::support::endian;
using namespace llvm::ELF;
using namespace lld;
using namespace lld::elf;

namespace {

class Xtensa final : public TargetInfo {
public:
  Xtensa();
  RelExpr getRelExpr(RelType type, const Symbol &s,
                     const uint8_t *loc) const override;
  void relocate(uint8_t *loc, const Relocation &rel,
                uint64_t val) const override;
};

} // namespace

Xtensa::Xtensa() {}

RelExpr Xtensa::getRelExpr(RelType type, const Symbol &s,
                           const uint8_t *loc) const {
  switch (type) {
  case R_XTENSA_32:
    return R_ABS;
  case R_XTENSA_SLOT0_OP:
    // This relocation is used for various instructions, with varying ways to
    // calculate the relocation value. This is unlike most ELF architectures,
    // and is arguably bad design (see the comment on R_386_GOT32 in X86.cpp).
    // But that's what compilers emit, so it needs to be supported.
    //
    // We work around this by returning R_PC here and calculating the PC address
    // in Xtensa::relocate based on the relative value. That's ugly. A better
    // solution would be to look at the instruction here and emit various
    // Xtensa-specific RelTypes, but that has another problem: the RelExpr enum
    // is at its maximum size of 64. This will need to be fixed eventually, but
    // for now hack around it and return R_PC.
    return R_PC;
  case R_XTENSA_ASM_EXPAND:
    // This relocation appears to be emitted by the GNU Xtensa compiler as a
    // linker relaxation hint. For example, for the following code:
    //
    //   .section .foo
    //   .align  4
    //   foo:
    //       nop
    //       nop
    //       call0 bar
    //   .align  4
    //       bar:
    //
    // The call0 instruction is compiled to a l32r and callx0 instruction.
    // The LLVM Xtensa backend does not emit this relocation.
    // Because it's a relaxation hint, this relocation can be ignored for now
    // until linker relaxations are implemented.
    return R_NONE;
  case R_XTENSA_DIFF8:
  case R_XTENSA_DIFF16:
  case R_XTENSA_DIFF32:
  case R_XTENSA_PDIFF8:
  case R_XTENSA_PDIFF16:
  case R_XTENSA_PDIFF32:
  case R_XTENSA_NDIFF8:
  case R_XTENSA_NDIFF16:
  case R_XTENSA_NDIFF32:
    // > Xtensa relocations to mark the difference of two local symbols.
    // > These are only needed to support linker relaxation and can be ignored
    // > when not relaxing.
    // Source:
    // https://github.com/espressif/binutils-gdb/commit/30ce8e47fad9b057b6d7af9e1d43061126d34d20:
    // Because we don't do linker relaxation, we can ignore these relocations.
    return R_NONE;
  default:
    error(getErrorLocation(loc) + "unknown relocation (" + Twine(type) +
          ") against symbol " + toString(s));
    return R_NONE;
  }
}

static inline bool isRRI8Branch(uint8_t *loc) {
  if ((loc[0] & 0x0f) == 0b0111) {
    // instructions: ball, bany, bbc, bbci, bbs, bbsi, beq, bge, bgeu, blt,
    // bltu, bnall, bne, bnone
    return true;
  }
  if ((loc[0] & 0b11'1111) == 0b10'0110) {
    // instructions: beqi, bgei, bnei, blti
    return true;
  }
  if ((loc[0] & 0b1011'1111) == 0b1011'0110) {
    // instructions: bgeui, bltui
    return true;
  }
  // some other instruction
  return false;
}

void Xtensa::relocate(uint8_t *loc, const Relocation &rel, uint64_t val) const {
  switch (rel.type) {
  case R_XTENSA_32:
    write32le(loc, val);
    break;
  case R_XTENSA_SLOT0_OP: {
    // HACK: calculate the instruction location based on the PC-relative
    // relocation value.
    uint64_t dest = rel.sym->getVA(rel.addend);
    uint64_t p = dest - val;

    // This relocation is used for various instructions.
    // Look at the instruction to determine how to do the relocation.
    uint8_t opcode = loc[0] & 0x0f;
    if (opcode == 0b0001) { // RI16 format: l32r
      uint64_t val = dest - ((p + 3) & (uint64_t)0xfffffffc);
      checkInt(loc, static_cast<int64_t>(val) >> 2, 16, rel);
      checkAlignment(loc, val, 4, rel);
      write16le(loc + 1, static_cast<int64_t>(val) >> 2);
    } else if (opcode == 0b0101) { // call0, call4, call8, call12 (CALL format)
      uint64_t val = dest - ((p + 4) & (uint64_t)0xfffffffc);
      checkInt(loc, static_cast<int64_t>(val) >> 2, 18, rel);
      checkAlignment(loc, val, 4, rel);
      const int64_t target = static_cast<int64_t>(val) >> 2;
      loc[0] = (loc[0] & 0b0011'1111) | ((target & 0b0000'0011) << 6);
      loc[1] = target >> 2;
      loc[2] = target >> 10;
    } else if ((loc[0] & 0x3f) == 0b00'0110) { // j (CALL format)
      uint64_t valJ = val - 4;
      checkInt(loc, static_cast<int64_t>(valJ), 18, rel);
      loc[0] = (loc[0] & 0b0011'1111) | ((valJ & 0b0000'0011) << 6);
      loc[1] = valJ >> 2;
      loc[2] = valJ >> 10;
    } else if (isRRI8Branch(loc)) { // RRI8 format (various branch instructions)
      uint64_t v = val - 4;
      checkInt(loc, static_cast<int64_t>(v), 8, rel);
      loc[2] = v & 0xff;
    } else if ((loc[0] & 0b1000'1111) == 0b1000'1100) { // RI16 format: beqz.n, bnez.n
      uint64_t v = val - 4;
      checkUInt(loc, v, 6, rel);
      loc[0] = (loc[0] & 0xcf) | (v & 0x30);
      loc[1] = (loc[1] & 0x0f) | ((v & 0x0f) << 4);
    } else if ((loc[0] & 0b0011'1111) == 0b0001'0110) { // BRI12 format: beqz, bgez, bltz, bnez
      uint64_t v = val - 4;
      checkInt(loc, static_cast<int64_t>(v), 12, rel);
      loc[1] = ((loc[1] & 0x0f)) | ((v & 0x0f) << 4);
      loc[2] = (v >> 4) & 0xff;
    } else {
      error(getErrorLocation(loc) +
            "unknown opcode for relocation: " + std::to_string(loc[0]));
    }
    break;
  }
  default:
    llvm_unreachable("unknown relocation");
  }
}

TargetInfo *elf::getXtensaTargetInfo() {
  static Xtensa target;
  return &target;
}

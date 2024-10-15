//===- XtensaConstantIslandPass.cpp - Emit Pc Relative loads
//----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass is used to make Pc relative loads of constants.
//
// Loading constants inline is expensive on Xtensa and it's in general better
// to place the constant nearby in code space and then it can be loaded with a
// simple l32r instruction.
//
// The constants can be not just numbers but addresses of functions and labels.
// This can be particularly helpful in static relocation mode for embedded
// non-linux targets.
//
//===----------------------------------------------------------------------===//

#include "Xtensa.h"
#include "XtensaConstantPoolValue.h"
#include "XtensaMachineFunctionInfo.h"
#include "XtensaSubtarget.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iterator>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "xtensa-constant-islands"

STATISTIC(NumCPEs, "Number of constpool entries");
STATISTIC(NumSplit, "Number of uncond branches inserted");
STATISTIC(NumCBrFixed, "Number of cond branches fixed");
STATISTIC(NumUBrFixed, "Number of uncond branches fixed");

// FIXME: This option should be removed once it has received sufficient testing.
static cl::opt<bool>
    AlignConstantIslands("xtensa-align-constant-islands", cl::Hidden,
                         cl::init(true),
                         cl::desc("Align constant islands in code"));

// Rather than do make check tests with huge amounts of code, we force
// the test to use this amount.
static cl::opt<int> ConstantIslandsSmallOffset(
    "xtensa-constant-islands-small-offset", cl::init(0),
    cl::desc("Make small offsets be this amount for testing purposes"),
    cl::Hidden);

// TODO
// This defines for L32R and J instruction displacemnt for 
// testing purposes only
#define MAX_DISP_L32R 262144
#define BITS_JUMP 18

static unsigned int branchTargetOperand(MachineInstr *MI) {
  switch (MI->getOpcode()) {
  case Xtensa::J:
    return 0;
  case Xtensa::BEQ:
  case Xtensa::BNE:
  case Xtensa::BLT:
  case Xtensa::BLTU:
  case Xtensa::BGE:
  case Xtensa::BGEU:
  case Xtensa::BEQI:
  case Xtensa::BNEI:
  case Xtensa::BLTI:
  case Xtensa::BLTUI:
  case Xtensa::BGEI:
  case Xtensa::BGEUI:
    return 2;
  case Xtensa::BEQZ:
  case Xtensa::BNEZ:
  case Xtensa::BLTZ:
  case Xtensa::BGEZ:
    return 1;
  case Xtensa::BT:
  case Xtensa::BF:
    return 1;
  }
  llvm_unreachable("Unknown branch type");
}

namespace {

using Iter = MachineBasicBlock::iterator;
using ReverseIter = MachineBasicBlock::reverse_iterator;

/// XtensaConstantIslands - Due to limited PC-relative displacements, Xtensa
/// requires constant pool entries to be scattered among the instructions
/// inside a function.  To do this, it completely ignores the normal LLVM
/// constant pool; instead, it places constants wherever it feels like with
/// special instructions.
///
/// The terminology used in this pass includes:
///   Islands - Clumps of constants placed in the function.
///   Water   - Potential places where an island could be formed.
///   CPE     - A constant pool entry that has been placed somewhere, which
///             tracks a list of users.

class XtensaConstantIslands : public MachineFunctionPass {
  /// BasicBlockInfo - Information about the offset and size of a single
  /// basic block.
  struct BasicBlockInfo {
    /// Offset - Distance from the beginning of the function to the beginning
    /// of this basic block.
    ///
    /// Offsets are computed assuming worst case padding before an aligned
    /// block. This means that subtracting basic block offsets always gives a
    /// conservative estimate of the real distance which may be smaller.
    ///
    /// Because worst case padding is used, the computed offset of an aligned
    /// block may not actually be aligned.
    unsigned Offset = 0;

    /// Size - Size of the basic block in bytes.  If the block contains
    /// inline assembly, this is a worst case estimate.
    ///
    /// The size does not include any alignment padding whether from the
    /// beginning of the block, or from an aligned jump table at the end.
    unsigned Size = 0;

    BasicBlockInfo() = default;

    unsigned postOffset() const { return Offset + Size; }
  };

  std::vector<BasicBlockInfo> BBInfo;

  /// WaterList - A sorted list of basic blocks where islands could be placed
  /// (i.e. blocks that don't fall through to the following block, due
  /// to a return, unreachable, or unconditional branch).
  std::vector<MachineBasicBlock *> WaterList;

  /// NewWaterList - The subset of WaterList that was created since the
  /// previous iteration by inserting unconditional branches.
  SmallSet<MachineBasicBlock *, 4> NewWaterList;

  using water_iterator = std::vector<MachineBasicBlock *>::iterator;

  /// CPUser - One user of a constant pool, keeping the machine instruction
  /// pointer, the constant pool being referenced, and the max displacement
  /// allowed from the instruction to the CP.  The LowWaterMark records the
  /// lowest basic block where a new CPEntry can be placed.  To ensure this
  /// pass terminates, the CP entries are initially placed at the second block
  /// of the function and then move monotonically to higher addresses.  The
  /// exception to this rule is when the current CP entry for a particular
  /// CPUser is out of range, but there is another CP entry for the same
  /// constant value in range.  We want to use the existing in-range CP
  /// entry, but if it later moves out of range, the search for new water
  /// should resume where it left off.  The LowWaterMark is used to record
  /// that point.
  struct CPUser {
    MachineInstr *MI;
    MachineInstr *CPEMI;
    MachineBasicBlock *LowWaterMark;

  private:
    unsigned MaxDisp;

  public:
    CPUser(MachineInstr *mi, MachineInstr *cpemi, unsigned maxdisp)
        : MI(mi), CPEMI(cpemi), MaxDisp(maxdisp) {
      LowWaterMark = CPEMI->getParent();
    }

    /// getMaxDisp - Returns the maximum displacement supported by MI.
    unsigned getMaxDisp() const {
      unsigned xMaxDisp =
          ConstantIslandsSmallOffset ? ConstantIslandsSmallOffset : MaxDisp;
      return xMaxDisp;
    }

    void setMaxDisp(unsigned val) { MaxDisp = val; }
  };

  /// CPUsers - Keep track of all of the machine instructions that use various
  /// constant pools and their max displacement.
  std::vector<CPUser> CPUsers;

  /// CPEntry - One per constant pool entry, keeping the machine instruction
  /// pointer, the constpool index, and the number of CPUser's which
  /// reference this entry.
  struct CPEntry {
    MachineInstr *CPEMI;
    unsigned CPI;
    unsigned RefCount;

    CPEntry(MachineInstr *cpemi, unsigned cpi, unsigned rc = 0)
        : CPEMI(cpemi), CPI(cpi), RefCount(rc) {}
  };

  /// CPEntries - Keep track of all of the constant pool entry machine
  /// instructions. For each original constpool index (i.e. those that
  /// existed upon entry to this pass), it keeps a vector of entries.
  /// Original elements are cloned as we go along; the clones are
  /// put in the vector of the original element, but have distinct CPIs.
  std::vector<std::vector<CPEntry>> CPEntries;

  /// ImmBranch - One per immediate branch, keeping the machine instruction
  /// pointer, conditional or unconditional, the max displacement,
  /// and (if isCond is true) the corresponding unconditional branch
  /// opcode.
  struct ImmBranch {
    MachineInstr *MI;
    unsigned MaxDisp : 31;
    bool isCond : 1;
    int UncondBr;

    ImmBranch(MachineInstr *mi, unsigned maxdisp, bool cond, int ubr)
        : MI(mi), MaxDisp(maxdisp), isCond(cond), UncondBr(ubr) {}
  };

  /// ImmBranches - Keep track of all the immediate branch instructions.
  ///
  std::vector<ImmBranch> ImmBranches;

  const XtensaSubtarget *STI = nullptr;
  const XtensaInstrInfo *TII;
  const TargetRegisterInfo *TRI;
  XtensaFunctionInfo *MFI;
  MachineFunction *MF = nullptr;
  MachineConstantPool *MCP = nullptr;
  MachineBasicBlock *InitConstantMBB = nullptr;
  std::unique_ptr<RegScavenger> RS;
  LivePhysRegs LiveRegs;

  unsigned PICLabelUId;
  bool PrescannedForConstants = false;

  void initPICLabelUId(unsigned UId) { PICLabelUId = UId; }

  unsigned createPICLabelUId() { return PICLabelUId++; }

public:
  static char ID;

  XtensaConstantIslands() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override { return "Xtensa Constant Islands"; }

  bool runOnMachineFunction(MachineFunction &F) override;

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::NoVRegs);
  }

  void doInitialPlacement(std::vector<MachineInstr *> &CPEMIs);
  CPEntry *findConstPoolEntry(unsigned CPI, const MachineInstr *CPEMI);
  Align getCPEAlign(const MachineInstr &CPEMI);
  void initializeFunctionInfo(const std::vector<MachineInstr *> &CPEMIs);
  unsigned getOffsetOf(MachineInstr *MI) const;
  unsigned getUserOffset(CPUser &) const;
  void dumpBBs();

  bool isOffsetInRange(unsigned UserOffset, unsigned TrialOffset,
                       unsigned Disp);

  void computeBlockSize(MachineBasicBlock *MBB);
  MachineBasicBlock *splitBlockBeforeInstr(MachineInstr &MI);
  void updateForInsertedWaterBlock(MachineBasicBlock *NewBB);
  void adjustBBOffsetsAfter(MachineBasicBlock *BB);
  bool decrementCPEReferenceCount(unsigned CPI, MachineInstr *CPEMI);
  int findInRangeCPEntry(CPUser &U, unsigned UserOffset);
  bool findAvailableWater(CPUser &U, unsigned UserOffset,
                          water_iterator &WaterIter);
  void createNewWater(unsigned CPUserIndex, unsigned UserOffset,
                      MachineBasicBlock *&NewMBB);
  bool handleConstantPoolUser(unsigned CPUserIndex);
  void removeDeadCPEMI(MachineInstr *CPEMI);
  bool removeUnusedCPEntries();
  bool isCPEntryInRange(MachineInstr *MI, unsigned UserOffset,
                        MachineInstr *CPEMI, unsigned Disp,
                        bool DoDump = false);
  bool isWaterInRange(unsigned UserOffset, MachineBasicBlock *Water, CPUser &U,
                      unsigned &Growth);
  bool isBBInRange(MachineInstr *MI, MachineBasicBlock *BB, unsigned Disp);
  bool fixupImmediateBr(ImmBranch &Br);
  bool fixupConditionalBr(ImmBranch &Br);
  bool fixupUnconditionalBr(ImmBranch &Br);
  void removeEntryJump();
};

} // end anonymous namespace

char XtensaConstantIslands::ID = 0;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
/// print block size and offset information - debugging
LLVM_DUMP_METHOD void XtensaConstantIslands::dumpBBs() {
  for (unsigned J = 0, E = BBInfo.size(); J != E; ++J) {
    const BasicBlockInfo &BBI = BBInfo[J];
    dbgs() << format("%08x %bb.%u\t", BBI.Offset, J)
           << format(" size=%#x\n", BBInfo[J].Size);
  }
}
#endif

bool XtensaConstantIslands::runOnMachineFunction(MachineFunction &mf) {
  MF = &mf;
  MCP = mf.getConstantPool();
  STI = &mf.getSubtarget<XtensaSubtarget>();
  LLVM_DEBUG(dbgs() << "constant island machine function "
                    << "\n");
  TII = (const XtensaInstrInfo *)STI->getInstrInfo();
  MFI = MF->getInfo<XtensaFunctionInfo>();

  TRI = STI->getRegisterInfo();

  if (!STI->useTextSectionLiterals())
    return false;

  if (TRI->trackLivenessAfterRegAlloc(*MF))
    RS.reset(new RegScavenger());

  LLVM_DEBUG(dbgs() << "constant island processing "
                    << "\n");

  // Renumber all of the machine basic blocks in the function, guaranteeing that
  // the numbers agree with the position of the block in the function.
  MF->RenumberBlocks();

  bool MadeChange = false;

  // Perform the initial placement of the constant pool entries.  To start with,
  // we put them all at the end of the function.
  std::vector<MachineInstr *> CPEMIs;
  doInitialPlacement(CPEMIs);

  // Renumber all of the machine basic blocks in the function, guaranteeing
  // that the numbers agree with the position of the block in the function.
  MF->RenumberBlocks();

  /// The next UID to take is the first unused one.
  initPICLabelUId(CPEMIs.size());

  // Do the initial scan of the function, building up information about the
  // sizes of each block, the location of all the water, and finding all of the
  // constant pool users.
  initializeFunctionInfo(CPEMIs);
  CPEMIs.clear();
  LLVM_DEBUG(dumpBBs());

  /// Remove dead constant pool entries.
  MadeChange |= removeUnusedCPEntries();

  // Iteratively place constant pool entries and fix up branches until there
  // is no change.
  unsigned NoCPIters = 0, NoBRIters = 0;
  (void)NoBRIters;
  while (true) {
    LLVM_DEBUG(dbgs() << "Beginning CP iteration #" << NoCPIters << '\n');
    bool CPChange = false;
    for (unsigned i = 0, e = CPUsers.size(); i != e; ++i) {
      CPChange |= handleConstantPoolUser(i);
    }
    if (CPChange && ++NoCPIters > 30)
      report_fatal_error("Constant Island pass failed to converge!");
    LLVM_DEBUG(dumpBBs());

    // Clear NewWaterList now.  If we split a block for branches, it should
    // appear as "new water" for the next iteration of constant pool placement.
    NewWaterList.clear();

    LLVM_DEBUG(dbgs() << "Beginning BR iteration #" << NoBRIters << '\n');
    bool BRChange = false;
    for (unsigned i = 0, e = ImmBranches.size(); i != e; ++i)
      BRChange |= fixupImmediateBr(ImmBranches[i]);
    if (BRChange && ++NoBRIters > 30)
      report_fatal_error("Branch Fix Up pass failed to converge!");
    LLVM_DEBUG(dumpBBs());

    if (!CPChange && !BRChange)
      break;

    MadeChange = true;
  }
  removeEntryJump();
  LLVM_DEBUG(dbgs() << '\n'; dumpBBs());
  BBInfo.clear();
  WaterList.clear();
  CPUsers.clear();
  CPEntries.clear();
  ImmBranches.clear();
  return MadeChange;
}

/// BBHasFallthrough - Return true if the specified basic block can fallthrough
/// into the block immediately after it.
static bool BBHasFallthrough(MachineBasicBlock *MBB) {
  // Get the next machine basic block in the function.
  MachineFunction::iterator MBBI = MBB->getIterator();
  // Can't fall off end of function.
  if (std::next(MBBI) == MBB->getParent()->end())
    return false;

  MachineBasicBlock *NextBB = &*std::next(MBBI);
  return llvm::is_contained(MBB->successors(), NextBB);
}

/// doInitialPlacement - Perform the initial placement of the constant pool
/// entries.  To start with, we put them all at the end of the function.
void XtensaConstantIslands::doInitialPlacement(
    std::vector<MachineInstr *> &CPEMIs) {
  // Create the basic block to hold the CPE's.
  MachineBasicBlock *BB = MF->CreateMachineBasicBlock();

  // TODO
  MachineBasicBlock *Entry = &MF->front();
  MachineBasicBlock *NewEntry = MF->CreateMachineBasicBlock();

  MF->insert(Entry->getIterator(), NewEntry);
  BuildMI(NewEntry, DebugLoc(), TII->get(Xtensa::J)).addMBB(Entry);
  NewEntry->addSuccessor(Entry);
  NewEntry->setAlignment(Entry->getAlignment());

  // Copy live-in information to new block.
  for (const MachineBasicBlock::RegisterMaskPair &RegMaskPair :
       Entry->liveins())
    NewEntry->addLiveIn(RegMaskPair);

  // MachineConstantPool measures alignment in bytes. We measure in log2(bytes).
  const Align MaxAlign = MCP->getConstantPoolAlign();

  BB->setAlignment(AlignConstantIslands ? MaxAlign : Align(4));

  MF->insert(Entry->getIterator(), BB);

  // The function needs to be as aligned as the basic blocks. The linker may
  // move functions around based on their alignment.
  MF->ensureAlignment(BB->getAlignment());

  // Order the entries in BB by descending alignment.  That ensures correct
  // alignment of all entries as long as BB is sufficiently aligned.  Keep
  // track of the insertion point for each alignment.  We are going to bucket
  // sort the entries as they are created.
  SmallVector<MachineBasicBlock::iterator, 8> InsPoint(Log2(MaxAlign) + 1,
                                                       BB->end());

  // Add all of the constants from the constant pool to the end block, use an
  // identity mapping of CPI's to CPE's.
  const std::vector<MachineConstantPoolEntry> &CPs = MCP->getConstants();

  const DataLayout &TD = MF->getDataLayout();
  for (unsigned i = 0, e = CPs.size(); i != e; ++i) {
    unsigned Size = CPs[i].getSizeInBytes(TD);
    assert(Size >= 4 && "Too small constant pool entry");
    Align Alignment = CPs[i].getAlign();
    // Verify that all constant pool entries are a multiple of their alignment.
    // If not, we would have to pad them out so that instructions stay aligned.
    assert(isAligned(Alignment, Size) && "CP Entry not multiple of 4 bytes!");

    // Insert CONSTPOOL_ENTRY before entries with a smaller alignment.
    unsigned LogAlign = Log2(Alignment);
    MachineBasicBlock::iterator InsAt = InsPoint[LogAlign];

    MachineInstr *CPEMI =
        BuildMI(*BB, InsAt, DebugLoc(), TII->get(Xtensa::CONSTPOOL_ENTRY))
            .addImm(i)
            .addConstantPoolIndex(i)
            .addImm(Size);

    CPEMIs.push_back(CPEMI);

    // Ensure that future entries with higher alignment get inserted before
    // CPEMI. This is bucket sort with iterators.
    for (unsigned a = LogAlign + 1; a <= Log2(MaxAlign); ++a)
      if (InsPoint[a] == InsAt)
        InsPoint[a] = CPEMI;
    // Add a new CPEntry, but no corresponding CPUser yet.
    CPEntries.emplace_back(1, CPEntry(CPEMI, i));
    ++NumCPEs;
    LLVM_DEBUG(dbgs() << "Moved CPI#" << i << " to end of function, size = "
                      << Size << ", align = " << Alignment.value() << '\n');
  }
  InitConstantMBB = BB;
  LLVM_DEBUG(BB->dump());
}

/// findConstPoolEntry - Given the constpool index and CONSTPOOL_ENTRY MI,
/// look up the corresponding CPEntry.
XtensaConstantIslands::CPEntry *
XtensaConstantIslands::findConstPoolEntry(unsigned CPI,
                                          const MachineInstr *CPEMI) {
  std::vector<CPEntry> &CPEs = CPEntries[CPI];
  // Number of entries per constpool index should be small, just do a
  // linear search.
  for (CPEntry &CPE : CPEs) {
    if (CPE.CPEMI == CPEMI)
      return &CPE;
  }
  return nullptr;
}

/// getCPEAlign - Returns the required alignment of the constant pool entry
/// represented by CPEMI.  Alignment is measured in log2(bytes) units.
Align XtensaConstantIslands::getCPEAlign(const MachineInstr &CPEMI) {
  assert(CPEMI.getOpcode() == Xtensa::CONSTPOOL_ENTRY);

  // Everything is 4-byte aligned unless AlignConstantIslands is set.
  if (!AlignConstantIslands)
    return Align(4);

  unsigned CPI = CPEMI.getOperand(1).getIndex();
  assert(CPI < MCP->getConstants().size() && "Invalid constant pool index.");
  return MCP->getConstants()[CPI].getAlign();
}

/// initializeFunctionInfo - Do the initial scan of the function, building up
/// information about the sizes of each block, the location of all the water,
/// and finding all of the constant pool users.
void XtensaConstantIslands::initializeFunctionInfo(
    const std::vector<MachineInstr *> &CPEMIs) {
  BBInfo.clear();
  BBInfo.resize(MF->getNumBlockIDs());

  // First thing, compute the size of all basic blocks, and see if the function
  // has any inline assembly in it. If so, we have to be conservative about
  // alignment assumptions, as we don't know for sure the size of any
  // instructions in the inline assembly.
  for (MachineBasicBlock &MBB : *MF)
    computeBlockSize(&MBB);

  // Compute block offsets.
  adjustBBOffsetsAfter(&MF->front());

  // Now go back through the instructions and build up our data structures.
  for (MachineBasicBlock &MBB : *MF) {
    // If this block doesn't fall through into the next MBB, then this is
    // 'water' that a constant pool island could be placed.
    if (!BBHasFallthrough(&MBB))
      WaterList.push_back(&MBB);
    for (MachineInstr &MI : MBB) {
      if (MI.isDebugInstr())
        continue;

      int Opc = MI.getOpcode();
      if (MI.isBranch()) {
        bool isCond = false;
        unsigned Bits = 0;
        unsigned Scale = 1;
        int UOpc = Xtensa::J;
        switch (Opc) {
        default:
          continue; // Ignore other branches for now
        case Xtensa::J:
          Bits = BITS_JUMP;
          Scale = 1;
          isCond = false;
          break;
        case Xtensa::BEQ:
        case Xtensa::BNE:
        case Xtensa::BLT:
        case Xtensa::BLTU:
        case Xtensa::BGE:
        case Xtensa::BGEU:
        case Xtensa::BEQI:
        case Xtensa::BNEI:
        case Xtensa::BLTI:
        case Xtensa::BLTUI:
        case Xtensa::BGEI:
        case Xtensa::BGEUI:
          Bits = 8;
          Scale = 1;
          isCond = true;
          break;
        case Xtensa::BEQZ:
        case Xtensa::BNEZ:
        case Xtensa::BLTZ:
        case Xtensa::BGEZ:
          Bits = 12;
          Scale = 1;
          isCond = true;
          break;
        case Xtensa::BT:
        case Xtensa::BF:
          Bits = 8;
          Scale = 1;
          isCond = true;
          break;
        }
        // Record this immediate branch.
        unsigned MaxOffs = ((1 << (Bits - 1)) - 1) * Scale - 4;
        ImmBranches.push_back(ImmBranch(&MI, MaxOffs, isCond, UOpc));
      }

      if (Opc == Xtensa::CONSTPOOL_ENTRY)
        continue;

      // Scan the instructions for constant pool operands.
      for (const MachineOperand &MO : MI.operands())
        if (MO.isCPI()) {
          // We found one.  The addressing mode tells us the max displacement
          // from the PC that this instruction permits.
          unsigned CPI = MO.getIndex();
          MachineInstr *CPEMI = CPEMIs[CPI];

          switch (Opc) {
          default:
            llvm_unreachable("Unknown addressing mode for CP reference!");
          case Xtensa::L32R:
            CPUsers.push_back(CPUser(&MI, CPEMI, MAX_DISP_L32R));
            break;
          }

          // Increment corresponding CPEntry reference count.
          CPEntry *CPE = findConstPoolEntry(CPI, CPEMI);
          assert(CPE && "Cannot find a corresponding CPEntry!");
          CPE->RefCount++;

          // Instructions can only use one CP entry, don't bother scanning the
          // rest of the operands.
          break;
        }
    }
  }
}

/// computeBlockSize - Compute the size and some alignment information for MBB.
/// This function updates BBInfo directly.
void XtensaConstantIslands::computeBlockSize(MachineBasicBlock *MBB) {
  BasicBlockInfo &BBI = BBInfo[MBB->getNumber()];
  BBI.Size = 0;

  for (const MachineInstr &MI : *MBB) {
    if (MI.getOpcode() == Xtensa::CONSTPOOL_ENTRY) {
      BBI.Size += 4;
    } else {
      BBI.Size += TII->getInstSizeInBytes(MI);
    }
  }
}

/// getOffsetOf - Return the current offset of the specified machine instruction
/// from the start of the function.  This offset changes as stuff is moved
/// around inside the function.
unsigned XtensaConstantIslands::getOffsetOf(MachineInstr *MI) const {
  MachineBasicBlock *MBB = MI->getParent();

  // The offset is composed of two things: the sum of the sizes of all MBB's
  // before this instruction's block, and the offset from the start of the block
  // it is in.
  unsigned Offset = BBInfo[MBB->getNumber()].Offset;

  // Sum instructions before MI in MBB.
  for (MachineBasicBlock::iterator I = MBB->begin(); &*I != MI; ++I) {
    assert(I != MBB->end() && "Didn't find MI in its own basic block?");
    Offset += TII->getInstSizeInBytes(*I);
  }
  return Offset;
}

/// CompareMBBNumbers - Little predicate function to sort the WaterList by MBB
/// ID.
static bool CompareMBBNumbers(const MachineBasicBlock *LHS,
                              const MachineBasicBlock *RHS) {
  return LHS->getNumber() < RHS->getNumber();
}

/// updateForInsertedWaterBlock - When a block is newly inserted into the
/// machine function, it upsets all of the block numbers.  Renumber the blocks
/// and update the arrays that parallel this numbering.
void XtensaConstantIslands::updateForInsertedWaterBlock(
    MachineBasicBlock *NewBB) {
  // Renumber the MBB's to keep them consecutive.
  NewBB->getParent()->RenumberBlocks(NewBB);

  // Insert an entry into BBInfo to align it properly with the (newly
  // renumbered) block numbers.
  BBInfo.insert(BBInfo.begin() + NewBB->getNumber(), BasicBlockInfo());

  // Next, update WaterList.  Specifically, we need to add NewMBB as having
  // available water after it.
  water_iterator IP = llvm::lower_bound(WaterList, NewBB, CompareMBBNumbers);
  WaterList.insert(IP, NewBB);
}

unsigned XtensaConstantIslands::getUserOffset(CPUser &U) const {
  return getOffsetOf(U.MI);
}

/// Split the basic block containing MI into two blocks, which are joined by
/// an unconditional branch.  Update data structures and renumber blocks to
/// account for this change and returns the newly created block.
MachineBasicBlock *
XtensaConstantIslands::splitBlockBeforeInstr(MachineInstr &MI) {
  MachineBasicBlock *OrigBB = MI.getParent();

  // Collect liveness information at MI.
  LivePhysRegs LRs(*MF->getSubtarget().getRegisterInfo());
  LRs.addLiveOuts(*OrigBB);
  auto LivenessEnd = ++MachineBasicBlock::iterator(MI).getReverse();
  for (MachineInstr &LiveMI : make_range(OrigBB->rbegin(), LivenessEnd))
    LRs.stepBackward(LiveMI);

  // Create a new MBB for the code after the OrigBB.
  MachineBasicBlock *NewBB =
      MF->CreateMachineBasicBlock(OrigBB->getBasicBlock());
  MachineFunction::iterator MBBI = ++OrigBB->getIterator();
  MF->insert(MBBI, NewBB);

  // Splice the instructions starting with MI over to NewBB.
  NewBB->splice(NewBB->end(), OrigBB, MI, OrigBB->end());

  // Add an unconditional branch from OrigBB to NewBB.
  // Note the new unconditional branch is not being recorded.
  // There doesn't seem to be meaningful DebugInfo available; this doesn't
  // correspond to anything in the source.
  BuildMI(OrigBB, DebugLoc(), TII->get(Xtensa::J)).addMBB(NewBB);
  ++NumSplit;

  // Update the CFG.  All succs of OrigBB are now succs of NewBB.
  NewBB->transferSuccessors(OrigBB);

  // OrigBB branches to NewBB.
  OrigBB->addSuccessor(NewBB);

  // Update live-in information in the new block.
  MachineRegisterInfo &MRI = MF->getRegInfo();
  for (MCPhysReg L : LRs)
    if (!MRI.isReserved(L))
      NewBB->addLiveIn(L);

  // Update internal data structures to account for the newly inserted MBB.
  // This is almost the same as updateForInsertedWaterBlock, except that
  // the Water goes after OrigBB, not NewBB.
  MF->RenumberBlocks(NewBB);

  // Insert an entry into BBInfo to align it properly with the (newly
  // renumbered) block numbers.
  BBInfo.insert(BBInfo.begin() + NewBB->getNumber(), BasicBlockInfo());

  // Next, update WaterList.  Specifically, we need to add OrigMBB as having
  // available water after it (but not if it's already there, which happens
  // when splitting before a conditional branch that is followed by an
  // unconditional branch - in that case we want to insert NewBB).
  water_iterator IP = llvm::lower_bound(WaterList, OrigBB, CompareMBBNumbers);
  MachineBasicBlock *WaterBB = *IP;
  if (WaterBB == OrigBB)
    WaterList.insert(std::next(IP), NewBB);
  else
    WaterList.insert(IP, OrigBB);
  NewWaterList.insert(OrigBB);

  // Figure out how large the OrigBB is.  As the first half of the original
  // block, it cannot contain a tablejump.  The size includes
  // the new jump we added.  (It should be possible to do this without
  // recounting everything, but it's very confusing, and this is rarely
  // executed.)
  computeBlockSize(OrigBB);

  // Figure out how large the NewMBB is.  As the second half of the original
  // block, it may contain a tablejump.
  computeBlockSize(NewBB);

  // All BBOffsets following these blocks must be modified.
  adjustBBOffsetsAfter(OrigBB);

#if 0
  //TODO
  // Need to fix live-in lists if we track liveness.
  if (TRI->trackLivenessAfterRegAlloc(*MF))
    computeAndAddLiveIns(LiveRegs, *NewBB);
#endif
  return NewBB;
}

/// isOffsetInRange - Checks whether UserOffset (the location of a constant pool
/// reference) is within MaxDisp of TrialOffset (a proposed location of a
/// constant pool entry).
bool XtensaConstantIslands::isOffsetInRange(unsigned UserOffset,
                                            unsigned TrialOffset,
                                            unsigned MaxDisp) {
  UserOffset = (UserOffset + 3) & (~0x3);
  if ((UserOffset >= TrialOffset) && (UserOffset - TrialOffset <= MaxDisp)) {
    return true;
  }
  return false;
}

/// isWaterInRange - Returns true if a CPE placed after the specified
/// Water (a basic block) will be in range for the specific MI.
///
/// Compute how much the function will grow by inserting a CPE after Water.
bool XtensaConstantIslands::isWaterInRange(unsigned UserOffset,
                                           MachineBasicBlock *Water, CPUser &U,
                                           unsigned &Growth) {
  unsigned CPEOffset = BBInfo[Water->getNumber()].postOffset();
  unsigned NextBlockOffset;
  Align NextBlockAlignment;
  MachineFunction::const_iterator NextBlock = ++Water->getIterator();
  if (NextBlock == MF->end()) {
    NextBlockOffset = BBInfo[Water->getNumber()].postOffset();
    NextBlockAlignment = Align(1);
  } else {
    NextBlockOffset = BBInfo[NextBlock->getNumber()].Offset;
    NextBlockAlignment = NextBlock->getAlignment();
  }
  unsigned Size = U.CPEMI->getOperand(2).getImm();
  unsigned CPEEnd = CPEOffset + Size;

  // The CPE may be able to hide in the alignment padding before the next
  // block. It may also cause more padding to be required if it is more aligned
  // that the next block.
  if (CPEEnd > NextBlockOffset) {
    Growth = CPEEnd - NextBlockOffset;
    // Compute the padding that would go at the end of the CPE to align the next
    // block.
    Growth += offsetToAlignment(CPEEnd, NextBlockAlignment);

    // If the CPE is to be inserted before the instruction, that will raise
    // the offset of the instruction. Also account for unknown alignment padding
    // in blocks between CPE and the user.
    if (CPEOffset < UserOffset)
      UserOffset += Growth;
  } else
    // CPE fits in existing padding.
    Growth = 0;

  return isOffsetInRange(UserOffset, CPEOffset, U.getMaxDisp());
}

/// isCPEntryInRange - Returns true if the distance between specific MI and
/// specific ConstPool entry instruction can fit in MI's displacement field.
bool XtensaConstantIslands::isCPEntryInRange(MachineInstr *MI,
                                             unsigned UserOffset,
                                             MachineInstr *CPEMI,
                                             unsigned MaxDisp, bool DoDump) {
  unsigned CPEOffset = getOffsetOf(CPEMI);

  if (DoDump) {
    LLVM_DEBUG({
      unsigned Block = MI->getParent()->getNumber();
      const BasicBlockInfo &BBI = BBInfo[Block];
      dbgs() << "User of CPE#" << CPEMI->getOperand(0).getImm()
             << " max delta=" << MaxDisp
             << format(" insn address=%#x", UserOffset) << " in "
             << printMBBReference(*MI->getParent()) << ": "
             << format("%#x-%x\t", BBI.Offset, BBI.postOffset()) << *MI
             << format("CPE address=%#x offset=%+d: ", CPEOffset,
                       int(CPEOffset - UserOffset));
    });
  }

  return isOffsetInRange(UserOffset, CPEOffset, MaxDisp);
}

#ifndef NDEBUG
/// BBIsJumpedOver - Return true of the specified basic block's only predecessor
/// unconditionally branches to its only successor.
static bool BBIsJumpedOver(MachineBasicBlock *MBB) {
  if (MBB->pred_size() != 1 || MBB->succ_size() != 1)
    return false;
  MachineBasicBlock *Succ = *MBB->succ_begin();
  MachineBasicBlock *Pred = *MBB->pred_begin();
  MachineInstr *PredMI = &Pred->back();

  if (PredMI->getOpcode() == Xtensa::J)
    return PredMI->getOperand(0).getMBB() == Succ;
  return false;
}
#endif

void XtensaConstantIslands::adjustBBOffsetsAfter(MachineBasicBlock *BB) {
  unsigned BBNum = BB->getNumber();
  for (unsigned i = BBNum + 1, e = MF->getNumBlockIDs(); i < e; ++i) {
    // Get the offset and known bits at the end of the layout predecessor.
    // Include the alignment of the current block.
    unsigned Offset = BBInfo[i - 1].Offset + BBInfo[i - 1].Size;
    Align BlockAlignment = MF->getBlockNumbered(i)->getAlignment();
    BBInfo[i].Offset = Offset + offsetToAlignment(Offset, BlockAlignment);
  }
}

/// decrementCPEReferenceCount - find the constant pool entry with index CPI
/// and instruction CPEMI, and decrement its refcount.  If the refcount
/// becomes 0 remove the entry and instruction.  Returns true if we removed
/// the entry, false if we didn't.
bool XtensaConstantIslands::decrementCPEReferenceCount(unsigned CPI,
                                                       MachineInstr *CPEMI) {
  // Find the old entry. Eliminate it if it is no longer used.
  CPEntry *CPE = findConstPoolEntry(CPI, CPEMI);
  assert(CPE && "Unexpected!");
  if (--CPE->RefCount == 0) {
    removeDeadCPEMI(CPEMI);
    CPE->CPEMI = nullptr;
    --NumCPEs;
    return true;
  }
  return false;
}

/// LookForCPEntryInRange - see if the currently referenced CPE is in range;
/// if not, see if an in-range clone of the CPE is in range, and if so,
/// change the data structures so the user references the clone.  Returns:
/// 0 = no existing entry found
/// 1 = entry found, and there were no code insertions or deletions
/// 2 = entry found, and there were code insertions or deletions
int XtensaConstantIslands::findInRangeCPEntry(CPUser &U, unsigned UserOffset) {
  MachineInstr *UserMI = U.MI;
  MachineInstr *CPEMI = U.CPEMI;

  // Check to see if the CPE is already in-range.
  if (isCPEntryInRange(UserMI, UserOffset, CPEMI, U.getMaxDisp())) {
    LLVM_DEBUG(dbgs() << "In range\n");
    return 1;
  }

  // No.  Look for previously created clones of the CPE that are in range.
  unsigned CPI = CPEMI->getOperand(1).getIndex();
  std::vector<CPEntry> &CPEs = CPEntries[CPI];
  for (CPEntry &CPE : CPEs) {
    // We already tried this one
    if (CPE.CPEMI == CPEMI)
      continue;
    // Removing CPEs can leave empty entries, skip
    if (CPE.CPEMI == nullptr)
      continue;
    if (isCPEntryInRange(UserMI, UserOffset, CPE.CPEMI, U.getMaxDisp())) {
      LLVM_DEBUG(dbgs() << "Replacing CPE#" << CPI << " with CPE#" << CPE.CPI
                        << "\n");
      // Point the CPUser node to the replacement
      U.CPEMI = CPE.CPEMI;
      // Change the CPI in the instruction operand to refer to the clone.
      for (MachineOperand &MO : UserMI->operands())
        if (MO.isCPI()) {
          MO.setIndex(CPE.CPI);
          break;
        }
      // Adjust the refcount of the clone...
      CPE.RefCount++;
      // ...and the original.  If we didn't remove the old entry, none of the
      // addresses changed, so we don't need another pass.
      return decrementCPEReferenceCount(CPI, CPEMI) ? 2 : 1;
    }
  }
  return 0;
}

/// getUnconditionalBrDisp - Returns the maximum displacement that can fit in
/// the specific unconditional branch instruction.
static inline unsigned getUnconditionalBrDisp(int Opc) {
  // Currently only J instruction is used
  return (1 << (BITS_JUMP - 1));
}

/// findAvailableWater - Look for an existing entry in the WaterList in which
/// we can place the CPE referenced from U so it's within range of U's MI.
/// Returns true if found, false if not.  If it returns true, WaterIter
/// is set to the WaterList entry.
/// To ensure that this pass
/// terminates, the CPE location for a particular CPUser is only allowed to
/// move to a lower address, so search backward from the end of the list and
/// prefer the first water that is in range.
bool XtensaConstantIslands::findAvailableWater(CPUser &U, unsigned UserOffset,
                                               water_iterator &WaterIter) {
  if (WaterList.empty())
    return false;

  unsigned BestGrowth = ~0u;
  for (water_iterator IP = std::prev(WaterList.end()), B = WaterList.begin();;
       --IP) {
    MachineBasicBlock *WaterBB = *IP;
    // Check if water is in range and is either at a higher address than the
    // current "low water mark" or a new water block that was created since
    // the previous iteration by inserting an unconditional branch.  In the
    // latter case, we want to allow resetting the low water mark back to
    // this new water since we haven't seen it before.  Inserting branches
    // should be relatively uncommon and when it does happen, we want to be
    // sure to take advantage of it for all the CPEs near that block, so that
    // we don't insert more branches than necessary.
    unsigned Growth;
    if (isWaterInRange(UserOffset, WaterBB, U, Growth) &&
        (WaterBB->getNumber() > U.LowWaterMark->getNumber() ||
         NewWaterList.count(WaterBB)) &&
        Growth < BestGrowth) {
      // This is the least amount of required padding seen so far.
      BestGrowth = Growth;
      WaterIter = IP;
      LLVM_DEBUG(dbgs() << "Found water after " << printMBBReference(*WaterBB)
                        << " Growth=" << Growth << '\n');

      // Keep looking unless it is perfect.
      if (BestGrowth == 0)
        return true;
    }
    if (IP == B)
      break;
  }
  return BestGrowth != ~0u;
}

/// createNewWater - No existing WaterList entry will work for
/// CPUsers[CPUserIndex], so create a place to put the CPE.  The end of the
/// block is used if in range, and the conditional branch munged so control
/// flow is correct.  Otherwise the block is split to create a hole with an
/// unconditional branch around it.  In either case NewMBB is set to a
/// block following which the new island can be inserted (the WaterList
/// is not adjusted).
void XtensaConstantIslands::createNewWater(unsigned CPUserIndex,
                                           unsigned UserOffset,
                                           MachineBasicBlock *&NewMBB) {
  CPUser &U = CPUsers[CPUserIndex];
  MachineInstr *UserMI = U.MI;
  MachineBasicBlock *UserMBB = UserMI->getParent();
  NewMBB = splitBlockBeforeInstr(*UserMI);
}

/// handleConstantPoolUser - Analyze the specified user, checking to see if it
/// is out-of-range.  If so, pick up the constant pool value and move it some
/// place in-range.  Return true if we changed any addresses (thus must run
/// another pass of branch lengthening), false otherwise.
bool XtensaConstantIslands::handleConstantPoolUser(unsigned CPUserIndex) {
  CPUser &U = CPUsers[CPUserIndex];
  MachineInstr *UserMI = U.MI;
  MachineInstr *CPEMI = U.CPEMI;
  unsigned CPI = CPEMI->getOperand(1).getIndex();
  unsigned Size = CPEMI->getOperand(2).getImm();
  // Compute this only once, it's expensive.
  unsigned UserOffset = getUserOffset(U);

  // See if the current entry is within range, or there is a clone of it
  // in range.
  int result = findInRangeCPEntry(U, UserOffset);
  if (result == 1)
    return false;
  else if (result == 2)
    return true;

  // Look for water where we can place this CPE.
  MachineBasicBlock *NewIsland = MF->CreateMachineBasicBlock();
  MachineBasicBlock *NewMBB;
  water_iterator IP;
  if (findAvailableWater(U, UserOffset, IP)) {
    LLVM_DEBUG(dbgs() << "Found water in range\n");
    MachineBasicBlock *WaterBB = *IP;

    // If the original WaterList entry was "new water" on this iteration,
    // propagate that to the new island.  This is just keeping NewWaterList
    // updated to match the WaterList, which will be updated below.
    if (NewWaterList.erase(WaterBB))
      NewWaterList.insert(NewIsland);

    // The new CPE goes before the following block (NewMBB).
    NewMBB = &*++WaterBB->getIterator();
  } else {
    // No water found.
    LLVM_DEBUG(dbgs() << "No water found\n");
    createNewWater(CPUserIndex, UserOffset, NewMBB);

    // splitBlockBeforeInstr adds to WaterList, which is important when it is
    // called while handling branches so that the water will be seen on the
    // next iteration for constant pools, but in this context, we don't want
    // it.  Check for this so it will be removed from the WaterList.
    // Also remove any entry from NewWaterList.
    MachineBasicBlock *WaterBB = &*--NewMBB->getIterator();
    IP = llvm::find(WaterList, WaterBB);
    if (IP != WaterList.end())
      NewWaterList.erase(WaterBB);

    // We are adding new water.  Update NewWaterList.
    NewWaterList.insert(NewIsland);
  }

  // Remove the original WaterList entry; we want subsequent insertions in
  // this vicinity to go after the one we're about to insert.  This
  // considerably reduces the number of times we have to move the same CPE
  // more than once and is also important to ensure the algorithm terminates.
  if (IP != WaterList.end())
    WaterList.erase(IP);

  // Okay, we know we can put an island before NewMBB now, do it!
  MF->insert(NewMBB->getIterator(), NewIsland);

  // Update internal data structures to account for the newly inserted MBB.
  updateForInsertedWaterBlock(NewIsland);

  // Decrement the old entry, and remove it if refcount becomes 0.
  decrementCPEReferenceCount(CPI, CPEMI);

  // No existing clone of this CPE is within range.
  // We will be generating a new clone.  Get a UID for it.
  unsigned ID = createPICLabelUId();

  // Now that we have an island to add the CPE to, clone the original CPE and
  // add it to the island.
  U.LowWaterMark = NewIsland;
  U.CPEMI = BuildMI(NewIsland, DebugLoc(), TII->get(Xtensa::CONSTPOOL_ENTRY))
                .addImm(ID)
                .addConstantPoolIndex(CPI)
                .addImm(Size);
  CPEntries[CPI].push_back(CPEntry(U.CPEMI, ID, 1));
  ++NumCPEs;

  // Mark the basic block as aligned as required by the const-pool entry.
  NewIsland->setAlignment(getCPEAlign(*U.CPEMI));

  // Increase the size of the island block to account for the new entry.
  BBInfo[NewIsland->getNumber()].Size += Size;
  adjustBBOffsetsAfter(&*--NewIsland->getIterator());

  // Finally, change the CPI in the instruction operand to be ID.
  for (MachineOperand &MO : UserMI->operands())
    if (MO.isCPI()) {
      MO.setIndex(ID);
      break;
    }

  LLVM_DEBUG(
      dbgs() << "  Moved CPE to #" << ID << " CPI=" << CPI
             << format(" offset=%#x\n", BBInfo[NewIsland->getNumber()].Offset));

  return true;
}

/// removeDeadCPEMI - Remove a dead constant pool entry instruction. Update
/// sizes and offsets of impacted basic blocks.
void XtensaConstantIslands::removeDeadCPEMI(MachineInstr *CPEMI) {
  MachineBasicBlock *CPEBB = CPEMI->getParent();
  unsigned Size = CPEMI->getOperand(2).getImm();
  CPEMI->eraseFromParent();
  BBInfo[CPEBB->getNumber()].Size -= Size;
  // All succeeding offsets have the current size value added in, fix this.
  if (CPEBB->empty()) {
    BBInfo[CPEBB->getNumber()].Size = 0;

    // This block no longer needs to be aligned.
    CPEBB->setAlignment(Align(1));
  } else {
    // Entries are sorted by descending alignment, so realign from the front.
    CPEBB->setAlignment(getCPEAlign(*CPEBB->begin()));
  }

  adjustBBOffsetsAfter(CPEBB);
  // An island has only one predecessor BB and one successor BB. Check if
  // this BB's predecessor jumps directly to this BB's successor. This
  // shouldn't happen currently.
  assert(!BBIsJumpedOver(CPEBB) && "How did this happen?");
  // FIXME: remove the empty blocks after all the work is done?
}

/// removeUnusedCPEntries - Remove constant pool entries whose refcounts
/// are zero.
bool XtensaConstantIslands::removeUnusedCPEntries() {
  unsigned MadeChange = false;
  for (std::vector<CPEntry> &CPEs : CPEntries) {
    for (CPEntry &CPE : CPEs) {
      if (CPE.RefCount == 0 && CPE.CPEMI) {
        removeDeadCPEMI(CPE.CPEMI);
        CPE.CPEMI = nullptr;
        MadeChange = true;
      }
    }
  }
  return MadeChange;
}

/// isBBInRange - Returns true if the distance between specific MI and
/// specific BB can fit in MI's displacement field.
bool XtensaConstantIslands::isBBInRange(MachineInstr *MI,
                                        MachineBasicBlock *DestBB,
                                        unsigned MaxDisp) {
  unsigned PCAdj = 4;
  unsigned BrOffset = getOffsetOf(MI) + PCAdj;
  unsigned DestOffset = BBInfo[DestBB->getNumber()].Offset;

  LLVM_DEBUG(dbgs() << "Branch of destination " << printMBBReference(*DestBB)
                    << " from " << printMBBReference(*MI->getParent())
                    << " max delta=" << MaxDisp << " from " << getOffsetOf(MI)
                    << " to " << DestOffset << " offset "
                    << int(DestOffset - BrOffset) << "\t" << *MI);
  if (BrOffset <= DestOffset) {
    // Branch before the Dest.
    if (DestOffset - BrOffset < MaxDisp)
      return true;
  } else {
    if (BrOffset - DestOffset <= MaxDisp)
      return true;
  }
  return false;
}

/// fixupImmediateBr - Fix up an immediate branch whose destination is too far
/// away to fit in its displacement field.
bool XtensaConstantIslands::fixupImmediateBr(ImmBranch &Br) {
  MachineInstr *MI = Br.MI;

  if (MI->getOpcode() == Xtensa::JX)
    return false;

  // TOOO: currently we don't fix J in start block
  if (MI->getParent()->getNumber() == 0)
    return false;

  unsigned TargetOperand = branchTargetOperand(MI);
  MachineBasicBlock *DestBB = MI->getOperand(TargetOperand).getMBB();

  // Check to see if the DestBB is already in-range.
  if (isBBInRange(MI, DestBB, Br.MaxDisp))
    return false;

  if (!Br.isCond)
    return fixupUnconditionalBr(Br);
  return fixupConditionalBr(Br);
}

/// fixupUnconditionalBr - Fix up an unconditional branch whose destination is
/// too far away to fit in its displacement field. If the LR register has been
/// spilled in the epilogue, then we can use BL to implement a far jump.
/// Otherwise, add an intermediate branch instruction to a branch.
/// fixupUnconditionalBr - Fix up an unconditional branch whose destination is
/// too far away to fit in its displacement field. If the LR register has been
/// spilled in the epilogue, then we can use BSR to implement a far jump.
/// Otherwise, add an intermediate branch instruction to a branch.
bool XtensaConstantIslands::fixupUnconditionalBr(ImmBranch &Br) {
  MachineInstr *MI = Br.MI;
  MachineBasicBlock *MBB = MI->getParent();
  MachineBasicBlock *DestBB = TII->getBranchDestBlock(*MI);
  MachineFunction *MF = MBB->getParent();
  MachineRegisterInfo &MRI = MF->getRegInfo();
  MachineConstantPool *ConstantPool = MF->getConstantPool();

  XtensaConstantPoolValue *C =
      XtensaConstantPoolMBB::Create(MF->getFunction().getContext(), DestBB, 0);
  unsigned CPSize = ConstantPool->getConstants().size();
  unsigned Idx = ConstantPool->getConstantPoolIndex(C, Align(4));
  Register DestReg = MRI.createVirtualRegister(&Xtensa::ARRegClass);

  MachineInstr *CPEMI = nullptr;
  // Add a new CPEntry, but no corresponding CPUser yet.
  unsigned ID = Idx;
  if (CPSize == Idx) {
    ID = createPICLabelUId();
    CPEMI =
        BuildMI(InitConstantMBB, DebugLoc(), TII->get(Xtensa::CONSTPOOL_ENTRY))
            .addImm(ID)
            .addConstantPoolIndex(Idx)
            .addImm(4);
    CPEntries.emplace_back(1, CPEntry(CPEMI, Idx, 1));
    ++NumCPEs;
    BBInfo[InitConstantMBB->getNumber()].Size += 4;
    adjustBBOffsetsAfter(InitConstantMBB);
  } else {
    std::vector<CPEntry> &CPEs = CPEntries[Idx];
    for (CPEntry &CPE : CPEs) {
      if ((CPE.CPEMI != nullptr) &&
          isCPEntryInRange(MI, MAX_DISP_L32R, CPE.CPEMI, MAX_DISP_L32R, true)) {
        CPEMI = CPE.CPEMI;
        CPE.RefCount++;
      }
    }
    if (CPEMI == nullptr) {
      ID = createPICLabelUId();
      CPEMI = BuildMI(InitConstantMBB, DebugLoc(),
                      TII->get(Xtensa::CONSTPOOL_ENTRY))
                  .addImm(ID)
                  .addConstantPoolIndex(Idx)
                  .addImm(4);
      CPEntries[Idx].push_back(CPEntry(CPEMI, ID, 1));
      ++NumCPEs;
      BBInfo[InitConstantMBB->getNumber()].Size += 4;
      adjustBBOffsetsAfter(InitConstantMBB);
    } else {
      ID = CPEMI->getOperand(0).getImm();
    }
  }
  MachineInstr *L32R =
      BuildMI(*MBB, MI, DebugLoc(), TII->get(Xtensa::L32R), DestReg)
          .addConstantPoolIndex(ID);

  MI->setDesc(TII->get(Xtensa::JX));
  MI->removeOperand(0);
  MI->addOperand(MachineOperand::CreateReg(DestReg, true));

  RegScavenger RS;
  RS.enterBasicBlockEnd(*MBB);
  unsigned Scav = RS.scavengeRegisterBackwards(Xtensa::ARRegClass,
                                               L32R->getIterator(), false, 0);
  MRI.replaceRegWith(DestReg, Scav);
  MRI.clearVirtRegs();
  RS.setRegUsed(Scav);
  CPUsers.push_back(CPUser(L32R, CPEMI, MAX_DISP_L32R));
  BBInfo[MBB->getNumber()].Size += 3;
  adjustBBOffsetsAfter(MBB);
  ++NumUBrFixed;

  LLVM_DEBUG(dbgs() << "  Changed B to long jump " << *MI);
  return true;
}

// TODO
/// fixupConditionalBr - Fix up a conditional branch whose destination is too
/// far away to fit in its displacement field. It is converted to an inverse
/// conditional branch + an unconditional branch to the destination.
bool XtensaConstantIslands::fixupConditionalBr(ImmBranch &Br) {
  MachineInstr *MI = Br.MI;
  MachineBasicBlock *DestBB = TII->getBranchDestBlock(*MI);

  SmallVector<MachineOperand, 4> Cond;
  Cond.push_back(MachineOperand::CreateImm(MI->getOpcode()));
  Cond.push_back(MI->getOperand(0));
  TII->reverseBranchCondition(Cond);
  // Add an unconditional branch to the destination and invert the branch
  // condition to jump over it:
  // bteqz L1
  // =>
  // bnez L2
  // b   L1
  // L2:

  // If the branch is at the end of its MBB and that has a fall-through block,
  // direct the updated conditional branch to the fall-through block. Otherwise,
  // split the MBB before the next instruction.
  MachineBasicBlock *MBB = MI->getParent();
  MachineInstr *BMI = &MBB->back();
  bool NeedSplit = (BMI != MI) || !BBHasFallthrough(MBB);

  ++NumCBrFixed;
  if (BMI != MI) {
    if (std::next(MachineBasicBlock::iterator(MI)) == std::prev(MBB->end()) &&
        BMI->isUnconditionalBranch()) {
      // Last MI in the BB is an unconditional branch. Can we simply invert the
      // condition and swap destinations:
      // beqz L1
      // b   L2
      // =>
      // bnez L2
      // b   L1
      MachineBasicBlock *NewDest = TII->getBranchDestBlock(*BMI);
      if (isBBInRange(MI, NewDest, Br.MaxDisp)) {
        LLVM_DEBUG(
            dbgs() << "  Invert Bcc condition and swap its destination with "
                   << *BMI);
        BMI->getOperand(BMI->getNumExplicitOperands() - 1).setMBB(DestBB);
        MI->getOperand(MI->getNumExplicitOperands() - 1).setMBB(NewDest);

        MI->setDesc(TII->get(Cond[0].getImm()));
        return true;
      }
    }
  }

  if (NeedSplit) {
    splitBlockBeforeInstr(*MI);
    // No need for the branch to the next block. We're adding an unconditional
    // branch to the destination.
    int Delta = TII->getInstSizeInBytes(MBB->back());
    BBInfo[MBB->getNumber()].Size -= Delta;
    MBB->back().eraseFromParent();

    // The conditional successor will be swapped between the BBs after this, so
    // update CFG.
    MBB->addSuccessor(DestBB);
    std::next(MBB->getIterator())->removeSuccessor(DestBB);
  }
  MachineBasicBlock *NextBB = &*++MBB->getIterator();

  LLVM_DEBUG(dbgs() << "  Insert B to " << printMBBReference(*DestBB)
                    << " also invert condition and change dest. to "
                    << printMBBReference(*NextBB) << "\n");

  // Insert a new conditional branch and a new unconditional branch.
  // Also update the ImmBranch as well as adding a new entry for the new branch.
  switch (MI->getOpcode()) {
  case Xtensa::BEQ:
  case Xtensa::BNE:
  case Xtensa::BLT:
  case Xtensa::BLTU:
  case Xtensa::BGE:
  case Xtensa::BGEU:
    BuildMI(MBB, DebugLoc(), TII->get(Cond[0].getImm()))
        .addReg(MI->getOperand(0).getReg())
        .addReg(MI->getOperand(1).getReg())
        .addMBB(NextBB);
    break;
  case Xtensa::BEQI:
  case Xtensa::BNEI:
  case Xtensa::BLTI:
  case Xtensa::BLTUI:
  case Xtensa::BGEI:
  case Xtensa::BGEUI:
    BuildMI(MBB, DebugLoc(), TII->get(Cond[0].getImm()))
        .addReg(MI->getOperand(0).getReg())
        .addImm(MI->getOperand(1).getImm())
        .addMBB(NextBB);
    break;
  case Xtensa::BEQZ:
  case Xtensa::BNEZ:
  case Xtensa::BLTZ:
  case Xtensa::BGEZ:
    BuildMI(MBB, DebugLoc(), TII->get(Cond[0].getImm()))
        .addReg(MI->getOperand(0).getReg())
        .addMBB(NextBB);
    break;
  case Xtensa::BT:
  case Xtensa::BF:
    BuildMI(MBB, DebugLoc(), TII->get(Cond[0].getImm()))
        .addReg(MI->getOperand(0).getReg())
        .addMBB(NextBB);
    break;
  }

  Br.MI = &MBB->back();
  BBInfo[MBB->getNumber()].Size += TII->getInstSizeInBytes(MBB->back());
  BuildMI(MBB, DebugLoc(), TII->get(Br.UncondBr)).addMBB(DestBB);
  BBInfo[MBB->getNumber()].Size += TII->getInstSizeInBytes(MBB->back());
  unsigned MaxDisp = getUnconditionalBrDisp(Br.UncondBr);
  ImmBranches.push_back(ImmBranch(&MBB->back(), MaxDisp, false, Br.UncondBr));

  // Remove the old conditional branch.  It may or may not still be in MBB.
  BBInfo[MI->getParent()->getNumber()].Size -= TII->getInstSizeInBytes(*MI);
  MI->eraseFromParent();
  adjustBBOffsetsAfter(MBB);
  return true;
}

// Check first constant island. If it is empty, then we can remove first block,
// which contains jump instruction to the third block and first constant
// insland.
void XtensaConstantIslands::removeEntryJump() {
  MachineFunction *MF = InitConstantMBB->getParent();
  MachineBasicBlock *Entry = &MF->front();
  if (InitConstantMBB->empty()) {
    Entry->removeSuccessor(Entry->getSingleSuccessor());
    MF->remove(Entry);
    MF->remove(InitConstantMBB);
    MF->RenumberBlocks();
  }
}

/// Returns a pass that converts branches to long branches.
FunctionPass *llvm::createXtensaConstantIslandPass() {
  return new XtensaConstantIslands();
}

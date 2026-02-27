#!/usr/bin/env python3
"""
Regenerate xtheadmatrix test files by extracting instructions from the .td file
and running them through llvm-mc to get correct encodings.

Usage:
  python3 regen_xtheadmatrix_tests.py [--build-dir BUILD_DIR]

This script:
1. Reads RISCVInstrInfoXTHeadMatrix.td to extract all instruction definitions
2. Generates assembly for each instruction with varied registers/immediates
3. Runs through llvm-mc --show-encoding to get actual encodings
4. Produces updated test files with correct CHECK-ENCODING lines
"""

import subprocess
import re
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(SCRIPT_DIR, 'build')
TD_FILE = os.path.join(SCRIPT_DIR, 'llvm/lib/Target/RISCV/RISCVInstrInfoXTHeadMatrix.td')
VALID_TEST = os.path.join(SCRIPT_DIR, 'llvm/test/MC/RISCV/xtheadmatrix-valid.s')
INVALID_TEST = os.path.join(SCRIPT_DIR, 'llvm/test/MC/RISCV/xtheadmatrix-invalid.s')
CSR_TEST = os.path.join(SCRIPT_DIR, 'llvm/test/MC/RISCV/xtheadmatrix-csr.s')

ERROR_MSG = "# CHECK-ERROR: instruction requires the following: 'XTHeadMatrix' (T-Head Matrix Extension){{$}}"

# Register cycling lists
MREGS = ['tr0', 'tr1', 'tr2', 'tr3', 'acc0', 'acc1', 'acc2', 'acc3']
GPRS = ['a0', 'a1', 'a2', 'a3', 'a4', 'a5', 's0', 's1', 't0', 't1', 't2', 't3']
IMM3_VALS = [0, 1, 2, 3, 4, 5, 6, 7]


def extract_instructions_from_td(td_path):
    """Parse the .td file to extract all instruction mnemonics and their operand patterns."""
    with open(td_path) as f:
        content = f.read()

    instructions = []

    # Match patterns like: "th.mnemonic", "$operands"
    # Look for string patterns in def/defm declarations
    # Pattern: "th.XXX" appears as the mnemonic in instruction defs
    for match in re.finditer(r'"(th\.[a-z0-9._]+)",\s*"([^"]*)"', content):
        mnemonic = match.group(1)
        operand_pattern = match.group(2)
        instructions.append((mnemonic, operand_pattern))

    return instructions


def generate_asm_for_instruction(mnemonic, operand_pattern, idx):
    """Generate assembly text for an instruction given its mnemonic and operand pattern."""
    mr_idx = idx % len(MREGS)
    gpr_idx = idx % len(GPRS)
    imm3_idx = idx % len(IMM3_VALS)

    # Parse operand pattern to determine what to substitute
    asm = mnemonic
    if not operand_pattern:
        return asm

    parts = operand_pattern.strip()
    # Replace operand placeholders
    md_count = 0
    ms1_count = 0
    ms2_count = 0
    ms3_count = 0
    rd_count = 0
    rs1_count = 0
    rs2_count = 0
    imm3_count = 0
    imm10_count = 0

    result_parts = []
    for part in parts.split(','):
        part = part.strip()
        if part == '$md':
            result_parts.append(MREGS[(mr_idx + md_count) % len(MREGS)])
            md_count += 1
        elif part == '$ms1':
            result_parts.append(MREGS[(mr_idx + 1 + ms1_count) % len(MREGS)])
            ms1_count += 1
        elif part == '$ms2':
            result_parts.append(MREGS[(mr_idx + 2 + ms2_count) % len(MREGS)])
            ms2_count += 1
        elif part == '$ms3':
            result_parts.append(MREGS[(mr_idx + ms3_count) % len(MREGS)])
            ms3_count += 1
        elif part == '$rd':
            result_parts.append(GPRS[(gpr_idx + rd_count) % len(GPRS)])
            rd_count += 1
        elif part == '$rs1':
            result_parts.append(GPRS[(gpr_idx + 1 + rs1_count) % len(GPRS)])
            rs1_count += 1
        elif part == '(${rs1})':
            result_parts.append('(' + GPRS[(gpr_idx + 1 + rs1_count) % len(GPRS)] + ')')
            rs1_count += 1
        elif part == '$rs2':
            result_parts.append(GPRS[(gpr_idx + 2 + rs2_count) % len(GPRS)])
            rs2_count += 1
        elif part == '$imm3':
            result_parts.append(str(IMM3_VALS[(imm3_idx + imm3_count) % len(IMM3_VALS)]))
            imm3_count += 1
        elif part == '$imm':
            # uimm10 - use various values
            vals = [0, 8, 16, 32, 64, 128, 256, 512]
            result_parts.append(str(vals[imm10_count % len(vals)]))
            imm10_count += 1
        else:
            result_parts.append(part)

    asm += ' ' + ', '.join(result_parts)
    return asm


def get_encoding_from_llvm_mc(asm_lines, llvm_mc_path):
    """Run assembly through llvm-mc and extract encodings."""
    asm_text = '\n'.join(asm_lines)
    try:
        result = subprocess.run(
            [llvm_mc_path, '-triple=riscv64', '-show-encoding',
             '--mattr=+experimental-xtheadmatrix'],
            input=asm_text, capture_output=True, text=True, timeout=30
        )
    except subprocess.TimeoutExpired:
        print("ERROR: llvm-mc timed out", file=sys.stderr)
        return {}

    encodings = {}
    lines = result.stdout.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line or line.startswith('.'):
            continue
        # Parse: "th.mnemonic operands  # encoding: [0x2b,...]"
        m = re.match(r'(th\.\S+.*?)\s+#\s+encoding:\s+(\[.*?\])', line)
        if m:
            instr = m.group(1).strip()
            # Normalize whitespace: collapse tabs/spaces to single space
            instr = re.sub(r'\s+', ' ', instr)
            encoding = m.group(2)
            encodings[instr] = encoding

    if result.stderr:
        for errline in result.stderr.strip().split('\n'):
            if 'error:' in errline:
                print(f"  ASSEMBLY ERROR: {errline}", file=sys.stderr)

    return encodings


def generate_valid_test(instructions, encodings):
    """Generate the complete xtheadmatrix-valid.s test file."""
    lines = []
    lines.append("# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+experimental-xtheadmatrix %s \\")
    lines.append("# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST")
    lines.append("# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \\")
    lines.append("# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR")
    lines.append("")

    current_section = None
    for asm, section in instructions:
        if section != current_section:
            if current_section is not None:
                lines.append("")
            lines.append(f"# {section}")
            current_section = section

        # Normalize asm for lookup
        asm_normalized = re.sub(r'\s+', ' ', asm.strip())
        encoding = encodings.get(asm_normalized, None)

        lines.append(asm)
        lines.append(f"# CHECK-INST: {asm}")
        if encoding:
            lines.append(f"# CHECK-ENCODING: {encoding}")
        else:
            lines.append(f"# CHECK-ENCODING: [ENCODING_MISSING]")
            print(f"  WARNING: No encoding found for: {asm}", file=sys.stderr)
        lines.append(ERROR_MSG)
        lines.append("")

    return '\n'.join(lines)


def main():
    llvm_mc = os.path.join(BUILD_DIR, 'bin', 'llvm-mc')
    if not os.path.exists(llvm_mc):
        print(f"ERROR: llvm-mc not found at {llvm_mc}", file=sys.stderr)
        print("Build first: cd build && ninja -j12 llvm-mc", file=sys.stderr)
        sys.exit(1)

    print("Step 1: Extracting instructions from .td file...")
    raw_instructions = extract_instructions_from_td(TD_FILE)
    print(f"  Found {len(raw_instructions)} instruction definitions")

    # Generate assembly for each instruction
    print("Step 2: Generating assembly for each instruction...")
    asm_lines = []
    instruction_info = []  # (asm, section_name)

    # Categorize instructions by section
    section = "Configuration instructions"
    idx = 0
    for mnemonic, operand_pattern in raw_instructions:
        # Determine section based on mnemonic
        if 'release' in mnemonic or 'settile' in mnemonic:
            section = "Configuration instructions"
        elif mnemonic.startswith('th.ml'):
            section = "Load instructions"
        elif mnemonic.startswith('th.ms') and ('ae' in mnemonic or 'be' in mnemonic or 'ce' in mnemonic or 'me' in mnemonic or 'ate' in mnemonic or 'bte' in mnemonic or 'cte' in mnemonic):
            section = "Store instructions"
        elif 'macc' in mnemonic or 'mmacc' in mnemonic:
            section = "Matrix multiply instructions"
        elif 'zero' in mnemonic or 'mov' in mnemonic or 'dup' in mnemonic or 'pack' in mnemonic or 'slide' in mnemonic or 'bca' in mnemonic:
            section = "Misc instructions"
        elif 'fcvt' in mnemonic or 'ucvt' in mnemonic or 'scvt' in mnemonic or 'clip' in mnemonic:
            section = "Element-wise conversion instructions"
        elif 'mfadd' in mnemonic or 'mfsub' in mnemonic or 'mfmul' in mnemonic or 'mfmax' in mnemonic or 'mfmin' in mnemonic:
            section = "FP element-wise arithmetic"
        elif 'madd' in mnemonic or 'msub' in mnemonic or 'mmul' in mnemonic or 'mmax' in mnemonic or 'mmin' in mnemonic or 'msrl' in mnemonic or 'msll' in mnemonic or 'msra' in mnemonic:
            section = "Integer element-wise arithmetic"

        asm = generate_asm_for_instruction(mnemonic, operand_pattern, idx)
        asm_lines.append(asm)
        instruction_info.append((asm, section))
        idx += 1

    print(f"  Generated {len(asm_lines)} assembly lines")

    # Get encodings from llvm-mc
    print("Step 3: Getting encodings from llvm-mc...")
    encodings = get_encoding_from_llvm_mc(asm_lines, llvm_mc)
    print(f"  Got {len(encodings)} encodings")

    # Generate test file
    print("Step 4: Generating test file...")
    test_content = generate_valid_test(instruction_info, encodings)

    with open(VALID_TEST, 'w') as f:
        f.write(test_content)
    print(f"  Written to {VALID_TEST}")

    # Count entries
    count = test_content.count('# CHECK-INST:')
    print(f"  Total test entries: {count}")

    # Verify
    print("Step 5: Verifying with FileCheck...")
    result = subprocess.run(
        f'{llvm_mc} -triple=riscv64 -show-encoding --mattr=+experimental-xtheadmatrix {VALID_TEST} 2>&1 | '
        f'{os.path.join(BUILD_DIR, "bin", "FileCheck")} {VALID_TEST} --check-prefixes=CHECK-ENCODING,CHECK-INST',
        shell=True, capture_output=True, text=True
    )
    if result.returncode == 0:
        print("  ENCODING PASS")
    else:
        print(f"  ENCODING FAIL: {result.stderr[:500]}")

    result2 = subprocess.run(
        f'{llvm_mc} -triple=riscv64 -show-encoding {VALID_TEST} 2>&1 | '
        f'{os.path.join(BUILD_DIR, "bin", "FileCheck")} {VALID_TEST} --check-prefix=CHECK-ERROR',
        shell=True, capture_output=True, text=True
    )
    if result2.returncode == 0:
        print("  ERROR PASS")
    else:
        print(f"  ERROR FAIL: {result2.stderr[:500]}")


if __name__ == '__main__':
    main()

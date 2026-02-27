#!/bin/bash
# Regenerate CHECK-ENCODING lines in xtheadmatrix-valid.s
# by running the assembly through llvm-mc and capturing actual encodings.
#
# Usage: ./regen_encodings.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
LLVM_MC="$BUILD_DIR/bin/llvm-mc"
FILECHECK="$BUILD_DIR/bin/FileCheck"
LIT="$BUILD_DIR/bin/llvm-lit"
TEST_FILE="$SCRIPT_DIR/llvm/test/MC/RISCV/xtheadmatrix-valid.s"
INVALID_FILE="$SCRIPT_DIR/llvm/test/MC/RISCV/xtheadmatrix-invalid.s"
CSR_FILE="$SCRIPT_DIR/llvm/test/MC/RISCV/xtheadmatrix-csr.s"

echo "=== Step 1: Build llvm-mc ==="
cd "$BUILD_DIR"
ninja -j12 llvm-mc llvm-objdump FileCheck llvm-config 2>&1 | tail -5
echo ""

echo "=== Step 2: Get actual encodings from llvm-mc ==="
# Run llvm-mc and capture the encoding output
ACTUAL_OUTPUT=$($LLVM_MC -triple=riscv64 -show-encoding --mattr=+experimental-xtheadmatrix "$TEST_FILE" 2>&1)

# Check for assembly errors
ERRORS=$(echo "$ACTUAL_OUTPUT" | grep "error:" || true)
if [ -n "$ERRORS" ]; then
    echo "ASSEMBLY ERRORS detected:"
    echo "$ERRORS"
    echo ""
    echo "These instructions need to be fixed or removed from the test file."
fi

# Count successful encodings
ENCODING_COUNT=$(echo "$ACTUAL_OUTPUT" | grep -c "encoding:" || true)
echo "Got $ENCODING_COUNT encodings from llvm-mc"
echo ""

echo "=== Step 3: Regenerate CHECK-ENCODING lines ==="
# Create a temporary file with updated encodings
python3 - "$TEST_FILE" "$LLVM_MC" <<'PYEOF'
import sys
import subprocess
import re

test_file = sys.argv[1]
llvm_mc = sys.argv[2]

# Read the test file
with open(test_file) as f:
    lines = f.readlines()

# Extract all assembly lines (lines that don't start with # and aren't empty)
asm_lines = []
asm_indices = []
for i, line in enumerate(lines):
    stripped = line.strip()
    if stripped and not stripped.startswith('#') and not stripped.startswith('.'):
        asm_lines.append(stripped)
        asm_indices.append(i)

# Run through llvm-mc to get encodings
asm_text = '\n'.join(asm_lines)
result = subprocess.run(
    [llvm_mc, '-triple=riscv64', '-show-encoding', '--mattr=+experimental-xtheadmatrix'],
    input=asm_text, capture_output=True, text=True, timeout=30
)

# Parse encodings from output
encoding_map = {}  # line_content -> encoding
output_lines = result.stdout.strip().split('\n')
enc_idx = 0
for out_line in output_lines:
    out_line = out_line.strip()
    if not out_line:
        continue
    m = re.search(r'#\s+encoding:\s+(\[.*?\])', out_line)
    if m and enc_idx < len(asm_lines):
        encoding_map[asm_lines[enc_idx]] = m.group(1)
        enc_idx += 1

# Update CHECK-ENCODING lines in the test file
output_lines_new = []
i = 0
while i < len(lines):
    line = lines[i]

    # Check if the next lines contain CHECK-ENCODING that needs updating
    if line.strip().startswith('# CHECK-ENCODING:'):
        # Find the preceding assembly line
        prev_asm = None
        for j in range(i-1, -1, -1):
            stripped = lines[j].strip()
            if stripped and not stripped.startswith('#'):
                prev_asm = stripped
                break
            # Also check if CHECK-INST is before this
            if stripped.startswith('# CHECK-INST:'):
                prev_asm_from_check = stripped.replace('# CHECK-INST:', '').strip()
                break

        if prev_asm and prev_asm in encoding_map:
            output_lines_new.append(f'# CHECK-ENCODING: {encoding_map[prev_asm]}\n')
        else:
            output_lines_new.append(line)
    else:
        output_lines_new.append(line)
    i += 1

# Write updated file
with open(test_file, 'w') as f:
    f.writelines(output_lines_new)

# Report
print(f"Updated {len(encoding_map)} encodings in {test_file}")
# Check for missing encodings
missing = [asm for asm in asm_lines if asm not in encoding_map]
if missing:
    print(f"WARNING: {len(missing)} instructions had no encoding:")
    for m in missing[:10]:
        print(f"  {m}")
PYEOF

echo ""
echo "=== Step 4: Verify with FileCheck ==="
$LLVM_MC -triple=riscv64 -show-encoding --mattr=+experimental-xtheadmatrix "$TEST_FILE" 2>&1 | \
    $FILECHECK "$TEST_FILE" --check-prefixes=CHECK-ENCODING,CHECK-INST && \
    echo "ENCODING PASS" || echo "ENCODING FAIL"

$LLVM_MC -triple=riscv64 -show-encoding "$TEST_FILE" 2>&1 | \
    $FILECHECK "$TEST_FILE" --check-prefix=CHECK-ERROR && \
    echo "ERROR PASS" || echo "ERROR FAIL"

echo ""
echo "=== Step 5: Run invalid and CSR tests ==="
$LLVM_MC -triple=riscv64 --mattr=+experimental-xtheadmatrix "$INVALID_FILE" 2>&1 | \
    $FILECHECK "$INVALID_FILE" && echo "INVALID PASS" || echo "INVALID FAIL"

$LLVM_MC -triple=riscv64 --mattr=+experimental-xtheadmatrix -show-encoding "$CSR_FILE" 2>&1 | \
    $FILECHECK "$CSR_FILE" --check-prefix=CHECK-INST && echo "CSR PASS" || echo "CSR FAIL"

echo ""
echo "=== Step 6: Run full RISCV MC test suite ==="
$LIT ../llvm/test/MC/RISCV/ -j4 2>&1 | grep -E "Total|Passed|Failed|Unsupported"

echo ""
echo "=== Done ==="

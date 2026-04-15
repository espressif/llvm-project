//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: *
// TODO: This test unexpectedly passes in ESP test environment. Investigate and fix.
// UNSUPPORTED: LIBCXX-ESP-FIXME

// Make sure the test DOES NOT pass if it fails at runtime.

int main(int, char**) {
    return 1;
}

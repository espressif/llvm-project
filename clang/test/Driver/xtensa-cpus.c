// Check target CPUs are correctly passed.

// RUN: %clang -target xtensa -### -c %s 2>&1 -mcpu=esp8266 | FileCheck -check-prefix=MCPU-ESP8266 %s
// MCPU-ESP8266: "-target-cpu" "esp8266"
// MCPU-ESP8266: "-target-feature" "+density" "-target-feature" "+nsa" "-target-feature" "+mul32" "-target-feature" "+extendedl32r"
// MCPU-ESP8266: "-target-feature" "+debug" "-target-feature" "+exception" "-target-feature" "+highpriinterrupts"
// MCPU-ESP8266: "-target-feature" "+interrupt" "-target-feature" "+rvector" "-target-feature" "+timerint" "-target-feature" "+prid"
// MCPU-ESP8266: "-target-feature" "+regprotect"

// RUN: %clang -target xtensa -### -c %s 2>&1 -mcpu=esp32 | FileCheck -check-prefix=MCPU-ESP32 %s
// MCPU-ESP32: "-target-cpu" "esp32"
// MCPU-ESP32: "-target-feature" "+density" "-target-feature" "+fp" "-target-feature" "+windowed" "-target-feature" "+bool"
// MCPU-ESP32: "-target-feature" "+loop" "-target-feature" "+sext" "-target-feature" "+nsa" "-target-feature" "+clamps"
// MCPU-ESP32: "-target-feature" "+minmax" "-target-feature" "+mul32" "-target-feature" "+mul32high"
// MCPU-ESP32: "-target-feature" "+div32" "-target-feature" "+mac16" "-target-feature" "+dfpaccel"
// MCPU-ESP32: "-target-feature" "+s32c1i" "-target-feature" "+threadptr" "-target-feature" "+atomctl" "-target-feature" "+memctl"
// MCPU-ESP32: "-target-feature" "+debug" "-target-feature" "+exception" "-target-feature" "+highpriinterrupts"
// MCPU-ESP32: "-target-feature" "+coprocessor" "-target-feature" "+interrupt" "-target-feature" "+rvector" "-target-feature" "+timerint"
// MCPU-ESP32: "-target-feature" "+prid" "-target-feature" "+regprotect" "-target-feature" "+miscsr"

// RUN: %clang -target xtensa -### -c %s 2>&1 -mcpu=esp32s2 | FileCheck -check-prefix=MCPU-ESP32S2 %s
// MCPU-ESP32S2: "-target-cpu" "esp32s2"
// MCPU-ESP32S2: "-target-feature" "+density" "-target-feature" "+windowed" "-target-feature" "+sext" "-target-feature" "+nsa"
// MCPU-ESP32S2: "-target-feature" "+clamps" "-target-feature" "+minmax" "-target-feature" "+mul32"
// MCPU-ESP32S2: "-target-feature" "+mul32high" "-target-feature" "+div32" "-target-feature" "+threadptr"
// MCPU-ESP32S2: "-target-feature" "+memctl" "-target-feature" "+debug" "-target-feature" "+exception" "-target-feature" "+highpriinterrupts"
// MCPU-ESP32S2: "-target-feature" "+coprocessor" "-target-feature" "+interrupt" "-target-feature" "+rvector" "-target-feature" "+timerint"
// MCPU-ESP32S2: "-target-feature" "+prid" "-target-feature" "+regprotect" "-target-feature" "+miscsr" "-target-feature" "+esp32s2"

// RUN: %clang -target xtensa -### -c %s 2>&1 -mcpu=esp32s3 | FileCheck -check-prefix=MCPU-ESP32S3 %s
// MCPU-ESP32S3: "-target-cpu" "esp32s3"
// MCPU-ESP32S3: "-target-feature" "+density" "-target-feature" "+fp" "-target-feature" "+windowed" "-target-feature" "+bool"
// MCPU-ESP32S3: "-target-feature" "+loop" "-target-feature" "+sext" "-target-feature" "+nsa" "-target-feature" "+clamps"
// MCPU-ESP32S3: "-target-feature" "+minmax" "-target-feature" "+mul32" "-target-feature" "+mul32high" "-target-feature" "+div32"
// MCPU-ESP32S3: "-target-feature" "+mac16" "-target-feature" "+dfpaccel" "-target-feature" "+s32c1i"
// MCPU-ESP32S3: "-target-feature" "+threadptr" "-target-feature" "+atomctl" "-target-feature" "+memctl"
// MCPU-ESP32S3: "-target-feature" "+debug" "-target-feature" "+exception" "-target-feature" "+highpriinterrupts"
// MCPU-ESP32S3: "-target-feature" "+coprocessor" "-target-feature" "+interrupt" "-target-feature" "+rvector" "-target-feature" "+timerint"
// MCPU-ESP32S3: "-target-feature" "+prid" "-target-feature" "+regprotect" "-target-feature" "+miscsr" "-target-feature" "+esp32s3"

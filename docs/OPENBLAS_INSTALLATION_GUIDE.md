# Solution 1: Install OpenBLAS via vcpkg

## Step 1: Install vcpkg (if not already installed)
```bash
# Clone vcpkg
git clone https://github.com/Microsoft/vcpkg.git C:/vcpkg
cd C:/vcpkg
./bootstrap-vcpkg.bat

# Integrate with Visual Studio
./vcpkg integrate install
```

## Step 2: Install OpenBLAS
```bash
# Install OpenBLAS for x64-windows
C:/vcpkg/vcpkg install openblas:x64-windows

# This will install to: C:/vcpkg/installed/x64-windows/
```

## Step 3: Update xmake/cpu.lua to use vcpkg

Replace the `add_requires("openblas")` line with manual include/link paths:

```lua
target("llaisys-ops-cpu")
    -- ... existing config ...

    -- Phase 3: OpenBLAS via vcpkg
    if is_plat("windows") then
        add_includedirs("C:/vcpkg/installed/x64-windows/include")
        add_linkdirs("C:/vcpkg/installed/x64-windows/lib")
        add_links("openblas")
    else
        add_packages("openblas")
    end
```

## Step 4: Build
```bash
cd F:/Project/llaisys
xmake f -c
xmake
xmake install
```

---

# Solution 2: Download Prebuilt OpenBLAS Binaries

## Step 1: Download from official releases
```bash
# Download from: https://github.com/OpenMathLib/OpenBLAS/releases
# Get: OpenBLAS-0.3.30-x64.zip (or latest version)
```

## Step 2: Extract to a known location
```bash
# Extract to: C:/OpenBLAS/
# Structure should be:
#   C:/OpenBLAS/include/cblas.h
#   C:/OpenBLAS/lib/openblas.lib
#   C:/OpenBLAS/bin/openblas.dll
```

## Step 3: Update xmake/cpu.lua

```lua
target("llaisys-ops-cpu")
    -- ... existing config ...

    -- Phase 3: OpenBLAS manual installation
    if is_plat("windows") then
        add_includedirs("C:/OpenBLAS/include")
        add_linkdirs("C:/OpenBLAS/lib")
        add_links("openblas")
        -- Copy DLL to output directory
        after_build(function (target)
            os.cp("C:/OpenBLAS/bin/openblas.dll", "$(buildir)/bin/")
        end)
    else
        add_packages("openblas")
    end
```

---

# Solution 3: Use Intel MKL Instead (Alternative)

Intel MKL is highly optimized for Intel CPUs and easier to install on Windows:

## Step 1: Install Intel MKL
```bash
# Download from: https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html
# Or use chocolatey:
choco install intel-mkl
```

## Step 2: Update code to use MKL

In `src/ops/linear/cpu/linear_cpu.cpp`, replace:
```cpp
#include <cblas.h>
```

With:
```cpp
#include <mkl.h>
#include <mkl_cblas.h>
```

## Step 3: Update xmake/cpu.lua

```lua
target("llaisys-ops-cpu")
    -- ... existing config ...

    -- Phase 3: Intel MKL
    if is_plat("windows") then
        add_includedirs("C:/Program Files (x86)/Intel/oneAPI/mkl/latest/include")
        add_linkdirs("C:/Program Files (x86)/Intel/oneAPI/mkl/latest/lib")
        add_links("mkl_intel_lp64", "mkl_sequential", "mkl_core")
    end
```

---

# Solution 4: Fix xmake Package (Advanced)

If you want to fix the xmake package issue:

## Step 1: Clear xmake cache
```bash
cd F:/Project/llaisys
xmake clean -a
rm -rf .xmake
rm -rf ~/.xmake/packages/o/openblas
```

## Step 2: Try with explicit version
```lua
add_requires("openblas 0.3.30")
```

## Step 3: Use xrepo with verbose output
```bash
xrepo install -v openblas
```

---

# Recommended Approach for Your System

Given that you have:
- Windows 11
- MSVC 2022
- Intel Core Ultra 7 265K

**I recommend Solution 1 (vcpkg)** because:
1. ✅ Well-maintained and tested with MSVC
2. ✅ Automatic dependency resolution
3. ✅ Easy integration with Visual Studio
4. ✅ Prebuilt binaries for Windows

**Alternative: Solution 3 (Intel MKL)** because:
1. ✅ Highly optimized for Intel CPUs (your 265K)
2. ✅ Often faster than OpenBLAS on Intel hardware
3. ✅ Professional support and documentation
4. ✅ Free for developers

---

# Quick Start: vcpkg Method

```bash
# 1. Install vcpkg (one-time setup)
git clone https://github.com/Microsoft/vcpkg.git C:/vcpkg
cd C:/vcpkg
./bootstrap-vcpkg.bat
./vcpkg integrate install

# 2. Install OpenBLAS
C:/vcpkg/vcpkg install openblas:x64-windows

# 3. Update xmake/cpu.lua (remove add_requires, add manual paths)
# See Solution 1 above

# 4. Build
cd F:/Project/llaisys
xmake f -c
xmake
```

This should resolve the package dependency issues and allow you to complete Phase 3 of the optimization.


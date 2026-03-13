target("llaisys-device-cpu")
    set_kind("static")
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("../src/device/cpu/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-ops-cpu")
    set_kind("static")
    add_deps("llaisys-tensor")
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC")  -- removed -Wno-unknown-pragmas: omp pragmas now recognized
    end

    -- Phase 1: OpenMP
    if is_plat("windows") then
        add_cxflags("/openmp")  -- MSVC links OpenMP automatically; no ldflags needed
    else
        add_cxflags("-fopenmp")
        add_ldflags("-fopenmp")
    end

    -- Phase 2: AVX2 + FMA
    if is_plat("windows") then
        add_cxflags("/arch:AVX2")  -- implicitly enables FMA on MSVC
    else
        add_cxflags("-mavx2", "-mfma")
    end

    -- Phase 3: Intel MKL (F32 path only)
    if is_plat("windows") then
        add_includedirs("D:/ProgramData/miniconda3/Library/include")
        add_linkdirs("D:/ProgramData/miniconda3/Library/lib")
        add_links("mkl_intel_lp64_dll", "mkl_sequential_dll", "mkl_core_dll")
    else
        add_packages("openblas")
    end

    add_files("../src/ops/*/cpu/*.cpp")

    on_install(function (target) end)
target_end()


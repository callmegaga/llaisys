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

    add_files("../src/ops/*/cpu/*.cpp")

    on_install(function (target) end)
target_end()


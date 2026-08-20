#!/usr/bin/env julia
# run_pdcs_all.jl
# PDCS-GPU batch test on all CBLIB instances
#
# Usage:
#   julia run_pdcs_all.jl <input_dir> <output_dir> [time_limit] [rel_tol] [abs_tol] [gpu_device]
#
# Example (300 s, CBLIB default):
#   julia --project=$HOME/PDCS run_pdcs_all.jl \
#       $HOME/CBLIB/represent_data_unzip/ \
#       $HOME/test1/results/pdcs_$(date +%Y%m%d_%H%M%S)/ \
#       300.0 1e-4 1e-4 4
#
# Runs in the background. Track progress with:
#   tail -f <output_dir>/pdcs_all.log
#   cat  <output_dir>/pdcs_all.json

using Pkg
Pkg.activate(joinpath(ENV["HOME"], "PDCS"))

using Mmap
using SparseArrays
include(joinpath(ENV["HOME"], "HPRSOCP/src/utils/cbf_io.jl"))

using PDCS: PDCS_GPU
using JuMP
import MathOptInterface as MOI
using Printf
using JSON
using Dates

const DEFAULT_TIME_LIMIT  = 300.0
const DEFAULT_REL_TOL     = 1e-4
const DEFAULT_ABS_TOL     = 1e-4
const DEFAULT_GPU_DEVICE  = 4          # your H100 device

# =====================================================================
# Core solve function (mirrors run_pdcs_one.jl)
# =====================================================================
function solve_one(
    input_file::String,
    output_dir::String;
    time_limit::Float64 = DEFAULT_TIME_LIMIT,
    rel_tol::Float64   = DEFAULT_REL_TOL,
    abs_tol::Float64   = DEFAULT_ABS_TOL,
    gpu_device::Int    = DEFAULT_GPU_DEVICE,
)
    name = splitext(basename(input_file))[1]
    json_file = joinpath(output_dir, "$(name).json")
    isfile(input_file) || error("Input file not found: $input_file")
    mkpath(output_dir)

    t0 = time_ns()

    # ── Read ──────────────────────────────────────────────────────
    Q_unused, c, A, rhs, SOC_con_idx, number_eq, number_ineq, lb, ub, SOC_var_idx, obj_constant =
        read_cbf(input_file)
    t_parse = (time_ns() - t0) / 1e9

    n = length(c)
    m = size(A, 1)
    soc_count = length(SOC_con_idx) - 1

    # ── Build model ────────────────────────────────────────────────
    model = Model(PDCS_GPU.Optimizer)
    set_optimizer_attribute(model, "time_limit_secs", time_limit)
    set_optimizer_attribute(model, "rel_tol", rel_tol)
    set_optimizer_attribute(model, "abs_tol", abs_tol)
    set_optimizer_attribute(model, "verbose", 0)
    set_optimizer_attribute(model, "logfile", nothing)

    n_vars = n
    @variable(model, x[i=1:n_vars])
    for i in 1:n_vars
        isfinite(lb[i]) && set_lower_bound(x[i], lb[i])
        isfinite(ub[i]) && set_upper_bound(x[i], ub[i])
    end
    @objective(model, Min, sum(c[i] * x[i] for i in 1:n_vars))

    for i in 1:number_eq
        @constraint(model, sum(A[i,j] * x[j] for j in 1:n_vars) == rhs[i])
    end

    # CBF v3: L+ = {g|g>=0}, L- = {g|g<=0}, g = A_cbf*x + b_cbf.
    # rhs = -b_cbf (L+) or +b_cbf (L-, A already negated).
    # Both require A*x >= rhs.
    for i in (number_eq+1):(number_eq+number_ineq)
        @constraint(model, sum(A[i,j] * x[j] for j in 1:n_vars) >= rhs[i])
    end

    if soc_count > 0
        for k in 1:soc_count
            row_start = SOC_con_idx[k]
            row_end   = SOC_con_idx[k+1] - 1
            dim       = row_end - row_start + 1
            exprs = [sum(A[r,j] * x[j] for j in 1:n_vars) - rhs[r]
                     for r in row_start:row_end]
            @constraint(model, exprs in MOI.SecondOrderCone(dim))
        end
    end
    t_build = (time_ns() - t0 - t_parse * 1e9) / 1e9
    t_startup = t_parse + t_build

    # ── Solve ──────────────────────────────────────────────────────
    optimize!(model)
    t_end   = time_ns()
    t_wall  = (t_end - t0) / 1e9
    t_solve = (t_end - t0 - (t_parse + t_build) * 1e9) / 1e9

    status     = termination_status(model)
    obj_val    = objective_value(model) + obj_constant
    solve_time = solve_time(model)   # reported by PDCS GPU kernel

    result = Dict(
        "problem"          => name,
        "solver"           => "PDCS-GPU",
        "status"           => string(status),
        "objective"        => obj_val,
        "solve_time_sec"   => solve_time,
        "wall_time_sec"    => t_wall,
        "startup_time_sec" => t_startup,
        "build_time_sec"   => t_build,
        "parse_time_sec"   => t_parse,
        "n_vars"           => n_vars,
        "n_constraints"    => m,
        "number_eq"        => number_eq,
        "number_ineq"      => number_ineq,
        "soc_count"        => soc_count,
        "obj_constant"     => obj_constant,
        "rel_tol"          => rel_tol,
        "abs_tol"          => abs_tol,
        "time_limit"       => time_limit,
        "gpu_device"       => gpu_device,
        "start_time"       => "",
        "end_time"         => "",
    )

    open(json_file, "w") do f
        JSON.print(f, result, 4)
    end

    return (;
        name,
        status,
        objective = obj_val,
        solve_time,
        wall_time = t_wall,
        startup_time = t_startup,
        json_file,
    )
end

# =====================================================================
# Main batch driver
# =====================================================================
function main()
    if length(ARGS) < 2
        println("Usage: julia run_pdcs_all.jl <input_dir> <output_dir> [time_limit] [rel_tol] [abs_tol] [gpu_device]")
        exit(1)
    end

    input_dir  = ARGS[1]
    output_dir = ARGS[2]
    time_limit = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : DEFAULT_TIME_LIMIT
    rel_tol    = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : DEFAULT_REL_TOL
    abs_tol    = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : DEFAULT_ABS_TOL
    gpu_device = length(ARGS) >= 6 ? parse(Int,    ARGS[6]) : DEFAULT_GPU_DEVICE

    isdir(input_dir) || error("Input directory not found: $input_dir")
    mkpath(output_dir)

    timestamp = Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS")
    log_file  = joinpath(output_dir, "pdcs_all.log")

    # ── Discover all .cbf files ───────────────────────────────────
    cbf_files = sort([f for f in readdir(input_dir, join=true)
                      if endswith(f, ".cbf")])
    n_total = length(cbf_files)

    if n_total == 0
        println("No .cbf files found in $input_dir")
        exit(1)
    end

    # ── Open shared log ───────────────────────────────────────────
    log_io = open(log_file, "w")
    start_iso = Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS")

    function log(msg::String)
        ts = Dates.format(Dates.now(), "HH:MM:SS")
        line = "[$ts] $msg"
        println(log_io, line)
        flush(log_io)
        println(line)   # also print to stdout (captured by nohup)
    end

    log("=" ^ 62)
    log("PDCS-GPU Batch Runner  started $start_iso")
    log("Input:   $input_dir  ($n_total problems)")
    log("Output:  $output_dir")
    log("Params:  time_limit=$time_limit, rel_tol=$rel_tol, abs_tol=$abs_tol, gpu=$gpu_device")
    log("=" ^ 62)

    all_results = Dict{String, Any}()
    ok_statuses = ["OPTIMAL", "LOCALLY_SOLVED"]

    t_batch_start = time_ns()

    for (idx, input_file) in enumerate(cbf_files)
        name = splitext(basename(input_file))[1]
        log("[$idx/$n_total] >>> START $name")

        flush(log_io)
        t_prob_start = time_ns()

        try
            res = solve_one(input_file, output_dir;
                            time_limit, rel_tol, abs_tol, gpu_device)
            t_prob = (time_ns() - t_prob_start) / 1e9

            all_results[name] = res
            log("[$idx/$n_total] <<< DONE  $name  status=$(res.status)  " *
                "obj=$(fmt_obj(res.objective))  " *
                "solve=$(fmt_time(res.solve_time))s  " *
                "wall=$(fmt_time(res.wall_time))s  " *
                "startup=$(fmt_time(res.startup_time))s")

        catch e
            t_prob = (time_ns() - t_prob_start) / 1e9
            err_msg = sprint(showerror, e)
            log("[$idx/$n_total] !!! FAIL  $name  ($t_prob s)")
            log("         ERROR: $err_msg")
            all_results[name] = Dict(
                "problem" => name,
                "status"  => "ERROR",
                "error"   => err_msg,
                "wall_time_sec" => t_prob,
            )
        end

        flush(log_io)
    end

    t_batch = (time_ns() - t_batch_start) / 1e9

    # ── Print summary ──────────────────────────────────────────────
    log("")
    log("=" ^ 62)
    log("BATCH SUMMARY  ($n_total problems in $(fmt_time(t_batch)) total)")
    log("=" ^ 62)

    # Count statuses
    status_counts = Dict{String,Int}()
    for res in values(all_results)
        s = res isa NamedTuple ? string(res.status) : string(get(res, "status", "UNKNOWN"))
        status_counts[s] = get(status_counts, s, 0) + 1
    end
    for (s, cnt) in sort(status_counts)
        log("  $s : $cnt")
    end

    log("")
    header = "  # | Problem                       | Status         | Objective        | Solve(s) | Wall(s) | Startup(s)"
    log(header)
    log("  " * "-"^75)

    for (idx, input_file) in enumerate(cbf_files)
        name = splitext(basename(input_file))[1]
        res  = get(all_results, name, nothing)
        if res === nothing
            log("  $idx | $name | MISSING | — | — | —")
            continue
        end
        if res isa NamedTuple
            log("  $idx | $(rpad(name,30)) | $(rpad(string(res.status),13)) | " *
                "$(fmt_obj(res.objective)) | " *
                "$(rpad(fmt_time(res.solve_time),8)) | " *
                "$(rpad(fmt_time(res.wall_time),8)) | " *
                "$(fmt_time(res.startup_time))")
        else
            log("  $idx | $(rpad(name,30)) | $(rpad(string(get(res,"status","?")),13)) | — | — | —")
        end
    end

    # ── Save combined JSON ─────────────────────────────────────────
    combined = Dict(
        "batch_start"    => start_iso,
        "batch_end"      => Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"),
        "batch_time_sec" => t_batch,
        "input_dir"      => input_dir,
        "output_dir"     => output_dir,
        "time_limit"     => time_limit,
        "rel_tol"        => rel_tol,
        "abs_tol"        => abs_tol,
        "gpu_device"     => gpu_device,
        "n_total"        => n_total,
        "status_counts"  => status_counts,
        "results"        => all_results,
    )
    combined_file = joinpath(output_dir, "pdcs_all.json")
    open(combined_file, "w") do f
        JSON.print(f, combined, 4)
    end
    log("")
    log("Combined JSON: $combined_file")
    log("Done.")

    close(log_io)
end

# =====================================================================
# Formatting helpers
# =====================================================================
fmt_time(t::Float64)   = @sprintf("%7.3f", t)
fmt_time(t::Real)       = fmt_time(Float64(t))
fmt_time(t::Missing)    = "    —  "
fmt_time(t::Nothing)    = "    —  "
fmt_time(t)             = "    —  "

function fmt_obj(v::Real)
    v, s = isnan(v) ? (0.0, "NaN") : (abs(v), sign(v) < 0 ? "-" : "")
    if v == 0.0       return "$(s)0.000e+00"
    elseif v <  1e-3  return "$s$(v)"
    elseif v >= 1e6   return @sprintf("%s%.4e", s, v)
    else              return @sprintf("%s%.4e", s, v)
    end
end
fmt_obj(::Missing) = "—"

main()

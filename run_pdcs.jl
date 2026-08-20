#!/usr/bin/env julia
# run_pdcs.jl
# PDCS-GPU single-instance test script
# Usage: julia run_pdcs.jl <input.cbf> <output_dir> [time_limit] [rel_tol] [abs_tol] [gpu_device]
#
# Example:
#   julia run_pdcs.jl /path/to/10_std.cbf ./results 60.0 1e-4 1e-4 0

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

# Default parameters
const DEFAULT_TIME_LIMIT = 300.0
const DEFAULT_REL_TOL = 1e-4
const DEFAULT_ABS_TOL = 1e-4
const DEFAULT_GPU_DEVICE = 0

function main()
    # Parse command line arguments
    if length(ARGS) < 2
        println("Usage: julia run_pdcs.jl <input.cbf> <output_dir> [time_limit] [rel_tol] [abs_tol] [gpu_device]")
        println()
        println("Arguments:")
        println("  input.cbf   - Path to CBF file (required)")
        println("  output_dir  - Directory for output files (required)")
        println("  time_limit  - Time limit in seconds (default: 300.0)")
        println("  rel_tol     - Relative tolerance for primal/dual residual and gap (default: 1e-4)")
        println("  abs_tol     - Absolute tolerance for primal/dual residual and gap (default: 1e-4)")
        println("  gpu_device  - GPU device ID (default: 0)")
        return 1
    end
    
    input_file = ARGS[1]
    output_dir = ARGS[2]
    time_limit = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : DEFAULT_TIME_LIMIT
    rel_tol = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : DEFAULT_REL_TOL
    abs_tol = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : DEFAULT_ABS_TOL
    gpu_device = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : DEFAULT_GPU_DEVICE
    
    name = splitext(basename(input_file))[1]
    json_file = joinpath(output_dir, "$(name).json")
    
    isfile(input_file) || error("Input file not found: $input_file")
    mkpath(output_dir)
    
    start_wall = time()
    start_iso = Dates.format(Dates.now(), "yyyy-mm-ddTHH:MM:SS")
    
    println("="^60)
    println("PDCS-GPU Solver")
    println("="^60)
    println("Problem: $name")
    println("Input:   $input_file")
    println("Output:  $output_dir")
    println("Params:  time_limit=$time_limit, rel_tol=$rel_tol, abs_tol=$abs_tol")
    println("GPU:     device=$gpu_device")
    println("="^60)
    
    try
        # ========== Read CBF ==========
        println("\n[1] Reading CBF file...")
        Q, c, A, rhs, SOC_con_idx, number_eq, number_ineq, l, u, SOC_var_idx, obj_constant = 
            read_cbf(input_file)
        
        n = length(c)
        m = size(A, 1)
        
        println("  Variables: $n")
        println("  Constraints: $m")
        println("  number_eq: $number_eq")
        println("  number_ineq: $number_ineq")
        println("  obj_constant: $obj_constant")
        println("  SOC_con_idx length: $(length(SOC_con_idx))")
        println("  SOC_var_idx length: $(length(SOC_var_idx))")
        
        # ========== Build JuMP model ==========
        println("\n[2] Building JuMP model...")
        
        model = Model(PDCS_GPU.Optimizer)
        set_optimizer_attribute(model, "time_limit_secs", time_limit)
        set_optimizer_attribute(model, "rel_tol", rel_tol)
        set_optimizer_attribute(model, "abs_tol", abs_tol)
        set_optimizer_attribute(model, "verbose", 1)
        
        n_vars = length(c)
        @variable(model, x[1:n_vars])
        @objective(model, Min, c' * x)
        
        # Equality constraints
        println("  Adding $number_eq equality constraints...")
        for i in 1:number_eq
            @constraint(model, sum(A[i,j] * x[j] for j in 1:n_vars) == rhs[i])
        end
        
        # Inequality constraints
        println("  Adding $number_ineq inequality constraints...")
        for i in (number_eq+1):(number_eq+number_ineq)
            @constraint(model, sum(A[i,j] * x[j] for j in 1:n_vars) <= rhs[i])
        end
        
        # SOC constraints
        soc_count = length(SOC_con_idx) - 1
        if soc_count > 0
            println("  Adding $soc_count SOC constraints...")
            for k in 1:soc_count
                con_start = SOC_con_idx[k]
                con_end = SOC_con_idx[k+1] - 1
                dim = con_end - con_start + 1
                
                # Build SOC constraint: t >= ||x||
                # CBF format: con_start contains t, remaining are x variables
                soc_vars = [x[con_start]]
                for idx in (con_start+1):con_end
                    push!(soc_vars, x[idx])
                end
                
                @constraint(model, soc in MOI.SecondOrderCone(dim), soc_vars)
            end
        end
        
        # ========== Solve ==========
        println("\n[3] Solving...")
        flush(stdout)
        
        optimize!(model)
        
        # ========== Extract results ==========
        status = termination_status(model)
        obj_val = objective_value(model)
        solve_time_sec = solve_time(model)
        
        println("\n[4] Results:")
        println("  Status:       $status")
        println("  Objective:    $obj_val")
        println("  Solve time:   $(solve_time_sec) s")
        println("  Wall time:    $(time() - start_wall) s")
        
        # ========== Save JSON ==========
        result = Dict(
            "problem" => name,
            "solver" => "PDCS-GPU",
            "status" => string(status),
            "objective" => obj_val,
            "solve_time_sec" => solve_time_sec,
            "wall_time_sec" => time() - start_wall,
            "n_vars" => n_vars,
            "n_constraints" => m,
            "number_eq" => number_eq,
            "number_ineq" => number_ineq,
            "soc_count" => soc_count,
            "obj_constant" => obj_constant,
            "rel_tol" => rel_tol,
            "abs_tol" => abs_tol,
            "time_limit" => time_limit,
            "gpu_device" => gpu_device,
            "start_time" => start_iso,
            "end_time" => Dates.format(Dates.now(), "yyyy-mm-ddTHH:MM:SS")
        )
        
        open(json_file, "w") do f
            JSON.print(f, result, 4)
        end
        println("  JSON saved:   $json_file")
        
        return 0
        
    catch e
        error_msg = sprint(showerror, e)
        println("\nERROR: $error_msg")
        
        result = Dict(
            "problem" => name,
            "solver" => "PDCS-GPU",
            "status" => "ERROR",
            "error" => error_msg,
            "wall_time_sec" => time() - start_wall,
            "start_time" => start_iso,
            "end_time" => Dates.format(Dates.now(), "yyyy-mm-ddTHH:MM:SS")
        )
        
        open(json_file, "w") do f
            JSON.print(f, result, 4)
        end
        
        return 1
    end
end

exit(main())

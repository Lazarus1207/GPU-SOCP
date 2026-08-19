# run_pdcs_one.jl
# PDCS-GPU single-instance test script
# Usage: julia run_pdcs_one.jl <input.cbf> <output_dir> [time_limit] [rel_tol] [abs_tol]

using Pkg
Pkg.activate(joinpath(ENV["HOME"], "PDCS"))

using PDCS: PDCS_GPU
using ConicBenchmarkUtilities
using SparseArrays
using JuMP
import MathOptInterface as MOI
using CUDA
using Printf
using JSON
using Dates

const DEFAULT_TIME_LIMIT = 300.0
const DEFAULT_REL_TOL = 1e-4
const DEFAULT_ABS_TOL = 1e-4

function parse_cones(aconcones, vcones)
    mGzero = mGnonnegative = 0
    socG_list = Int[]
    rsocG_list = Int[]
    expG_total = 0

    for (cone_type, idxs) in aconcones
        if cone_type == :Zero
            mGzero += length(idxs)
        elseif cone_type in (:NonNeg, :NonPos)
            mGnonnegative += length(idxs)
        elseif cone_type == :SOC
            push!(socG_list, length(idxs))
        elseif cone_type == :RSOC
            push!(rsocG_list, length(idxs))
        elseif cone_type == :ExpPrimal
            expG_total += div(length(idxs), 3)
        end
    end

    soc_x_list = Int[]
    rsoc_x_list = Int[]
    exp_x_total = 0

    for (cone_type, idxs) in vcones
        if cone_type == :SOC
            push!(soc_x_list, length(idxs))
        elseif cone_type == :RSOC
            push!(soc_x_list, length(idxs))
        elseif cone_type == :ExpPrimal
            exp_x_total += div(length(idxs), 3)
        end
    end

    return mGzero, mGnonnegative, socG_list, rsocG_list, expG_total,
           soc_x_list, rsoc_x_list, exp_x_total
end

function main()
    @assert length(ARGS) >= 2 "Usage: julia run_pdcs_one.jl <input.cbf> <output_dir> [time_limit] [rel_tol] [abs_tol]"

    input_file = ARGS[1]
    output_dir = ARGS[2]
    time_limit = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : DEFAULT_TIME_LIMIT
    rel_tol = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : DEFAULT_REL_TOL
    abs_tol = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : DEFAULT_ABS_TOL

    name = splitext(basename(input_file))[1]
    log_file = joinpath(output_dir, "$(name)_pdcs_log.txt")
    json_file = joinpath(output_dir, "$(name).json")

    isfile(input_file) || error("Input file not found: $input_file")
    mkpath(output_dir)

    start_wall = time()
    start_iso = Dates.format(Dates.now(), "yyyy-mm-ddTHH:MM:SS")

    println("========================================")
    println("PDCS-GPU: $name")
    println("  input:   $input_file")
    println("  output:  $output_dir")
    println("  time_limit=$time_limit, rel_tol=$rel_tol, abs_tol=$abs_tol")
    println("========================================")

    try
        # Load CBF
        cbf = readcbfdata(input_file)
        c, G, b, aconcones, vcones, vtypes, sense, obj_offset = cbftoppb(cbf)

        n = length(c)
        m = size(G, 1)

        mGzero, mGnonnegative, socG_list, rsocG_list, expG_total,
            soc_x_list, rsoc_x_list, exp_x_total = parse_cones(aconcones, vcones)

        @printf "  n=%d, m=%d, mGzero=%d, mGnonneg=%d, socG=%s, rsocG=%s, expG=%d\n"
            n m mGzero mGnonnegative socG_list rsocG_list expG_total
        @printf "  soc_x=%s, rsoc_x=%s, exp_x=%d\n"
            soc_x_list rsoc_x_list exp_x_total

        # Call PDCS-GPU
        t0 = time()
        sol = PDCS_GPU.rpdhg_gpu_solve(
            b=n, m=m, nb=n,
            c=c, G=G, h=b,
            mGzero=mGzero,
            mGnonnegative=mGnonnegative,
            socG=socG_list,
            rsocG=rsocG_list,
            expG=expG_total,
            dual_expG=0,
            bl=Float64[], bu=Float64[],
            soc_x=soc_x_list,
            rsoc_x=rsoc_x_list,
            exp_x=exp_x_total,
            dual_exp_x=0,
            use_preconditioner=true,
            rescaling_method=:ruiz_pock_chambolle,
            method=:average,
            use_adaptive_restart=true,
            use_adaptive_step_size_weight=true,
            use_resolving=true,
            use_accelerated=false,
            use_aggressive=true,
            time_limit=time_limit,
            rel_tol=rel_tol,
            abs_tol=abs_tol,
            print_freq=5000,
            check_terminate_freq=2000,
            max_outer_iter=50000,
            max_inner_iter=500000,
            verbose=1,
            logfile_name=log_file,
        )
        elapsed = time() - t0
        wall_sec = time() - start_wall
        end_iso = Dates.format(Dates.now(), "yyyy-mm-ddTHH:MM:SS")

        # Extract results
        status = string(sol.exit_status)
        pObj = sol.objective_value
        dObj = sol.dual_objective_value
        rel_gap = abs(pObj - dObj) / max(1.0, abs(pObj), abs(dObj))
        iter = sol.iterations
        runtime = sol.solve_time_sec

        @printf "\n[PDCS-GPU] status=%s iter=%d time=%.3fs wall=%.3fs\n" status iter runtime wall_sec
        @printf "[PDCS-GPU] pObj=%.10e dObj=%.10e gap=%.2e\n" pObj dObj rel_gap

        # Write JSON
        record = (;
            problem=name,
            algorithm="PDCS-GPU",
            instance=input_file,
            status=status,
            iterations=iter,
            runtime_sec=runtime,
            wall_sec=wall_sec,
            primal_obj=pObj,
            dual_obj=dObj,
            rel_gap=rel_gap,
            primal_feas=0.0,
            dual_feas=0.0,
            timestamp_start=start_iso,
            timestamp_end=end_iso,
            exit_code=sol.exit_code,
            error="",
        )

        open(json_file, "w") do jf
            JSON.print(jf, record, 2)
        end
        println("[PDCS-GPU] JSON -> $json_file")

    catch e
        wall_sec = time() - start_wall
        end_iso = Dates.format(Dates.now(), "yyyy-mm-ddTHH:MM:SS")
        msg = sprint(showerror, e)
        println("[ERROR] $msg")
        status = "ERROR"

        record = (;
            problem=name,
            algorithm="PDCS-GPU",
            instance=input_file,
            status=status,
            iterations=0,
            runtime_sec=wall_sec,
            wall_sec=wall_sec,
            primal_obj=NaN,
            dual_obj=NaN,
            rel_gap=NaN,
            primal_feas=NaN,
            dual_feas=NaN,
            timestamp_start=start_iso,
            timestamp_end=end_iso,
            exit_code=1,
            error=msg,
        )

        open(json_file, "w") do jf
            JSON.print(jf, record, 2)
        end
        println("[PDCS-GPU] ERROR -> $json_file")
    end

    println("Done: $name")
end

main()
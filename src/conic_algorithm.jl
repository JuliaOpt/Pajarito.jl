#  Copyright 2016, Los Alamos National Laboratory, LANS LLC, and Chris Coey.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, you can obtain one at http://mozilla.org/MPL/2.0/.

#=========================================================
This mixed-integer conic programming algorithm is described in:
  Lubin, Yamangil, Bent, Vielma (2016), Extended formulations
  in Mixed-Integer Convex Programming, IPCO 2016, Liege, Belgium
  (available online at http://arxiv.org/abs/1511.06710)

Model MICP with JuMP.jl conic format or Convex.jl DCP format
http://mathprogbasejl.readthedocs.org/en/latest/conic.html


TODO issues
- MPB issue - can't call supportedcones on defaultConicsolver
- maybe want two zero tols: one for discarding cut if largest value is too small, and one for setting near zeros to zero (the former should be larger)

TODO features
- implement warm-starting: use set_best_soln!
- enable querying logs information etc
- log primal cuts added

=========================================================#

using JuMP

type PajaritoConicModel <: MathProgBase.AbstractConicModel
    # Solver parameters
    log_level::Int              # Verbosity flag: -1 for no output, 0 for minimal solution information, 1 for basic OA iteration and solve statistics, 2 for cone summary information, 3 for infeasibilities of duals, cuts, and OA solutions
    timeout::Float64            # Time limit for outer approximation algorithm not including initial load (in seconds)
    rel_gap::Float64            # Relative optimality gap termination condition

    mip_solver_drives::Bool     # Let MIP solver manage convergence and conic subproblem calls (to add lazy cuts and heuristic solutions in branch and cut fashion)
    mip_solver::MathProgBase.AbstractMathProgSolver # MIP solver (MILP or MISOCP)
    mip_subopt_solver::MathProgBase.AbstractMathProgSolver # MIP solver for suboptimal solves, with appropriate options (gap or timeout) specified directly
    mip_subopt_count::Int       # (Conic only) Number of times to solve MIP suboptimally with time limit between zero gap solves
    round_mip_sols::Bool        # (Conic only) Round the integer variable values from the MIP solver before passing to the conic subproblems
    pass_mip_sols::Bool         # (Conic only) Give best feasible solutions constructed from conic subproblem solution to MIP

    cont_solver::MathProgBase.AbstractMathProgSolver # Continuous solver (conic or nonlinear)
    solve_relax::Bool           # (Conic only) Solve the continuous conic relaxation to add initial dual cuts
    dualize_relax::Bool         # (Conic only) Solve the conic dual of the continuous conic relaxation
    dualize_sub::Bool           # (Conic only) Solve the conic duals of the continuous conic subproblems

    soc_disagg::Bool            # (Conic only) Disaggregate SOC cones in the MIP only
    soc_in_mip::Bool            # (Conic only) Use SOC cones in the MIP outer approximation model (if MIP solver supports MISOCP)
    sdp_eig::Bool               # (Conic SDP only) Use SDP eigenvector-derived cuts
    sdp_soc::Bool               # (Conic SDP only) Use SDP eigenvector SOC cuts (if MIP solver supports MISOCP; except during MIP-driven solve)
    init_soc_one::Bool          # (Conic only) Start with disaggregated L_1 outer approximation cuts for SOCs (if soc_disagg)
    init_soc_inf::Bool          # (Conic only) Start with disaggregated L_inf outer approximation cuts for SOCs (if soc_disagg)
    init_exp::Bool              # (Conic Exp only) Start with several outer approximation cuts on the exponential cones
    init_sdp_lin::Bool          # (Conic SDP only) Use SDP initial linear cuts
    init_sdp_soc::Bool          # (Conic SDP only) Use SDP initial SOC cuts (if MIP solver supports MISOCP)

    viol_cuts_only::Bool        # (Conic only) Only add cuts that are violated by the current MIP solution (may be useful for MSD algorithm where many cuts are added)
    proj_dual_infeas::Bool      # (Conic only) Project dual cone infeasible dual vectors onto dual cone boundaries
    proj_dual_feas::Bool        # (Conic only) Project dual cone strictly feasible dual vectors onto dual cone boundaries
    scale_dual_cuts::Bool       # (Conic only) Rescale a dual vector so largest abs value is 1
    prim_cuts_only::Bool        # (Conic only) Do not add dual cuts
    prim_cuts_always::Bool      # (Conic only) Add primal cuts at each iteration or in each lazy callback
    prim_cuts_assist::Bool      # (Conic only) Add primal cuts only when integer solutions are repeating
    prim_viol_cuts_only::Bool   # (Conic only) Only add primal cuts that are violated (including individual disaggregated cuts)
    prim_max_viol_only::Bool    # (Conic only) Only add primal cuts for the cone with largest absolute violation
    prim_soc_disagg::Bool       # (Conic only) Use disaggregated primal cuts for SOCs
    prim_sdp_eig::Bool          # (Conic only) Use eigenvector cuts for SDPs

    tol_zero::Float64           # (Conic only) Tolerance for setting small absolute values in duals to zeros
    tol_prim_infeas::Float64    # (Conic only) Tolerance level for cone outer infeasibilities for primal cut adding functions (must be at least 1e-5)
    tol_sdp_eigvec::Float64     # (Conic SDP only) Tolerance for setting small values in SDP eigenvectors to zeros (for cut sanitation)
    tol_sdp_eigval::Float64     # (Conic SDP only) Tolerance for ignoring eigenvectors corresponding to small (positive) eigenvalues

    # Initial data
    num_var_orig::Int           # Initial number of variables
    num_con_orig::Int           # Initial number of constraints
    c_orig                      # Initial objective coefficients vector
    A_orig                      # Initial affine constraint matrix (sparse representation)
    b_orig                      # Initial constraint right hand side
    cone_con_orig               # Initial constraint cones vector (cone, index)
    cone_var_orig               # Initial variable cones vector (cone, index)
    var_types::Vector{Symbol}   # Variable types vector on original variables (only :Bin, :Cont, :Int)
    # var_start::Vector{Float64}  # Variable warm start vector on original variables

    # Conic subproblem data
    cone_con_sub::Vector{Tuple{Symbol,Vector{Int}}} # Constraint cones data in conic subproblem
    cone_var_sub::Vector{Tuple{Symbol,Vector{Int}}} # Variable cones data in conic subproblem
    A_sub_cont::SparseMatrixCSC{Float64,Int64} # Submatrix of A containing full rows and continuous variable columns
    A_sub_int::SparseMatrixCSC{Float64,Int64} # Submatrix of A containing full rows and integer variable columns
    b_sub::Vector{Float64}      # Subvector of b containing full rows
    c_sub_cont::Vector{Float64} # Subvector of c for continuous variables
    c_sub_int::Vector{Float64}  # Subvector of c for integer variables
    b_sub_int::Vector{Float64}  # Slack vector that we operate on in conic subproblem

    # MIP data
    model_mip::JuMP.Model       # JuMP MIP (outer approximation) model
    x_int::Vector{JuMP.Variable} # JuMP (sub)vector of integer variables
    x_cont::Vector{JuMP.Variable} # JuMP (sub)vector of continuous variables

    # SOC data
    num_soc::Int                # Number of SOCs
    summ_soc::Dict{Symbol,Real} # Data and infeasibilities
    dim_soc::Vector{Int}        # Dimensions
    rows_sub_soc::Vector{Vector{Int}} # Row indices in subproblem
    vars_soc::Vector{Vector{JuMP.Variable}} # Slack variables (newly added or detected)
    vars_dagg_soc::Vector{Vector{JuMP.Variable}} # Disaggregated variables

    # Miscellaneous for algorithms
    update_conicsub::Bool       # Indicates whether to use setbvec! to update an existing conic subproblem model
    model_conic::MathProgBase.AbstractConicModel # Conic subproblem model: persists when the conic solver implements MathProgBase.setbvec!
    isnew_feas::Bool            # Indicator for incumbent/best feasible solution not yet added by MIP-solver-driven heuristic callback

    # Solve information
    mip_obj::Float64            # Latest MIP (outer approx) objective value
    best_obj::Float64           # Best feasible objective value
    best_int::Vector{Float64}   # Best feasible integer solution
    best_cont::Vector{Float64}  # Best feasible continuous solution
    best_slck::Vector{Float64}  # Best feasible slack vector (for calculating MIP solution)
    gap_rel_opt::Float64        # Relative optimality gap = |mip_obj - best_obj|/|best_obj|
    final_soln::Vector{Float64} # Final solution on original variables
    solve_time::Float64         # Time between starting loadproblem and ending optimize (seconds)

    # Current Pajarito status
    status::Symbol

    # Model constructor
    function PajaritoConicModel(log_level, timeout, rel_gap, mip_solver_drives, mip_solver, mip_subopt_solver, mip_subopt_count, round_mip_sols, pass_mip_sols, cont_solver, solve_relax, dualize_relax, dualize_sub, soc_disagg, soc_in_mip, sdp_eig, sdp_soc, init_soc_one, init_soc_inf, init_exp, init_sdp_lin, init_sdp_soc, viol_cuts_only, proj_dual_infeas, proj_dual_feas, scale_dual_cuts, prim_cuts_only, prim_cuts_always, prim_cuts_assist, prim_viol_cuts_only, prim_max_viol_only, prim_soc_disagg, prim_sdp_eig, tol_zero, tol_prim_infeas, tol_sdp_eigvec, tol_sdp_eigval)
        # Errors
        if !mip_solver_drives
            error("This branch of Pajarito is only for the MSD algorithm\n")
        end
        if !isa(mip_solver, CPLEX.CplexSolver)
            error("This branch of Pajarito requires CplexSolver as the MIP solver\n")
        end
        if soc_in_mip
            error("This branch of Pajarito cannot do SOC in MIP\n")
        end
        if !prim_cuts_assist
            error("This branch of Pajarito requires primal cuts assist\n")
        end
        if prim_cuts_only || prim_cuts_always
            error("This branch of Pajarito cannot do primal cuts only or always\n")
        end

        # Warnings
        if log_level > 1
            if !solve_relax
                warn("Not solving the conic continuous relaxation problem; Pajarito may return status :MIPFailure if the outer approximation MIP is unbounded\n")
            end
            warn("For the MIP-solver-driven algorithm, optimality tolerance must be specified as MIP solver option, not Pajarito option\n")
        end

        # Initialize model
        m = new()

        m.log_level = log_level
        m.mip_solver_drives = mip_solver_drives
        m.solve_relax = solve_relax
        m.dualize_relax = dualize_relax
        m.dualize_sub = dualize_sub
        m.pass_mip_sols = pass_mip_sols
        m.round_mip_sols = round_mip_sols
        m.mip_subopt_count = mip_subopt_count
        m.mip_subopt_solver = mip_subopt_solver
        m.soc_in_mip = soc_in_mip
        m.soc_disagg = soc_disagg
        m.init_soc_one = init_soc_one
        m.init_soc_inf = init_soc_inf
        m.init_exp = init_exp
        m.init_sdp_lin = init_sdp_lin
        m.init_sdp_soc = init_sdp_soc
        m.proj_dual_infeas = proj_dual_infeas
        m.proj_dual_feas = proj_dual_feas
        m.viol_cuts_only = viol_cuts_only
        m.scale_dual_cuts = scale_dual_cuts
        m.mip_solver = mip_solver
        m.cont_solver = cont_solver
        m.timeout = timeout
        m.rel_gap = rel_gap
        m.tol_zero = tol_zero
        m.prim_cuts_only = prim_cuts_only
        m.prim_cuts_always = prim_cuts_always
        m.prim_cuts_assist = prim_cuts_assist
        m.prim_viol_cuts_only = prim_viol_cuts_only
        m.prim_max_viol_only = prim_max_viol_only
        m.prim_soc_disagg = prim_soc_disagg
        m.prim_sdp_eig = prim_sdp_eig
        m.tol_prim_infeas = tol_prim_infeas
        m.sdp_eig = sdp_eig
        m.sdp_soc = sdp_soc
        m.tol_sdp_eigvec = tol_sdp_eigvec
        m.tol_sdp_eigval = tol_sdp_eigval

        m.var_types = Symbol[]
        # m.var_start = Float64[]
        m.num_var_orig = 0
        m.num_con_orig = 0

        m.update_conicsub = false
        m.isnew_feas = false

        m.best_obj = Inf
        m.mip_obj = -Inf
        m.gap_rel_opt = NaN
        m.best_int = Float64[]
        m.best_cont = Float64[]
        m.final_soln = Float64[]
        m.solve_time = 0.

        m.status = :NotLoaded

        return m
    end
end

# Used a lot for scaling PSD cone elements (converting between smat and svec)
const sqrt2 = sqrt(2)
const sqrt2inv = 1/sqrt2


#=========================================================
 MathProgBase functions
=========================================================#

# Verify initial conic data and convert appropriate types and store in Pajarito model
function MathProgBase.loadproblem!(m::PajaritoConicModel, c, A, b, cone_con, cone_var)
    # Verify consistency of conic data
    verify_data(c, A, b, cone_con, cone_var)

    # Verify cone compatibility with solver (if solver is not defaultConicsolver: an MPB issue)
    if m.cont_solver != MathProgBase.defaultConicsolver
        # Get cones supported by conic solver
        conic_spec = MathProgBase.supportedcones(m.cont_solver)

        # Pajarito converts rotated SOCs to standard SOCs
        if :SOC in conic_spec
            push!(conic_spec, :SOCRotated)
        end

        # Error if a cone in data is not supported
        for (spec, _) in vcat(cone_con, cone_var)
            if !(spec in conic_spec)
                error("Cones $spec are not supported by the specified conic solver\n")
            end
        end
    end

    # Save original data
    m.num_con_orig = length(b)
    m.num_var_orig = length(c)
    m.c_orig = c
    m.A_orig = A
    m.b_orig = b
    m.cone_con_orig = cone_con
    m.cone_var_orig = cone_var

    m.final_soln = fill(NaN, m.num_var_orig)
    m.status = :Loaded
end

# Store warm-start vector on original variables in Pajarito model
function MathProgBase.setwarmstart!(m::PajaritoConicModel, var_start::Vector{Real})
    error("Warm-starts are not currently implemented in Pajarito (submit an issue)\n")
    # # Check if vector can be loaded
    # if m.status != :Loaded
    #     error("Must specify warm start right after loading problem\n")
    # end
    # if length(var_start) != m.num_var_orig
    #     error("Warm start vector length ($(length(var_start))) does not match number of variables ($(m.num_var_orig))\n")
    # end
    #
    # m.var_start = var_start
end

# Store variable type vector on original variables in Pajarito model
function MathProgBase.setvartype!(m::PajaritoConicModel, var_types::Vector{Symbol})
    if m.status != :Loaded
        error("Must specify variable types right after loading problem\n")
    end
    if length(var_types) != m.num_var_orig
        error("Variable types vector length ($(length(var_types))) does not match number of variables ($(m.num_var_orig))\n")
    end
    if any((var_type -> (var_type != :Bin) && (var_type != :Int) && (var_type != :Cont)), var_types)
        error("Some variable types are not in :Bin, :Int, :Cont\n")
    end
    if !any((var_type -> (var_type == :Bin) || (var_type == :Int)), var_types)
        error("No variables are in :Bin, :Int; use conic solver directly if problem is continuous\n")
    end

    m.var_types = var_types
end

# Solve, given the initial conic model data and the variable types vector and possibly a warm-start vector
function MathProgBase.optimize!(m::PajaritoConicModel)
    if m.status != :Loaded
        error("Must call optimize! function after loading conic data and setting variable types\n")
    end
    if isempty(m.var_types)
        error("Variable types were not specified; must call setvartype! function\n")
    end

    logs = create_logs()
    logs[:total] = time()

    # Transform data
    if m.log_level > 1
        @printf "\nTransforming original data..."
    end
    tic()
    (c_new, A_new, b_new, cone_con_new, cone_var_new, keep_cols, var_types_new) = transform_data(m.c_orig, m.A_orig, m.b_orig, m.cone_con_orig, m.cone_var_orig, m.var_types, m.solve_relax)
    logs[:data_trans] += toq()
    if m.log_level > 1
        @printf "...Done %8.2fs\n" logs[:data_trans]
    end

    # Create conic subproblem data
    if m.log_level > 1
        @printf "\nCreating conic model data..."
    end
    tic()
    (map_rows_sub, cols_cont, cols_int) = create_conicsub_data!(m, c_new, A_new, b_new, cone_con_new, cone_var_new, var_types_new)
    logs[:data_conic] += toq()
    if m.log_level > 1
        @printf "...Done %8.2fs\n" logs[:data_conic]
    end

    # Create MIP model
    if m.log_level > 1
        @printf "\nCreating MIP model..."
    end
    tic()
    (rows_relax_soc) = create_mip_data!(m, c_new, A_new, b_new, cone_con_new, cone_var_new, var_types_new, map_rows_sub, cols_cont, cols_int)
    logs[:data_mip] += toq()
    if m.log_level > 1
        @printf "...Done %8.2fs\n" logs[:data_mip]
    end

    print_cones(m)

    if m.solve_relax
        # Solve relaxed conic problem, proceed with algorithm if optimal or suboptimal, else finish
        if m.log_level > 0
            @printf "\nSolving conic relaxation..."
        end
        tic()
        if m.dualize_relax
            solver_relax = ConicDualWrapper(conicsolver=m.cont_solver)
        else
            solver_relax = m.cont_solver
        end
        model_relax = MathProgBase.ConicModel(solver_relax)
        MathProgBase.loadproblem!(model_relax, c_new, A_new, b_new, cone_con_new, cone_var_new)
        MathProgBase.optimize!(model_relax)
        logs[:relax_solve] += toq()
        if m.log_level > 0
            @printf "...Done %8.2fs\n" logs[:relax_solve]
        end

        status_relax = MathProgBase.status(model_relax)
        if status_relax == :Infeasible
            warn("Initial conic relaxation status was $status_relax: terminating Pajarito\n")
            m.status = :Infeasible
        elseif status_relax == :Unbounded
            warn("Initial conic relaxation status was $status_relax: terminating Pajarito\n")
            m.status = :UnboundedRelaxation
        elseif (status_relax != :Optimal) && (status_relax != :Suboptimal)
            warn("Apparent conic solver failure with status $status_relax\n")
        else
            obj_relax = MathProgBase.getobjval(model_relax)
            if m.log_level >= 1
                @printf " - Relaxation status    = %14s\n" status_relax
                @printf " - Relaxation objective = %14.6f\n" obj_relax
            end

            # Add initial dual cuts to MIP model
            dual_conic = MathProgBase.getdual(model_relax)

            for n in 1:m.num_soc
                vars = m.vars_soc[n]
                vars_dagg = m.vars_dagg_soc[n]
                dual = dual_conic[rows_relax_soc[n]]
                dim = length(dual)

                # Optionally rescale dual cut
                if m.scale_dual_cuts
                    # Feasible dual: rescale cut by number of cones / absval of full conic objective
                    scale!(dual, m.num_soc / (abs(obj_relax) + 1e-5))
                end

                # Sanitize and discard if all values are small, project dual, discard cut if epigraph variable is tiny
                keep = false
                for j in 2:dim
                    if abs(dual[j]) < m.tol_zero
                        dual[j] = 0.
                    else
                        keep = true
                    end
                end
                dual[1] = vecnorm(dual[j] for j in 2:dim)
                if !keep || (dual[1] <= m.tol_zero)
                    continue
                end

                add_full = false
                # Add disaggregated dual cuts
                for j in 2:dim
                    if dual[j] == 0.
                        # Zero cut
                        continue
                    elseif (dim - 1) * dual[j]^2 / (2. * dual[1]) < m.tol_zero
                        # Coefficient is too small
                        add_full = true
                        continue
                    elseif (dual[j] / dual[1])^2 < 1e-5
                        # Cut is poorly conditioned, add it but also add full cut
                        add_full = true
                    end

                    # Add disaggregated cut
                    @constraint(m.model_mip, (dim - 1) * (dual[j]^2 / (2. * dual[1]) * vars[1] + dual[1] * vars_dagg[j-1] + dual[j] * vars[j]) >= 0.)
                end

                # Add full cut if any cuts were poorly conditioned
                if add_full
                    @constraint(m.model_mip, vecdot(dual, vars) >= 0.)
                end
            end
        end

        # Free the conic model
        if applicable(MathProgBase.freemodel!, model_relax)
            MathProgBase.freemodel!(model_relax)
        end
    end

    if (m.status != :Infeasible) && (m.status != :UnboundedRelaxation)
        if m.log_level > 1
            @printf "\nCreating conic subproblem model..."
        end
        if m.dualize_sub
            solver_conicsub = ConicDualWrapper(conicsolver=m.cont_solver)
        else
            solver_conicsub = m.cont_solver
        end
        m.model_conic = MathProgBase.ConicModel(solver_conicsub)
        if method_exists(MathProgBase.setbvec!, (typeof(m.model_conic), Vector{Float64}))
            # Can use setbvec! on the conic subproblem model: load it
            m.update_conicsub = true
            MathProgBase.loadproblem!(m.model_conic, m.c_sub_cont, m.A_sub_cont, m.b_sub_int, m.cone_con_sub, m.cone_var_sub)
        end
        if m.log_level > 1
            @printf "...Done\n"
        end

        # Initialize and begin iterative or MIP-solver-driven algorithm
        m.best_slck = zeros(length(m.b_sub))
        if m.log_level > 0
            @printf "\nStarting MIP-solver-driven outer approximation algorithm\n"
        end

        logs[:oa_alg] = time()
        solve_mip_driven!(m, logs)
        logs[:oa_alg] = time() - logs[:oa_alg]

        if m.best_obj < Inf
            # Have a best feasible solution, update final solution on original variables
            soln_new = zeros(length(c_new))
            soln_new[cols_int] = m.best_int
            soln_new[cols_cont] = m.best_cont
            m.final_soln = zeros(m.num_var_orig)
            m.final_soln[keep_cols] = soln_new
        end
    end

    # Finish timer and print summary
    logs[:total] = time() - logs[:total]
    m.solve_time = logs[:total]
    print_finish(m, logs)
end

MathProgBase.numconstr(m::PajaritoConicModel) = m.num_con_orig

MathProgBase.numvar(m::PajaritoConicModel) = m.num_var_orig

MathProgBase.status(m::PajaritoConicModel) = m.status

MathProgBase.getsolvetime(m::PajaritoConicModel) = m.solve_time

MathProgBase.getobjval(m::PajaritoConicModel) = m.best_obj

MathProgBase.getobjbound(m::PajaritoConicModel) = m.mip_obj

MathProgBase.getsolution(m::PajaritoConicModel) = m.final_soln


#=========================================================
 Data functions
=========================================================#

# Verify consistency of conic data
function verify_data(c, A, b, cone_con, cone_var)
    # Check dimensions of conic problem
    num_con_orig = length(b)
    num_var_orig = length(c)
    if size(A) != (num_con_orig, num_var_orig)
        error("Dimensions of matrix A $(size(A)) do not match lengths of vector b ($(length(b))) and c ($(length(c)))\n")
    end
    if isempty(cone_con) || isempty(cone_var)
        error("Variable or constraint cones are missing\n")
    end

    # Check constraint cones
    inds_con = zeros(Int, num_con_orig)
    for (spec, inds) in cone_con
        if spec == :Free
            error("A cone $spec is in the constraint cones\n")
        end

        if any(inds .> num_con_orig)
            error("Some indices in a constraint cone do not correspond to indices of vector b\n")
        end

        inds_con[inds] += 1
    end
    if any(inds_con .== 0)
        error("Some indices in vector b do not correspond to indices of a constraint cone\n")
    end
    if any(inds_con .> 1)
        error("Some indices in vector b appear in multiple constraint cones\n")
    end

    # Check variable cones
    inds_var = zeros(Int, num_var_orig)
    for (spec, inds) in cone_var
        if any(inds .> num_var_orig)
            error("Some indices in a variable cone do not correspond to indices of vector c\n")
        end

        inds_var[inds] += 1
    end
    if any(inds_var .== 0)
        error("Some indices in vector c do not correspond to indices of a variable cone\n")
    end
    if any(inds_var .> 1)
        error("Some indices in vector c appear in multiple variable cones\n")
    end

    # Verify consistency of cone indices
    for (spec, inds) in vcat(cone_con, cone_var)
        if isempty(inds)
            error("A cone $spec has no associated indices\n")
        end
        if spec == :SOC && (length(inds) < 2)
            error("A cone $spec has fewer than 2 indices ($(length(inds)))\n")
        elseif spec == :SOCRotated && (length(inds) < 3)
            error("A cone $spec has fewer than 3 indices ($(length(inds)))\n")
        end
    end
end

# Transform/preprocess data
function transform_data(c_orig, A_orig, b_orig, cone_con_orig, cone_var_orig, var_types, solve_relax)
    A = sparse(A_orig)
    dropzeros!(A)
    (A_I, A_J, A_V) = findnz(A)

    num_con_new = length(b_orig)
    b_new = b_orig
    cone_con_new = Tuple{Symbol,Vector{Int}}[(spec, collect(inds)) for (spec, inds) in cone_con_orig]

    num_var_new = 0
    cone_var_new = Tuple{Symbol,Vector{Int}}[]

    old_new_col = zeros(Int, length(c_orig))
    bin_vars_new = Int[]

    vars_nonneg = Int[]
    vars_nonpos = Int[]
    vars_free = Int[]
    for (spec, cols) in cone_var_orig
        # Ignore zero variable cones
        if spec != :Zero
            vars_nonneg = Int[]
            vars_nonpos = Int[]
            vars_free = Int[]

            for j in cols
                if var_types[j] == :Bin
                    # Put binary vars in NonNeg var cone, unless the original var cone was NonPos in which case the binary vars are fixed at zero
                    if spec != :NonPos
                        num_var_new += 1
                        old_new_col[j] = num_var_new
                        push!(vars_nonneg, j)
                        push!(bin_vars_new, j)
                    end
                else
                    # Put non-binary vars in NonNeg or NonPos or Free var cone
                    num_var_new += 1
                    old_new_col[j] = num_var_new
                    if spec == :NonNeg
                        push!(vars_nonneg, j)
                    elseif spec == :NonPos
                        push!(vars_nonpos, j)
                    else
                        push!(vars_free, j)
                    end
                end
            end

            if !isempty(vars_nonneg)
                push!(cone_var_new, (:NonNeg, old_new_col[vars_nonneg]))
            end
            if !isempty(vars_nonpos)
                push!(cone_var_new, (:NonPos, old_new_col[vars_nonpos]))
            end
            if !isempty(vars_free)
                push!(cone_var_new, (:Free, old_new_col[vars_free]))
            end

            if (spec != :Free) && (spec != :NonNeg) && (spec != :NonPos)
                # Convert nonlinear var cone to constraint cone
                push!(cone_con_new, (spec, collect((num_con_new + 1):(num_con_new + length(cols)))))
                for j in cols
                    num_con_new += 1
                    push!(A_I, num_con_new)
                    push!(A_J, j)
                    push!(A_V, -1.)
                    push!(b_new, 0.)
                end
            end
        end
    end

    A = sparse(A_I, A_J, A_V, num_con_new, length(c_orig))
    keep_cols = find(old_new_col)
    c_new = c_orig[keep_cols]
    A = A[:, keep_cols]
    var_types_new = var_types[keep_cols]

    # Convert SOCRotated cones to SOC cones (MathProgBase definitions)
    # (y,z,x) in RSOC <=> (y+z,-y+z,sqrt2*x) in SOC, y >= 0, z >= 0
    socr_rows = Vector{Int}[]
    for n_cone in 1:length(cone_con_new)
        (spec, rows) = cone_con_new[n_cone]
        if spec == :SOCRotated
            cone_con_new[n_cone] = (:SOC, rows)
            push!(socr_rows, rows)
        end
    end

    (A_I, A_J, A_V) = findnz(A)
    row_to_nzind = map(_ -> Int[], 1:num_con_new)
    for (ind, i) in enumerate(A_I)
        push!(row_to_nzind[i], ind)
    end

    for rows in socr_rows
        inds_1 = row_to_nzind[rows[1]]
        inds_2 = row_to_nzind[rows[2]]

        # Add new constraint cones for y >= 0, z >= 0
        push!(cone_con_new, (:NonNeg, collect((num_con_new + 1):(num_con_new + 2))))

        append!(A_I, fill((num_con_new + 1), length(inds_1)))
        append!(A_J, A_J[inds_1])
        append!(A_V, A_V[inds_1])
        push!(b_new, b_new[rows[1]])

        append!(A_I, fill((num_con_new + 2), length(inds_2)))
        append!(A_J, A_J[inds_2])
        append!(A_V, A_V[inds_2])
        push!(b_new, b_new[rows[2]])

        num_con_new += 2

        # Use old constraint cone SOCRotated for (y+z,-y+z,sqrt2*x) in SOC
        append!(A_I, fill(rows[1], length(inds_2)))
        append!(A_J, A_J[inds_2])
        append!(A_V, A_V[inds_2])
        b_new[rows[1]] += b_new[rows[2]]

        append!(A_I, fill(rows[2], length(inds_1)))
        append!(A_J, A_J[inds_1])
        append!(A_V, -A_V[inds_1])
        b_new[rows[2]] -= b_new[rows[1]]

        for i in rows[3:end]
            for ind in row_to_nzind[i]
                A_V[ind] *= sqrt2
            end
        end
        b_new[rows[2:end]] .*= sqrt2
    end

    if solve_relax
        # Preprocess to tighten bounds on binary and integer variables in conic relaxation
        # Detect isolated row nonzeros with nonzero b
        row_slck_count = zeros(Int, num_con_new)
        for (ind, i) in enumerate(A_I)
            if (A_V[ind] != 0.) && (b_new[i] != 0.)
                if row_slck_count[i] == 0
                    row_slck_count[i] = ind
                elseif row_slck_count[i] > 0
                    row_slck_count[i] = -1
                end
            end
        end

        bin_set_upper = falses(length(bin_vars_new))
        j = 0
        type_j = :Cont
        bound_j = 0.0

        # For each bound-type constraint, tighten by rounding
        for (spec, rows) in cone_con_new
            if (spec != :NonNeg) && (spec != :NonPos)
                continue
            end

            for i in rows
                if row_slck_count[i] > 0
                    # Isolated variable x_j with b_i - a_ij*x_j in spec, b_i & a_ij nonzero
                    j = A_J[row_slck_count[i]]
                    type_j = var_types[keep_cols[j]]
                    bound_j = b_new[i] / A_V[row_slck_count[i]]

                    if (spec == :NonNeg) && (A_V[row_slck_count[i]] > 0) || (spec == :NonPos) && (A_V[row_slck_count[i]] < 0)
                        # Upper bound: b_i/a_ij >= x_j
                        if (type_j == :Bin) && (bound_j >= 1.)
                            # Tighten binary upper bound to 1
                            if spec == :NonNeg
                                # 1 >= x_j
                                b_new[i] = 1.
                                A_V[row_slck_count[i]] = 1.
                            else
                                # -1 <= -x_j
                                b_new[i] = -1.
                                A_V[row_slck_count[i]] = -1.
                            end

                            bin_set_upper[j] = true
                        elseif type_j != :Cont
                            # Tighten binary or integer upper bound by rounding down
                            # TODO this may cause either fixing or infeasibility: detect this and remove variable (at least for binary)
                            if spec == :NonNeg
                                # floor >= x_j
                                b_new[i] = floor(bound_j)
                                A_V[row_slck_count[i]] = 1.
                            else
                                # -floor <= -x_j
                                b_new[i] = -floor(bound_j)
                                A_V[row_slck_count[i]] = -1.
                            end

                            if type_j == :Bin
                                bin_set_upper[j] = true
                            end
                        end
                    else
                        # Lower bound: b_i/a_ij <= x_j
                        if type_j != :Cont
                            # Tighten binary or integer lower bound by rounding up
                            # TODO this may cause either fixing or infeasibility: detect this and remove variable (at least for binary)
                            if spec == :NonPos
                                # ceil <= x_j
                                b_new[i] = ceil(bound_j)
                                A_V[row_slck_count[i]] = 1.
                            else
                                # -ceil >= -x_j
                                b_new[i] = -ceil(bound_j)
                                A_V[row_slck_count[i]] = -1.
                            end
                        end
                    end
                end
            end
        end

        # For any binary variables without upper bound set, add 1 >= x_j to constraint cones
        num_con_prev = num_con_new
        for ind in 1:length(bin_vars_new)
            if !bin_set_upper[ind]
                num_con_new += 1
                push!(A_I, num_con_new)
                push!(A_J, bin_vars_new[ind])
                push!(A_V, 1.)
                push!(b_new, 1.)
            end
        end
        if num_con_new > num_con_prev
            push!(cone_con_new, (:NonNeg, collect((num_con_prev + 1):num_con_new)))
        end
    end

    A_new = sparse(A_I, A_J, A_V, num_con_new, num_var_new)
    dropzeros!(A_new)

    return (c_new, A_new, b_new, cone_con_new, cone_var_new, keep_cols, var_types_new)
end

# Create conic subproblem data
function create_conicsub_data!(m::PajaritoConicModel, c_new::Vector{Float64}, A_new::SparseMatrixCSC{Float64,Int64}, b_new::Vector{Float64}, cone_con_new::Vector{Tuple{Symbol,Vector{Int}}}, cone_var_new::Vector{Tuple{Symbol,Vector{Int}}}, var_types_new::Vector{Symbol})
    # Build new subproblem variable cones by removing integer variables
    cols_cont = Int[]
    cols_int = Int[]
    num_cont = 0
    cone_var_sub = Tuple{Symbol,Vector{Int}}[]

    for (spec, cols) in cone_var_new
        cols_cont_new = Int[]
        for j in cols
            if var_types_new[j] == :Cont
                push!(cols_cont, j)
                num_cont += 1
                push!(cols_cont_new, num_cont)
            else
                push!(cols_int, j)
            end
        end
        if !isempty(cols_cont_new)
            push!(cone_var_sub, (spec, cols_cont_new))
        end
    end

    # Determine "empty" rows with no nonzero coefficients on continuous variables
    (A_cont_I, _, A_cont_V) = findnz(A_new[:, cols_cont])
    num_con_new = size(A_new, 1)
    rows_nz = falses(num_con_new)
    for (i, v) in zip(A_cont_I, A_cont_V)
        if !rows_nz[i] && (v != 0)
            rows_nz[i] = true
        end
    end

    # Build new subproblem constraint cones by removing empty rows
    num_full = 0
    rows_full = Int[]
    cone_con_sub = Tuple{Symbol,Vector{Int}}[]
    map_rows_sub = Vector{Int}(num_con_new)

    for (spec, rows) in cone_con_new
        if (spec == :Zero) || (spec == :NonNeg) || (spec == :NonPos)
            rows_full_new = Int[]
            for i in rows
                if rows_nz[i]
                    push!(rows_full, i)
                    num_full += 1
                    push!(rows_full_new, num_full)
                end
            end
            if !isempty(rows_full_new)
                push!(cone_con_sub, (spec, rows_full_new))
            end
        else
            map_rows_sub[rows] = collect((num_full + 1):(num_full + length(rows)))
            push!(cone_con_sub, (spec, collect((num_full + 1):(num_full + length(rows)))))
            append!(rows_full, rows)
            num_full += length(rows)
        end
    end

    # Store conic data
    m.cone_var_sub = cone_var_sub
    m.cone_con_sub = cone_con_sub

    # Build new subproblem A, b, c data by removing empty rows and integer variables
    m.A_sub_cont = A_new[rows_full, cols_cont]
    m.A_sub_int = A_new[rows_full, cols_int]
    m.b_sub = b_new[rows_full]
    m.c_sub_cont = c_new[cols_cont]
    m.c_sub_int = c_new[cols_int]
    m.b_sub_int = zeros(length(rows_full))

    return (map_rows_sub, cols_cont, cols_int)
end

# Generate MIP model and maps relating conic model and MIP model variables
function create_mip_data!(m::PajaritoConicModel, c_new::Vector{Float64}, A_new::SparseMatrixCSC{Float64,Int64}, b_new::Vector{Float64}, cone_con_new::Vector{Tuple{Symbol,Vector{Int}}}, cone_var_new::Vector{Tuple{Symbol,Vector{Int}}}, var_types_new::Vector{Symbol}, map_rows_sub::Vector{Int}, cols_cont::Vector{Int}, cols_int::Vector{Int})
    # Initialize JuMP model for MIP outer approximation problem
    model_mip = JuMP.Model(solver=m.mip_solver)

    # Create variables and set types
    x_all = @variable(model_mip, [1:length(var_types_new)])
    for j in cols_int
        setcategory(x_all[j], var_types_new[j])
    end

    # Set objective function
    @objective(model_mip, :Min, dot(c_new, x_all))

    # Add variable cones to MIP
    for (spec, cols) in cone_var_new
        if spec == :NonNeg
            for j in cols
                setname(x_all[j], "v$(j)")
                setlowerbound(x_all[j], 0.)
            end
        elseif spec == :NonPos
            for j in cols
                setname(x_all[j], "v$(j)")
                setupperbound(x_all[j], 0.)
            end
        elseif spec == :Free
            for j in cols
                setname(x_all[j], "v$(j)")
            end
        elseif spec == :Zero
            error("Bug: Zero cones should have been removed by transform data function (submit an issue)\n")
        end
    end

    # Loop through nonlinear cones to count and summarize
    num_soc = 0
    summ_soc = Dict{Symbol,Real}(:max_dim => 0, :min_dim => 0)

    for (spec, rows) in cone_con_new
        if spec == :SOC
            num_soc += 1
            if summ_soc[:max_dim] < length(rows)
                summ_soc[:max_dim] = length(rows)
            end
            if (summ_soc[:min_dim] == 0) || (summ_soc[:min_dim] > length(rows))
                summ_soc[:min_dim] = length(rows)
            end
        end
    end

    # Allocate data for nonlinear cones
    rows_relax_soc = Vector{Vector{Int}}(num_soc)
    rows_sub_soc = Vector{Vector{Int}}(num_soc)
    dim_soc = Vector{Int}(num_soc)
    vars_soc = Vector{Vector{JuMP.Variable}}(num_soc)
    vars_dagg_soc = Vector{Vector{JuMP.Variable}}(num_soc)

    # Set up a SOC cone in the MIP
    function add_soc!(n_soc, len, rows, vars)
        dim_soc[n_soc] = len
        rows_relax_soc[n_soc] = rows
        rows_sub_soc[n_soc] = map_rows_sub[rows]
        vars_soc[n_soc] = vars
        vars_dagg_soc[n_soc] = Vector{JuMP.Variable}(0)

        # Set bounds
        setlowerbound(vars[1], 0.)

        # Set names
        for j in 1:len
            setname(vars[j], "s$(j)_soc$(n_soc)")
        end

        # Add disaggregated SOC variables
        # 2*d_j >= y_j^2/x
        vars_dagg = @variable(model_mip, [j in 1:(len - 1)], lowerbound=0.)
        vars_dagg_soc[n_soc] = vars_dagg

        # Add disaggregated SOC constraint
        # x >= sum(2*d_j)
        @constraint(model_mip, 2. * vars[1] >= 4. * sum(vars_dagg))

        # Set names
        for j in 1:(len - 1)
            setname(vars_dagg[j], "d$(j+1)_soc$(n_soc)")
        end

        # Add initial SOC linearizations
        if m.init_soc_one
            # Add initial L_1 SOC cuts
            # 2*d_j >= 2*|y_j|/sqrt(len - 1) - x/(len - 1)
            # for all j, implies x*sqrt(len - 1) >= sum(|y_j|)
            # linearize y_j^2/x at x = 1, y_j = 1/sqrt(len - 1) for all j
            for j in 2:len
              @constraint(model_mip, 2. * (len - 1) * vars_dagg[j-1] - 2. * sqrt(len - 1) * vars[j] + vars[1] >= 0)
              @constraint(model_mip, 2. * (len - 1) * vars_dagg[j-1] + 2. * sqrt(len - 1) * vars[j] + vars[1] >= 0)
            end
        end
        if m.init_soc_inf
            # Add initial L_inf SOC cuts
            # 2*d_j >= 2|y_j| - x
            # implies x >= |y_j|, for all j
            # linearize y_j^2/x at x = 1, y_j = 1 for each j (y_k = 0 for k != j)
            # equivalent to standard 3-dim rotated SOC linearizations x + d_j >= 2|y_j|
            for j in 2:len
              @constraint(model_mip, (len - 1) * (2. * vars_dagg[j-1] - 2. * vars[j] + vars[1]) >= 0)
              @constraint(model_mip, (len - 1) * (2. * vars_dagg[j-1] + 2. * vars[j] + vars[1]) >= 0)
            end
        end
    end

    n_soc = 0
    @expression(model_mip, lhs_expr, b_new - A_new * x_all)

    # Add constraint cones to MIP; if linear, add directly, else create slacks if necessary
    for (spec, rows) in cone_con_new
        if spec == :NonNeg
            @constraint(model_mip, lhs_expr[rows] .>= 0.)
        elseif spec == :NonPos
            @constraint(model_mip, lhs_expr[rows] .<= 0.)
        elseif spec == :Zero
            @constraint(model_mip, lhs_expr[rows] .== 0.)
        else
            # Set up nonlinear cone slacks and data
            vars = @variable(model_mip, [1:length(rows)])
            @constraint(model_mip, lhs_expr[rows] - vars .== 0.)

            # Set up MIP cones
            if spec == :SOC
                n_soc += 1
                add_soc!(n_soc, length(rows), rows, vars)
            else
                error("Cone type $spec not valid for this branch of Pajarito\n")
            end
        end
    end

    # Store MIP data
    m.model_mip = model_mip
    m.x_int = x_all[cols_int]
    m.x_cont = x_all[cols_cont]
    # @show model_mip

    m.num_soc = num_soc
    m.summ_soc = summ_soc
    m.dim_soc = dim_soc
    m.rows_sub_soc = rows_sub_soc
    m.vars_soc = vars_soc
    m.vars_dagg_soc = vars_dagg_soc

    return rows_relax_soc
end


#=========================================================
 Algorithm functions
=========================================================#

# Solve the MIP model using true MIP-solver-driven callback algorithm with incumbent callbacks
function solve_mip_driven!(m::PajaritoConicModel, logs::Dict{Symbol,Real})
    if isfinite(m.timeout) && applicable(MathProgBase.setparameters!, m.mip_solver)
        MathProgBase.setparameters!(m.mip_solver, TimeLimit=max(0., m.timeout - (time() - logs[:total])))
        setsolver(m.model_mip, m.mip_solver)
    end

    cache_cuts = Dict{Vector{Float64},Vector{JuMP.AffExpr}}()
    viol_cones = trues(m.num_soc)

    # Add lazy cuts callback to add dual and primal conic cuts
    function callback_lazy(cb)
        # println("doing lazy cb")
        # If any SOC variables are SOC infeasible, must continue
        fill!(viol_cones, false)
        maxviol = 0.
        maxviolcone = 0
        for n in 1:m.num_soc
            vars = m.vars_soc[n]
            viol = vecnorm(getvalue(vars[j]) for j in 2:length(vars)) - getvalue(vars[1])
            if viol > m.tol_prim_infeas
                viol_cones[n] = true
                if viol > maxviol
                    maxviol = viol
                    maxviolcone = n
                end
            end
        end
        if maxviolcone == 0
            return
        end

        # Get integer solution, round if option
        viol_cut = false
        soln_int = getvalue(m.x_int)
        if m.round_mip_sols
            soln_int = map!(round, soln_int)
        end

        # Add new dual cuts if new solution, else add existing dual cuts
        if !haskey(cache_cuts, soln_int)
            # New integer solution
            cuts = Vector{JuMP.AffExpr}()
            cache_cuts[copy(soln_int)] = cuts

            # Solve conic subproblem and save dual in dict (empty if conic failure)
            (status_conic, dual_conic, obj_conic) = solve_conicsub!(m, soln_int, logs)

            if m.scale_dual_cuts && (status_conic == :Infeasible)
                # Find rescaling factor for ray
                ray_value = vecdot(dual_conic, m.b_sub_int)  # sum(vecdot([m.rows_sub_soc[n]], m.b_sub_int[m.rows_sub_soc[n]]) for n in 1:num_soc)
                if ray_value > -m.tol_zero
                    error("Conic solver failure: b'y not sufficiently negative for infeasible ray y\n")
                end
            end

            if (status_conic == :Optimal) || (status_conic == :Infeasible)
                cuts = Vector{JuMP.AffExpr}(m.num_soc)

                for n in 1:m.num_soc
                    vars = m.vars_soc[n]
                    vars_dagg = m.vars_dagg_soc[n]
                    dual = dual_conic[m.rows_sub_soc[n]]
                    dim = length(dual)

                    # Optionally rescale dual cut
                    if m.scale_dual_cuts
                        if status_conic == :Infeasible
                            # Infeasible ray: rescale cut so that infeasible point will be cut off
                            scale!(dual, m.num_soc / ray_value)
                        else
                            # Feasible dual: rescale cut by number of cones / absval of full conic objective
                            scale!(dual, m.num_soc / (abs(obj_conic) + 1e-5))
                        end
                    end

                    # Sanitize and discard if all values are small, project dual, discard cut if epigraph variable is tiny
                    keep = false
                    for j in 2:dim
                        if abs(dual[j]) < m.tol_zero
                            dual[j] = 0.
                        else
                            keep = true
                        end
                    end
                    dual[1] = vecnorm(dual[j] for j in 2:dim)
                    if !keep || (dual[1] <= m.tol_zero)
                        cuts[n] = JuMP.AffExpr(0)
                        continue
                    end

                    add_full = false
                    # Add disaggregated dual cuts
                    for j in 2:dim
                        if dual[j] == 0.
                            # Zero cut
                            continue
                        elseif (dim - 1) * dual[j]^2 / (2. * dual[1]) < m.tol_zero
                            # Coefficient is too small
                            add_full = true
                            continue
                        elseif (dual[j] / dual[1])^2 < 1e-5
                            # Cut is poorly conditioned, add it but also add full cut
                            add_full = true
                        end

                        # Add disaggregated cut (optionally if violated)
                        @expression(m.model_mip, cut_expr, (dim - 1) * (dual[j]^2 / (2. * dual[1]) * vars[1] + dual[1] * vars_dagg[j-1] + dual[j] * vars[j]))
                        if -getvalue(cut_expr) > m.tol_prim_infeas
                            viol_cut = true
                            @lazyconstraint(cb, cut_expr >= 0.)
                        elseif !m.viol_cuts_only
                            @lazyconstraint(cb, cut_expr >= 0.)
                        end
                    end

                    @expression(m.model_mip, cut_expr, vecdot(dual, vars))
                    cuts[n] = cut_expr
                    # Add full cut if any cuts were poorly conditioned
                    if add_full
                        if -getvalue(cut_expr) > m.tol_prim_infeas
                            viol_cut = true
                            @lazyconstraint(cb, cut_expr >= 0.)
                        elseif !m.viol_cuts_only
                            @lazyconstraint(cb, cut_expr >= 0.)
                        end
                    end
                end
            end
        else
            # Repeat integer solution: get dual cuts if they exist
            logs[:n_repeat] += 1
            cuts = cache_cuts[soln_int]
            if !isempty(cuts)
                # Add infeasible full dual cut for each infeasible cone
                for n in 1:num_soc
                    if !viol_cones[n]
                        continue
                    end

                    cut_expr = cuts[n]
                    if -getvalue(cut_expr) > m.tol_prim_infeas
                        viol_cut = true
                        @lazyconstraint(cb, cut_expr >= 0.)
                    end
                end
            end
        end

        # Finish lazy callback if added a violated dual cut already
        if viol_cut
            return
        end

        # Add primal cuts on infeasible cones
        if m.prim_max_viol_only
            # Most violated cone only
            cut_cones = maxviolcone:maxviolcone
        else
            # All infeasible cones
            cut_cones = 1:m.num_soc
        end

        for n in cut_cones
            if !viol_cones[n]
                continue
            end

            vars = m.vars_soc[n]
            prim = getvalue(vars)
            dim = length(vars)

            # Remove near-zeros, discard if all values are small
            keep = false
            for j in 1:dim
                if abs(prim[j]) < m.tol_zero
                    prim[j] = 0.
                else
                    keep = true
                end
            end
            if !keep
                continue
            end

            xnorm = vecnorm(prim[j] for j in 2:dim)
            add_full = false

            # Add primal disagg cuts
            if m.prim_soc_disagg
                vars_dagg = m.vars_dagg_soc[n]
                for j in 2:dim
                    if prim[j] == 0.
                        # Zero cut
                        continue
                    elseif (dim - 1) * (prim[j] / xnorm)^2 / 2. < m.tol_zero
                        # Coefficient is too small
                        add_full = true
                        continue
                    elseif (prim[j] / xnorm)^2 < 1e-5
                        # Cut is poorly conditioned, add it but also add full cut
                        add_full = true
                    end

                    # Disagg cut
                    # 2*dj >= 2xj'/||x'||*xj - (xj'/||x'||)^2*y
                    @expression(m.model_mip, cut_expr, (dim - 1) * ((prim[j] / xnorm)^2 / 2. * vars[1] + vars_dagg[j-1] - prim[j] / xnorm * vars[j]))
                    if !m.prim_viol_cuts_only || (-getvalue(cut_expr) > m.tol_prim_infeas)
                        @lazyconstraint(cb, cut_expr >= 0.)
                        viol_cut = true
                    end
                end
            end

            # Add primal full cut
            if add_full || !m.prim_soc_disagg
                # Full primal cut
                # x'*x / ||x'|| <= y
                @expression(m.model_mip, cut_expr, vars[1] - sum(prim[j] / xnorm * vars[j] for j in 2:dim))
                if !m.prim_viol_cuts_only || (-getvalue(cut_expr) > m.tol_prim_infeas)
                    @lazyconstraint(cb, cut_expr >= 0.)
                    viol_cut = true
                end
            end
        end

        if !viol_cut
            warn("No dual cuts or primal cuts were added on an infeasible solution\n")
        end
    end
    addlazycallback(m.model_mip, callback_lazy)

    if m.pass_mip_sols
        # Add heuristic callback to give MIP solver feasible solutions from conic solves
        function callback_heur(cb)
            # If have a new best feasible solution since last heuristic solution added
            println("doing heuristic cb")
            if m.isnew_feas
                println("adding new sol")
                # Set MIP solution to the new best feasible solution
                set_best_soln!(m, cb, logs)
                addsolution(cb)
                m.isnew_feas = false
                m.best_obj = Inf
            end
        end
        addheuristiccallback(m.model_mip, callback_heur)
    end

    sol_incum = Set{Vector{Float64}}()

    # Add incumbent callback to tell MIP solver whether solutions are conic feasible incumbents or not
    function callback_incumbent(cb)
        push!(sol_incum, getvalue(m.x_cont))

        println("doing incumbent cb")
        # If any SOC variables are SOC infeasible, return false
        for vars in m.vars_soc
            prim_inf = vecnorm(getvalue(vars[j]) for j in 2:length(vars))^2 - getvalue(vars[1])^2
            @show prim_inf
            if prim_inf > m.tol_prim_infeas
                println("checked feas: rejecting")
                CPLEX.rejectincumbent(cb)
                return
            end
        end

        # No conic infeasibility: allow solution as new incumbent
        println("checked feas: accepting")
        CPLEX.acceptincumbent(cb)
    end
    CPLEX.addincumbentcallback(m.model_mip, callback_incumbent)

    # Start MIP solver
    logs[:mip_solve] = time()
    status_mip = solve(m.model_mip)#, suppress_warnings=true)
    logs[:mip_solve] = time() - logs[:mip_solve]

    if (status_mip == :Infeasible) || (status_mip == :InfeasibleOrUnbounded)
        m.status = :Infeasible
    elseif status_mip == :Unbounded
        # Shouldn't happen - initial conic relax solve should detect this
        warn("MIP solver returned status $status_mip, which could indicate that the initial dual cuts added were too weak\n")
        m.status = :MIPFailure
    elseif (status_mip == :UserLimit) || (status_mip == :Optimal) || (status_mip == :Suboptimal)
        m.best_int = getvalue(m.x_int)
        m.best_cont = getvalue(m.x_cont)

        if !in(m.best_cont, sol_incum)
            error("solution did not go thru incumbent cb\n")
        end

        m.best_obj = getobjectivevalue(m.model_mip)
        m.mip_obj = getobjbound(m.model_mip)
        m.gap_rel_opt = getobjgap(m.model_mip) #(m.best_obj - m.mip_obj) / (abs(m.best_obj) + 1e-5)

        m.status = status_mip
    else
        error("MIP solver returned status $status_mip, which Pajarito does not handle (please submit an issue)\n")
    end
end

# Solve conic subproblem given some solution to the integer variables, update incumbent
function solve_conicsub!(m::PajaritoConicModel, soln_int::Vector{Float64}, logs::Dict{Symbol,Real})
    # Calculate new b vector from integer solution and solve conic model
    m.b_sub_int = m.b_sub - m.A_sub_int*soln_int

    # Load/solve conic model
    tic()
    if m.update_conicsub
        # Reuse model already created by changing b vector
        MathProgBase.setbvec!(m.model_conic, m.b_sub_int)
    else
        # Load new model
        if m.dualize_sub
            solver_conicsub = ConicDualWrapper(conicsolver=m.cont_solver)
        else
            solver_conicsub = m.cont_solver
        end
        m.model_conic = MathProgBase.ConicModel(solver_conicsub)
        MathProgBase.loadproblem!(m.model_conic, m.c_sub_cont, m.A_sub_cont, m.b_sub_int, m.cone_con_sub, m.cone_var_sub)
    end
    # Mosek.writedata(m.model_conic.task, "mosekfail.task")         # For MOSEK debug only: dump task
    MathProgBase.optimize!(m.model_conic)
    logs[:conic_solve] += toq()
    logs[:n_conic] += 1

    status_conic = MathProgBase.status(m.model_conic)

    # Get dual vector
    if (status_conic == :Optimal) || (status_conic == :Infeasible)
        dual_conic = MathProgBase.getdual(m.model_conic)
    else
        dual_conic = Float64[]
    end

    # Check if have new feasible solution
    if status_conic == :Optimal
        logs[:n_feas] += 1

        # Calculate full objective value
        soln_cont = MathProgBase.getsolution(m.model_conic)
        obj_conic = dot(m.c_sub_int, soln_int) + dot(m.c_sub_cont, soln_cont)

        # Check if new full objective beats best incumbent
        if m.pass_mip_sols && (obj_conic < m.best_obj)
            # Save new incumbent info and indicate new solution for heuristic callback
            m.best_obj = obj_conic
            m.best_int = soln_int
            m.best_cont = soln_cont
            m.best_slck = m.b_sub_int - m.A_sub_cont * m.best_cont
            m.isnew_feas = true
        end
    else
        obj_conic = Inf
    end

    # Free the conic model if not saving it
    if !m.update_conicsub && applicable(MathProgBase.freemodel!, m.model_conic)
        MathProgBase.freemodel!(m.model_conic)
    end

    return (status_conic, dual_conic, obj_conic)
end

# Construct and warm-start MIP solution using best solution
function set_best_soln!(m::PajaritoConicModel, cb, logs::Dict{Symbol,Real})
    set_soln!(m, cb, m.x_int, m.best_int)
    set_soln!(m, cb, m.x_cont, m.best_cont)

    for n in 1:m.num_soc
        set_soln!(m, cb, m.vars_soc[n], m.best_slck, m.rows_sub_soc[n])
        set_dagg_soln!(m, cb, m.vars_dagg_soc[n], m.best_slck, m.rows_sub_soc[n])
    end
end

# Call setvalue or setsolutionvalue solution for a vector of variables and a solution vector and corresponding solution indices
function set_soln!(m::PajaritoConicModel, cb, vars::Vector{JuMP.Variable}, soln::Vector{Float64}, inds::Vector{Int})
    for (j, ind) in enumerate(inds)
        setsolutionvalue(cb, vars[j], soln[ind])
    end
end

# Call setvalue or setsolutionvalue solution for a vector of variables and a solution vector
function set_soln!(m::PajaritoConicModel, cb, vars::Vector{JuMP.Variable}, soln::Vector{Float64})
    for j in 1:length(vars)
        setsolutionvalue(cb, vars[j], soln[j])
    end
end

# Call setvalue or setsolutionvalue solution for a vector of SOC disaggregated variables and a solution vector and corresponding solution indices
function set_dagg_soln!(m::PajaritoConicModel, cb, vars_dagg::Vector{JuMP.Variable}, soln::Vector{Float64}, inds)
    if soln[inds[1]] == 0.
        for j in 2:length(inds)
            setsolutionvalue(cb, vars_dagg[j-1], 0.)
        end
    else
        for j in 2:length(inds)
            setsolutionvalue(cb, vars_dagg[j-1], (soln[inds[j]]^2 / (2. * soln[inds[1]])))
        end
    end
end


#=========================================================
 Logging and printing functions
=========================================================#

# Create dictionary of logs for timing and iteration counts
function create_logs()
    logs = Dict{Symbol,Real}()

    # Timers
    logs[:total] = 0.       # Performing total optimize algorithm
    logs[:data_trans] = 0.  # Transforming data
    logs[:data_conic] = 0.  # Generating conic data
    logs[:data_mip] = 0.    # Generating MIP data
    logs[:relax_solve] = 0. # Solving initial conic relaxation model
    logs[:oa_alg] = 0.      # Performing outer approximation algorithm
    logs[:mip_solve] = 0.   # Solving the MIP model
    logs[:conic_solve] = 0. # Solving conic subproblem model

    # Counters
    logs[:n_conic] = 0      # Number of conic subproblem solves
    logs[:n_feas] = 0       # Number of feasible solutions encountered
    logs[:n_repeat] = 0     # Number of times integer solution repeats

    return logs
end

# Print cone dimensions summary
function print_cones(m::PajaritoConicModel)
    if m.log_level <= 1
        return
    end

    @printf "\nCone types summary:"
    @printf "\n%-10s | %-8s | %-8s | %-8s\n" "Cone" "Count" "Min dim" "Max dim"
    if m.num_soc > 0
        @printf "%10s | %8d | %8d | %8d\n" "SOC" m.num_soc m.summ_soc[:min_dim] m.summ_soc[:max_dim]
    end
    flush(STDOUT)
end

# Print after finish
function print_finish(m::PajaritoConicModel, logs::Dict{Symbol,Real})
    if m.log_level < 0
        @printf "\n"
        flush(STDOUT)
        return
    end

    @printf "\nPajarito MICP solve summary:\n"
    @printf " - Total time (s)       = %14.2e\n" logs[:total]
    @printf " - Status               = %14s\n" m.status
    @printf " - Best feasible obj.   = %+14.6e\n" m.best_obj
    @printf " - Final OA obj. bound  = %+14.6e\n" m.mip_obj
    @printf " - Relative opt. gap    = %14.3e\n" m.gap_rel_opt

    if m.log_level == 0
        @printf "\n"
        flush(STDOUT)
        return
    end

    @printf " - Conic solve count    = %14d\n" logs[:n_conic]
    @printf " - Feas. solution count = %14d\n" logs[:n_feas]
    @printf " - Integer repeat count = %14d\n" logs[:n_repeat]
    @printf "\nTimers (s):\n"
    @printf " - Setup                = %14.2e\n" (logs[:total] - logs[:oa_alg])
    @printf " -- Transform data      = %14.2e\n" logs[:data_trans]
    @printf " -- Create conic data   = %14.2e\n" logs[:data_conic]
    @printf " -- Create MIP data     = %14.2e\n" logs[:data_mip]
    @printf " -- Load/solve relax    = %14.2e\n" logs[:relax_solve]
    @printf " - MIP-driven algorithm = %14.2e\n" logs[:oa_alg]
    @printf " -- Solve conic model   = %14.2e\n" logs[:conic_solve]
    @printf "\n"
    flush(STDOUT)
end

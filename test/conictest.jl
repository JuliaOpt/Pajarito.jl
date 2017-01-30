#  Copyright 2016, Los Alamos National Laboratory, LANS LLC.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

function runconictests(mip_solver_drives, mip_solver, conic_solver, log)
    algorithm = mip_solver_drives ? "MIP-driven" : "Iterative"

    facts("Infeasible conic problem") do
        context("With $algorithm, $(typeof(mip_solver)) and $(typeof(conic_solver))") do
            x = Convex.Variable(1,:Int)

            problem = Convex.maximize(3x,
                                x >= 4,
                                x^2 <= 9)

            Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=conic_solver, log_level=log))

            @fact problem.status --> :Infeasible
        end
    end

    facts("Univariate maximization problem") do
        context("With $algorithm, $(typeof(mip_solver)) and $(typeof(conic_solver))") do
            x = Convex.Variable(1,:Int)

            problem = Convex.maximize(3x,
                                x <= 10,
                                x^2 <= 9)

            Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=conic_solver, log_level=log))

            @fact problem.optval --> roughly(9.0, TOL)
            @fact problem.status --> :Optimal
        end
    end

    facts("Variable not in zero cone problem") do
        context("With $algorithm, $(typeof(mip_solver)) and $(typeof(conic_solver))") do
            # max  y + z
            # st   x == 1
            #     (x,y,z) in SOC
            #      x in {0,1}
            m = MathProgBase.ConicModel(PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=conic_solver, log_level=log))
            MathProgBase.loadproblem!(m,
            [ 0.0, -1.0, -1.0],
            [ 1.0  0.0  0.0;
             -1.0  0.0  0.0;
              0.0 -1.0  0.0;
              0.0  0.0 -1.0],
            [ 1.0, 0.0, 0.0, 0.0],
            Any[(:Zero,1:1),(:SOC,2:4)],
            Any[(:Free,[1,2,3])])
            MathProgBase.setvartype!(m, [:Int,:Cont,:Cont])

            MathProgBase.optimize!(m)
            @fact MathProgBase.status(m) --> :Optimal
            @fact MathProgBase.getobjval(m) --> roughly(-sqrt(2.0), TOL)
            vals = MathProgBase.getsolution(m)
            @fact vals[1] --> roughly(1, TOL)
            @fact vals[2] --> roughly(1.0/sqrt(2.0), TOL)
            @fact vals[3] --> roughly(1.0/sqrt(2.0), TOL)
       end
    end

    facts("Variable in zero cone problem") do
        context("With $algorithm, $(typeof(mip_solver)) and $(typeof(conic_solver))") do
            # Same as "Variable not in zero cone problem" but with variables 2 and 4 added and in zero cones
            m = MathProgBase.ConicModel(PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=conic_solver, log_level=log))
            MathProgBase.loadproblem!(m,
            [ 0.0, 0.0, -1.0, 1.0, -1.0],
            [ 1.0  1.0  0.0  0.0  0.0;
             -1.0  0.0  0.0 -0.5  0.0;
              0.0  2.0 -1.0  0.0  0.0;
              0.0  0.0  0.0 0.5  -1.0],
            [ 1.0, 0.0, 0.0, 0.0],
            Any[(:Zero,1:1),(:SOC,2:4)],
            Any[(:Free,[1,3,5]),(:Zero,[2,4])])
            MathProgBase.setvartype!(m, [:Int,:Int,:Cont,:Cont,:Cont])

            MathProgBase.optimize!(m)
            @fact MathProgBase.status(m) --> :Optimal
            @fact MathProgBase.getobjval(m) --> roughly(-sqrt(2.0), TOL)
            vals = MathProgBase.getsolution(m)
            @fact vals[1] --> roughly(1, TOL)
            @fact vals[2] --> roughly(0, TOL)
            @fact vals[3] --> roughly(1.0/sqrt(2.0), TOL)
            @fact vals[4] --> roughly(0.0, TOL)
            @fact vals[5] --> roughly(1.0/sqrt(2.0), TOL)
       end
    end

    facts("Rotated SOC problem") do
        context("With $algorithm, $(typeof(mip_solver)) and $(typeof(conic_solver))") do
            problem = MathProgBase.ConicModel(PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=conic_solver, log_level=log))

            c = [-3.0, 0.0, 0.0, 0.0]
            A = zeros(4,4)
            A[1,1] = 1.0
            A[2,2] = 1.0
            A[3,3] = 1.0
            A[4,1] = 1.0
            A[4,4] = -1.0
            b = [10.0, 1.5, 3.0, 0.0]

            constr_cones = Any[(:NonNeg,[1,2,3]),(:Zero,[4])]
            var_cones = Any[(:SOCRotated,[2,3,1]),(:Free,[4])]
            vartypes = [:Cont, :Cont, :Cont, :Int]

            MathProgBase.loadproblem!(problem, c, A, b, constr_cones, var_cones)
            MathProgBase.setvartype!(problem, vartypes)
            MathProgBase.optimize!(problem)

            @fact MathProgBase.getobjval(problem) --> roughly(-9.0, TOL)
        end
    end
end

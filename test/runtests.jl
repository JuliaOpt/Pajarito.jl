#  Copyright 2016, Los Alamos National Laboratory, LANS LLC.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

using Pajarito
using CPLEX
using Mosek, ECOS
using JuMP
import Convex
using FactCheck

include("conictest.jl")
include("sdptest.jl")

# Set fact check tolerance
TOL = 1e-3

# Option to print with log_level
log = 3

# Run tests
mip = CPLEX.CplexSolver(
    CPX_PARAM_MIPDISPLAY=3,
    CPX_PARAM_EPGAP=1e-5,
    CPX_PARAM_EPRHS=1e-8,
    CPX_PARAM_EPINT=1e-8,
    CPX_PARAM_THREADS=1,
    CPX_PARAM_TILIM=60.0,
    CPX_PARAM_REDUCE=1,
    CPX_PARAM_MIPCBREDLP=0
    )
runconictests(true, mip, ECOS.ECOSSolver(verbose=false), log)
# runsdptests(true, mip, Mosek.MosekSolver(LOG=0), log)

FactCheck.exitstatus()

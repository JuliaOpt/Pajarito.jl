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
log = 0

# Run tests
mip = CPLEX.CplexSolver()
runconictests(true, mip, ECOS.ECOSSolver(verbose=false), log)
# runsdptests(true, mip, Mosek.MosekSolver(LOG=0), log)

FactCheck.exitstatus()

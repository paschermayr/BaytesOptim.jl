############################################################################################
# Import External PackagesJK
using Test

using Random: Random, AbstractRNG, seed!
using Statistics
using Bijectors
using SimpleUnPack, ArgCheck

############################################################################################
# Import Baytes Packages
using ModelWrappers
using BaytesOptim

############################################################################################
# Include Files
include("TestHelper.jl")

############################################################################################
# Run Tests
@testset "All tests" begin
    include("test-optim.jl")
end

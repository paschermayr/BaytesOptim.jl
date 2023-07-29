############################################################################################
"""
$(TYPEDEF)

Stores information used throughout optimization algorithms.

# Fields
$(TYPEDFIELDS)
"""
struct OptimTune{
    T<:Tagged, 
    K,
    B<:BaytesCore.UpdateBool,
} <: AbstractTune
    "Tagged Parameter for Optimization routine"
    tagged::T
    "Tuning arguments for individual Optimizer"
    kernel::K
    "Boolean if generated quantities should be generated while sampling"
    generated::B
    "Current iteration number"
    iter::Iterator
    function OptimTune(
        tagged::T, 
        kernel::K,
        generated::B
    ) where {T<:Tagged, K, B}
        #!NOTE: Start with 0, so first proposal step will update iter to 1
        iter = Iterator(0)
        return new{T, K, B}(
            tagged, kernel, generated, iter
        )
    end
end

############################################################################################
"""
$(SIGNATURES)
Update Optim tuning fields at current iteration.

# Examples
```julia
```

"""
function update!(
    tune::OptimTune, result::S
) where {S<:ℓObjectiveResult}
    ##  Update Current iteration counter
    update!(tune.iter)
    ## Update Kernel Tuning container
    update!(tune.kernel, result)
    ## Pack container
    return nothing
end

################################################################################
export OptimTune
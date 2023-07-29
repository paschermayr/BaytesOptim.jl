############################################################################################
struct OptimDiagnostics{P, K<:OptimKernelDiagnostics, G, A} <: AbstractDiagnostics
    "Diagnostics used for all Baytes kernels"
    base::BaytesCore.BaseDiagnostics{P}
    "Kernel specific diagnostics."
    kernel::K
    "Generated quantities specified for objective"
    generated::G
    "Generated quantities specified for algorithm"
    generated_algorithm::A
    function OptimDiagnostics(
        base::BaytesCore.BaseDiagnostics{P},
        kerneldiagnostics::K,
        generated::G,
        generated_algorithm::A
    ) where {P, K<:OptimKernelDiagnostics, G, A}
        return new{P,K,G,A}(
            base, kerneldiagnostics, generated, generated_algorithm
        )
    end
end

function generate_showvalues(diagnostics::D) where {D<:OptimDiagnostics}
    kernel = generate_showvalues(diagnostics.kernel)
    return function showvalues()
        return (:optimizer, "diagnostics"),
        (:iter, diagnostics.base.iter),
        (:logobjective, diagnostics.base.ℓobjective),
        (:Temperature, diagnostics.base.temperature),
        (:generated, diagnostics.generated),
        (:generated_algorithm, diagnostics.generated_algorithm),
        kernel()...
    end
end

############################################################################################
export OptimDiagnostics, generate_showvalues
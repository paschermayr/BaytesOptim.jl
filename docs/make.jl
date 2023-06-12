using BaytesOptim
using Documenter

DocMeta.setdocmeta!(BaytesOptim, :DocTestSetup, :(using BaytesOptim); recursive=true)

makedocs(;
    modules=[BaytesOptim],
    authors="Patrick Aschermayr <p.aschermayr@gmail.com>",
    repo="https://github.com/paschermayr/BaytesOptim.jl/blob/{commit}{path}#{line}",
    sitename="BaytesOptim.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://paschermayr.github.io/BaytesOptim.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/paschermayr/BaytesOptim.jl",
    devbranch="master",
)

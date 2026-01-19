using Documenter
using IFTDuals

makedocs(
    sitename = "IFTDuals.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
    modules = [IFTDuals],
    pages = [
        "Home" => "index.md",
        "API" => "api.md",
    ]
)

deploydocs(
    repo = "github.com/Garren-H/IFTDuals.jl.git",
)

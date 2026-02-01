using Documenter
using IFTDuals

makedocs(
    sitename = "IFTDuals.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
    modules = [IFTDuals],
    checkdocs = :exports,
    pages = [
        "Home" => "index.md",
        "Examples" => "examples.md",
        "Advanced Usage" => "advanced.md",
        "API Reference" => "api.md",
    ]
)

deploydocs(
    repo = "github.com/Garren-H/IFTDuals.jl.git",
)

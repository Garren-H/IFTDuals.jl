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
        "Introduction" => "index.md",
        "User Documentation" => ["Limitations" => "UserGuide/limitations.md",
                                 "API Reference" => "UserGuide/api.md",
                                 "Advanced Usage" => "UserGuide/advanced.md"],
        "Developer Documentation" => "dev.md",
    ]
)

deploydocs(
    repo = "github.com/Garren-H/IFTDuals.jl.git",
)

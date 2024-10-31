using Literate

name = ARGS[1]
@info joinpath(@__DIR__, name)
cd(joinpath(@__DIR__, name))

Literate.markdown(
    "README.jl";
    credit = false,
    execute = true,
    flavor = Literate.CommonMarkFlavor(),
    mdstrings = true,
)

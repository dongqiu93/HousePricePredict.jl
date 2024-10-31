#=
# Dataset Explorations
=#

using Revise
using CSV
using DataFrames
using ColorSchemes
using CairoMakie

#=
## Import Train Data
=#

train = DataFrame(
    CSV.File("C:\\Users\\Dong\\Project\\HousePricesPredict.jl\\assets\\train_cleaned.csv"),
);

#=
## Analysis 
=#

#=
#### LotFrontage
=#
cats = unique(train.MSSubClass)

f = Figure()
Axis(f[1, 1])
for (i, cat) in enumerate(cats)
    density!(
        skipnan(train[train.MSSubClass.==cat, "LotFrontage"]);
        label = string(cat),
        offset = -i * 0.1,
    )
end
axislegend(position = :rb, colormap = :)
f


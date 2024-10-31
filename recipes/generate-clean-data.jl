using Revise
using HousePricesPredict
using CSV
using DataFrames
using Dates 
using StatsBase: mean

import HousePricesPredict:
    parse_or_nan,
    skipnan,
    access_price,
    quality_of_material,
    exposure_bsmt,
    rate_bsmt,
    home_functionality,
    garage_finished,
    paved_drive_score,
    fence_quality

train = DataFrame(CSV.File("assets\\train.csv"))

cat_vars = [
    :MSSubClass,
    :MSZoning,
    :LotShape,
    :LandContour,
    :Utilities,
    :LotConfig,
    :LandSlope,
    :Neighborhood,
    :Condition1,
    :Condition2,
    :BldgType,
    :HouseStyle,
    :RoofStyle,
    :RoofMatl,
    :Exterior1st,
    :Exterior2nd,
    :MasVnrType,
    :Foundation,
    :Heating,
    :Electrical,
    :GarageType,
    :MiscFeature,
    :SaleType,
    :SaleCondition,
]
bool_vars = [:CentralAir]
num_vars =
    string.([
        :Street,
        :Alley,
        :LotFrontage,
        :LotArea,
        :OverallQual,
        :OverallCond,
        :MasVnrArea,
        :ExterQual,
        :ExterCond,
        :BsmtQual,
        :BsmtCond,
        :BsmtExposure,
        :BsmtFinType1,
        :BsmtFinSF1,
        :BsmtFinType2,
        :BsmtFinSF2,
        :BsmtUnfSF,
        :TotalBsmtSF,
        :HeatingQC,
        "1stFlrSF",
        "2ndFlrSF",
        :LowQualFinSF,
        :GrLivArea,
        :BsmtFullBath,
        :BsmtHalfBath,
        :FullBath,
        :HalfBath,
        :BedroomAbvGr,
        :KitchenAbvGr,
        :KitchenQual,
        :TotRmsAbvGrd,
        :Functional,
        :Fireplaces,
        :FireplaceQu,
        :GarageFinish,
        :GarageCars,
        :GarageArea,
        :GarageQual,
        :GarageCond,
        :PavedDrive,
        :WoodDeckSF,
        :OpenPorchSF,
        :EnclosedPorch,
        "3SsnPorch",
        :ScreenPorch,
        :PoolArea,
        :PoolQC,
        :Fence,
        :MiscVal,
        :YearBuilt, 
        :YearRemodAdd, 
        :GarageYrBlt, 
        :MoSold, 
        :YrSold
    ])
time_vars = [:YearBuilt, :YearRemodAdd, :GarageYrBlt, :MoSold, :YrSold];

#=
## Standardize data types
=#
# Category & Boolean
transform!(
    train,
    cat_vars .=> (x -> string.(x)) .=> cat_vars,
    bool_vars .=> (x -> ifelse.(x .== "Y", 1, 0)) .=> "DE_CAT_" .* string.(bool_vars),
)

# Numeric 
train.DE_Price_Street = access_price.(train.Street)
train.DE_Price_Alley = access_price.(train.Alley)

parse_vars = [
    :LotFrontage,
    :LotArea,
    :OverallQual,
    :MasVnrArea,
    :BsmtFinSF1,
    :BsmtFinSF2,
    :BsmtUnfSF,
    :TotalBsmtSF,
    "1stFlrSF",
    "2ndFlrSF",
    :LowQualFinSF,
    :GrLivArea,
    :BsmtFullBath,
    :BsmtHalfBath,
    :FullBath,
    :HalfBath,
    :BedroomAbvGr,
    :KitchenAbvGr,
    :TotRmsAbvGrd,
    :Fireplaces,
    :GarageCars,
    :GarageArea,
    :WoodDeckSF,
    :OpenPorchSF,
    :EnclosedPorch,
    "3SsnPorch",
    :ScreenPorch,
    :PoolArea,
    :MiscVal,
    :YearBuilt, :YearRemodAdd, :GarageYrBlt, :MoSold, :YrSold
]
transform!(train, parse_vars .=> (x -> parse_or_nan.(x)) .=> parse_vars)

train.DE_CAT_Bsmt = ifelse.(train.BsmtCond .== "NA", 0.0, 1.0)
quality_vars_in = [
    :ExterQual,
    :ExterCond,
    :BsmtQual,
    :BsmtCond,
    :HeatingQC,
    :KitchenQual,
    :FireplaceQu,
    :GarageQual,
    :GarageCond,
    :PoolQC,
]
quality_vars_out = "DE_Quality_" .* string.(quality_vars_in)
transform!(train, quality_vars_in .=> (x -> quality_of_material.(x)) .=> quality_vars_out)

train.DE_Exposure_Bsmt = exposure_bsmt.(train.BsmtExposure)

transform!(
    train,
    [:BsmtFinType1, :BsmtFinType2] .=>
        (x -> rate_bsmt.(x)) .=> [:DE_Rate_BsmtFinType1, :DE_Rate_BsmtFinType2],
)

train.DE_Rate_Functional = home_functionality.(train.Functional)
train.DE_Rate_GarageFinish = garage_finished.(train.GarageFinish)
train.DE_Rate_PavedDrive = paved_drive_score.(train.PavedDrive)
train.DE_Rate_Fence = fence_quality.(train.Fence);

## feature engineering 
### cat features
for cat in cat_vars
    vars = unique(train[!, cat]) 
    one_hot_encoded = DataFrame(transpose(vars.== permutedims(train.MSSubClass)), :auto)
    rename!(one_hot_encoded, "DE_CAT_" .* string(cat) .* "_" .*  vars)
    train = hcat(train, one_hot_encoded)
end

### create ratios regarding sqft
normalize_vars_in = [:LotFrontage,
    :LotArea,
    :OverallQual,
    :MasVnrArea,
    :BsmtFinSF1,
    :BsmtFinSF2,
    :BsmtUnfSF,
    :TotalBsmtSF,
    "1stFlrSF",
    "2ndFlrSF",
    :LowQualFinSF,
    :GrLivArea,
    :GarageCars,
    :GarageArea,
    :WoodDeckSF,
    :OpenPorchSF,
    :EnclosedPorch,
    "3SsnPorch",
    :ScreenPorch,
    :PoolArea,
    :MiscVal
]
normalize_vars_out = "DE_Area_" .* string.(normalize_vars_in)

transform!(train,
    normalize_vars_in .=> (x->clamp.(log.(x),0,10)) .=> normalize_vars_out
)

# feature engineering 
## by building type
train.DE_Size_LotArea_per_building = train.LotArea ./ train.BldgType

## by MasVnrType
dfg = groupby(train, :MasVnrType)
df = transform!(dfg, :MasVnrArea => (x-> x .- mean(skipnan(x))) => "DE_Size_MasVnrArea_per_type")

## by basement type
bsmt_area = [:BsmtFinSF1,
    :BsmtFinSF2,
    :BsmtUnfSF]
bsmt_area_pairs = [[x, :TotalBsmtSF] for x in bsmt_area]
bsmt_area_out = "DE_Rate_" .* bsmt_area

transform!(train, bsmt_area_pairs .=> 
    ((x,y)->clamp.(x ./ y, 0,1)) .=> 
        bsmt_area_out)

dfg = groupby(train, :BldgType)
df = transform(dfg,
    bsmt_area_out .=> (x-> x .- mean(skipnan(x))) .=> bsmt_area_out .* "_bldgtype"
)

## # of rooms should be keept as between [0,1]
rms = [:BsmtFullBath, :BsmtHalfBath, 
    :FullBath, :HalfBath,:BedroomAbvGr,:KitchenAbvGr,:TotRmsAbvGrd,
    :Fireplaces]
train.DE_Rate_bedroomsabvgr = clamp.(train.BedroomAbvGr ./ train.TotRmsAbvGrd, 0, 1)
train.DE_Rate_fullbathabvgr = clamp.(train.FullBath ./ train.TotRmsAbvGrd, 0, 1)
train.DE_Rate_halfbathabvgr = clamp.(train.HalfBath ./ train.TotRmsAbvGrd, 0, 1)
train.DE_Rate_kitchenabvgr = clamp.(train.KitchenAbvGr ./ train.TotRmsAbvGrd, 0, 1)

transform!(train,
    rms .=> identity .=> "DE_Count_" .* rms
)

### transform time to numerical 
time_vars = [:YearBuilt, :YearRemodAdd, :GarageYrBlt, :MoSold, :YrSold];
train[!,time_vars]

train.DE_Time_built_till_now = (Int64(year(today())) .- train.YearBuilt) / 50
train.DE_Time_remod_till_now = (Int64(year(today())) .- train.YearRemodAdd) / 40
train.DE_Time_garage_built_till_now = (Int64(year(today())) .- train.GarageYrBlt) / 45
train.DE_Time_sold_till_now = Dates.value.(today() .- Date.(train.YrSold, train.MoSold, 1)) / 6000
train.DE_Time_sold_since_built = (train.YrSold .- train.YearBuilt) / 35

### target
train._target = log.(train.SalePrice)

out_vars = names(train)[occursin.("DE_",names(train))]

train_cleaned = select(train, ["Id"; out_vars; "_target"])
transform!(train_cleaned,
    out_vars .=> (x->Float64.(x)) .=> out_vars,
    :Id => (x->string.(x)) => :Id
)

print(typeof.(eachcol(train_cleaned)))

### save 
CSV.write("assets/train_cleaned.csv", train_cleaned)
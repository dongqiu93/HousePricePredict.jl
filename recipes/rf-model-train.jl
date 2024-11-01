using CSV
using DataFrames
using MLJ 
using Random 
using StatsBase 
using HousePricesPredict

import HousePricesPredict:
    skipnan

train = DataFrame(CSV.File("assets/train_cleaned.csv"))
test = DataFrame(CSV.File("assets/test_cleaned.csv"))
mutual_vars = intersect(names(train),names(test))
train = select(train, [mutual_vars; "_target"])
test = select(test, mutual_vars)

# searching for a model that matches the datatypes
_train = select(train, Not(["Id","_target"]))
_y = train."_target"

# task(model) = matching(model, X, y)
# models(task)

# XGBoostRegressor is suggested 
# create and train a gradient boosted tree model of 5 trees
RandomForestRegressor = @load RandomForestRegressor pkg=DecisionTree

grid = Dict(
    :max_depth => [8,16,32,64,128], 
    :min_samples_leaf => [1,2,4,8],
    :min_samples_split => [12,16,20,24,28,32], 
    :n_subfeatures => [0],
    :n_trees => 64:32:128,
    :sampling_fraction => [0.7,0.8,0.9,1.0]
)

Random.seed!(123)
hyper_list = Dict{Symbol,Dict}()
for i in 1:256
    hyper = Dict{Symbol,Any}()
    for key ∈ keys(grid)
        hyper[key] = sample(grid[key])
    end
    hyper_list[Symbol("M_$i")] = hyper
end 

cvs = collect(1:4)
_rows = collect(1:nrow(_train))
cutoff_lists = []
for cv in cvs 
    @info cv
    cutoff = sample(_rows, Int64(round(nrow(_train) / 4; digits=0)), replace=false)
    setdiff!(_rows, cutoff)
    push!(cutoff_lists, cutoff)
end

logger_list = DataFrame()
for cv in cvs
    # clip the data into 4 pieces 
    rdn = cutoff_lists[cv]

    X_train = _train[setdiff(1:1:nrow(_train), rdn),:]
    y_train = _y[setdiff(1:1:nrow(_train), rdn)]

    X_eval = _train[rdn,:]
    y_eval = _y[rdn]

    for key in collect(keys(hyper_list))
        @info key
        hyper = hyper_list[key]
        bst = RandomForestRegressor(; hyper...)
        mach = machine(bst, X_train, y_train)
        MLJ.fit!(mach, verbosity=2)
        pred = MLJ.predict(mach, X_eval);
        rmse = rms(pred, y_eval)
        logger = Dict(
            :cv => cv,
            :model_id => key,
            :rmse => rmse
        )
        append!(logger_list, logger)
    end
    _logger_list = logger_list[logger_list.cv .== cv,:]
    sort!(_logger_list, :rmse)
    top_n = Int64(round(length(collect(keys(hyper_list))) / 2; digits=0))
    _logger_list = _logger_list[1:top_n,:]
    _model_ids = unique(_logger_list.model_id)
    filter!(kv -> kv[1] in _model_ids, hyper_list)
end

logger_list_df = DataFrame(logger_list)
low_rmse_df = logger_list_df[(logger_list_df.cv .== 4 .&& logger_list_df.model_id .∈ Ref(collect(keys(hyper_list)))),:]
sort!(low_rmse_df, :rmse)
low_rmse_df = low_rmse_df[1:10,:]
hyper_list_selected = filter(kv -> kv[1] in low_rmse_df.model_id, hyper_list)

@info DataFrame([value for (key,value) in hyper_list_selected])

# check feature importance

## train the model and make predictions using test data 
_test = select(test, Not(["Id"]))

out = DataFrame()
for key in collect(keys(hyper_list_selected))
    @info key
    hyper = hyper_list_selected[key]
    bst = RandomForestRegressor(; hyper...)
    mach = machine(bst, _train, _y)
    MLJ.fit!(mach, verbosity=2)
    pred = MLJ.predict(mach, _test)
    _pred = exp.(pred)
    out_iter = DataFrame(
        :model_id => key,
        :Id => test.Id,
        :SalePrice => _pred,
    )
    append!(out, out_iter)
end

# aggregate the results 
_out = combine(groupby(out,:Id),
    :SalePrice => (x->mean(skipnan(Float64.(x)))) => :SalePrice
)

CSV.write("res/rf_res_3.csv", _out)
CSV.write("res/rf_hyper_selected.csv", DataFrame([value for (key,value) in hyper_list_selected]))

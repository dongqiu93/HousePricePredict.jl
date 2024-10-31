using CSV
using DataFrames
using MLJ 
using Random 
using StatsBase 

train = DataFrame(CSV.File("assets/train_cleaned.csv"))
test = DataFrame(CSV.File("assets/test_cleaned.csv"))
mutual_vars = intersect(names(train),names(test))
train = select(train, [mutual_vars; "_target"])
test = select(test, mutual_vars)

# searching for a model that matches the datatypes
_train = select(train, Not(["Id","_target"]))
_y = train."_target"

rdn = sample(1:1:nrow(_train), 1200; replace=false)

X_train = _train[rdn,:]
y_train = _y[rdn]

X_eval = _train[setdiff(1:1:nrow(_train), rdn),:]
y_eval = _y[setdiff(1:1:nrow(_train), rdn)]

# task(model) = matching(model, X, y)
# models(task)

# XGBoostRegressor is suggested 
# create and train a gradient boosted tree model of 5 trees
XGBoostRegressor = @load XGBoostRegressor pkg=XGBoost

grid_xgboost = Dict(
    :num_round => 200:200:3000,
    :eta => [0.1,0.01,0.001],
    :max_depth => 2:2:12,
    :lambda => [0, 0.01, 0.1, 1, 10],
    :gamma => [0, 0.0005, 0.001, 0.002, 0.01],
    :alpha => [0, 0.01, 0.1, 1, 10],
    :early_stopping_rounds => [0, 100],
    :subsample => [0.25, 0.5, 0.75, 1],
    :max_bin => [64, 128, 256],
    :seed => [123],
)

Random.seed!(123)
hyper_list = Dict{Symbol,Dict}()
for i in 1:256
    hyper = Dict{Symbol,Any}()
    for key ∈ keys(grid_xgboost)
        hyper[key] = sample(grid_xgboost[key])
    end
    hyper_list[Symbol("M_$i")] = hyper
end 

cvs = collect(1:4)
logger_list = DataFrame()
for cv in cvs
    for key in collect(keys(hyper_list))
        @info key
        hyper = hyper_list[key]
        bst = XGBoostRegressor(; hyper...)
        mach = machine(bst, X_train, y_train)
        fit!(mach, verbosity=2)
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

# select final hypers 
logger_list_df = DataFrame(logger_list)
low_rmse_df = logger_list_df[(logger_list_df.cv .== 4 .&& logger_list_df.model_id .∈ Ref(collect(keys(hyper_list)))),:]
sort!(low_rmse_df, :rmse)
low_rmse_df = low_rmse_df[1:10,:]
hyper_list_selected = filter(kv -> kv[1] in low_rmse_df.model_id, hyper_list)

## train the model and make predictions using test data 
_X_test = select(test, Not(["Id"]))

out = DataFrame()
for key in collect(keys(hyper_list_selected))
    @info key
    hyper = hyper_list[key]
    bst = XGBoostRegressor(; hyper...)
    mach = machine(bst, _train, _y)
    fit!(mach, verbosity=2)
    pred = MLJ.predict(mach, _X_test)
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
    :SalePrice => mean => :SalePrice
)

CSV.write("res/xgboost_res_3.csv", _out)

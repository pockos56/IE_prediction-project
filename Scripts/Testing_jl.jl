using Pkg
Pkg.add(url="https://github.com/pockos56/IE_prediction.jl")

using IE_prediction
using Statistics

mean()
logIE_from_CNLs([20.3],200.5,"positive",10)
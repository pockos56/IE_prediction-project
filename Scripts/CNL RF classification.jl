## import packages ##
using ScikitLearn
using BSON
using Plots
using Statistics
using DataFrames
using CSV
using ScikitLearn.CrossValidation: cross_val_score
using ScikitLearn.CrossValidation: train_test_split
using PyCall
using Conda
using JLD
using LaTeXStrings
using LinearAlgebra
using Random
using StatsBase
@sk_import linear_model: LinearRegression
@sk_import ensemble: RandomForestClassifier



#imports
using NearestNeighbors, Statistics, LinearAlgebra, Random, Flux#master
using Flux: onehotbatch, onecold, onehot, crossentropy, throttle, NNlib, @functor
using Zygote:@nograd
using Statistics: mean
using Base.Iterators: partition


#args
root = "./data/"
K = 10 # k nearest-neighbors
batch_size = 2
num_classes = 10 #possible values {10,40}
npoints = 128


#data and model
include("./data.jl")
include("./model.jl")


# input: width*height*channel*minibatch

# Fetching the train and validation data and getting them into proper shape
datapath, classes = get_data(num_classes)
X = [datapoint(p, classes) for p in datapath["train"]]
data = [X[i][1] for i in 1:length(X)]
labels = onehotbatch(cat([X[i][2] for i in 1:length(X)]..., dims=1), 1:num_classes)
train = [(cat(data[i]..., dims = 3), labels[:,i]) for i in partition(1:length(data), batch_size)]

VAL = [datapoint(p, classes) for p in datapath["test"]]
valX = cat([VAL[i][1] for i in 1:length(VAL)]..., dims=3)
valY = onehotbatch(cat([VAL[i][2] for i in 1:length(VAL)]..., dims=1), 1:num_classes)

# Defining the loss and accuracy functions

m = DGCNN_cls(num_classes, K, npoints)

loss(x, y) = crossentropy(m(x), y)

accuracy(x, y) = mean(onecold(cpu(m(x)), 1:num_classes) .== onecold(cpu(y), 1:num_classes))

# Defining the callback and the optimizer

evalcb = throttle(() -> @show(accuracy(valX, valY)), 10)

opt = ADAM()

# Starting to train models

# print(m(rand(3,npoints,batch_size)))
print(typeof(train[1][1]))
# print(loss(train[1]...))
# gradient(() -> loss(train[1]...), params(m))

print("Training Start\n")
Flux.train!(loss, params(m), train, opt, cb = evalcb)
print("Training Finished\n")


# Accuracy of valset of ModelNet dataset
@show(accuracy(valX, valY))
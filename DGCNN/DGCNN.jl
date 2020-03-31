#imports
using NearestNeighbors, Statistics, LinearAlgebra, Random, Flux#master
using Flux: onehotbatch, onecold, onehot, crossentropy, throttle, NNlib, @functor
using Zygote: @nograd, Params
using Statistics: mean
using Base.Iterators: partition

#args
include("./config.jl")

dataset = ARGS[1]
root = "./$(dataset)/"

#data and model
include("./data.jl")
include("./model.jl")


# input: width*height*channel*minibatch

# Fetching the train and validation data and getting them into proper shape
train, (valX, valY) = get_data(dataset, num_classes)

# Defining the loss and accuracy functions

m = DGCNN_cls(num_classes, K, npoints)

loss(x, y) = crossentropy(m(x), y)

accuracy(x, y) = mean(onecold(cpu(m(x)), 1:num_classes) .== onecold(cpu(y), 1:num_classes))

# Defining the callback and the optimizer

opt = ADAM(0.003)

# Starting to train models

function custom_train!(loss, ps, data, opt, epochs)
    ps = Zygote.Params(ps)
    for epoch in 1:epochs
        running_loss = 0
        for d in data
        gs = gradient(ps) do
            training_loss = loss(d...)
            running_loss += training_loss
            return training_loss
        end
        Flux.update!(opt, ps, gs)
        end
        print("Epoch: $(epoch), epoch_loss: $(running_loss), accuracy: $(accuracy(valX, valY))\n")
    end
  end

print("Training Start\n")
custom_train!(loss, params(m), train, opt, epochs)
print("Training Finished\n")

# Accuracy of valset
@show(accuracy(valX, valY))
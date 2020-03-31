function get_data(dataset::String = "3dmnist", num_classes::Int = 10) 
    
    if dataset == "3dmnist"
        
        class_file = Dict()
        shapeids = Dict()
        class = Dict()
        datapath = Dict()

        classes = Dict{String, Int}()
        for i in 0:9
            classes["$(i)"] = i
        end

        shapeids["train"] = ["$(i)" for i in 1:5000]
        shapeids["test"] = ["$(i)" for i in 1:1000]

        for split_ in ["train","test"]
            class_file[split_] = joinpath(root, "labels_$(split_).txt")
            class[split_] = [line::String for line in readlines(class_file[split_])]
            datapath[split_] = [(class[split_][i], joinpath(root, split_, (shapeids[split_][i])*".txt")) for i in 1:length(shapeids[split_])]
        end

        #shuffling trainset Point
        rng = MersenneTwister(1234);
        shuffle!(rng, datapath["train"])


        # Fetching the train and validation data and getting them into proper shape
        datapath, classes = get_data(num_classes)
        X = [datapoint(p, classes) for p in datapath["train"]]
        data = [X[i][1] for i in 1:length(X)]
        labels = onehotbatch(cat([X[i][2][1] for i in 1:length(X)]..., dims=1), 0:9)
        train = [(cat(data[i]..., dims = 3), labels[:,i]) for i in partition(1:length(data), batch_size)]

        VAL = [datapoint(p, classes) for p in datapath["test"]]
        valX = cat([VAL[i][1] for i in 1:length(VAL)]..., dims=3)
        valY = onehotbatch(cat([VAL[i][2][1] for i in 1:length(VAL)]..., dims=1), 0:9)
    
    elseif dataset == "modelnet"    
        catFile = joinpath(root, "modelnet$(num_classes)_shape_names.txt")
        categories = [line::String for line in readlines(catFile)]

        classes = Dict{String, Int}()
        for i in 1:length(categories)
            classes[categories[i]] = i
        end

        shapeids = Dict()
        shape_names = Dict()
        datapath = Dict()

        shapeids["train"] = [line::String for line in readlines(joinpath(root, "modelnet$(num_classes)_train.txt"))]
        shapeids["test"] = [line::String for line in readlines(joinpath(root, "modelnet$(num_classes)_test.txt"))]

        for split_ in ["train","test"]
            shape_names[split_] = [join(split(shapeids[split_][i], "_")[1:end-1], "_") for i in 1:length(shapeids[split_])]
            datapath[split_] = [(shape_names[split_][i], joinpath(root, shape_names[split_][i], (shapeids[split_][i])*".txt")) for i in 1:length(shapeids[split_])]
        end

        #shuffling trainset Point
        rng = MersenneTwister(1234);
        shuffle!(rng, datapath["train"])

        # Fetching the train and validation data and getting them into proper shape
        datapath, classes = get_data(num_classes)
        X = [datapoint(p, classes) for p in datapath["train"]]
        data = [X[i][1] for i in 1:length(X)]
        labels = onehotbatch(cat([X[i][2] for i in 1:length(X)]..., dims=1), 1:num_classes)
        train = [(cat(data[i]..., dims = 3), labels[:,i]) for i in partition(1:length(data), batch_size)]

        VAL = [datapoint(p, classes) for p in datapath["test"]]
        valX = cat([VAL[i][1] for i in 1:length(VAL)]..., dims=3)
        valY = onehotbatch(cat([VAL[i][2] for i in 1:length(VAL)]..., dims=1), 1:num_classes)
    else
        print("invalid dataset name choose between modelnet and 3dmnist")
        exit()
    end
        
    return train, (valX, valY)
end

function pcNormalize(pc)
    centroid = mean((x->x), pc, dims = 1)
    pc = pc .- centroid
    m = max((sum(pc.^2, dims=2).^.5)...)
    pc = pc ./ convert(Float32, m)
    return pc
end

function datapoint(fn, classes)
    #fn = datapath[index]
    cls = [classes[fn[1]]]
    Pointset = Array{Float32}(undef, npoints, 3)
    stream = open(fn[2], "r")
    for i in 1:npoints
        tmp = readline(stream, keep=false)
        Pointset[i, :] = map((x->parse(Float32, x)), split(tmp, " ")[1:3])
    end
    Pointset = pcNormalize(Pointset)
    return (Pointset,cls)
end
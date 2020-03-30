function get_data(num_classes::Int) 
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

    return datapath, classes
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
        Pointset[i, :] = map((x->parse(Float32, x)), split(tmp, ",")[1:3])
    end
    Pointset = pcNormalize(Pointset)
    return (Pointset,cls)
end
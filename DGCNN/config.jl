#args
dataset = "3dmnist"
root = "./$(dataset)/"
K = 10 # k nearest-neighbors
batch_size = 32
num_classes = 10 #possible values {10,40}
npoints = 1024
epochs = 50
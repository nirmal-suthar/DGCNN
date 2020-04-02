# DGCNN
Implementation of DGCNN classification in Flux machine learning library written in Julia.

For downloading and processing the modelnet40 dataset.
```bash
$ URL=https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip
$ ZIP_FILE=./modelnet/modelnet40.zip
$ mkdir -p ./modelnet/
$ wget -N $URL -O $ZIP_FILE
$ unzip $ZIP_FILE -d ./modelnet/
$ rm $ZIP_FILE
```

For downloading and processing the 3dmnist dataset.
```bash
$ URL=https://github.com/nirmal-suthar/dataset/raw/master/3dmnist.zip
$ ZIP_FILE=./3dmnist.zip
$ wget -N $URL -O $ZIP_FILE
$ unzip $ZIP_FILE -d ./
$ rm $ZIP_FILE
```

For changing and viewing the config of model edit the file `DGCNN/config.jl`

For running the model

```bash
$ julia --project="." DGCNN/DGCNN.jl <name of dataset to train> 
```
## Results

Both model are trained with similar config
```
batch_size = 32
npoints = 1024
lr = 0.003 (ADAM)
K ( nearest-neighbour, used in DGCNN )
```

Dataset | ModelNet (10 classes) |
---|--|
DGCNN |<b>0.8744</b>|
PointNet |0.7544|

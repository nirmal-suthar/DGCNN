# DGCNN
Implementation of DGCNN classification in Flux machine learning library written in Julia.

For downloading and processing the modelnet40 dataset.
```bash
$ URL=https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip
$ ZIP_FILE=./data/modelnet40.zip
$ mkdir -p ./modelnet/
$ wget -N $URL -O $ZIP_FILE
$ unzip $ZIP_FILE -d ./modelnet/
$ rm $ZIP_FILE
```

For downloading and processing the 3dmnist dataset.
```bash
$ URL=https://github.com/nirmal-suthar/dataset/raw/master/3dmnist.zip
$ ZIP_FILE=./data/3dmnist.zip
$ mkdir -p ./3dmnist/
$ wget -N $URL -O $ZIP_FILE
$ unzip $ZIP_FILE -d ./3dmnist/
$ rm $ZIP_FILE
```

For changing and viewing the config of model edit the file `DGCNN/config.jl`

For running the model

```bash
$ julia --project="." DGCNN/DGCNN.jl <name of dataset to train> 
```


# cnn-nanoporous
![image](image_header.png)
![CI](https://github.com/sergei-zor/cnn-nanoporous/actions/workflows/ci.yml/badge.svg)

## Transferable 3D Convolutional Neural Networks for Elastic Constants Prediction in Nanoporous Metals

This repository contains the codebase for the paper 

*Transferable 3D Convolutional Neural Networks for Elastic Constants Prediction in Nanoporous Metals*

by S. Zorkaltsev, R. Topolnicki, T.E. Carmon, S. Mathesan, P. Dłotko, D. Mordehai and M. Haranczyk

[Mater. Des., vol. 260, p. 114896, 2025.](https://doi.org/10.1016/j.matdes.2025.114896)

**Note:** This is a fork of the repository accompanying the paper. It is extended with a FastAPI service, Docker
container, and CI pipeline.

### Nanoporous structures generation 
Three-dimensional periodic bicontinuous nanoporous structures were generated based on the method proposed by [Soyarslan et al.](https://doi.org/10.1016/j.actamat.2018.01.005), which involves superposition of standing sinusoidal waves with fixed wavelengths and varying phase. 

Three datasets generated in this study are provided in the `datasets` folder:

- **Dataset 1**: Contains 5000 nanoporous gold structures with a target solid volume fraction of 0.25.
- **Dataset 2**: Contains 1000 NPG structures with a target solid volume fraction of 0.35.
- **Dataset 3**: Contains 422 nanoporous silver structures with MD-computed elastic constants. It is divided into two parts: structures 1–250, and 251–422, have target solid volume fractions of 0.25 and 0.35, respectively.

Each structure in the datasets is labeled with Molecular Dynamics (MD) computed elastic constants in three principal directions, and a set of computed morphological and topological descriptors. All structures are accompanied by 30 phase values, required for identical regeneration. After the atomistic structure is generated in `.lmp` format, it is converted into a binary 3D array with the specified grid resolution.

The generator works with the following command
```
python datasets/grid_generator.py --dataset datasets/Dataset1_gold_0.25.xlsx --grid_resolution 80 --output_dir datasets/results --n_cells 125 --crystal_system FCC --keep_lmp_files 

```

### Training database preparation

To prepare the training data, follow these steps:

1. Rotate grids to match simulation directions (`x`, `y`, and `z`):

```
python database_preparation/npy_rotations.py --input_dir datasets/results/grids_80 --output_dir database_preparation/database_80 --rotation x
python database_preparation/npy_rotations.py --input_dir datasets/results/grids_80 --output_dir database_preparation/database_80 --rotation y
python database_preparation/npy_rotations.py --input_dir datasets/results/grids_80 --output_dir database_preparation/database_80 --rotation z

```
2. Generate a .csv database linking each rotated `.npy` grid to its target value (`cii`) and, optionally, a set of structural descriptors:

```
python database_preparation/prepare_database.py --dataset datasets/Dataset1_gold_0.25.xlsx --input_dir database_preparation/database_80 --include_descriptors

```

### Model training
To train the model, one needs to provide the:
   1. A folder containing all the `.npy` grids.
   2. A `.csv` database mapping each structure (column name `npy_path`) to its elastic constant values (stored in column `cii`), and optionally a list of additional descriptors.
```
npy_path	cii	Genus	Solid Volume Fraction	Number of Nodes	Mean Nodes Diameter	Nodes Diameter Standard Deviation	Mean Ligaments Diameter	….
/home/user/cnn-nanoporous/structures/dataset_md_5000/0_rot-x.npy	0.763035	6	0.265	28	75.809	13.066	60.165	….
/home/user/cnn-nanoporous/structures/dataset_md_5000/1_rot-x.npy	1.505392	5	0.275	30	77.56	14.807	55.442	….
/home/user/cnn-nanoporous/structures/dataset_md_5000/2_rot-x.npy	1.511555	4	0.267	30	73.027	14.271	64.65	….
/home/user/cnn-nanoporous/structures/dataset_md_5000/3_rot-x.npy	0.341111	5	0.233	19	71.369	15.981	67.228	….
/home/user/cnn-nanoporous/structures/dataset_md_5000/4_rot-x.npy	0.358744	1	0.243	19	77.502	12.793	63.799	….
/home/user/cnn-nanoporous/structures/dataset_md_5000/5_rot-x.npy	2.476169	9	0.28	31	77.736	11.347	68.959	….
/home/user/cnn-nanoporous/structures/dataset_md_5000/6_rot-x.npy	2.092207	7	0.258	32	67.656	16.139	59.48	….
/home/user/cnn-nanoporous/structures/dataset_md_5000/7_rot-x.npy	0.413300	3	0.24	20	72.718	14.825	68.353	….
/home/user/cnn-nanoporous/structures/dataset_md_5000/8_rot-x.npy	0.076880	2	0.222	21	68.661	14.375	62.859	….
```
Then all it takes is to run the following command (see `run_training.sh`), providing that directory `results` exists:
```
python train.py --outputdir results/ --database database_withfeatures.csv \
        --lr 0.001 --optimizer Adam --batch_size 40  --model_name densenet-201 \
        --p_flip 0.3 --aug_roll_ratio 0.3 --cv_folds 8 --folds 0 1 2 3 4 5 6 7 \
        --suffix 5k_flip0.3roll0.3_strat_nodesc --use_stratification --size 80
```
Use `--model_name` argument to specify CNN architecture to use. Options are:

```
    efficientnet-b0
    efficientnet-b4
    densenet-121
    densenet-169
    densenet-201
    densenet-264
    resnet-18
    resnet-50
    resnet-101
    cnn
    mobilenet
```

## Running Inference
The champion DenseNet-201 model is available in the models/ directory. 
There are 40 exemplary voxelized structures provided in the `exemplary_structures/` directory.
You can run inference directly using the provided model **without any training** (GPU is not required).
To predict the Elastic Modulus for these 40 structures, run:
```
python inference.py --outputfile inference_results.csv --model_path models/model_DenseNet201.pth  --txt_path inference_filelist.txt
```
The predictions will be saved to the `inference_results.csv` file, which will look like this:
```
path,prediction
exemplary_structures/155_rot-z.npy,0.92881143
exemplary_structures/1689_rot-z.npy,0.35452503
exemplary_structures/2599_rot-z.npy,0.701846
exemplary_structures/2193_rot-z.npy,1.0405514
exemplary_structures/3678_rot-y.npy,0.9879305
exemplary_structures/3217_rot-y.npy,0.4050305
[....]
```
The expected predicted Elastic Modulus values, along with the exact values obtained using MD, are available in the `expected_inference_results.csv` file.
The R2 coefficient for these 40 exemplary structures is **0.9488**.

## Serving the model (API)

The same DenseNet-201 model is also available as an API, integrating the model inference into other applications without a local Python setup.

### Run locally
```
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Check the API state at `http://localhost:8000/health`, and make a prediction with:
```
curl -X POST -F "file=@exemplary_structures/155_rot-z.npy" http://localhost:8000/predict
```

### Run as a Docker container
```
docker build -t nanoporous-api .
docker run -p 8000:8000 nanoporous-api
```

### API endpoints
- `GET /health` - service and model status
- `POST /predict` - takes a `.npy` voxelized structure, returns the predicted elastic modulus

A GitHub Actions workflow builds and verifies on every push (`.github/workflows/ci.yml`).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

### Credits
Some CNN implementations used in this repo modifies the code from [xmuyzz/3D-CNN-PyTorch](https://github.com/xmuyzz/3D-CNN-PyTorch) repository. 
 

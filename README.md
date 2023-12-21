🚀 Good news! We have have created a demo showcasing the capabilities of our illumination correction model whithin the full document refinement pipeline. [Check it out here!](https://felixhertlein.de/docrefine/home)

# Template-guided Illumination Correction for Document Images with Imperfect Geometric Reconstruction

This is the repository will contain the code for our [paper](https://openaccess.thecvf.com/content/ICCV2023W/NIVT/html/Hertlein_Template-Guided_Illumination_Correction_for_Document_Images_with_Imperfect_Geometric_Reconstruction_ICCVW_2023_paper.html) which has been accepted at the International Conference on Computer Vision Workshop on New Ides in Vision Transformers ([NIVT-ICCV2023](https://sites.google.com/view/nivt-iccv2023/)).

The project page can be found [here](https://felixhertlein.github.io/illtrtemplate/).

## Usage

### VS-Code Devcontainer

We highly recommend to use the provided Devcontainer to make the usage as easy as possible:

- Install [Docker](https://www.docker.com/) and [VS Code](https://code.visualstudio.com/)
- Install VS Code Devcontainer extension `ms-vscode-remote.remote-containers`
- Clone the repository
  ```shell
  git clone https://github.com/FelixHertlein/inv3d-model.git
  ```
- Press `F1` (or `CTRL + SHIFT + P`) and select `Dev Containers: Rebuild and Reopen Container`
- Go to `Run and Debug (CTRL + SHIFT + D)` and press the run button, alternatively press `F5`

## Inference

### Start the inference

`python3 inference.py --model illtr_template@inv3d@full --dataset inv3d_real_unwarp --gpu 0`

### Models

The models will be downloaded automatically before the inference starts.

Available models are:

- illtr@inv3d
- illtr_template@inv3d@full
- illtr_template@inv3d@pad=0
- illtr_template@inv3d@pad=64
- illtr_template@inv3d@pad=128

### Datasets

#### Inv3DRealUnwarped

Inv3DRealUnwarped will be downloaded automatically when passing `inv3d_real_unwarp` as the dataset argument.

#### Custom dataset

To unwarp your own data, you can mout your data inside the container using the `.devcontainer/devcontainer.json` config.

Mount your data folder to `/workspaces/inv3d-model/input/YOUR_DATA_DIRECTORY`.
Make sure, all samples contain an image `norm_image.png` and the corresponding template `template.png` (only for template-based models) within the sample subdirectory.

### Output: Unwarped images

All unwarped images are placed in the `output` folder.

## Training

### Training datasets

#### Inv3D

Download Inv3D [here](https://publikationen.bibliothek.kit.edu/1000161884), combine all downloads and mount it using the devcontainer.json, such that the file tree looks as follows:

```
input/inv3d
|-- data
|   |-- test
|   |-- train
|   |-- val
|   `-- wc_min_max.json
|-- log.txt
`-- settings.json
```

### Start a new training

`python3 train.py --model illtr_template --dataset inv3d --version v1 --gpu 0 --num_workers 32`

### Resume a training

`python3 train.py --model illtr_template --dataset inv3d --version v1 --gpu 0 --num_workers 32 --resume`

### Training output

```
models/TRAINED_MODEL/
|-- checkpoints
|   |-- checkpoint-epoch=00-val_mse_loss=0.0015.ckpt
|   `-- last.ckpt
|-- logs
|   |-- events.out.tfevents.1698250741.d6258ba74799.433.0
|   |-- ...
|   `-- hparams.yaml
`-- model.py
```

### Help

```
train.py [-h]
--model MODEL
--dataset DATASET
--gpu GPU
--num_workers NUM_WORKERS
[--version VERSION]
[--fast_dev_run]
[--model_kwargs MODEL_KWARGS]
[--resume]

Training script

options:
  -h, --help            show this help message and exit
  --model {illtr,illtr_template}
                        Select the model for training.
  --dataset {inv3d}     Select the dataset to train on.
  --gpu GPU             The index of the GPU to use for training.
  --num_workers NUM_WORKERS
                        The number of workers as an integer.
  --version VERSION     Specify a version id for given training. Optional.
  --fast_dev_run        Enable fast development run (default is False).
  --model_kwargs MODEL_KWARGS
                        Optional model keyword arguments as a JSON string.
  --resume              Resume from a previous run (default is False).
```

## Evaluation

### Models

The pretrained models will be downloaded automatically before the evaluation starts.

Available models are:

- illtr@inv3d
- illtr_template@inv3d@full
- illtr_template@inv3d@pad=0
- illtr_template@inv3d@pad=64
- illtr_template@inv3d@pad=128

### Datasets

#### Inv3DRealUnwarped

Inv3DRealUnwarped will be downloaded automatically when passing `inv3d_real_unwarp` as the dataset argument.

#### Inv3DTest

Include the Inv3D dataset as described in section Training > Training dataset > Inv3d.

### Start an evaluation

`python3 eval.py --trained_model geotr_template@inv3d@v1 --dataset inv3d_real --gpu 0 --num_workers 16`

Evaluation output:

```
models/TRAINED_MODEL
|-- checkpoints
|   `-- ...
|-- eval
|   `-- inv3d_real_unwarp
|       `-- results.csv
`-- inference
    `-- inv3d_real_unwarp
        |-- 00411
        |   |-- warped_document_crumpleseasy_bright
        |   |   |-- ill_image.png
        |   |   `-- orig_image.png
        |   |-- ...
        |-- ...
```

### Help

```
eval.py [-h]
--trained_model MODEL
--dataset DATASET
--gpu GPU
--num_workers NUM_WORKERS

Evaluation script

options:
  -h, --help            show this help message and exit
  --trained_model {illtr@inv3d,
                   illtr_template@inv3d@full,
                   illtr_template@inv3d@pad=0,
                   illtr_template@inv3d@pad=128,
                   illtr_template@inv3d@pad=64,
                   ... own trained models}
                        Select the model for evaluation.
  --dataset {inv3d_real_unwarp}
                        Select the dataset to evaluate on.
  --gpu GPU             The index of the GPU to use for training.
  --num_workers NUM_WORKERS
                        The number of workers as an integer.
```

## Citation

If you use the code of our paper for scientific research, please consider citing

```latex
@inproceedings{Hertlein2023,
  title={Template-Guided Illumination Correction for Document Images with Imperfect Geometric Reconstruction},
  author={Hertlein, Felix and Naumann, Alexander},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={904--913},
  year={2023}
}
```

## Acknowledgements

The model IllTr is part of [DocTr](https://github.com/fh2019ustc/DocTr). IllTrTemplate is based on IllTr.

## Affiliations

<p align="center">
    <img src="https://upload.wikimedia.org/wikipedia/de/thumb/4/44/Fzi_logo.svg/1200px-Fzi_logo.svg.png?raw=true" alt="FZI Logo" height="200"/>
</p>

## License

This project is licensed under [MIT](LICENSE).

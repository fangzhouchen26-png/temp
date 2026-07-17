# PARE-Net protein density-point-cloud adapter

This directory adapts upstream [PARE-Net](https://github.com/yaorz97/PARENet) to register a simulated single-chain density point cloud against spherical candidate crops extracted from the complete protein or complex density point cloud stored under `dataset/`.

## Task definition

For each valid chain:

- reference: `<case>/<case>_tgt_2.00.txt`
- source: `<case>/<case>_model*_chain*_src_2.00.txt`
- output: a rigid `4 x 4` transform mapping the augmented source into the reference coordinate system

Coordinates stay in Angstroms. The target crop diameter is fixed to `1.25 x` the robust source-chain diameter (`r_crop = 1.25 x r_chain`, with `r_chain` computed from the 99th percentile of distances to the chain centroid). Training and validation use an oracle positive crop around the known chain location. Testing generates target-supported sliding-window centers across the complete map and runs the complete PARE-Net pipeline independently for every candidate crop.

During training, the source is rotated around its centroid, translated by up to 30 Angstrom per axis, optionally thinned, and perturbed by coordinate noise. The inverse augmentation is supplied as the ground-truth transform.

Cases, rather than individual chains, are assigned to train/validation/test splits. This prevents different chains sharing the same complete target map from leaking across splits.

## Install the experiment into PARE-Net

From the root of this repository:

```bash
bash parenet_protein/install_into_parenet.sh
```

The script clones PARE-Net to `external/PARENet` when needed and creates:

```text
external/PARENet/experiments/ProteinFit/
```

It reuses the upstream `model.py`, `backbone.py`, and `trainval.py`, while installing protein-specific data, configuration, loss, inspection, and testing files.

## Install upstream dependencies

Use the environment versions recommended by upstream PARE-Net. From `external/PARENet`:

```bash
pip install -r requirements.txt
python setup.py build develop
cd pareconv/extentions/pointops
python setup.py install
```

PARE-Net's CUDA extensions must match the PyTorch/CUDA environment on the training server.

## Inspect data and create a fixed split file

```bash
cd external/PARENet/experiments/ProteinFit
python inspect_dataset.py \
  --dataset_root ../../../dataset \
  --write_splits ../../../parenet_protein/protein_splits.json
```

The inspection reports valid cases, valid chain pairs, point-count ranges, skipped files, identity overlap, and the numerical inverse-transform error.

To force training and testing to use the saved split:

```bash
export PROTEINFIT_SPLIT_FILE="$(realpath ../../../parenet_protein/protein_splits.json)"
```

## Preflight before training

Run the exact coarse-patch audit across multiple deterministic augmentations:

```bash
cd external/PARENet/experiments/ProteinFit
export PROTEINFIT_DATASET_ROOT="$(realpath ../../../dataset)"
CUDA_VISIBLE_DEVICES=0 python audit_training_windows.py --subset train --augmentation-repeats 3
CUDA_VISIBLE_DEVICES=0 python audit_training_windows.py --subset val --augmentation-repeats 3
```

Then verify a real forward/loss/backward/optimizer path:

```bash
CUDA_VISIBLE_DEVICES=0 python smoke_train.py --steps 20
```

Do not start a full run unless both commands report `PASS`.

## Train

```bash
cd external/PARENet/experiments/ProteinFit
export PROTEINFIT_DATASET_ROOT="$(realpath ../../../dataset)"
CUDA_VISIBLE_DEVICES=0 python trainval.py
```

The initial settings are deliberately conservative for 2 Angstrom density samples:

- spherical crop diameter: `1.25 x` source-chain diameter
- sliding-window center stride: `0.25 x` source-chain diameter, minimum 2 Angstrom
- all valid sliding windows are compared by default (`crop.max_candidates = None`)
- stage voxel scales: approximately 2, 4, and 8 Angstrom
- ground-truth node radius: 4 Angstrom
- fine positive radius: 3 Angstrom
- fine negative radius: 8 Angstrom
- validation/test augmentation: deterministic
- minimum source points: 128
- minimum target points: 256

## Test

Use a snapshot produced by training:

```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
  --benchmark test \
  --snapshot ../../output/ProteinFit/snapshots/<snapshot>.pth.tar
```

Per-candidate results are saved as `chain<id>_candidate<nnnn>.npz` under `external/PARENet/output/ProteinFit/features/test/`. Each file records the window center, chain/crop diameter, estimated transform, correspondences, and scores.

## Important limitations

The repository currently contains only a small number of protein cases. This is enough for parser, transform-direction, overfitting, and pipeline validation, but not for a final generalizable model. Random rotations create pose diversity, not structural diversity. Expand the training set and split it by sequence/structure homology before reporting benchmark performance.

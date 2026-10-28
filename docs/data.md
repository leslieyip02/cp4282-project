# Datasets

## Mip-NeRF 360 bonsai

Download the [NerfBaselines scene](https://github.com/nerfbaselines/nerfbaselines#downloading-data)
from the repository root:

```sh
python -m pip install nerfbaselines
nerfbaselines download-dataset external://mipnerf360/bonsai -o data/bonsai
```

The training input in `data/bonsai/` is:

```text
nb-info.json
train_list.txt
test_list.txt
images_2/
sparse_2/0/cameras.bin
sparse_2/0/images.bin
sparse_2/0/points3D.bin
```

Copy these files and directories to the cluster. The loader reads poses and
per-view intrinsics from the COLMAP binaries and initializes splats from
`points3D.bin`; it supports undistorted `PINHOLE` cameras. The same layout works
for other NerfBaselines Mip-NeRF 360 scenes, such as `counter`.

`config/real.yaml` resizes bonsai to 390 × 260. The loader reserves eight of
the official training views for validation and uses the official test views
only for final evaluation. Each run writes its exact split to `split.json`.
Scores from resized images and fewer training views are not directly comparable
with published full-resolution scores.

The [published Mip-NeRF 360 checkpoint and predictions](https://nerfbaselines.github.io/m-mipnerf360)
are optional reference material. They are not training inputs and do not need
to be copied to the cluster:

```sh
mkdir -p data/bonsai/reference/mipnerf360
curl -fL -o data/bonsai/reference/mipnerf360.zip \
  https://huggingface.co/nerfbaselines/nerfbaselines/resolve/main/mipnerf360/mipnerf360/bonsai.zip
unzip data/bonsai/reference/mipnerf360.zip -d data/bonsai/reference/mipnerf360
```

Everything under `data/` is ignored by Git.

## NeRF-synthetic scenes

A scene needs `transforms_train.json`, `transforms_val.json`, and
`transforms_test.json` plus their referenced images. The NeRF-synthetic download
does not need to contain `init.ply`: the synthetic preparation job generates it
when missing, before preparing filtered targets. All 21 method/seed runs for a
scene then read the same initialization. The initialization seed (default 7) is
separate from the training seeds (7, 17, 27).

Generate it explicitly from the repository root if you want to inspect or copy
the scene before running preparation:

```sh
python generate_synthetic_init.py --data data/lego --seed 7 --points 50000
```

The generator samples positions uniformly in `[-1.3, 1.3]^3`, uses near-gray
random colors, and writes the Gaussian properties required by this trainer.
It writes the parameters to `init.json` beside `init.ply`. The default point
count is 50,000, matching `model.initial_splats` in the shared training defaults;
the preparation job uses the count from the selected config. Keep the count
below `model.capacity` to allow densification. The generator refuses to replace
an existing `init.ply` unless passed `--overwrite`; preparation always keeps an
existing file. To choose another seed for a missing file in a Slurm preparation
job, submit with `INIT_SEED=123 sbatch ...`.

For an experiment comparing image filters, use one initialization per scene so
all methods start from the same points. If replacing a previously supplied
initialization, save it under another name first and generate a new `init.ply`.

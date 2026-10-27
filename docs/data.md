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
`transforms_test.json`, their referenced images, and `init.ply` (or pass
`--init-ply` when training). For example, the Assignment 2 Lego scene can be
used with `--data ../cp4282-as2/data/lego`.

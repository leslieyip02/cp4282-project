# Image filtering during 3D Gaussian splatting training

This project asks whether filtering training images early in 3D Gaussian
splatting improves reconstruction on **unfiltered novel views**. It compares
seven target-image methods on NeRF-synthetic and Mip-NeRF 360 scenes. Each
method uses the same posed views and seeded camera sequence; PSNR and SSIM are
measured on held-out images.

## Cluster playbook

The same five steps work for a Mip-NeRF 360 scene such as bonsai and a
NeRF-synthetic scene such as Lego. The examples assume local `data/bonsai/` or
`data/lego/` and a copy of this repository at `~/cp4282/cp4282-project` on
the SoC cluster. See [dataset layout](docs/data.md) before copying a scene.
Replace `your_soc_id` and `REMOTE_PROJECT` if needed. Run steps 1 and 5
locally; run steps 2–4 on the cluster.

Before transferring data, sync the project code to the cluster. On the
cluster, from the repository root, install `requirements.txt` in its existing
`.venv` if needed and check imports:

```sh
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/check_training_env.py
```

1. **Copy training data to the cluster.** Set the destination from the local
   repository root:

   ```sh
   CLUSTER=your_soc_id@xlogin.comp.nus.edu.sg
   REMOTE_PROJECT=cp4282/cp4282-project
   ```

   For bonsai, copy the scene files but omit the optional reference checkpoint
   and predictions:

   ```sh
   ssh "$CLUSTER" "mkdir -p $REMOTE_PROJECT/data/bonsai"
   scp -r data/bonsai/nb-info.json data/bonsai/train_list.txt \
     data/bonsai/test_list.txt data/bonsai/images_2 data/bonsai/sparse_2 \
     "$CLUSTER:$REMOTE_PROJECT/data/bonsai/"
   ```

   For a synthetic Lego scene containing the three `transforms_*.json` files,
   referenced images, and `init.ply`, copy its whole directory instead:

   ```sh
   ssh "$CLUSTER" "mkdir -p $REMOTE_PROJECT/data"
   scp -r data/lego "$CLUSTER:$REMOTE_PROJECT/data/"
   ```

2. **Prepare filtered training targets.** Connect with `ssh "$CLUSTER"` and
   keep that session open while queuing steps 2–4 so the job ID variables
   remain available. From the cluster repository root, choose **one** scene:

   ```sh
   # Bonsai
   SCENE=bonsai
   DATA_DIR=data/bonsai
   CONFIG=config/real.yaml
   TRAIN_SCRIPT=scripts/slurm_train_real.sh
   ```

   ```sh
   # Synthetic Lego
   SCENE=lego
   DATA_DIR=data/lego
   CONFIG=config/synthetic.yaml
   TRAIN_SCRIPT=scripts/slurm_train_synthetic.sh
   ```

   Then create `logs/` and submit the CPU preparation job:

   ```sh
   cd ~/cp4282/cp4282-project
   mkdir -p logs
   prep_job=$(sbatch --parsable scripts/slurm_prepare_synthetic.sh "$DATA_DIR" "$CONFIG")
   ```

3. **Train all methods and seeds.** This queues the 21-task GPU array only if
   preparation succeeds. The training script checks CUDA inside each job.

   ```sh
   train_job=$(sbatch --parsable --dependency="afterok:$prep_job" \
     "$TRAIN_SCRIPT" "$DATA_DIR" "$CONFIG")
   ```

4. **Analyze completed runs.** This CPU job starts after the entire training
   array ends, including if some tasks failed. Check `missing_jobs.csv` before
   treating the results as complete.

   ```sh
   analysis_job=$(sbatch --parsable --dependency="afterany:$train_job" \
     scripts/slurm_analyze_synthetic.sh --scenes "$SCENE")
   printf 'prep=%s train=%s analysis=%s\n' "$prep_job" "$train_job" "$analysis_job"
   ```

   After the analysis job finishes:

   ```sh
   sacct -j "$analysis_job" --format=JobID,State,ExitCode
   cat outputs/analysis/missing_jobs.csv
   ```

   If the CSV has rows beyond its header, rerun those method/seed tasks and
   run analysis again. See [cluster notes](docs/cluster.md) for array indices,
   node diagnostics, and retries. For combined bonsai and Lego results, wait
   for both arrays and run analysis without `--scenes`; a second per-scene
   analysis would replace the first set of CSVs.

5. **Copy the analysis CSVs back to the local machine** once the analysis job
   has finished and `missing_jobs.csv` is clear.

   ```sh
   CLUSTER=your_soc_id@xlogin.comp.nus.edu.sg
   REMOTE_PROJECT=cp4282/cp4282-project
   mkdir -p outputs
   scp -r "$CLUSTER:$REMOTE_PROJECT/outputs/analysis" outputs/
   ```

Results are under `outputs/<scene>/<method>/seed_<seed>/` on the cluster;
`outputs/analysis/` contains paired runs, method summaries, validation curves,
and the missing-job report. If you need final PLYs or images locally, copy the
corresponding run directories as well.

## Further reading

- [Datasets](docs/data.md): bonsai download, expected files, optional reference
  bundle, and synthetic-scene layout.
- [Experiment and outputs](docs/experiment.md): methods, phase switch,
  evaluation, and result files.
- [Cluster notes](docs/cluster.md): Slurm options, GPU checks, retries, and
  synthetic-scene jobs.

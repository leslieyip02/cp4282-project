# Final submission checklist

The eight-scene synthetic matrix (72 runs) and nine-scene 20k/200k real matrix
(81 runs) are saved. The [report draft](final_report_draft.md) contains the
current numerical story. Finish the submission from the saved evidence before
starting new experiments.

## For the report

- [ ] Verify synthetic source manifests and initialization bytes if the original
  datasets become available. The saved run configs and final metrics have been
  checked; the report states the remaining provenance limit.
- [x] Add a sourced related-work comparison that distinguishes this study's
  fixed kernels from other splatting systems.
- [ ] Export matched held-out test crops with
  `scripts/slurm_export_test_crops.sh` showing representative gains, ties, and
  losses, including at least one synthetic scene. Keep the existing bonsai
  DSCF5648 validation image labelled as qualitative evidence only.
- [x] Add kernel-curve, paired scene/seed, and matched-hardware quality-versus-
  render-time figures from the existing run files.
- [ ] Insert the required format, references, and authorship or tool-use
  declaration from the course submission brief.

## Later extensions, if time permits

- [ ] Plot final versus best validation PSNR for each scene/kernel/seed and
  inspect late declines. Keep the fixed final checkpoint as the primary result.
- [ ] Extend the matched 100k-versus-200k comparison to another scene and test
  one change at a time for capacity, resolution, or training schedule.

Keep synthetic and real findings scoped to their own scene sets, image sizes,
backgrounds, initialization methods, capacities, and training budgets. The
project explores this renderer's behavior; it does not require a universally
better kernel.

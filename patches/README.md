# Third-party reproducibility patches

`rppg-toolbox-minimal-imports.patch` narrows eager package imports to the two datasets and five Table 1/2 baseline models used by this reproduction. It also defers optional POS pseudo-label imports until that feature is called, limits local preprocessing to two processes, exposes the DataLoader worker count for the local 8 GB machine, and numerically sorts UBFC/MMPD subjects for deterministic 6:2:2 splits. It does not modify model, loss, preprocessing mathematics, training, or metric code. This avoids requiring unrelated PhysMamba/SCAMPS/YOLO dependencies when running the selected supervised baselines.

Apply to a fresh official rPPG-Toolbox checkout with:

```bash
cd third_party/rPPG-Toolbox
patch -p1 < ../../patches/rppg-toolbox-minimal-imports.patch
```

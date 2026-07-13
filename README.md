# SuperSimpleNet

Official implementation of SuperSimpleNet.

[![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg)](https://huggingface.co/papers/2508.19060)

Original: [SuperSimpleNet : Unifying Unsupervised and Supervised Learning for Fast and Reliable Surface Defect Detection](https://arxiv.org/abs/2408.03143) - ICPR (International Conference on Pattern Recognition) 2024.

Extension to mixed supervision: [No Label Left Behind: A Unified Surface Defect Detection Model for all Supervision Regimes](https://arxiv.org/abs/2508.19060) - JIMS (Journal of Intelligent Manufacturing) 2025.

---

Unsupervised ICPR version of SuperSimpleNet is also available in [Anomalib](https://github.com/open-edge-platform/anomalib).

## Environment
```bash
conda create -n ssn_env python=3.10
pip install -r requirements.txt
```

The project uses wandb for logging, but it's optional. 
To enable this: uncomment wandb from requirements.txt to install and set `LOG_WANDB=True` at the top of train.py.

## Datasets

Follow the steps below to prepare all 4 datasets used in the paper. The code used to download datasets requires the env from the previous step.
If you already have the files prepared for a specific dataset, you can change the path in `eval.py`/`train.py`.

Note that for the VisA, the data needs to be correctly split and stored inside `visa/visa_pytorch`. 
This is handled automatically with the provided script. Ensure that the splits are correct if you are using existing VisA data.

1. Change directory to `./datamodules/setup/`.
2. Run `prepare_mvtec.py` to download and extract MVTec files.
3. Run `prepare_visa.py` to download, extract, and **prepare splits** for VisA files.
4. Run `prepare_ksdd2.py` to download and extract KSDD2 files.
5. To download SensumSODF, request a link on the official site.
   - Download the data from the link you receive [here](https://www.sensum.eu/sensumsodf-dataset/) and extract it to the dataset folder.
   
   - Then download SensumSODF 3-fold CV [split files](https://drive.google.com/file/d/1CrolrOHHm3wHaKu6JKqQ62qQGclwDKBM/view?usp=sharing). Extract them and place the `sensum_splits` folder inside the SensumSODF root.
   
   - If you are evaluating your method on SensumSODF, use the provided split files within the 3-fold CV setting for fair comparison.

The final structure should then look like this (case-sensitive):

```
datasets/
    KolektorSDD2/
        train/...
        test/...
        split_weakly_0.pyb
        ...
    SensumSODF/
        capsule/...
        softgel/...
        sensum_splits/
            capsule/
                0/...
                ...
            softgel/...
    mvtec/
         bottle/...
         ...
    visa/
        visa_pytorch/
            candle/
            ....
```


## Checkpoints

[![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm.svg)](https://huggingface.co/papers/2508.19060)

Checkpoints are available [on HuggingFace](https://huggingface.co/papers/2508.19060) and on [GDrive](https://drive.google.com/drive/folders/1bBKL7-xFgNrzOZVnED0jBgqT5poeYf0d). We recommend that you use the latest JIMS weights and JIMS code. 
Extract checkpoints into `./weights` path and ensure they are all inside a directory with run_id 0: 
```
./weights/
   0/
      ksdd2/
         ksdd2/
            <ratio> (e.g. 246)/
               weights.pt
      sensum/
      mvtec/
      visa/
```

The original ICPR checkpoints don't have the `ratio` subdirectory, 
while the latest JIMS version also has ratio subdirectory for each mixed supervision scenario.

We report an average of 5 runs in our paper, but the weights from the link are only for a single run.
Therefore, the results won't exactly match the ones reported in the paper.

We also include the reported mean and std as a json inside `paper_results` for all datasets in the paper.

## Evaluate

Evaluate using the checkpoints:

```bash
python eval.py
```

Slurm script `run_slurm_eval.sh` is also provided to execute evaluation on a slurm based system.

---
Config for the model and datasets is contained within the eval.py file. 

## Train

Train the model:

```bash
python train.py <dataset_name>
```
Possible dataset names are: `mvtec`, `visa`, `sensum`, and `ksdd2`.

Slurm script `run_slurm_train.sh` is also provided to execute training on a slurm based system.

---

Config for the model and datasets is contained within train.py file. If you want to modify training params, change the values there.

**The ICPR and JIMS versions mostly differ in 3 parameters:** 
`dt` (distance transform), `dilate` (label dilation), and `adapt_cls_feat` (use adapted features for cls head or not).

We recommend taking the MVTec parameters when training on your own **unsupervised** dataset and SenumSODF parameters for **supervised** dataset.

## ViT / foundation-model backbones

In addition to torchvision CNNs, the feature extractor supports frozen vision
foundation models (adapted from [AnomalyVFM](https://github.com/vicoslab/AnomalyVFM)),
selected purely by the `--backbone` name:

```bash
python train.py --dataset mvtec --backbone dinov2_vitl14_reg --image-size 448 448 --feat-scale 2 --batch 16 --amp
# or use the provided configs:
python train.py --config configs/mvtec_dinov2.txt
python train.py --config configs/custom_unsup_dinov2.txt --data-root /path/to/data
```

| Backbone | Patch | Recommended `--image-size` | Extra requirements |
|---|---|---|---|
| `dinov2_vitl14_reg` (also `_vitb14_reg`, `_vits14`, …) | 14 | 448 448 (or 252/518) | none (torch.hub download) |
| `dinov3_vitl16` | 16 | 512 512 | local [dinov3 repo](https://github.com/facebookresearch/dinov3) clone + gated weights: `--dinov3-path`, `--dinov3-weights` |
| `radio_v2.5-l` | 16 | 512 512 | `pip install timm einops` |
| `clip_vitl14_336` | 14 | 448 448 | `pip install transformers` |
| `siglip2_so400m` | 16 | 512 512 | `pip install transformers` |
| `tipsv2_l14` | 14 | 448 448 | `pip install transformers sentencepiece` (uses `trust_remote_code`) |

Notes:
- Optional dependencies for all ViT backbones at once: `pip install -r requirements-vit.txt`.
- The image size must be divisible by the backbone's patch size (14 or 16) — training fails fast with a suggestion otherwise. KSDD2's fixed native resolution (640x232) is incompatible with ViT backbones; Sensum works with patch-16 backbones only.
- The backbone is always **frozen** (same as the CNN path); only the adaptor and the seg/cls heads train, and backbone weights are excluded from checkpoints.
- `--feat-scale 2` doubles the feature-grid resolution (e.g. 32x32 → 64x64 at 448², matching the CNN path's working grid) at ~4x head memory; `--vit-layers` selects intermediate ViT blocks (dinov2/dinov3 only, e.g. `--vit-layers 17 23` for ViT-L multi-layer features).
- Default anomaly-generation hyperparameters (`--noise-std`, `--perlin-thr`) were tuned for the CNN feature space and may need retuning for ViT features.
- Smoke test any backbone without a dataset: `python tests/smoke_test_backbones.py dinov2_vitl14_reg`.

See `VIT_BACKBONE_INTEGRATION.md` for the full design, architecture comparison with AnomalyVFM, and expectation setting.

## Performance benchmark

Use the code inside `./perf` to evaluate performance metrics (inference speed, throughput, memory consumption, flops):

```bash
python perf_main.py <gpu_model>.
```

Slurm script `run_slurm_perf.sh` is also provided to execute benchmark on slurm based system.

Note that the results in paper are obtained with AMD Epyc 7272 CPU and NVIDIA Tesla V100S GPU and might therefore differ from the ones obtained on your system.

We also include the performance results from the paper inside `paper_results`.

## Citation

If you find our work useful, please cite our works:

```bibtex
@InProceedings{rolih2024supersimplenet,
  author={Rolih, Bla{\v{z}} and Fu{\v{c}}ka, Matic and Sko{\v{c}}aj, Danijel},
  booktitle={International Conference on Pattern Recognition}, 
  title={{S}uper{S}imple{N}et: {U}nifying {U}nsupervised and {S}upervised {L}earning for {F}ast and {R}eliable {S}urface {D}efect {D}etection},
  year={2024}
}

@article{rolih2025supersimplenet2,
  author={Rolih, Bla{\v{z}} and Fu{\v{c}}ka, Matic and Sko{\v{c}}aj, Danijel},
  journal={Journal of Intelligent Manufacturing}, 
  title={No Label Left Behind: A Unified Surface Defect Detection Model for all Supervision Regimes},
  year={2025}
}
```

## Acknowledgement

Thanks to [SimpleNet](https://github.com/DonaldRR/SimpleNet) for great inspiration.

# OCT retinal disease classification

Four-way classification of **retinal OCT** slices with **`torchvision` DenseNet-121** (ImageNet-1K weights), training pipelines in **`notebooks/oct-classification.ipynb`**, experiment tracking with [**Weights & Biases**](https://wandb.ai/), optional **Bayesian hyperparameter sweeps**, and **Grad-CAM** overlays for interpretability.

## Notebook overview

| Piece | Details |
|--------|--------|
| **Model** | `densenet121` (`IMAGENET1K_V1`), replaced classifier head: `Dropout` + `Linear` → **4 classes**. |
| **Modes** | **Frozen**: `AdamW` on classifier head only (full backbone receives gradients but only head weights update). **Finetune**: differential LR on **denseblock4** + **norm5** vs classifier head. |
| **Data** | `torchvision.datasets.ImageFolder` on `train/` and `test/`; **85/15 stratified** train/validation split from the training folder (`random_state=42`). |
| **Input** | Resize **224×224**, ImageNet normalization; train-time **random horizontal flip** and **±10° rotation**. |
| **Metrics** | Accuracy, **macro F1**, confusion matrix; extended test report with macro **precision** and **recall** (fine-tuned checkpoint section). |
| **MLOps** | W&B project **`oct-densenet`**; sweeps (**Bayes**, maximize **`val_f1`**) over `batch_size`, `lr`, `backbone_lr`, `weight_decay`, `dropout`. |
| **Interpretability** | **GradCAM** on `model.features.denseblock4.denselayer16.conv2` with `ClassifierOutputTarget`. |

Default training constants in the notebook include **`DEFAULT_BATCH_SIZE = 32`**, **`DEFAULT_EPOCHS = 8`**, **`NUM_CLASSES = 4`**, **`IMAGE_SIZE = 224`**. Reported accuracy in the mid-**90%** range and strong macro **F1** depend on your runs—check **W&B** and executed notebook outputs.

## Dataset layout

The notebook expects a root directory containing **`train/`** and **`test/`**, each with **one subdirectory per class** (labels = folder names).

On **Kaggle**, `DATA_DIR` is set to:

`/kaggle/input/datasets/lakshyashrivastava08/oct2017-kermany/OCT`

For **local** runs, change `DATA_DIR` in the notebook to your unpacked dataset path (same `train` / `test` structure). **Do not commit** raw image data to this repository.

The widely used **OCT2017** collection is described in the literature (e.g. Kermany et al., *Cell* **172**(5), 2018 — “Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning”). Use the dataset license and citation required by the bundle you download (e.g. from [Kaggle](https://www.kaggle.com/) or the original data release).

## Repository layout

| Path | Contents |
|------|----------|
| [`notebooks/oct-classification.ipynb`](notebooks/oct-classification.ipynb) | Main training, evaluation, W&B sweep, and Grad-CAM code (exported from Kaggle; Python **3.12**). |
| [`paper/`](paper/) | Course / project write-up (**PDF**) when you add it. |

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1    # Windows PowerShell
pip install -r requirements.txt
```

Install **GPU-enabled** `torch` / `torchvision` if needed using the selector at [pytorch.org](https://pytorch.org/get-started/locally/).

**W&B:** run `wandb login` before experiments that call `wandb.init`. Sweep cells call `wandb.sweep` / `wandb.agent`; reduce `count` or comment those cells if you only want single runs.

## Author

[Lakshya Shrivastava](https://github.com/LakshyaShrivastava) · [LinkedIn](https://www.linkedin.com/in/lakshya-shrivastava0803)

# OCT retinal disease classification

Deep learning for **optical coherence tomography (OCT)** retinal disease classification using **PyTorch** and **DenseNet-121**, with experiment tracking (**Weights & Biases**), hyperparameter sweeps, and **Grad-CAM** interpretability.

## Results (summary)

- Accuracy ~**93%**; strong macro **F1** (see notebook / paper for exact numbers on your split).
- Error and calibration analysis with confusion matrix and class-level failure modes (Grad-CAM).

## Repository layout

| Path | Contents |
|------|----------|
| [`notebooks/`](notebooks/) | Kaggle-exported **`.ipynb`** (primary training / eval notebook). |
| [`paper/`](paper/) | Project **write-up** (`paper.pdf` or similar). |

Add your files there, then commit (see **Setup** below).

## Dataset

Describe the dataset here (name, source, citation). **Do not commit** restricted raw data; link to the official download or competition page.

## Environment

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows
pip install -r requirements.txt
```

Adjust `requirements.txt` to match the notebook (PyTorch build, CUDA vs CPU).

## Author

[Lakshya Shrivastava](https://github.com/LakshyaShrivastava) · [LinkedIn](https://www.linkedin.com/in/lakshya-shrivastava0803)

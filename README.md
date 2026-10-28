# GestureSense

GestureSense is an educational static ASL alphabet recognizer. It extracts MediaPipe hand landmarks from the Kaggle ASL Alphabet images, trains classic scikit-learn classifiers on engineered geometric features, and provides image/camera inference views.

## Dataset

The public [Kaggle ASL Alphabet dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) page lists 87,000 200x200 training images across 29 classes and a 29-image test set. Its downloadable ZIP is listed as 1,100,887,034 bytes (about 1.10 GB). Dataset access may require a Kaggle account and acceptance of its terms/license.

Download from the project root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install kaggle
kaggle datasets download -d grassknoted/asl-alphabet -p data/raw --unzip
```

Or download and extract the ZIP manually under `data/raw/`. The extractor finds nested class folders. Python 3.10 or 3.11 is recommended; the pinned dependencies target Python 3.11.

The repository includes a sampled, processed landmark dataset at `data/processed/landmarks.csv` and two trained inference models at `outputs/models/`. The original image archive is intentionally not included: it is large and access is subject to Kaggle's terms. The processed CSV is sufficient to retrain models without downloading images or rerunning landmark extraction.

## Pipeline

1. `python -m src.extract_landmarks` samples up to 600 images/class (set `SAMPLES_PER_CLASS` in `config.py`), excludes J, Z, `del`, `nothing`, and `space` by default, detects hands, and writes `data/processed/landmarks.csv` plus detection counts.
2. `python -m src.eda` creates class-count, detection-rate, feature-correlation, and normalized-skeleton plots.
3. `python -m src.train_models` trains KNN, Logistic Regression, Decision Tree, Random Forest, and RBF SVM. Each uses a StandardScaler pipeline, stratified 80/20 split, and 5-fold parameter search over the training split only. The holdout is used only for final evaluation. Models and metrics are saved under `outputs/`.
4. `python -m src.live_demo` starts the OpenCV webcam demo (`q` quits, `m` switches model, `s` saves a screenshot).
5. `streamlit run app.py` opens the Streamlit app. Camera and upload pages classify a captured image and let you select a saved model; KNN is the default camera choice because dataset cross-validation does not identify which classifier best handles a new webcam signer. The app also displays metrics and provides a basic word builder.

## Streamlit deployment

Deploy `app.py` from the repository root with Python 3.11 and install `requirements.txt`. Before deploying, ensure these files are included in the GitHub commit (they are allow-listed by `.gitignore`):

- `outputs/models/best_model.joblib` and `outputs/models/knn.joblib` are required for image and camera inference.
- `outputs/metrics.json` enables the model-comparison page.
- `data/processed/landmarks.csv` is only needed if you want to retrain models; it is not loaded during app startup or inference.

The raw image dataset and a training run are not required to run inference. If these generated files exist locally but are missing from GitHub, add them to a commit and push it before deploying; Streamlit Cloud only receives files tracked by the repository.

All training and inference code uses the same feature transformation: wrist-centered and hand-size-normalized XYZ coordinates, five fingertip-to-wrist distances, five bend angles, and four thumb-to-other-fingertip distances (77 values total).

## Tests

```powershell
python -m unittest discover -s tests -v
```

## Outputs

- `outputs/models/`: per-classifier models and `best_model.joblib`
- `outputs/metrics.json`: per-model holdout metrics, classification reports, confusion matrices, timings, and CV scores
- `outputs/plots/`: confusion matrices, model comparison, and EDA plots

## Results and limitations

No performance result is published until the pipeline is run on the downloaded data; expected accuracy must be measured, not assumed. Image-level random splits may leak near-duplicate frames and background/signer characteristics, inflating results. For a stronger evaluation, collect data from multiple signers and split by signer or recording session, then evaluate on unseen webcam data. This prototype classifies static handshapes only: J and Z require movement and are omitted. It is not a complete ASL translator.

## Project layout

```text
config.py
app.py
src/       extraction, features, EDA, training, evaluation, inference, webcam
views/     Streamlit pages
tests/     feature and model-loading tests
data/raw/  downloaded image dataset
outputs/   models, metrics, plots
```

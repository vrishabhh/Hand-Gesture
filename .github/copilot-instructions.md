You are helping build GestureSense, an ASL alphabet recognizer using MediaPipe hand landmarks and scikit-learn.

- Use Python 3.10 or 3.11.
- Dataset images belong in data/raw/, organized into class folders. Extract landmarks to data/processed/landmarks.csv; do not train on image pixels.
- Exclude J, Z, del, nothing, and space by default. Keep the exclusion list and sample limit in config.py.
- Use the same wrist-centered, hand-size-normalized feature function in training and inference.
- Compare KNN, Logistic Regression, Decision Tree, Random Forest, and RBF SVM using a stratified holdout split and 5-fold GridSearchCV on training data only.
- Save models, metrics, and plots under outputs/. Do not use the test split during model selection.
- Use type hints, docstrings, logging, and tests. Prefer surgical, tested changes and explain dataset leakage limitations.

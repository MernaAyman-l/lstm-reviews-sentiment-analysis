# Product Review Sentiment Classification with LSTM

A Python NLP project that classifies product reviews as **positive** or **negative** using an LSTM neural network built with TensorFlow/Keras. The implementation is provided as a Jupyter Notebook.

> **Project scope:** This is an educational experiment using the first 500 rows of the Amazon Fine Food Reviews dataset, not a production sentiment-analysis service.

## Project Workflow

1. **Load review data:** Read the `Reviews.csv` file and select the first 500 records.
2. **Clean review text:** Convert text to lowercase, remove non-alphabetic characters, tokenize it with NLTK, and remove English stop words.
3. **Create sentiment labels:** Map ratings above 3 to positive (`1`) and ratings of 3 or below to negative (`0`).
4. **Tokenize and pad:** Keep a vocabulary of up to 5,000 tokens and pad/truncate sequences to 100 tokens.
5. **Split the data:** Use an 80/20 train/test split (400 training examples and 100 test examples in the saved run).
6. **Build the LSTM:** Use an embedding layer, an LSTM layer, and a sigmoid output for binary classification.
7. **Train and evaluate:** Train for five epochs and report classification accuracy on the held-out split.
8. **Visualize:** Plot training/validation accuracy and loss, and the sentiment-label distribution.

## Model Architecture

| Layer | Configuration | Purpose |
| --- | --- | --- |
| Embedding | Input dimension: 5,000; output dimension: 32 | Maps token IDs to dense vectors |
| LSTM | 32 units | Learns sequential patterns in review text |
| Dense | 1 unit, sigmoid activation | Outputs a positive-class probability |

The model is compiled with the Adam optimizer and binary cross-entropy loss, and trained for 5 epochs with a batch size of 32.

## Results

The notebook's saved run reports **81% test accuracy** on a test split of 100 reviews.

This is the recorded result from the current notebook run, not a claim of generalization to other datasets. The dataset subset is small, and the current notebook fits its tokenizer before the train/test split and uses the test split as validation during training. These choices can introduce information leakage and make the reported score optimistic. For a more reliable evaluation, split the raw reviews first, fit the tokenizer only on training text, and reserve the test set for final evaluation.

## Dataset

The notebook expects a file named `Reviews.csv`, based on the Amazon Fine Food Reviews dataset. The dataset is **not included in this repository**, so you need to obtain it separately and place it in the project directory (or update the path in the notebook).

Only the first 500 rows are currently loaded:

```python
df = pd.read_csv("Reviews.csv").head(500)
```

The label rule is:
- `Score > 3` → Positive (`1`)
- `Score <= 3` → Negative (`0`)

## Technologies

- Python
- TensorFlow / Keras
- NLTK
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Jupyter Notebook

## Repository Structure

```text
lstm-reviews-sentiment-analysis/
├── assignment_NLP.ipynb
└── README.md
```

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/MernaAyman-l/lstm-reviews-sentiment-analysis.git
cd lstm-reviews-sentiment-analysis
```

### 2. Create and activate a virtual environment (recommended)

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install pandas numpy tensorflow scikit-learn nltk matplotlib seaborn jupyter
```

### 4. Add the dataset

Download the Amazon Fine Food Reviews dataset separately, then place `Reviews.csv` in the repository's root directory. The dataset is not committed here.

### 5. Download NLTK resources

Run this in Python or in a notebook cell:

```python
import nltk
nltk.download("stopwords")
nltk.download("punkt")
```

### 6. Launch the notebook

```bash
jupyter notebook
```

Open `assignment_NLP.ipynb` and run its cells in order. If you store the CSV elsewhere, update the data-loading path accordingly.

## Current Limitations & Next Improvements

- Use the full dataset or a larger, documented sample instead of only 500 rows.
- Split data before fitting the tokenizer to prevent train/test information leakage.
- Use a validation split from training data and keep the test set for final evaluation only.
- Report precision, recall, F1-score, and a confusion matrix in addition to accuracy.
- Set random seeds for reproducibility and consider class imbalance.
- Save the trained model and tokenizer for repeatable inference.
- Add a small prediction function for classifying new review text.

## Project Note

This repository contains the notebook implementation and its saved outputs. It does not currently include a deployed application, a saved model file, or the source review dataset.

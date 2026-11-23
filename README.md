# Product Reviews Sentiment Analysis with LSTM

A Jupyter Notebook (`assignment_NLP.ipynb`) that performs end-to-end text preprocessing and binary sentiment classification on product reviews (`Reviews.csv`, first 500 rows) using an **LSTM** neural network built with TensorFlow/Keras.

## Project Structure & Key Files
* `assignment_NLP.ipynb`: Jupyter Notebook covering NLTK setup, dataset loading and text cleaning, sequence tokenization/padding, LSTM model building and training, test evaluation, and visualization plots.

## Pipeline & Functionality
* **Text Preprocessing (`clean_text`):** Converts text to lowercase, removes non-alphabetic characters with regex (`[^a-zA-Z\s]`), tokenizes via `nltk.word_tokenize`, and filters out English stop words.
* **Sentiment Labeling & Sequence Preparation:** Maps `Score > 3` to positive (`1`) and `<= 3` to negative (`0`), tokenizes cleaned text (`num_words=5000`), pads sequences to length `100`, and splits data `80/20` (`400` training samples, `100` testing samples).
* **LSTM Model & Training:** Builds a `Sequential` model with `Embedding(5000, 32)`, `LSTM(32)`, and a `Dense(1, activation='sigmoid')` output layer, compiled with `adam` and `binary_crossentropy`, trained for `5` epochs (`batch_size=32`), achieving `0.81` test accuracy.
* **Visualizations:** Plots training vs. validation accuracy over epochs, training vs. validation loss over epochs, and sentiment class distribution (`Negative` vs. `Positive`).

## Built With
* **Python**
* **TensorFlow / Keras** (`Tokenizer`, `pad_sequences`, `Sequential`, `Embedding`, `LSTM`, `Dense`)
* **NLTK** (`stopwords`, `word_tokenize`)
* **Pandas, NumPy & scikit-learn** (`train_test_split`)
* **Matplotlib & Seaborn**

## How to Run
1. Install the required libraries (`pandas`, `numpy`, `tensorflow`, `scikit-learn`, `nltk`, `matplotlib`, `seaborn`) and download NLTK `stopwords` and `punkt` resources.
2. Update the `Reviews.csv` file path in the data-loading cell so it points to your dataset location (or place `Reviews.csv` in your working directory).
3. Open `assignment_NLP.ipynb` in Jupyter or VS Code and run the cells sequentially.
```[cite: 15]


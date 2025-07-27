# ELECTRA Sentiment Analysis - README

## Project Overview
This project demonstrates sentiment analysis using the ELECTRA (Efficiently Learning an Encoder that Classifies Token Replacements Accurately) model. The goal is to classify product reviews as either positive (1) or negative (0) based on their text content. The model is trained on a subset of Amazon product reviews and evaluated for accuracy, precision, recall, and F1 score.

## Key Features
- Model: google/electra-base-discriminator (pre-trained ELECTRA model fine-tuned for binary classification)
- Dataset: Amazon product reviews (Reviews.csv), filtered to exclude neutral ratings (Score = 3)
- Training: 2 epochs with a batch size of 16
- Evaluation: Metrics computed on a held-out validation set (20% of data)

## Results
- Accuracy: 0.956 (95.6%)
- Precision: 0.974
- Recall: 0.974
- F1 Score: 0.974
- Training Time: 186.76 seconds
- Testing Time: 7.07 seconds

### Interpretation of Results
- Accuracy (95.6%): The model correctly predicts sentiment for 95.6% of reviews.
- Precision (97.4%): When the model predicts "positive," it is correct 97.4% of the time.
- Recall (97.4%): The model identifies 97.4% of all actual positive reviews.
- F1 Score (97.4%): A balanced measure of precision and recall, indicating strong performance.
- Training Time: The model trains in ~187 seconds (3.1 minutes).
- Testing Time: Predictions on the validation set take ~7 seconds.

## Code Structure
1. Install Dependencies
   - Installs transformers, datasets, and scikit-learn.

2. Load & Preprocess Data
   - Reads Reviews.csv, filters neutral reviews, and converts ratings to binary labels (0 = negative, 1 = positive).
   - Samples 5,000 reviews for faster experimentation.

3. Train-Validation Split
   - 80% for training, 20% for validation.

4. Tokenization
   - Uses ElectraTokenizer to convert text into numerical tokens (max length = 128).

5. Dataset Preparation
   - Wraps tokenized data into a PyTorch Dataset class for efficient training.

6. Model Loading
   - Initializes ElectraForSequenceClassification with 2 output classes (positive/negative).

7. Training Configuration
   - Sets hyperparameters (batch size = 16, epochs = 2).
   - Disables Weights & Biases (W&B) logging for simplicity.

8. Training & Evaluation
   - Trains the model and computes accuracy, precision, recall, and F1 score.

## How to Run
1. Requirements
   - Python 3.6+
   - PyTorch
   - Hugging Face transformers
   - pandas, scikit-learn

2. Steps
   - Upload Reviews.csv to the working directory.
   - Run the notebook cells sequentially.

## Possible Improvements
1. Hyperparameter Tuning
   - Adjust learning_rate, batch_size, or num_epochs for better performance.
2. Full Dataset Training
   - Train on the entire dataset (not just 5,000 samples) for higher accuracy.
3. Deployment
   - Save the trained model and deploy it as an API for real-time sentiment analysis.

## References
- ELECTRA Paper: https://arxiv.org/abs/2003.10555
- Hugging Face Transformers: https://huggingface.co/transformers/
- Scikit-learn Metrics: https://scikit-learn.org/stable/modules/classes.html#classification-metrics

## License
This project is open-source under the MIT License.

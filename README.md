# Wine Quality Prediction via Active Learning with Gaussian Processes

This project implements an **active learning framework** using **Bayesian Gaussian Process classifiers** to predict wine quality from physicochemical attributes. It includes multiple querying strategies and a fully interactive Streamlit interface to visualize learning progression.

---

## Project Motivation

In many machine learning settings, labeling data is expensive. **Active Learning** offers a solution by allowing the model to selectively query the most informative data points to label — reducing labeling costs while maintaining performance.

This project uses the **Wine Quality Dataset** from [Kaggle](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009) to demonstrate:

- Gaussian Process (GP) classification
- Query strategies like entropy, margin sampling, least confidence, and random
- Evaluation of learning curves across strategies
- A visual and interactive UI for live experimentation

---

## Features

- Active Learning loop from scratch
- **Bayesian modeling** using Gaussian Processes
- Multiple query strategies:
  - Entropy-based
  - Least confidence
  - Margin sampling
  - Random baseline
- Handles class imbalance using **SMOTE**
- Configurable pipeline via `config.yaml`
- Modular structure for extensibility
- Streamlit app for interactive visualization

---

## Project Structure
```
wine_GP_active_learning/
│
├── data/ # Raw data (e.g., winequality-red.csv)
├── results/ # Learning curve outputs and history logs
│
├── src/
│ ├── app.py # Streamlit app
│ ├── active_learning.py # Active learning loop logic
│ ├── evaluate_results.py # Evaluation and plot generation
│ ├── models.py # GP model builder
│ ├── preprocessing.py # Data loading, binning, SMOTE
│ └── run_active_learning.py# CLI to run active learning loop
│
├── config.yaml # Central configuration for preprocessing, GP, and learning
├── README.md
└── requirements.txt
```

## How It Works
Starts with a small seed set of labeled examples.

Trains a Gaussian Process classifier on those.

Scores all unlabeled data using a query strategy (e.g., entropy).

Queries a batch of most informative points.

Adds them to the labeled set and repeats.

Tracks metrics (Accuracy, F1) over rounds.

## Evaluation
    python src/evaluate_results.py \
      --inputs results/history_entropy.json results/history_random.json \
      --output results/learning_curve.png

## Streamlit Dashboard
  streamlit run src/app.py

  You can:
  > Choose query strategies
  
  > Watch the learning curve evolve live
  
  > Switch strategies between rounds
  
  > See labeled sample counts grow

## Installation
git clone https://github.com/arni-tech/wine-gp-active-learning.git
cd wine-gp-active-learning
pip install -r requirements.txt
Requires Python 3.8+
Recommended: Use a virtual environment or conda environment.


## TODO (Future Work)
 Add support for more kernels and hyperparameter tuning

 Evaluate on additional datasets

 Export model checkpoints

 Add uncertainty histogram visualizations in Streamlit





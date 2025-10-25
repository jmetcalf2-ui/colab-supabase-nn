# Colab Supabase NN Project

This repository contains code and documentation for training baseline logistic regression and neural network models on data fetched from a Supabase database.

## Project structure

- `notebooks/`: the Colab notebook `supabase_nn_project_2025-10-25.ipynb` used for data exploration, modeling, and evaluation.
- `data/`: cached exports of the Supabase tables (e.g., `leads.csv`, `leads_supplements.csv`).
- `models/`: saved model weights (`baseline_model.joblib`, `nn_model.pth`) and any preprocessing artifacts.
- `outputs/`: visualizations such as training and validation loss curves.
- `src/`: (optional) scripts or helper modules.

## Data source

Data is pulled from a Supabase project via the Python `supabase` client. Tables used include:
- **leads**: basic lead information, including `lead_id`, `full_name`, and `primary_role`.
- **leads_supplements**: supplemental attributes by `lead_id`; aggregated to a count of supplements per lead.
- **contacts**: excluded from modeling due to missing `lead_id` in the table snapshot.

The dataset used for modeling has 810 rows and 18 columns after filtering out label classes with fewer than two samples.

## Modeling

Two models are trained:
1. **Baseline Logistic Regression** – uses scikit‑learn’s `LogisticRegression` in a pipeline with one‑hot encoding and scaling. Provides a baseline for comparison.
2. **Neural Network** – a simple feed‑forward network implemented in PyTorch with one hidden layer. Trained for 15 epochs with Adam optimizer and cross‑entropy loss.

Performance metrics (accuracy and F1) are printed at the end of training. The neural network achieved approximately 0.88 accuracy and F1 on the test set.

## Running

To reproduce or modify the models:
1. Open the Colab notebook located in the `notebooks/` directory (available in your Google Drive project).
2. Mount Google Drive in Colab.
3. Run each cell sequentially to install dependencies, fetch data, preprocess, train, and evaluate the models.
4. Optional: modify hyperparameters (hidden units, epochs, learning rate) to experiment.

## Requirements

See `requirements.txt` for a list of Python packages used.

## Notes

- This repository does not include sensitive Supabase credentials or API keys.
- The project is intended for demonstration and educational purposes.

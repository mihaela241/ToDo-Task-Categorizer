# To-Do Task Categorizer

A small machine-learning project that predicts a task's **priority** (`low`, `medium`, `high`) and its **estimated duration category** (`short`, `medium`, `long`) from a text description. The dataset includes Bulgarian and English examples.

## Files
- `main.py`: trains the models, prints evaluation metrics, and offers an interactive prompt.
- `data.csv`: examples with task description, priority, and estimated minutes.
- `requirements.txt`: required Python packages.
- `Pismena_Teoria.docx` / `.pdf`: project theory document (the original PDF is retained).

## Setup and run
1. Install Python 3.9+.
2. In this folder, run: `pip install -r requirements.txt`
3. Run: `python main.py`
4. Enter a task description. Type `quit`, `exit`, or `q` to stop.

## How it works
1. **Load and validate:** reads CSV with pandas, checks required columns, removes missing/invalid rows and normalizes priority labels.
2. **Create time labels:** converts minutes into three classes: `short` (under 30 min), `medium` (30–90 min inclusive), and `long` (over 90 min).
3. **Convert text to features:** TF-IDF turns words and word pairs into numerical features. It weighs terms according to how informative they are in the dataset.
4. **Train classifiers:** two independent Logistic Regression pipelines learn patterns: one predicts priority and the other duration class. `class_weight="balanced"` helps compensate for unequal class frequencies.
5. **Evaluate:** where the data supports it, the code makes a stratified 75/25 train/test split and reports accuracy and a per-class precision/recall/F1 report. The evaluation is only an estimate, not a guarantee. If a class has too few examples, evaluation is skipped safely.
6. **Train final models:** each model is trained on the entire dataset for interactive predictions.
7. **Predict:** type a new task to get predicted priority and duration class.

## Model limitations
- This is a small educational dataset; predictions are not authoritative and may be inaccurate.
- Duration categories are based on labels supplied in the CSV, not on measured task times.
- TF-IDF does not understand meaning like a human; it learns associations from examples.
- Add representative examples and keep labels consistent to improve results.
- The program predicts categories, not a precise time estimate.

## CSV format
`description,priority,expected_minutes`

Priority values should be `low`, `medium`, or `high`. `expected_minutes` must be a numeric value.


## Interactive estimated time and feedback

The program now predicts a numeric time estimate in minutes using a TF-IDF + Ridge regression pipeline, alongside priority and duration category. After each task it asks whether the predictions are correct. If the user says no, they can enter corrected priority and minutes. The task and corrected labels (or confirmed prediction) are appended to `data.csv`, and the models are retrained immediately. Keep a backup of the CSV if you want to undo feedback.

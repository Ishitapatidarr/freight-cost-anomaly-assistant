# Freight Cost Anomaly Assistant

A Python data-analysis project that identifies unusual weekly freight costs on transport routes and explains possible causes using the supplied shipment records and context notes. It also includes a small command-line question-answering feature that returns findings from the generated report.

## Problem statement

Freight costs can change from week to week because of disruptions, route conditions, surcharges, or other operational factors. Looking at individual shipment records makes it difficult to spot sustained or unusual changes and connect them to available context notes.

The goal of this project is to:

- calculate a comparable weekly freight-cost metric for each route;
- compare a route's cost with its own earlier history and with similar routes in the same week;
- flag potentially unusual increases;
- connect findings to context notes only when route, date, and note content provide support;
- produce a structured CSV report and answer simple questions from the generated findings.

## What the project does

1. **Loads the input data** from `data/`.
2. **Prepares shipment records** and groups them by route and Monday-starting week.
3. **Calculates cost per tonne-kilometre** using total freight cost divided by the total of `quantity × distance` for the grouped shipments.
4. **Builds comparison baselines** using up to the previous eight weeks of available route history and the same-week peer comparison for similar route types. The current week is excluded from its own historical baseline.
5. **Flags potential anomalies** using a default relative threshold of 20% and a minimum positive rise of 5% compared with the previous observed week. These thresholds can be changed from the command line.
6. **Retrieves supporting context notes** using TF-IDF candidate ranking plus deterministic route/date/content checks. A note is attached only when it passes the project's evidence checks; unsupported findings are not given a made-up explanation.
7. **Writes the output report** to `output/freight_anomalies.csv` and creates run/evaluation information in the `output/` folder.
8. **Answers questions** about the generated findings through a simple command-line interface.

> The current implementation uses deterministic Python logic and TF-IDF retrieval. It does not call a hosted LLM API, so the run log reports zero LLM calls, tokens, and API cost. A missing supporting note means the cause is not established by the available evidence; it does not prove that no real-world cause existed.

## Tech stack

- **Python** — pipeline and command-line interface
- **Pandas** — data loading, cleaning, grouping, and CSV output
- **NumPy** — numerical operations
- **scikit-learn** — TF-IDF vectorization and cosine similarity for note retrieval
- **pytest** — automated tests
- **CSV** — input data and generated report format

## Project structure

```text
freight_cost_assistant/
├── data/
│   ├── shipment_records.csv
│   ├── context_notes.csv
│   └── sample_output_format_v2.csv
├── src/
│   ├── __init__.py
│   └── pipeline.py
├── tests/
│   └── test_pipeline.py
├── output/
│   ├── freight_anomalies.csv
│   ├── run_log.json
│   ├── evaluation_report.txt
│   ├── reproducibility_check.txt
│   └── qa_demo.txt
├── requirements.txt
├── .gitignore
└── README.md
```

## Requirements

- Python 3.10 or newer recommended
- `pip`
- Windows PowerShell, macOS Terminal, or Linux shell

## Installation

### Windows PowerShell

Open PowerShell in the project root—the folder containing `src`, `data`, and `requirements.txt`—then run:

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation, you can use this command for the current terminal session, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

If the `python` command is unavailable, install Python from <https://www.python.org/downloads/> and enable **Add Python to PATH** during installation. Restart the terminal afterwards.

### macOS / Linux

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the project

Run the pipeline from the project root:

```bash
python -m src.pipeline
```

The generated anomaly report is saved to:

```text
output/freight_anomalies.csv
```

The report follows the supplied output format and includes these columns:

- `route`
- `week_of`
- `cost_per_tonne_km`
- `vs_own_history`
- `vs_similar_routes`
- `flagged`
- `matched_note_id`
- `reason`

To change the anomaly threshold (for example, to 25%), run:

```bash
python -m src.pipeline --threshold 0.25
```

## Ask a question about the results

Use `--ask` to query the generated report. For example:

```bash
python -m src.pipeline --ask "Why did Chennai-Bangalore get pricier in March?"
python -m src.pipeline --ask "Did an earthquake increase Chennai-Bangalore freight costs?"
```

Answers are grounded in the pipeline's generated findings and available context notes. The system should not treat a cause as confirmed when the supplied notes do not support it.

## Run tests

From the project root, with the virtual environment activated:

```bash
pytest -q
```

## Evaluation and reproducibility

The `output/` folder contains run logs, evaluation information, a reproducibility check, and a sample Q&A transcript. Re-run the pipeline and tests if you change the code or input data, then inspect the generated report and logs before submitting the project.

## Assumptions and limitations

- A route/week is evaluated using the available shipment records; weeks without data may not appear.
- Historical comparisons use only earlier weeks, not the current week's value.
- Peer comparisons depend on the route-type information available in the input data.
- Context-note retrieval is evidence matching, not proof of causation. A note must be relevant to the route and effective time window and contain supporting information.
- The question-answering feature searches the generated anomaly report; it is not a general-purpose web search system and does not independently verify external events.
- The default anomaly thresholds are project settings, not universal industry standards.

## How to Download and Run the Project Locally

1. Download the Project

Open the GitHub repository.

Click the green Code button.

Select Download ZIP.

Extract the downloaded ZIP file.

Open the extracted project folder in VS Code.

2. Install Python

Make sure Python 3.10 or a compatible version is installed on your system.

Check the installation by running the following command in the terminal:

python --version

3. Open the Project Folder

Open the extracted project folder in VS Code. Open the terminal and navigate to the folder containing requirements.txt.

4. Create a Virtual Environment

Run the following commands:

python -m venv .venv

Activate the virtual environment.

For Windows (PowerShell):

.\.venv\Scripts\Activate.ps1

For macOS/Linux:

source .venv/bin/activate

5. Install Dependencies

Install the required Python libraries:

pip install -r requirements.txt

6. Run the Project

Run the following command to execute the freight cost anomaly detection pipeline:

python -m src.pipeline

The pipeline processes the shipment data, analyzes freight costs, detects anomalies, and generates the output report.

7. Ask Questions About the Results

You can query the generated anomaly results using natural-language questions:

python -m src.pipeline --ask "Why did Chennai-Bangalore get pricier in March?"

The system returns matching anomaly records and supporting evidence when available.

8. Check the Output

After execution, check the output folder for the generated freight anomaly report.

Note: Ensure that the required input CSV files are present in the data folder before running the project. Do not upload your virtual environment or secret credentials to GitHub.
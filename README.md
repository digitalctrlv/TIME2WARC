---
title: TIME2WARC Dashboard
emoji: 🔎
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---


# TIME2WARC

An analytical pipeline and interactive dashboard designed to extract, process, and temporally classify web archives (WARC files). This architecture parses raw web data, extracts HTML payloads, and utilizes a fine-tuned RoBERTa model to predict the creation period of specific domains.

## Project setup & heavy assets

Due to file size constraints, the machine learning models and raw WARC datasets are not hosted in this repository. You must download them manually before running the application.

1. **Download assets:** Go to the [Project Google Drive](https://drive.google.com/drive/folders/1hD4UJOc9w4nO8AkgEe01XhjA7MeO2G8t?usp=sharing).
2. **Model weights:** Download the model folder and place it in the root directory of this project so the path looks exactly like this:
   `TIME2WARC/models/v2_3_optimal_weights_fold1/`
3. **WARC files:** Download the `.warc` or `.warc.gz` files and drag them into the dashboard later. (These will be removed later but for demonstration purposes uploaded online)
4. **URL index**: Make sure to also download the `index_warcs.json` file from the /warcs directory in the Google Drive. Once your environment has set up, drag them into the first step of the interface, you need them to parse the warcs.

## How to run the application

1. **Clone the repository**
   ```bash
   git clone https://github.com/digitalctrlv/TIME2WARC.git

2. **Activate the virtual environment:**
   ```bash
   * Windows (PowerShell): `.venv\Scripts\Activate.ps1`
   * Windows (CMD): `.venv\Scripts\activate.bat`
   * Mac/Linux: `source .venv/bin/activate`

3. **Install dependencies (if not already installed):**
   ```bash
   pip install -r requirements.txt

4. **Launch the graphical interface via your terminal and open in a browser**
   ```bash
   streamlit run app.py

## Dashboard
**Tab 1**: Scans the uploaded_warcs/ directory, initializes the relational SQLite database (websites.db), and extracts the raw text/html payloads from the inputted archive. Also make sure to upload the index provided in the /warcs directory.

**Tab 2**: Loads the external RoBERTa model and evaluates the extracted HTML payloads. It assigns a predicted temporal period and a statistical confidence score to each domain, writing the outputs directly back to the database.

**Tab 3**: Displays the final results in an interactive data table. It provides a downloadable JSON file containing the fully annotated dataset, ready for downstream historical analysis.

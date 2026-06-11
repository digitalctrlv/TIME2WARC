# TIME2WARC

An analytical pipeline and interactive dashboard designed to extract, process, and temporally classify web archives (WARC files). This architecture parses raw web data, extracts HTML payloads, and utilizes a fine-tuned RoBERTa model to predict the creation period of specific domains.

## Project Setup & Heavy Assets

Due to file size constraints, the machine learning models and raw WARC datasets are not hosted in this repository. You must download them manually before running the application.

1. **Download assets:** Go to the [Project Google Drive](https://drive.google.com/drive/folders/1hD4UJOc9w4nO8AkgEe01XhjA7MeO2G8t?usp=sharing).
2. **Model weights:** Download the model folder and place it in the root directory of this project so the path looks exactly like this:
   `TIME2WARC/models/v2_3_optimal_weights_fold1/`
3. **WARC Files:** Download the `.warc` or `.warc.gz` files and place them in:
   `TIME2WARC/uploaded_warcs/`

## How to Run the Application

1. **Activate the virtual environment:**
   ```bash
   # Windows (PowerShell)
   .venv\Scripts\Activate.ps1

2. **Install dependencies (if not already installed):**
   ```bash
   pip install -r requirements.txt

3. **Run the dasboard**
   ```bash
   streamlit run app.py

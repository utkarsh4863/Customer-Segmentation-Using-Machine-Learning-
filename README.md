# Customer Segmentation Using Machine Learning

🛍️ **Customer Segmentation Using Machine Learning**  
This project uses K-Means clustering to segment customers based on their **Annual Income (k$)** and **Spending Score (1-100)**. It provides an interactive Streamlit app to predict clusters for single [...]

---

## Features
- Predict cluster for a single customer by entering Annual Income and Spending Score.
- Upload a CSV to classify multiple customers at once and download results.
- Automatically applies scaling and K-Means model to assign clusters.
- Provides human-readable business insights for each cluster.

---

## 🌐 Live Demo

You can try the app online here:  

[🔗 Customer Segmentation App](https://gjwdng8ddxehhlacfqjcoi.streamlit.app/)


---
## Tech Stack
- Python 3.x
- scikit-learn (KMeans, StandardScaler)
- pandas, numpy
- matplotlib & seaborn (visualizations)
- Streamlit (web interface)

---

## Project Structure
The project structure is shown below in a code block so it renders correctly on GitHub:

```text
Customer-Segmentation-Using-Machine-Learning-/
├── app.py                    # Main Streamlit application
├── kmeans_model.pkl          # Trained K-Means model (pickle)
├── scaler.pkl                # StandardScaler used to scale inputs (pickle)
├── requirements.txt          # Python dependencies
├── .gitignore
└── README.md                 # This file
```

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/utkarsh4863/Customer-Segmentation-Using-Machine-Learning-.git
cd Customer-Segmentation-Using-Machine-Learning-
```

2. (Optional) Create and activate a virtual environment:
- macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```
- Windows (PowerShell):
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Usage

### Run the Streamlit app
Start the app locally:
```bash
streamlit run app.py
```
Open the local URL shown in the terminal (usually http://localhost:8501).

### Single customer prediction
- Use the app's input fields to enter:
  - Annual Income (k$)
  - Spending Score (1-100)
- Click "Predict" to get the cluster number and business insight.

### CSV batch prediction
- Prepare a CSV file that exactly matches the required column names (see example below).
- Upload the CSV in the app.
- The app will return a CSV with an additional `Cluster` column and allow download.

---

## CSV format example
The CSV must include the following columns (case-sensitive):
- Annual Income (k$)
- Spending Score (1-100)

Example (customers.csv):
```csv
CustomerID,Annual Income (k$),Spending Score (1-100)
1,15,39
2,15,81
3,16,6
4,16,77
```

When you upload, the app will append a `Cluster` column with values 0–4.

---

## Cluster Insights / Business Interpretation
The model labels clusters 0–4. Interpretations (apply after confirming with your data):

- Cluster 0 — Average Customers: Moderate income and spending. Target with upsells, loyalty programs.
- Cluster 1 — High-Spending, High-Income: "Moneymakers". Prioritize premium offers and VIP treatment.
- Cluster 2 — High-Spending, Low-Income: "Careful Spenders". Promote value bundles and impulse purchases.
- Cluster 3 — Low-Spending, High-Income: "Miser". Offer curated high-value items and personalized promotions.
- Cluster 4 — Low-Spending, Low-Income: "General / Strugglers". Focus on essentials, discounts, and retention.

Note: The cluster numbers correspond to the trained model labels. If you retrain the model, labels may change — re-evaluate interpretations.

---

## Files required for prediction
- kmeans_model.pkl — trained KMeans model (pickle)
- scaler.pkl — StandardScaler used during training (pickle)

These two files must be in the same directory as app.py for predictions to work.

---

## Deployment
You can deploy to Streamlit Community Cloud:
1. Push your repository to GitHub.
2. Create a new app on Streamlit Cloud and connect your repo.
3. Set the entrypoint to `app.py` and the branch to `main`.
4. Make sure `kmeans_model.pkl` and `scaler.pkl` are present in the repo.

---

## Troubleshooting
- Missing model/scaler errors: ensure kmeans_model.pkl and scaler.pkl exist and are not corrupted.
- CSV upload errors: verify column names and CSV encoding (UTF-8 recommended).
- Version issues: recreate the environment and install `requirements.txt`.

---

## Contributing
Contributions, improvements, and bug fixes are welcome. Open an issue or submit a pull request with a clear description of changes.

---

## License
This project is open-source and free to use for educational purposes. (Specify a license file / type if you want a particular license, e.g., MIT.)

---

## Author
Utkarsh — GitHub: https://github.com/utkarsh4863

---

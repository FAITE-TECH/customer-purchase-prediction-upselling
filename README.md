# Customer Purchase Prediction & Upselling

This project implements a filtering-based product recommender system using the [LightFM](https://making.lyst.com/lightfm/docs/home.html) model. It is designed to analyze past transaction data and recommend products to customers for upselling and cross-selling.

## Project files
model/
- lightfm_model.pkl # Trained LightFM model
- user_map.pkl # Mapping: CustomerID → Internal user index
- item_map.pkl # Mapping: Itemname → Internal item index

data/
  Assignment-1_Data.csv # Sample demo dataset (Replace it with actual dataset)

train_model.ipynb # Jupyter Notebook used for training & testing
train_lightfm.py # Python script to retrain model from scratch
recommended.py # Script to generate recommendations
requirements.txt # Required Python packages
README.md # Project overview & usage instructions

## What It Does
- Analyzes historical customer purchase transactions.
- Builds a **user-item interaction matrix**.
- Trains a **LightFM model** with implicit feedback.
- Recommends top-N products to customers based on past behavior.

## Requirements
Install required packages using:

```bash
pip install -r requirements.txt
```
## Dataset Format
Expected CSV format (data/Assignment-1_Data.csv):

InvoiceNo	CustomerID	Itemname	Quantity	Date	   Price
10001	     13047.0	 Bread	       2	  01/01/2023   1,00

Ensure Price is formatted with a comma decimal (e.g., 1,50) if using European-style formatting.

## Training the Model

Option 1: Use Jupyter Notebook
Open train_model.ipynb in Google Colab or locally, and run all cells to:
- Preprocess the dataset
- Train the LightFM model
- Save the model & ID maps to /model/

Option 2: Use Python Script
Run the following python file.
```bash
python train_lightfm.py
```
This will:
- Load data from data/Assignment-1_Data.csv
- Train the model
- Save the outputs into /model/

## Making Recommendations
Run the recommended.py script to generate recommendations for a given CustomerID:
```bash
python recommended.py --user_id 13047 --top_k 5
```
Parameters:
--user_id: The customer ID to recommend products for.

--top_k: Number of products to recommend.

## Replacing Dataset
To use your own dataset:
- Replace data/Assignment-1_Data.csv with your full transactional CSV.
- Ensure the same columns: InvoiceNo, CustomerID, Itemname, Quantity, Date, Price
- Re-run train_model.ipynb or train_lightfm.py to retrain the model.
- Use recommended.py with your desired CustomerID.




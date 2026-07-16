# Use Case 1 — AWS-ready learner pack

## Files
- Notebook_1_Data_Ingestion_Exploration.ipynb
- online_retail_sample.csv
- retail_exploration_ready.csv (sample output)

## Live run order
1. Upload `online_retail_sample.csv` to `s3://<bucket>/<prefix>/usecase1/raw/`
2. Open the notebook in SageMaker Studio
3. Update the single setup cell with your bucket and optional prefix
4. Run all cells
5. Verify that `retail_exploration_ready.csv` is written to `usecase1/processed/`

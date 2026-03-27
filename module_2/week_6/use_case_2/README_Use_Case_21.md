# Use Case 2 — AWS-ready learner pack

## Files
- Notebook_2_Data_Cleaning_Transformation.ipynb
- Notebook_3_Simple_ETL_Pipeline.ipynb
- Notebook_3B_AWS_Glue_PySpark_ETL.ipynb
- retail_exploration_ready.csv
- retail_cleaned.csv
- daily_country_revenue.csv
- monthly_category_revenue.csv

## Live run order
1. Make sure Use Case 1 already created `retail_exploration_ready.csv` in S3
2. Open Notebook 2 in SageMaker Studio and update the bucket / prefix in the setup cell
3. Run Notebook 2 to create `retail_cleaned.csv` in `usecase2/processed/`
4. Run Notebook 3 to create daily and monthly output CSVs in `usecase2/output/`
5. Use Notebook 3B in AWS Glue Studio to show how the same ETL logic scales in PySpark

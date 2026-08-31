For liver function assessment, the columns in your table correspond to common Liver Function Tests (LFT). 

Below are the typical normal reference ranges for adults (may vary slightly by lab).
<img width="777" height="544" alt="image" src="https://github.com/user-attachments/assets/59b7e1d1-a54d-4637-9392-718984729f06" />

<img width="690" height="576" alt="image" src="https://github.com/user-attachments/assets/4951c861-3a7e-4b89-927c-c61afb409747" />


**Data Pipeline Note: Ingestion, Preprocessing, and Validation of Total Bilirubin**

**1. Data Ingestion**
The ingestion stage is responsible for loading the dataset from the source system into the processing environment. In this case, the liver patient dataset is read from a CSV file using Python and the Pandas library. During ingestion, the system ensures that the file is correctly loaded, the encoding is handled properly, and the data structure (rows and columns) is available for further processing. This stage establishes the raw dataset that will be used in subsequent pipeline steps.

**2. Data Preprocessing**
In the preprocessing stage, the dataset is cleaned and prepared for analysis. This includes handling missing or inconsistent values and ensuring that the data types are appropriate for processing. For the *Total Bilirubin* field, preprocessing involves identifying and removing records where the value is missing or invalid. Additional preprocessing steps may include renaming columns, converting numerical fields to the correct format, and ensuring that the dataset is ready for validation and downstream analytics or machine learning tasks.

**3. Data Validation (Total Bilirubin)**
Data validation ensures that the values in the dataset fall within acceptable medical or logical ranges. For the *Total Bilirubin* attribute, records are validated to ensure the value is within a reasonable threshold. In this pipeline, records where *Total Bilirubin* is greater than or equal to 10 are treated as invalid or extreme values and are removed from the dataset. This step helps maintain data quality and prevents anomalous records from affecting downstream analysis or model training.

Together, these stages—ingestion, preprocessing, and validation—form a simple data pipeline that ensures the dataset is clean, reliable, and ready for further processing or machine learning workflows.

import sys
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.utils import getResolvedOptions
from awsglue.job import Job
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType

args = getResolvedOptions(sys.argv, ['JOB_NAME', 'SOURCE_PATH', 'TARGET_PATH'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

df = spark.read.option('header', True).option('inferSchema', True).csv(args['SOURCE_PATH'])

df = (
    df.withColumn('TotalCharges', F.when(F.trim(F.col('TotalCharges')) == '', None).otherwise(F.col('TotalCharges')).cast(DoubleType()))
      .withColumn('MonthlyCharges', F.col('MonthlyCharges').cast(DoubleType()))
      .withColumn('tenure', F.col('tenure').cast(IntegerType()))
      .withColumn('label', F.when(F.col('Churn') == 'Yes', F.lit(1)).otherwise(F.lit(0)))
      .withColumn('is_new_customer', F.when(F.col('tenure') <= 6, F.lit(1)).otherwise(F.lit(0)))
      .withColumn('monthly_charge_band',
          F.when(F.col('MonthlyCharges') <= 35, F.lit('Low'))
           .when(F.col('MonthlyCharges') <= 70, F.lit('Medium'))
           .otherwise(F.lit('High'))
      )
      .withColumn('avg_monthly_spend_gap',
          F.col('TotalCharges') - (F.col('tenure').cast(DoubleType()) * F.col('MonthlyCharges'))
      )
)

# Validate before writing
null_labels = df.filter(F.col('label').isNull()).count()
if null_labels > 0:
    raise ValueError(f"Validation failed: {null_labels} rows have null label. Aborting job.")

df.write.mode('overwrite').option('header', True).csv(args['TARGET_PATH'])
job.commit()

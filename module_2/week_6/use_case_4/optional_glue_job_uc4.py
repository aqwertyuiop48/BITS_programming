import sys
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType, DoubleType

args = getResolvedOptions(sys.argv, ['JOB_NAME', 'INPUT_PATH', 'BASELINE_PATH', 'OUTPUT_PATH'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

incoming = spark.read.option('header', True).option('inferSchema', True).csv(args['INPUT_PATH'])
baseline = spark.read.option('header', True).option('inferSchema', True).csv(args['BASELINE_PATH'])


def numeric_cols(df, exclude=None):
    """Return only numeric column names to avoid AnalysisException on F.mean of StringType columns."""
    exclude = set(exclude or [])
    return [f.name for f in df.schema.fields if isinstance(f.dataType, NumericType) and f.name not in exclude]


baseline_num_cols = numeric_cols(baseline, exclude=['customerID'])
incoming_num_cols = numeric_cols(incoming, exclude=['customerID'])

baseline_profile = baseline.select([F.mean(c).alias(c) for c in baseline_num_cols])
incoming_profile = incoming.select([F.mean(c).alias(c) for c in incoming_num_cols])

# Compare profiles and flag drift (>10% mean shift)
baseline_row = baseline_profile.first().asDict() if baseline_profile.count() > 0 else {}
incoming_row = incoming_profile.first().asDict() if incoming_profile.count() > 0 else {}
common_cols = [c for c in baseline_num_cols if c in incoming_num_cols]

findings = []
for col in common_cols:
    b_val = baseline_row.get(col)
    i_val = incoming_row.get(col)
    if b_val is not None and b_val != 0:
        pct_change = (i_val - b_val) / b_val
        flagged = abs(pct_change) > 0.10
    else:
        pct_change = None
        flagged = False
    findings.append({
        'column_name': col,
        'check_type': 'mean_shift',
        'baseline_mean': b_val,
        'incoming_mean': i_val,
        'pct_change': round(pct_change, 4) if pct_change is not None else None,
        'flag': flagged
    })

findings_df = spark.createDataFrame(findings)
flagged_df = findings_df.filter(F.col('flag') == True)

baseline_profile.write.mode('overwrite').json(args['OUTPUT_PATH'] + '/baseline_profile_json')
incoming_profile.write.mode('overwrite').json(args['OUTPUT_PATH'] + '/incoming_profile_json')
findings_df.write.mode('overwrite').option('header', True).csv(args['OUTPUT_PATH'] + '/drift_findings')
flagged_df.write.mode('overwrite').option('header', True).csv(args['OUTPUT_PATH'] + '/flagged_drift_findings')
job.commit()

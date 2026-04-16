import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from parser import load_contracts

from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp
from delta import configure_spark_with_delta_pip

builder = SparkSession.builder \
    .appName("DocuSense-Ingestion") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .config("spark.driver.memory", "2g")

spark = configure_spark_with_delta_pip(builder).getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

DELTA_PATH = "data/delta/documents"

def run_pipeline():
    print("Loading contracts...")
    records = load_contracts("data/raw/contracts/cuad_contracts.jsonl")

    rows = [{k: v for k, v in r.items()} for r in records]
    df = spark.createDataFrame(rows)
    df = df.withColumn("ingested_at", current_timestamp())

    print(f"Writing {df.count()} records to Delta Lake...")
    df.write.format("delta").mode("overwrite").save(DELTA_PATH)
    print(f"Done. Schema:")
    df.printSchema()

    verify = spark.read.format("delta").load(DELTA_PATH)
    print(f"Verified: {verify.count()} records in Delta Lake at {DELTA_PATH}")

if __name__ == "__main__":
    run_pipeline()

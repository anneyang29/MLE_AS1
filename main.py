# main.py
import os
import glob
from datetime import datetime
import pyspark

# 只用三個模組
import utils.data_processing_bronze_table as bronze_mod
import utils.data_processing_silver_table as silver_mod
import utils.data_processing_gold_table   as gold_mod

# -----------------------------
# Spark
# -----------------------------
spark = pyspark.sql.SparkSession.builder.appName("dev").master("local[*]").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# -----------------------------
# Config
# -----------------------------
START_DATE_STR = "2023-01-01"
END_DATE_STR   = "2024-12-01"

# Bronze 輸出路徑
BRONZE_LMS_DIR  = "datamart/bronze/lms/"
BRONZE_BASE_DIR = "datamart/bronze/"

# Silver 輸出路徑
SILVER_LMS_DIR  = "datamart/silver/loan_daily/"
SILVER_ATTR_DIR = "datamart/silver/features_attributes/"
SILVER_FIN_DIR  = "datamart/silver/features_financials/"

# Gold 輸出路徑
GOLD_LABEL_DIR  = "datamart/gold/label_store/"
GOLD_FEAT_DIR   = "datamart/gold/feature_store/"

# 其他設定
LABEL_DPD = 30
LABEL_MOB = 6
OUTLIER_ACTION = "null"   # or "winsor"

# -----------------------------
# Helpers
# -----------------------------
def month_range(start_date_str, end_date_str):
    start = datetime.strptime(start_date_str, "%Y-%m-%d")
    end   = datetime.strptime(end_date_str,   "%Y-%m-%d")
    cur = datetime(start.year, start.month, 1)
    out = []
    while cur <= end:
        out.append(cur.strftime("%Y-%m-%d"))
        year  = cur.year + (1 if cur.month == 12 else 0)
        month = 1 if cur.month == 12 else cur.month + 1
        cur = datetime(year, month, 1)
    return out

DATES = month_range(START_DATE_STR, END_DATE_STR)
print("dates:", DATES)

# -----------------------------
# Ensure output dirs exist
# -----------------------------
for d in [BRONZE_LMS_DIR, SILVER_LMS_DIR, SILVER_ATTR_DIR, SILVER_FIN_DIR, GOLD_LABEL_DIR, GOLD_FEAT_DIR]:
    os.makedirs(d, exist_ok=True)

# -----------------------------
# Bronze backfill
# -----------------------------
for ds in DATES:
    # 用位置參數（snapshot_date_str, bronze_lms_directory, spark）
    bronze_mod.process_bronze_table(ds, BRONZE_LMS_DIR, spark)

# -----------------------------
# Silver backfill
# -----------------------------
for ds in DATES:
    # A) loan_daily：用位置參數（snapshot_date_str, bronze_loan_daily_directory, silver_loan_daily_directory, spark）
    print("process silver", ds)
    silver_mod.process_silver_table(ds, BRONZE_LMS_DIR, SILVER_LMS_DIR, spark)

    # B) attributes：用位置參數（snapshot_date_str, bronze_directory, silver_directory, spark）
    silver_mod.process_silver_features_attributes(ds, BRONZE_BASE_DIR, SILVER_ATTR_DIR, spark)

    # C) financials：用位置參數（snapshot_date_str, bronze_directory, silver_directory, spark, outlier_action）
    silver_mod.process_silver_features_financials(ds, BRONZE_BASE_DIR, SILVER_FIN_DIR, spark, OUTLIER_ACTION)
print("silv finish")
# -----------------------------
# Gold backfill
# -----------------------------
for ds in DATES:
    # A) Label Store：用位置參數（snapshot_date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd, mob）
    gold_mod.process_labels_gold_table(ds, SILVER_LMS_DIR, GOLD_LABEL_DIR, spark, LABEL_DPD, LABEL_MOB)

    # B) Feature Store：用位置參數（snapshot_date_str, silver_attr_directory, silver_fin_directory, gold_feature_store_directory, spark, silver_loan_daily_directory=None/dir）
    gold_mod.process_gold_feature_table(ds, SILVER_ATTR_DIR, SILVER_FIN_DIR, GOLD_FEAT_DIR, spark, SILVER_LMS_DIR)

# -----------------------------
# Quick sanity checks (optional)
# -----------------------------
lbl_paths  = glob.glob(os.path.join(GOLD_LABEL_DIR, "*"))
feat_paths = glob.glob(os.path.join(GOLD_FEAT_DIR,  "*"))

if lbl_paths:
    df_lbl = spark.read.parquet(*lbl_paths)
    print("label rows:", df_lbl.count())
    df_lbl.show(5, truncate=False)
else:
    print("No label files found under:", GOLD_LABEL_DIR)

if feat_paths:
    df_feat = spark.read.parquet(*feat_paths)
    print("feature rows:", df_feat.count())
    df_feat.show(5, truncate=False)
else:
    print("No feature files found under:", GOLD_FEAT_DIR)

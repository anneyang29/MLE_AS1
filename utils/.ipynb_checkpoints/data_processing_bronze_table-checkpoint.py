from pyspark.sql.functions import col, lit
from datetime import datetime
import os

def process_bronze_table(snapshot_date_str, bronze_lms_directory, spark):
    # 1) 來源
    csv_file_lms         = "data/lms_loan_daily.csv"
    csv_file_clickstream = "data/feature_clickstream.csv"
    csv_file_attributes  = "data/features_attributes.csv"
    csv_file_financials  = "data/features_financials.csv"

    # 2) 讀＋同日切片（以字串比對，避免型別不一致）
    df_lms = spark.read.csv(csv_file_lms, header=True, inferSchema=True) \
        .filter(col("snapshot_date") == lit(snapshot_date_str))
    df_clk = spark.read.csv(csv_file_clickstream, header=True, inferSchema=True) \
        .filter(col("snapshot_date") == lit(snapshot_date_str))
    df_att = spark.read.csv(csv_file_attributes, header=True, inferSchema=True) \
        .filter(col("snapshot_date") == lit(snapshot_date_str))
    df_fin = spark.read.csv(csv_file_financials, header=True, inferSchema=True) \
        .filter(col("snapshot_date") == lit(snapshot_date_str))

    print(f"[Bronze][{snapshot_date_str}] rows -> lms={df_lms.count()}, clickstream={df_clk.count()}, attributes={df_att.count()}, financials={df_fin.count()}")

    # 3) 寫 Bronze
    sub = snapshot_date_str.replace('-', '_')
    partition_name = f"bronze_loan_daily_{sub}.csv"
    filepath = os.path.join(bronze_lms_directory, partition_name)

    # 轉成 pandas 再存成單一 CSV
    df_lms.toPandas().to_csv(filepath, index=False)
    print(f"[Bronze][loan_daily] saved to: {filepath}")

    # 其餘三表用資料湖分區風格
    bronze_base = os.path.dirname(os.path.dirname(bronze_lms_directory)) or "datamart/bronze"
    out_clk = os.path.join(bronze_base, "clickstream",  f"snapshot_date={sub}")
    out_att = os.path.join(bronze_base, "attributes",   f"snapshot_date={sub}")
    out_fin = os.path.join(bronze_base, "financials",   f"snapshot_date={sub}")

    df_clk.write.mode("overwrite").parquet(out_clk)
    df_att.write.mode("overwrite").parquet(out_att)
    df_fin.write.mode("overwrite").parquet(out_fin)

    return {"lms": df_lms, "clickstream": df_clk, "attributes": df_att, "financials": df_fin}

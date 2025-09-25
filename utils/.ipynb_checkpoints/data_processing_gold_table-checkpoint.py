import os
from datetime import datetime
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType, StringType
from pyspark.sql.window import Window

# ------------------------
# Gold: Label Store
# ------------------------

###def process_labels_gold_table(snapshot_date_str,silver_loan_daily_directory,
                              #gold_label_store_directory,
                              #spark,
                              #dpd=30,
                              #mob=6):
    #sub = snapshot_date_str.replace("-", "_")
    #inpath = os.path.join(silver_loan_daily_directory, f"silver_loan_daily_{sub}.parquet")
    #df = spark.read.parquet(inpath)
    #print("[Gold][label] loaded:", inpath, "rows:", df.count())

    #df = df.filter(col("mob") == mob) \
           #.withColumn("label", F.when(col("dpd") >= F.lit(dpd), 1).otherwise(0).cast(IntegerType())) \
           #.withColumn("label_def", F.lit(f"{dpd}dpd_{mob}mob").cast(StringType()))

    #df_out = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")

    #outname = f"gold_label_store_{sub}.parquet"
    #outpath = os.path.join(gold_label_store_directory, outname)
    #df_out.write.mode("overwrite").parquet(outpath)
    #print("[Gold][label] saved:", outpath)
    #return df_out

def process_labels_gold_table(snapshot_date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd, mob):
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    # get customer at mob
    df = df.filter(col("mob") == mob)

    # get label
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(str(dpd)+'dpd_'+str(mob)+'mob').cast(StringType()))

    # select columns to save
    df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")

    # save gold table - IRL connect to database to write
    partition_name = "gold_label_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_label_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df

# ------------------------
# Gold: Feature Store
# ------------------------
def process_gold_feature_table(snapshot_date_str,
                               silver_attr_directory,
                               silver_fin_directory,
                               gold_feature_store_directory,
                               spark,
                               silver_loan_daily_directory=None):  # 可選：若要帶 loan_daily 欄位
    sub = snapshot_date_str.replace("-", "_")

    # 讀 Silver features
    attr_in = os.path.join(silver_attr_directory, f"silver_features_attributes_{sub}.parquet")
    fin_in  = os.path.join(silver_fin_directory,  f"silver_features_financials_{sub}.parquet")
    df_attr = spark.read.parquet(attr_in)
    df_fin  = spark.read.parquet(fin_in)
    print("[Gold][feature] loaded attr:", attr_in, "rows:", df_attr.count())
    print("[Gold][feature] loaded fin :", fin_in,  "rows:", df_fin.count())

    # join（customer_id + snapshot_date）
    df = df_attr.join(df_fin, on=["customer_id", "snapshot_date"], how="inner")

    # 可選：帶入 loan_daily 的連結欄位（例如 balance/due_amt/overdue_amt）
    if silver_loan_daily_directory:
        lms_in = os.path.join(silver_loan_daily_directory, f"silver_loan_daily_{sub}.parquet")
        df_lms = spark.read.parquet(lms_in).select("Customer_ID", "snapshot_date", "balance", "due_amt", "overdue_amt")
        df = df.join(df_lms.withColumnRenamed("Customer_ID", "customer_id"),
                     on=["customer_id", "snapshot_date"], how="left")

    # 類別編碼
    # occupation
    if "occupation" in df.columns:
        occ_map = (df.select("occupation").distinct()
                     .withColumn("occupation_index",
                                 (F.row_number().over(Window.orderBy(F.col("occupation").asc_nulls_last())) - 1)
                                 .cast(IntegerType())))
        df = df.join(occ_map, on="occupation", how="left")
    else:
        df = df.withColumn("occupation_index", F.lit(None).cast(IntegerType()))

    # payment_behaviour
    if "payment_behaviour" in df.columns:
        pay_map = (df.select("payment_behaviour").distinct()
                     .withColumn("payment_behaviour_index",
                                 (F.row_number().over(Window.orderBy(F.col("payment_behaviour").asc_nulls_last())) - 1)
                                 .cast(IntegerType())))
        df = df.join(pay_map, on="payment_behaviour", how="left")
    else:
        df = df.withColumn("payment_behaviour_index", F.lit(None).cast(IntegerType()))

    # 選欄（若沒帶 loan_daily，就不要選 balance/due_amt/overdue_amt）
    base_cols = [
        F.col("customer_id").alias("Customer_ID"),
        "snapshot_date",
        "num_bank_accounts", "num_credit_card",
        "occupation_index", "payment_behaviour_index",
        "ssn_valid_flag",
        "num_bank_accounts_bucket", "num_credit_card_bucket",
        "credit_mix_underscore_flag", "payment_behaviour_noise_flag",
        "age_bucket", "age_invalid_flag", "occupation_underscore_flag"
    ]
    if silver_loan_daily_directory:
        base_cols += ["balance", "due_amt", "overdue_amt"]

    df_out = df.select(*base_cols)

    # 寫 Gold Feature Store
    outname = f"gold_feature_store_{sub}.parquet"
    outpath = os.path.join(gold_feature_store_directory, outname)
    df_out.write.mode("overwrite").parquet(outpath)
    print("[Gold][feature] saved:", outpath)
    return df_out

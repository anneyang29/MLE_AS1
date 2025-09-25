import os
from datetime import datetime
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

# ------------------------
# Silver: loan_daily（Lab2 強化版）
# ------------------------
#def process_silver_table(snapshot_date_str, bronze_loan_daily_directory, silver_loan_daily_directory, spark):
    #sub = snapshot_date_str.replace('-', '_')
    #inpath  = os.path.join(bronze_loan_daily_directory, f"bronze_loan_daily_{sub}")
    #df = spark.read.csv(inpath, header=True, inferSchema=True)
    #print("[Silver][loan_daily] loaded:", inpath, "rows:", df.count())

    # enforce schema
   # type_map = {
       # "loan_id": StringType(), "Customer_ID": StringType(),
        #"loan_start_date": DateType(), "tenure": IntegerType(),
       # "installment_num": IntegerType(), "loan_amt": FloatType(),
        #"due_amt": FloatType(), "paid_amt": FloatType(), "overdue_amt": FloatType(),
       # "balance": FloatType(), "snapshot_date": DateType(),
   # }
    #for c,t in type_map.items():
        #if c in df.columns:
           # df = df.withColumn(c, col(c).cast(t))

    # mob
   # df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

    # installments_missed（防呆）
    #df = df.withColumn(
       # "installments_missed",
        #F.when((col("due_amt").isNull()) | (col("due_amt") == 0), F.lit(0))
       #  .otherwise(F.ceil(F.coalesce(col("overdue_amt"), F.lit(0.0)) / col("due_amt")))
        # .cast(IntegerType())
   # )

    # first_missed_date / dpd
   # df = df.withColumn(
    #    "first_missed_date",
       # F.when(col("installments_missed") > 0,
           #    F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType())
   # )
   ## df = df.withColumn(
     #  "dpd",
      #  F.when(F.coalesce(col("overdue_amt"), F.lit(0.0)) > 0.0,
        #       F.datediff(col("snapshot_date"), col("first_missed_date")))
        # .otherwise(F.lit(0)).cast(IntegerType())
  #  )

    # write
   # outname = f"silver_loan_daily_{sub}.parquet"
   # outpath = os.path.join(silver_loan_daily_directory, outname)
   # df.write.mode("overwrite").parquet(outpath)
   # print("[Silver][loan_daily] saved:", outpath)
   # return df
def process_silver_table(snapshot_date_str, bronze_lms_directory, silver_loan_daily_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_loan_daily_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_lms_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # augment data: add month on book
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

    # augment data: add days past due
    df = df.withColumn("installments_missed", F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
    df = df.withColumn("first_missed_date", F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
    df = df.withColumn("dpd", F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))

    # save silver table - IRL connect to database to write
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df

# ------------------------
# Silver: features_attributes
# ------------------------
def process_silver_features_attributes(snapshot_date_str, bronze_directory, silver_directory, spark):
    sub = snapshot_date_str.replace('-', '_')
    inpath = os.path.join(bronze_directory, "attributes", f"snapshot_date={sub}")
    df = spark.read.parquet(inpath)
    print("[Silver][attributes] loaded:", inpath, "rows:", df.count())

    df = (
        df.withColumn("customer_id",   col("customer_id").cast(StringType()))
          .withColumn("Age",           col("Age").cast(StringType()))
          .withColumn("SSN",           col("SSN").cast(StringType()))
          .withColumn("Occupation",    col("Occupation").cast(StringType()))
          .withColumn("snapshot_date", col("snapshot_date").cast(DateType()))
    )

    # Age 清理：直接數值 / 由年份推算 / 其他→NULL
    age_raw = F.regexp_replace(F.trim(F.col("Age")), ",", "")
    age_num = F.when(F.length(age_raw) > 0, age_raw.cast("double"))
    current_year = F.year(F.current_date())
    age_from_year = F.when((age_num >= 1900) & (age_num <= current_year.cast("double")),
                           (current_year - age_num.cast("int")))
    age_direct = F.when((age_num >= 0) & (age_num <= 120), age_num)
    age_clean = F.coalesce(age_direct, age_from_year).cast(IntegerType())

    df = df.withColumn("age", age_clean) \
           .withColumn("age_invalid_flag", F.when(F.col("age").isNull(), 1).otherwise(0)) \
           .withColumn("age_bucket", F.when(F.col("age").isNotNull(),
                                            (F.floor(F.col("age")/10)*10).cast(IntegerType())))

    # SSN：格式旗標＋hash，去除原文
    valid_pat = r'^\d{3}-\d{2}-\d{4}$'
    df = df.withColumn("ssn_valid_flag", F.when(F.col("SSN").rlike(valid_pat), 1).otherwise(0)) \
           .withColumn("ssn_hash", F.when(F.col("SSN").rlike(valid_pat),
                                F.sha2(F.regexp_replace(F.lower(F.col("SSN")), "-", ""), 256))) \
           .drop("SSN")

    # Occupation：全底線→NULL
    underscore_regex = r'^_+$'
    df = df.withColumn("occupation_underscore_flag", F.when(F.col("Occupation").rlike(underscore_regex), 1).otherwise(0)) \
           .withColumn("occupation", F.when(F.col("Occupation").rlike(underscore_regex), None).otherwise(F.col("Occupation"))) \
           .drop("Age", "Occupation")

    # write
    outname = f"silver_features_attributes_{sub}.parquet"
    outpath = os.path.join(silver_directory, outname)
    df.write.mode("overwrite").parquet(outpath)
    print("[Silver][attributes] saved:", outpath)
    return df


# ------------------------
# Silver: features_financials
# ------------------------
def process_silver_features_financials(snapshot_date_str, bronze_directory, silver_directory, spark, outlier_action="null"):
    sub = snapshot_date_str.replace('-', '_')
    inpath = os.path.join(bronze_directory, "financials", f"snapshot_date={sub}")
    df = spark.read.parquet(inpath)
    print("[Silver][financials] loaded:", inpath, "rows:", df.count())

    df = (
        df.withColumn("customer_id",   col("customer_id").cast(StringType()))
          .withColumn("Num_Bank_Accounts", col("Num_Bank_Accounts").cast("double"))
          .withColumn("Num_Credit_Card",   col("Num_Credit_Card").cast("double"))
          .withColumn("Credit_Mix",        col("Credit_Mix").cast(StringType()))
          .withColumn("Payment_Behaviour", col("Payment_Behaviour").cast(StringType()))
          .withColumn("snapshot_date",     col("snapshot_date").cast(DateType()))
    )

    def tukey_upper_cap_local(df_local, colname, default_upper=100.0):
        qs = df_local.approxQuantile(colname, [0.25, 0.75], 0.01)
        if len(qs) < 2 or qs[0] is None or qs[1] is None:
            return float(default_upper)
        q1, q3 = qs; iqr = q3 - q1
        return float(q3 + 1.5 * iqr)

    ub_banks = tukey_upper_cap_local(df.where(F.col("Num_Bank_Accounts").isNotNull()), "Num_Bank_Accounts", 100.0)
    ub_cards = tukey_upper_cap_local(df.where(F.col("Num_Credit_Card").isNotNull()),   "Num_Credit_Card",   100.0)

    def clean_count(colname, upper, out_name):
        c = F.col(colname)
        neg_flag = F.when(c < 0, 1).otherwise(0)
        base = F.when(c < 0, None).otherwise(c)
        if outlier_action == "winsor":
            cleaned = F.when(base.isNull(), None).otherwise(F.when(base > F.lit(upper), F.lit(upper)).otherwise(base))
        else:
            cleaned = F.when((base.isNotNull()) & (base <= F.lit(upper)), base).otherwise(None)
        outlier_flag = F.when((base.isNotNull()) & (base > F.lit(upper)), 1).otherwise(0)
        return (cleaned.cast(IntegerType()).alias(out_name),
                neg_flag.alias(f"{out_name}_invalid_flag"),
                outlier_flag.alias(f"{out_name}_outlier_flag"),
                F.lit(float(upper)).alias(f"{out_name}_upper_cap"))

    nb, nb_inv, nb_out, nb_cap = clean_count("Num_Bank_Accounts", ub_banks, "num_bank_accounts")
    nc, nc_inv, nc_out, nc_cap = clean_count("Num_Credit_Card",   ub_cards, "num_credit_card")

    df = df.select(
        "customer_id", "snapshot_date",
        nb, nb_inv, nb_out, nb_cap,
        nc, nc_inv, nc_out, nc_cap,
        "Credit_Mix", "Payment_Behaviour"
    ).withColumn("credit_mix", F.when(F.col("Credit_Mix") == "_", None).otherwise(F.col("Credit_Mix"))) \
     .withColumn("credit_mix_underscore_flag", F.when(F.col("Credit_Mix") == "_", 1).otherwise(0)) \
     .withColumn("payment_behaviour", F.when(F.col("Payment_Behaviour") == "!@9#%8", None).otherwise(F.col("Payment_Behaviour"))) \
     .withColumn("payment_behaviour_noise_flag", F.when(F.col("Payment_Behaviour") == "!@9#%8", 1).otherwise(0)) \
     .drop("Credit_Mix", "Payment_Behaviour") \
     .withColumn("num_bank_accounts_bucket",  F.when(F.col("num_bank_accounts").isNotNull(),  (F.floor(F.col("num_bank_accounts")/2)*2).cast(IntegerType()))) \
     .withColumn("num_credit_card_bucket",    F.when(F.col("num_credit_card").isNotNull(),    (F.floor(F.col("num_credit_card")/2)*2).cast(IntegerType())))

    outname = f"silver_features_financials_{sub}.parquet"
    outpath = os.path.join(silver_directory, outname)
    df.write.mode("overwrite").parquet(outpath)
    print("[Silver][financials] saved:", outpath)
    return df

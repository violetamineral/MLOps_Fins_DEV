#****************************************************************************
# (C) Cloudera, Inc. 2020-2023 All rights reserved.
#
#  금융 사기거래 방지용 데이터 생성 파일
#    DB명 = "MA_MLOps"
#    테이블명 = "BANKING_TRANSACTIONS_{username}"
#    STORAGE = "s3a://go01-demo/user"
#    CONNECTION_NAME = "go01-aw-dl"
#
#  2024-06-07
#***************************************************************************/

import os
import numpy as np
import pandas as pd
from datetime import datetime
from pyspark.sql.types import LongType, IntegerType, StringType
from pyspark.sql import SparkSession
import dbldatagen as dg
import dbldatagen.distributions as dist
from dbldatagen import FakerTextFactory, DataGenerator, fakerText
from faker.providers import bank, credit_card, currency
import cml.data_v1 as cmldata


#은행데이터 생성 함수
class BankDataGen:

    '''Class to Generate Banking Data'''

    def __init__(self, username, dbname, storage, connectionName):
        self.username = username
        self.storage = storage
        self.dbname = dbname
        self.connectionName = connectionName


    def dataGen(self, spark, shuffle_partitions_requested = 5, partitions_requested = 2, data_rows = 10000):
        """
        Method to create credit card transactions in Spark Df
        """

        # setup use of Faker
        FakerTextUS = FakerTextFactory(locale=['en_US'], providers=[bank])

        # partition parameters etc.
        spark.conf.set("spark.sql.shuffle.partitions", shuffle_partitions_requested)

        fakerDataspec = (DataGenerator(spark, rows=data_rows, partitions=partitions_requested)
                    .withColumn("age", "float", minValue=10, maxValue=100, random=True)
                    .withColumn("credit_card_balance", "float", minValue=100, maxValue=30000, random=True)
                    .withColumn("bank_account_balance", "float", minValue=0.01, maxValue=100000, random=True)
                    .withColumn("mortgage_balance", "float", minValue=0.01, maxValue=1000000, random=True)
                    .withColumn("sec_bank_account_balance", "float", minValue=0.01, maxValue=100000, random=True)
                    .withColumn("savings_account_balance", "float", minValue=0.01, maxValue=500000, random=True)
                    .withColumn("sec_savings_account_balance", "float", minValue=0.01, maxValue=500000, random=True)
                    .withColumn("total_est_nworth", "float", minValue=10000, maxValue=500000, random=True)
                    .withColumn("primary_loan_balance", "float", minValue=0.01, maxValue=5000, random=True)
                    .withColumn("secondary_loan_balance", "float", minValue=0.01, maxValue=500000, random=True)
                    .withColumn("uni_loan_balance", "float", minValue=0.01, maxValue=10000, random=True)
                    .withColumn("longitude", "float", minValue=-180, maxValue=180, random=True)
                    .withColumn("latitude", "float", minValue=-90, maxValue=90, random=True)
                    .withColumn("transaction_amount", "float", minValue=0.01, maxValue=30000, random=True)
                    .withColumn("fraud", "integer", minValue=0, maxValue=1, random=True)
                    )
        df = fakerDataspec.build()

        return df


    def createSparkConnection(self):
        """
        Method to create a Spark Connection using CML Data Connections
        """

        from pyspark import SparkContext
        SparkContext.setSystemProperty('spark.executor.cores', '2')
        SparkContext.setSystemProperty('spark.executor.memory', '4g')

        import cml.data_v1 as cmldata
        conn = cmldata.get_connection(self.connectionName)
        spark = conn.get_spark_session()

        return spark


    def saveFileToCloud(self, df):
        """
        Method to save credit card transactions df as csv in cloud storage
        """

        df.write.format("csv").mode('overwrite').save(self.storage + "/bank_fraud_demo/" + self.username)


    def createDatabase(self, spark):
        """
        Method to create database before data generated is saved to new database and table
        """

        spark.sql("CREATE DATABASE IF NOT EXISTS {}".format(self.dbname))

        print("SHOW DATABASES LIKE '{}'".format(self.dbname))
        spark.sql("SHOW DATABASES LIKE '{}'".format(self.dbname)).show()


    def createOrReplace(self, df):
        """
        Method to create or append data to the BANKING TRANSACTIONS table
        The table is used to simulate batches of new data
        The table is meant to be updated periodically as part of a CML Job
        """

        try:
            df.writeTo("{0}.BANK_TX_XGB_ENG}".format(self.dbname))\
              .using("iceberg").tableProperty("write.format.default", "parquet").append()

        except:
            df.writeTo("{0}.BANK_TX_XGB_ENG".format(self.dbname))\
                .using("iceberg").tableProperty("write.format.default", "parquet").createOrReplace()


    def validateTable(self, spark):
        """
        Method to validate creation of table
        """
        print("SHOW TABLES FROM '{}'".format(self.dbname))
        spark.sql("SHOW TABLES FROM {}".format(self.dbname)).show()


def main():

    USERNAME = os.environ["PROJECT_OWNER"]
    DBNAME = "MA_MLOps"
    STORAGE = "s3a://go01-demo/user"
    CONNECTION_NAME = "go01-aw-dl"

    # Instantiate BankDataGen class
    dg = BankDataGen(USERNAME, DBNAME, STORAGE, CONNECTION_NAME)

    # Create CML Spark Connection
    spark = dg.createSparkConnection()

    # Create Banking Transactions DF
    df = dg.dataGen(spark)

    # Create Spark Database
    dg.createDatabase(spark)

    # Create Iceberg Table in Database
    dg.createOrReplace(df)

    # Validate Iceberg Table in Database
    dg.validateTable(spark)


if __name__ == '__main__':
    main()
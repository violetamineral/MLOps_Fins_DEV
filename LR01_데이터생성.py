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
                    .withColumn("나이", "float", minValue=10, maxValue=100, random=True)
                    .withColumn("카드잔액", "float", minValue=100, maxValue=30000, random=True)
                    .withColumn("계좌잔액", "float", minValue=0.01, maxValue=100000, random=True)
                    .withColumn("모기지잔액", "float", minValue=0.01, maxValue=1000000, random=True)
                    .withColumn("비밀계좌잔액", "float", minValue=0.01, maxValue=100000, random=True)
                    .withColumn("저축계좌잔액", "float", minValue=0.01, maxValue=500000, random=True)
                    .withColumn("비밀저축계좌잔액", "float", minValue=0.01, maxValue=500000, random=True)
                    .withColumn("총금액", "float", minValue=10000, maxValue=500000, random=True)
                    .withColumn("신용대출잔액", "float", minValue=0.01, maxValue=5000, random=True)
                    .withColumn("담보대출잔액", "float", minValue=0.01, maxValue=500000, random=True)
                    .withColumn("총대출잔액", "float", minValue=0.01, maxValue=10000, random=True)
                    .withColumn("경도", "float", minValue=-180, maxValue=180, random=True)
                    .withColumn("위도", "float", minValue=-90, maxValue=90, random=True)
                    .withColumn("거래금액", "float", minValue=0.01, maxValue=30000, random=True)
                    .withColumn("사기여부", "integer", minValue=0, maxValue=1, random=True)
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
            df.writeTo("{0}.BANK_TX_{1}".format(self.dbname, self.username))\
              .using("iceberg").tableProperty("write.format.default", "parquet").append()

        except:
            df.writeTo("{0}.BANK_TX_{1}".format(self.dbname, self.username))\
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
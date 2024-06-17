#****************************************************************************
# (C) Cloudera, Inc. 2020-2023
#  All rights reserved.
#
# XGBoost 학습 - 사기여부 예측 결과 출력
#***************************************************************************/

import os, warnings, sys, logging
import mlflow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, \
    precision_score, recall_score, \
    roc_auc_score, f1_score
import mlflow.sklearn
from xgboost import XGBClassifier
from datetime import date
import cml.data_v1 as cmldata
import pyspark.pandas as ps


USERNAME = os.environ["PROJECT_OWNER"]
DBNAME = "MA_MLOps"
STORAGE = "s3a://go01-demo/user"
CONNECTION_NAME = "go01-aw-dl"
DATE = date.today()
EXPERIMENT_NAME = "xgboost-bank-fraud-sh"

#실험생성
mlflow.set_experiment(EXPERIMENT_NAME)

conn = cmldata.get_connection(CONNECTION_NAME)
spark = conn.get_spark_session()

df_from_sql = ps.read_table('{0}.BANK_TX_XGB_ENG'.format(DBNAME))
df = df_from_sql.to_pandas()

X_train, X_test, y_train, y_test = train_test_split(df.drop("fraud", axis=1), df["fraud"], test_size=0.3)

#Hyperparameter 자동기록
#mlflow.autolog()
mlflow.xgboost.autolog() 

with mlflow.start_run():

    model = XGBClassifier(booster='gbtree',silent=True,min_child_weight=10, max_depth=8 , colsample_bytree=0.8,\
                          colsample_bylevel=0.9,objective = 'binary:logistic', eval_metric="logloss")
    #model = XGBClassifier(booster='gbtree',silent=True,min_child_weight=9, max_depth=7 , colsample_bytree=0.9,\
    #                      gamma =0, colsample_bylevel=0.6,objective = 'binary:logistic', eval_metric="logloss")
    #model = XGBClassifier(booster='gbtree',silent=True,min_child_weight=10, max_depth=6 , colsample_bytree=0.7,\
    #                      colsample_bylevel=0.7,objective = 'binary:logistic', eval_metric="logloss")
    #model = XGBClassifier(booster='gbtree',silent=False,min_child_weight=10, max_depth=8 , colsample_bytree=0.8,\
    #                      gamma = 1,colsample_bylevel=0.9,objective = 'binary:logistic', eval_metric="logloss")
    #model = XGBClassifier(booster='gbtree',silent=False,min_child_weight=9, max_depth=9 , colsample_bytree=0.6,\
    #                      colsample_bylevel=0.8,objective = 'binary:logistic', eval_metric="logloss")
   #학습
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    y_pred = model.predict(X_test)

    #예측결과값
    #model.predict(X_test, params={"predict_method": "predict_proba"})
    #print("predict_proba", model.predict_proba(response))
    
    matrix = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    
    #print("prob",model.predict_proba(y_pred))
    
    print('오차행렬')
    print(matrix)
    print('Accuracy : {0:.3f}'.format(accuracy))
    print('Precision : {0:.3f}'.format(precision))
    print('Recall : {0:.3f}'.format(recall))
    print('f1 : {0:.3f}'.format(f1))
    print('auc : {0:.3f}'.format(auc))
    
    #mlflow.log_metric("accuracy", accuracy)
    #mlflow.log_metric("recall", recall)
    #mlflow.log_metric("precision", precision)
    #mlflow.log_metric("f1", f1)
    #mlflow.log_metric("auc", auc)
    
    mlflow.xgboost.log_model(model, artifact_path="artifacts")#, registered_model_name="my_xgboost_model"

def getLatestExperimentInfo(experimentName):
    """
    Method to capture the latest Experiment Id and Run ID for the provided experimentName
    """
    experimentId = mlflow.get_experiment_by_name(experimentName).experiment_id
    runsDf = mlflow.search_runs(experimentId, run_view_type=1)
    experimentId = runsDf.iloc[-1]['experiment_id']
    experimentRunId = runsDf.iloc[-1]['run_id']

    return experimentId, experimentRunId

experimentId, experimentRunId = getLatestExperimentInfo(EXPERIMENT_NAME)

#Replace Experiment Run ID here:
run = mlflow.get_run(experimentRunId)

pd.DataFrame(data=[run.data.params], index=["Value"]).T
pd.DataFrame(data=[run.data.metrics], index=["Value"]).T

client = mlflow.tracking.MlflowClient()
client.list_artifacts(run_id=run.info.run_id)

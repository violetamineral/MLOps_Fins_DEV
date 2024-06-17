#****************************************************************************
# (C) Cloudera, Inc. 2020-2023
#  All rights reserved.
#
# LogisticRegression 학습
# 사기여부 예측 결과 출력
#***************************************************************************/

import os, warnings, sys, logging
import mlflow
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, \
    precision_score, recall_score, \
    roc_auc_score, f1_score
import mlflow.sklearn
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from datetime import date
import cml.data_v1 as cmldata
import pyspark.pandas as ps
import sys
import pickle


USERNAME = os.environ["PROJECT_OWNER"]
DBNAME = "MA_MLOps"
STORAGE = "s3a://go01-demo/user"
CONNECTION_NAME = "go01-aw-dl"
DATE = date.today()
EXPERIMENT_NAME = "LR-fraud-{0}".format(DATE)

#실험설정
mlflow.set_experiment(EXPERIMENT_NAME)

conn = cmldata.get_connection(CONNECTION_NAME)
spark = conn.get_spark_session()

df_from_sql = ps.read_table('{0}.BANK_TX_{1}'.format(DBNAME, USERNAME))
df = df_from_sql.to_pandas()

X_train, X_test, y_train, y_test = train_test_split(df.drop("사기여부", axis=1), df["사기여부"], test_size=0.3)

#실험 Hyperparameter자동 기록
mlflow.autolog()

with mlflow.start_run():
  
    #LogisticRegression학습
    lr_clf = LogisticRegression(penalty='l2', C=0.1, solver = 'lbfgs', max_iter = 1000, l1_ratio=0.2, tol=0.001)
    #lr_clf = LogisticRegression(penalty='l2', C=0.1, solver = 'newton-cg', max_iter = 1500, l1_ratio=0.2, tol=0.001)
    #lr_clf = LogisticRegression(penalty='l2', C=0.1, solver = 'saga', max_iter = 2000, l1_ratio=0.5, tol=0.01)
    lr_clf.fit(X_train, y_train)
    
    #예측
    y_pred = lr_clf.predict(X_test)

    # Output
    filename = 'model_lr_fraud.pkl'
    pickle.dump(lr_clf, open(filename, 'wb'))
    
    #mlflow.log_param("solver", 'lbfgs')
    #mlflow.log_metric("accuracy", accuracy)
    
    mlflow.sklearn.log_model(lr_clf, artifact_path="artifacts")#, registered_model_name="MR_Model"
    
def getLatestExperimentInfo(experimentName):
    #experimentName의 최신 Experiment Id, Run ID 찾기
    
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

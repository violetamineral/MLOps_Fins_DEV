#****************************************************************************
# (C) Cloudera, Inc. 2020-2023
#  All rights reserved.
#
#  모델 레지스트리 등록 및 배포
#***************************************************************************/
import os, warnings, sys, logging
from datetime import date
import cml.data_v1 as cmldata

from __future__ import print_function
import cmlapi
from cmlapi.rest import ApiException
from pprint import pprint
import json, secrets, os, time
import mlflow


class ModelDeployment():
    """
    Class to manage the model deployment of the xgboost model
    """

    def __init__(self, client, projectId, username, experimentName, experimentId):
        self.client = client
        self.projectId = projectId
        self.username = username
        self.experimentName = experimentName
        self.experimentId = experimentId

    def registerModelFromExperimentRun(self, modelName, experimentId, experimentRunId, modelPath, sessionId):
        """
        Method to register a model from an Experiment Run
        This is an alternative to the mlflow method to register a model via the register_model parameter in the log_model method
        Input: requires an experiment run
        Output:
        """

        model_name = 'XGB_Fraud_SH_'+ sessionId

        CreateRegisteredModelRequest = {
                                        "project_id": os.environ['CDSW_PROJECT_ID'],
                                        "experiment_id" : experimentId,
                                        "run_id": experimentRunId,
                                        "model_name": modelName,
                                        "model_path": modelPath
                                       }

        try:
            # Register a model.
            api_response = self.client.create_registered_model(CreateRegisteredModelRequest)
            pprint(api_response)
        except ApiException as e:
            print("Exception when calling CMLServiceApi->create_registered_model: %s\n" % e)

        return api_response

    def createPRDProject(self):
        """
        Method to create a PRD Project : 새로운 프로젝트를 생성
        """

        createProjRequest = {"name": "MLOps_PROD", "template":"git", "git_url":"https://github.com/violetamineral/MLOps_Fins_PROD.git"}

        try:
            # Create a new project
            api_response = self.client.create_project(createProjRequest)
            pprint(api_response)
        except ApiException as e:
            print("Exception when calling CMLServiceApi->create_project: %s\n" % e)

        return api_response

    def validatePRDProject(self, username):
        """
        Method to test successful project creation
        """

        try:
            # Return all projects, optionally filtered, sorted, and paginated.
            search_filter = {"owner.username" : username}
            search = json.dumps(search_filter)
            api_response = self.client.list_projects(search_filter=search)
            #pprint(api_response)
        except ApiException as e:
            print("Exception when calling CMLServiceApi->list_projects: %s\n" % e)

        return api_response

    def createModel(self, projectId, modelName, modelId, description = "My Spark Clf"):
        """
        Method to create a model
        """

        CreateModelRequest = {
                                "project_id": projectId,
                                "name" : modelName,
                                "description": description,
                                "registered_model_id": modelId
                             }

        try:
            # Create a model.
            api_response = self.client.create_model(CreateModelRequest, projectId)
            pprint(api_response)
        except ApiException as e:
            print("Exception when calling CMLServiceApi->create_model: %s\n" % e)

        return api_response

    def createModelBuild(self, projectId, modelVersionId, modelCreationId):
        """
        Method to create a Model build : 모델 빌드(패키징)
        """

        # Create Model Build
        CreateModelBuildRequest = {
                                    "registered_model_version_id": modelVersionId,
                                    "runtime_identifier": "docker.repository.cloudera.com/cloudera/cdsw/ml-runtime-workbench-python3.9-standard:2023.08.2-b8",
                                    "comment": "invoking model build",
                                    "model_id": modelCreationId
                                  }

        try:
            # Create a model build.
            api_response = self.client.create_model_build(CreateModelBuildRequest, projectId, modelCreationId)
            pprint(api_response)
        except ApiException as e:
            print("Exception when calling CMLServiceApi->create_model_build: %s\n" % e)

        return api_response

    def createModelDeployment(self, modelBuildId, projectId, modelCreationId):
        """
        Method to deploy a model build : 모델 배포
        """

        CreateModelDeploymentRequest = {
          "cpu" : "2",
          "memory" : "4"
        }

        try:
            # Create a model deployment.
            api_response = self.client.create_model_deployment(CreateModelDeploymentRequest, projectId, modelCreationId, modelBuildId)
            pprint(api_response)
        except ApiException as e:
            print("Exception when calling CMLServiceApi->create_model_deployment: %s\n" % e)

        return api_response


client = cmlapi.default_client()
client.list_projects()

#환경변수 설정
projectId = os.environ['CDSW_PROJECT_ID']
username = os.environ["PROJECT_OWNER"]
DBNAME = "MA_MLOps"
STORAGE = "s3a://go01-demo/user"
CONNECTION_NAME = "go01-aw-dl"
DATE = date.today()
experimentName = "xgboost-bank-fraud"

#실험ID 검색
experimentId = mlflow.get_experiment_by_name(experimentName).experiment_id
runsDf = mlflow.search_runs(experimentId, run_view_type=1)

experimentId = runsDf.iloc[-1]['experiment_id']
experimentRunId = runsDf.iloc[-1]['run_id']

#모델 배포
deployment = ModelDeployment(client, projectId, username, experimentName, experimentId)

#모델경로
sessionId = secrets.token_hex(nbytes=4)
modelPath = "artifacts"
modelName = 'XGB_Fraud_SH_'+ sessionId

registeredModelResponse = deployment.registerModelFromExperimentRun(modelName, experimentId, experimentRunId, modelPath, sessionId)
projectCreationResponse = deployment.createPRDProject()
apiResp = deployment.validatePRDProject(username)

prdProjId = projectCreationResponse.id
modelId = registeredModelResponse.model_id
modelVersionId = registeredModelResponse.model_versions[0].model_version_id

registeredModelResponse.model_versions[0].model_version_id

#모델 생성
createModelResponse = deployment.createModel(prdProjId, modelName, modelId)
modelCreationId = createModelResponse.id

#모델 빌드
createModelBuildResponse = deployment.createModelBuild(prdProjId, modelVersionId, modelCreationId)
modelBuildId = createModelBuildResponse.id

#모델 배포
deployment.createModelDeployment(modelBuildId, prdProjId, modelCreationId)

## NOW TRY A REQUEST WITH THIS PAYLOAD!

model_request = {"dataframe_split": {"columns": ["age", "credit_card_balance", "bank_account_balance", "mortgage_balance", "sec_bank_account_balance", "savings_account_balance", "sec_savings_account_balance", "total_est_nworth", "primary_loan_balance", "secondary_loan_balance", "uni_loan_balance", "longitude", "latitude", "transaction_amount"],
                                     "data":[[35.5, 20000.5, 3900.5, 14000.5, 2944.5, 3400.5, 12000.5, 29000.5, 1300.5, 15000.5, 10000.5, 2000.5, 90.5, 120.5]]}}

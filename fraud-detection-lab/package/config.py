import boto3
import sagemaker

AWS_REGION           = boto3.session.Session().region_name
SAGEMAKER_IAM_ROLE   = sagemaker.get_execution_role()
MODEL_DATA_S3_BUCKET = sagemaker.Session().default_bucket()
SOLUTION_NAME        = 'fraud-detection-using-machine-learning'
SOLUTIONS_S3_BUCKET  = 'sagemaker-solutions-prod'
SOLUTION_PREFIX      = 'sagemaker-soln-fdml'
SAGEMAKER_MODE       = 'Studio'

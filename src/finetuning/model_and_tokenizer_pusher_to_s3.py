import os
import boto3
from src.finetuning.merge_base_and_finetuned_model import MergeModels
from src.finetuning.logger import logging
from src.finetuning.exception import CustomException
from src.finetuning.model_trainer import ModelTrainer

class PushToS3:

    def __init__(self):
        try:
            # Set AWS credentials as environment variables (use AWS environment variables or IAM roles for security)
            os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID", "your_aws_access_key")
            os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY", "your_aws_secret_access_key")
            os.environ["AWS_REGION"] = os.getenv("AWS_REGION", "us-east-1")

            # Create an S3 client using boto3 with the credentials from environment variables
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
                aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
                region_name=os.environ["AWS_REGION"]
            )

            logging.info("S3 client initialized successfully.")

            # Set the local model path, S3 bucket, and S3 folder name permanently
            self.local_model_path = "/content/merged_model"
            self.bucket_name = "instruct"
            self.s3_folder_name = "files/"

        except Exception as e:
            logging.error(f"Failed to initialize AWS credentials and S3 client: {str(e)}")
            raise CustomException(f"Error during S3 client initialization: {str(e)}")

    def upload_files_to_s3(self):
        try:
            # Iterate over files in the local directory
            for filename in os.listdir(self.local_model_path):
                local_file_path = os.path.join(self.local_model_path, filename)

                # Ensure that only the 'model.safetensors' and 'tokenizer.json' files are uploaded
                if filename in ["model.safetensors", "tokenizer.json"]:
                    s3_file_path = os.path.join(self.s3_folder_name, filename)

                    logging.info(f"Uploading {local_file_path} to s3://{self.bucket_name}/{s3_file_path}")
                    self.s3_client.upload_file(local_file_path, self.bucket_name, s3_file_path)
                    logging.info(f"Uploaded {filename} successfully.")
                else:
                    logging.info(f"Skipping {filename}, as it is not the model.safetensors or tokenizer.json file.")

        except Exception as e:
            logging.error(f"Failed to upload files to S3: {str(e)}")
            raise CustomException(f"Error during file upload to S3: {str(e)}")

    def initiate_upload(self):
        self.upload_files_to_s3()


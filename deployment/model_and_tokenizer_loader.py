import os
from dotenv import load_dotenv
import boto3
from deployment.logger import logging
from deployment.exception import CustomException

# Load AWS credentials from .env
load_dotenv()

try:
    # Get AWS credentials from environment variables
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION = os.getenv("AWS_REGION")

    # Validate if credentials are loaded
    if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION]):
        raise CustomException("AWS credentials are missing or not set properly in .env file.")

    # Initialize S3 client with explicit credentials
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )
    logging.info("S3 client initialized successfully.")

    # S3 details
    BUCKET_NAME = "instruct"
    S3_MODEL_KEY = "files/model.safetensors"
    S3_TOKENIZER_KEY = "files/tokenizer.json"
    LOCAL_MODEL_DIR = "/tmp/model"

    # Ensure local directory exists
    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
    logging.info(f"Local directory created: {LOCAL_MODEL_DIR}")

    # Local file paths
    local_model_path = os.path.join(LOCAL_MODEL_DIR, "model.safetensors")
    local_tokenizer_path = os.path.join(LOCAL_MODEL_DIR, "tokenizer.json")

    # Function to download a single file from S3
    def download_file(s3_key, local_path):
        """
        Downloads a file from S3 to a local path.
        """
        try:
            s3_client.download_file(Bucket=BUCKET_NAME, Key=s3_key, Filename=local_path)
            logging.info(f"Downloaded {s3_key} to {local_path}")
        except Exception as e:
            logging.error(f"Failed to download {s3_key}: {str(e)}")
            raise CustomException(f"Error downloading {s3_key}: {str(e)}")

    # Download model and tokenizer
    download_file(S3_MODEL_KEY, local_model_path)
    download_file(S3_TOKENIZER_KEY, local_tokenizer_path)

    logging.info("Model and tokenizer downloaded successfully.")

except CustomException as ce:
    logging.error(f"CustomException: {str(ce)}")
except Exception as e:
    logging.error(f"An unexpected error occurred: {str(e)}")

from datasets import load_dataset
from unsloth.chat_templates import standardize_sharegpt
from src.finetuning.logger import logging
from src.finetuning.exception import CustomException


class DataPreparation:

    def __init__(self, dataset_name="charanhu/kannada-instruct-dataset-390-k", split="train"):
        """
        Initializes the DataPreparation class with optional dataset name and split.
        :param dataset_name: Name or path of the dataset to load.
        :param split: The split of the dataset to load (e.g., 'train', 'test').
        """
        self.dataset_name = dataset_name
        self.split = split

    def load_data(self):
        """
        Load the dataset based on the provided dataset name and split.
        :return: The loaded dataset.
        """
        try:
            dataset = load_dataset(self.dataset_name, split=self.split)
            logging.info(f"Dataset '{self.dataset_name}' loaded successfully with split '{self.split}'.")
            return dataset
        except Exception as e:
            logging.error(f"Error loading dataset '{self.dataset_name}': {e}")
            raise CustomException(f"Error loading dataset '{self.dataset_name}': {e}")

    def standardize(self, dataset):
        """
        Standardize the dataset using a specific sharegpt standardization method.
        :param dataset: The dataset to standardize.
        :return: The standardized dataset.
        """
        try:
            standardized_dataset = standardize_sharegpt(dataset)
            logging.info("Dataset standardized successfully.")
            return standardized_dataset
        except Exception as e:
            logging.error(f"Error during dataset standardization: {e}")
            raise CustomException(f"Error during dataset standardization: {e}")

    def initiate_datapreparation(self):
        """
        Initiates the data preparation process: loads and standardizes the dataset.
        :return: The standardized dataset.
        """
        try:
            dataset = self.load_data()  # Load the dataset
            standardized_dataset = self.standardize(dataset)  # Standardize the dataset
            return standardized_dataset
        except Exception as e:
            logging.error(f"Error in data preparation process: {e}")
            raise CustomException(f"Error in data preparation process: {e}")

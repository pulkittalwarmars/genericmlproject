import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    raw_data_path: str=os.path.join('artifacts',"raw.csv")
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")

class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config=DataIngestionConfig() #inside this variable you will have the 3 paths
    
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv("src/notebook/data/stud.csv")
            logging.info("Read the dataset in as DF")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            # saving raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # creating test train spit
            logging.info("train test initiated")
            train_set, test_set=train_test_split(df, test_size=0.2, random_state=42)

            # saving train and test split data
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            raise CustomException(e, sys)

if __name__=="__main__":
    obj=DataIngestion()
    obj.initiate_data_ingestion()
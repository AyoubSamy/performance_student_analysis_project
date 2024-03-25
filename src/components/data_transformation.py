import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd 
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler


from src.exception import CustomException
from src.logger import logging
import os

@dataclass
class DataTransformationConfig :
    #create a preprocessor pkl file 
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_column= ['Mother_Eductaion', 'Revision_Time', 'Father_Education','Age','Past_Class_Failures','Abscence']
            categorical_data=['Wants_Higher_Education','Week_Drugs_Consumption','Extra_Activities','Internet_Access']
 #pipline pour remplacer les valeures manquants par la mediane , et standariser les donnees       
            num_pieline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )
        

            cat_pipeline=Pipeline(
               
             steps=[
                 ("imputer",SimpleImputer(strategy="most_frequent")),
                 (one)
             ]
             )
           
           
        
        
        
        except:
            pass

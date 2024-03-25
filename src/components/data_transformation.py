import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd 
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer
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
            numerical_column=["Age","Mother_Eductaion","Past_Class_Failures","Day_Drugs_Consumption","Health_Status","Abscence"] 
        except:
            pass
    
['Age', 'Mother_Eductaion', 'Father_Education', 'Travel_Time', 'Revision_Time', 'Past_Class_Failures', 'Quality_Family_Relationship', 'Free_Time', 'Go_out', 'Day_Drugs_Consumption', 'Week_Drugs_Consumption', 'Health_Status', 'Abscence', 'First_Periode_Grade', 'Seconde_Periode_Grade', 'Final_Periode_Grade']
['School', 'Sex', 'Home_Address_Type', 'Family_Size', 'Parents_Cohabitation_Status', 'Mother_Job', 'Father_Job', 'Reason_Choosing_School', 'Guardien', 'Extra_Eductional_Support', 'Family_Educational_Support', 'Extra_Paid_Classes', 'Extra_Activities', 'Attended_Nursery', 'Wants_Higher_Education', 'Internet_Access', 'Romantic_Relationship', 'Final_grade', 'Binned_Revision_Time']
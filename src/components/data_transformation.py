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

from src.utils import save_object
@dataclass
class DataTransformationConfig :
    #create a preprocessor pkl file 
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        cette fonction est responsable sur la transformation des donnees 
        
        '''
        
        try:
            numerical_column= ['Mother_Eductaion', 'Revision_Time', 'Father_Education','Age','Past_Class_Failures','Abscence']
            categorical_column=['Wants_Higher_Education','Extra_Activities','Internet_Access']


           #pipline pour remplacer les valeures manquants par la mediane , et standariser les donnees       
            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
        


           #pipline pour les donnees categorielle remplacer les valeurs manquantes par la valeur la plus frequante puis coder les valeur non numeric apres standariser les donnees
            cat_pipeline=Pipeline(
               
             steps=[
                 ("imputer",SimpleImputer(strategy="most_frequent")),
                 ("one_hot_encoder",OneHotEncoder()),
                 ("scaler",StandardScaler(with_mean=False))
             ]
             
             )
            
            logging.info(f"numerical columns {numerical_column}")
            logging.info(f"categorical columns {categorical_column}")
           
            
            preprocessor=ColumnTransformer(

            [
                ("num_pipline",num_pipeline,numerical_column),
                ("cat_pipline",cat_pipeline,categorical_column)
            ]    
            )

            
            return preprocessor
            

           
        
        
        
        except Exception as e:

           raise  CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
          
          try:
              train_df=pd.read_csv(train_path)
              test_df=pd.read_csv(test_path)    

              logging.info("Read train and test data completed") 

              logging.info("obtainig preprocessing object") 
              
              preprocessing_object = self.get_data_transformer_object()
              
              target_column_name ="Final_grade"
              
              numerical_columns =['Mother_Eductaion', 'Revision_Time', 'Father_Education','Age','Past_Class_Failures','Abscence']
              
              input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
              
              target_feature_train_df=train_df[target_column_name]
              
              input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
              target_feature_test_df = test_df[target_column_name]
          
              logging.info(f"Applying preprocessing object on training dataframe and testing dataframe ")

              input_feature_train_arr=preprocessing_object.fit_transform(input_feature_train_df)
              input_feature_test_arr=preprocessing_object.transform(input_feature_test_df)
               
              train_arr=np.c_[
                  input_feature_train_arr , np.array(target_feature_train_df)
              ]
              
              test_arr = np.c_[ input_feature_test_arr, np.array(target_feature_test_df)]
        
              logging.info(f"Saved preprocessing object ")

              save_object(
                  file_path=self.data_transformation_config.preprocessor_obj_file_path,
                  obj=preprocessing_object

              )

              return(train_arr,test_arr,self.data_transformation_config.preprocessor_obj_file_path)

          
          except Exception as e: 
              raise CustomException(e,sys)

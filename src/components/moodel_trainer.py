from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import SVC
import numpy as np
import os
from src.logger import logging
from src.exception import CustomException
import sys
from src.components.data_transformation import DataTransformation

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
        
        
        
            self.models = {
                "Decision Tree": DecisionTreeClassifier(),
                "SGD Classifier": SGDClassifier(),
                "ADABoost Classifier": AdaBoostClassifier(),
                "Logistic Regression": LogisticRegression(),
                "Support Vector Classifier": SVC(),
                "Random Forest": RandomForestClassifier(),
            }

            self.params = {
                "Decision Tree": {
                    'criterion': ['gini', 'entropy'],
                    'splitter': ['best', 'random'],
                    'max_features': ['sqrt', 'log2', None],
                },
                "SGD Classifier": {
                    'loss': ['hinge', 'log', 'modified_huber'],
                    'penalty': ['l2', 'l1', 'elasticnet'],
                },
                "ADABoost Classifier": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 1],
                },
                "Logistic Regression": {
                    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                    'C': np.logspace(-4, 4, 20),
                },
                "Support Vector Classifier": {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['linear', 'rbf', 'poly'],
                },
                "Random Forest": {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                },
        }
        
        except Exception as e:
             raise CustomException(e,sys)
           
           
           
    def train_models(self, X_train, y_train, X_test, y_test):
        try:
            for name, model in self.models.items():
                params = self.params[name]
                grid_search = GridSearchCV(model, params, cv=5, n_jobs=-1, verbose=2)
                grid_search.fit(X_train, y_train)
                print(f"Best parameters for {name}: {grid_search.best_params_}")
                print(f"Best score: {grid_search.best_score_}")
        
        
        except Exception as e:
            raise CustomException(e,sys)



# Créez une instance de ModelTrainer
trainer = ModelTrainer()

# Initiez le ModelTrainer en lui passant vos données d'entraînement et de test
trainer.initiate_model_trainer(train_arr, test_arr)

# Appelez la méthode train_models avec vos données
trainer.train_models(X_train, y_train, X_test, y_test)
        








# @dataclass
# class ModelTrainerConfig:
#     trained_model_file_path=os.path.join("artifacts","model.pkl")

# class ModelTrainer:
#     def __init__(self):
#         self.model_trainer_config=ModelTrainerConfig()


#     def initiate_model_trainer(self,train_array,test_array):
#         try:
#             logging.info("Split training and test input data")
#             X_train,y_train,X_test,y_test=(
#                 train_array[:,:-1],
#                 train_array[:,-1],
#                 test_array[:,:-1],
#                 test_array[:,-1]
#             )
#             self.models = {

#             "Decision Tree": DecisionTreeClassifier(),
#             "SGD Classifier": SGDClassifier(),
#             "ADABoost Classifier": AdaBoostClassifier(),
#             "Logistic Regression": LogisticRegression(),
#             "Support Vector Classifier": SVC(),
#             "Random Forest": RandomForestClassifier(),
            
#             }
#             self.params = {
#             "Decision Tree": {
#                 'criterion': ['gini', 'entropy'],
#                 'splitter': ['best', 'random'],
#                 'max_features': ['sqrt', 'log2', None],
#             },
#             "SGD Classifier": {
#                 'loss': ['hinge', 'log', 'modified_huber'],
#                 'penalty': ['l2', 'l1', 'elasticnet'],
#             },
#             "ADABoost Classifier": {
#                 'n_estimators': [50, 100, 200],
#                 'learning_rate': [0.01, 0.1, 1],
#             },
#             "Logistic Regression": {
#                 'penalty': ['l1', 'l2', 'elasticnet', 'none'],
#                 'C': np.logspace(-4, 4, 20),
#             },
#             "Support Vector Classifier": {
#                 'C': [0.1, 1, 10, 100],
#                 'kernel': ['linear', 'rbf', 'poly'],
#             },
#             "Random Forest": {
#                 'n_estimators': [50, 100, 200],
#                 'max_depth': [None, 10, 20, 30],
#                 'min_samples_split': [2, 5, 10],
#             }
#         }

            

#             model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
#                                              models=models,param=params)
            
#             ## To get best model score from dict
#             best_model_score = max(sorted(model_report.values()))

#             ## To get best model name from dict

#             best_model_name = list(model_report.keys())[
#                 list(model_report.values()).index(best_model_score)
#             ]
#             best_model = models[best_model_name]

#             if best_model_score<0.6:
#                 raise CustomException("No best model found")
#             logging.info(f"Best found model on both training and testing dataset")

#             save_object(
#                 file_path=self.model_trainer_config.trained_model_file_path,
#                 obj=best_model
#             )

#             predicted=best_model.predict(X_test)

#             r2_square = r2_score(y_test, predicted)
#             return r2_square
            



            
#         except Exception as e:
#             raise CustomException(e,sys)
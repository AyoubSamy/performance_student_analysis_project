import logging
import os
from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%m_%d_%H_%M_%S')}.log"#creation de fichier de journalisation .log
logs_path=os.path.join(os.getcwd(),"logs",LOG_FILE)#return le repertoire actuelle de travail 
os.makedirs(logs_path,exist_ok=True)#cration dans le repertoire retourner le repertoire logs 


LOG_FILE_PATH=os.path.join(logs_path,LOG_FILE) 

logging.basicConfig( #configuration les option de journalisation 
   filename=LOG_FILE_PATH,
   format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,

)


#creation de notre application sous forme d'un package nomme:pfe_project
from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT='-e .'
def get_requirements(file_path:str)->List[str]:
    '''
      cette fonction retourne la liste des  exigences
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements ] #replace \n with the blancs 

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements        

setup(
name='pfe_project',
version='0.0.1',
author='AYOUB',
author_email='samy82696@gmail.com',   
packages=find_packages(),
install_requires=get_requirements('requirements.txt'),

   )
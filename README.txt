cd Desktop/Career/__SA/deepsurvk
python3 -m venv deepsurvk_fix
source deepsurvk_fix/bin/activate
pip install ipykernel
python -m ipykernel install --user --name=deepsurvk-fix --display-name "Python (deepsurvk-fix)"

##############################################################################################
!cd /path/to/deepsurvk_source_code
!pip install -e . 

##############################################################################################

pip install -r requirements.txt

tensorflow==2.2.0
keras==2.3.1
numpy==1.18.0
scipy==1.4.1
pandas==1.0.3
lifelines==0.24.15
matplotlib==3.2.1
seaborn==0.10.1
scikit-learn==0.22.2.post1
pydot==1.4.1
graphviz==0.14.1
autograd==1.3
autograd-gamma==0.5.0

##############################################################################################


 


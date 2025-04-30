cd Desktop/Career/__SA/deepsurvk
python3 -m venv deepsurvk_fix
source deepsurvk_fix/bin/activate
pip install ipykernel
python -m ipykernel install --user --name=deepsurvk-fix --display-name "Python (deepsurvk-fix)"
pip install -r requirements.txt

##############################################################################################
!cd /path/to/deepsurvk_source_code
!pip install -e . 

##############################################################################################


 


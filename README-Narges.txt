####################################################################################
***Approach 1: Melika


python3 -V 

# Create virtual environment with specific Python version
/usr/local/bin/python3 -m venv venv_38

# Activate the virtual environment
source venv_38/bin/activate

# Check version (should now show 3.8.10)
python -V
python3 -V

pip install -r requirements_38.txt
pip install protobuf==3.20.3

pip install -e .

python3 examples/01_deepsurvk_quickstart.py

####################################################################################
####################################################################################
***Approach  2: Modify DeepSurvK for modern TensorFlow

# Create a Dockerfile with older Python/TensorFlow
FROM python:3.8-slim
RUN pip install tensorflow==2.2.0 numpy==1.18.0
# ... rest of old requirements

####################################################################################
####################################################################################
***Approach  3: Modify DeepSurvK for modern TensorFlow

Update the DeepSurvK source code to work with TensorFlow 2.13+. This requires code changes in the package.

####################################################################################
####################################################################################
***Approach 4: Use an alternative modern library

pip install scikit-survival  # Modern, actively maintained
# or
pip install pycox  # PyTorch-based

####################################################################################
####################################################################################
***Approach  5: Narges - WORKING SOLUTION ✅

####################################################################################
***QUICK START SUMMARY:
conda create -n deepsurvk_env python=3.8 && conda activate deepsurvk_env
conda install -c conda-forge tensorflow=2.4.0 numpy=1.19.5 scipy=1.5.2 pandas=1.1.3 scikit-learn=0.23.2
pip install absl-py==0.10.0 six==1.15.0 termcolor==1.1.0 typing-extensions==3.7.4 wrapt==1.12.1
pip install keras==2.3.1 seaborn==0.10.1 pydot==1.4.1 graphviz==0.14.1 autograd==1.3 autograd-gamma==0.5.0 lifelines==0.24.15 "pygments>=2.5.1" protobuf==3.15.8 matplotlib==3.5.3
pip install -e . --no-deps
python3 examples/01_deepsurvk_quickstart.py

####################################################################################
***DETAILED SETUP STEPS:

# 1. Environment Setup
deactivate  # Exit any existing environment
brew install --cask miniconda
conda init zsh
conda create -n deepsurvk_env python=3.8
conda activate deepsurvk_env

# 2. Core Dependencies (conda for better ARM64 compatibility)
conda install -c conda-forge tensorflow=2.4.0
conda install -c conda-forge numpy=1.19.5 scipy=1.5.2 pandas=1.1.3 scikit-learn=0.23.2

# 3. TensorFlow Compatibility Packages
pip install absl-py==0.10.0 six==1.15.0 termcolor==1.1.0 typing-extensions==3.7.4 wrapt==1.12.1

# 4. Additional Dependencies
pip install keras==2.3.1 seaborn==0.10.1 pydot==1.4.1 graphviz==0.14.1 autograd==1.3 autograd-gamma==0.5.0 lifelines==0.24.15
pip install "pygments>=2.5.1" protobuf==3.15.8 matplotlib==3.5.3

# 5. Install DeepSurvK (bypass version conflicts)
pip install -e . --no-deps

# 6. Test & Run
python -c "import deepsurvk; print('✅ Success!')"
python3 examples/01_deepsurvk_quickstart.py

####################################################################################
***CRITICAL VERSION MATRIX (2020-2021 ERA COMPATIBILITY):
- Python: 3.8 (newer versions break TF 2.4.0)
- TensorFlow: 2.4.0 (target version for DeepSurvK 0.2.2)
- NumPy: 1.19.5 (1.20+ removed np.object, breaking TF 2.4.0)
- Protobuf: 3.15.8 (4.x+ incompatible with TF 2.4.0)
- Matplotlib: 3.5.3 (ARM64 compatible, older versions fail compilation)

####################################################################################
***EXPECTED BEHAVIORS:
✅ Dependency conflict warnings (SAFE - ignore these)
✅ Model trains and predicts correctly
✅ All survival analysis features work
⚠️  Script crashes at end (TensorFlow ARM64 grappler issue - harmless)

####################################################################################
***TROUBLESHOOTING:
- Dependency conflicts: Expected and safe - we use --no-deps to bypass
- NumPy errors: conda install numpy=1.19.5 --force-reinstall
- Protobuf issues: pip install protobuf==3.15.8
- End crash: Normal TF 2.4.0 ARM64 behavior, doesn't affect functionality

####################################################################################
***SYSTEM REQUIREMENTS:
- macOS with Apple Silicon (M1/M2/M3)
- Miniconda installed
- Python 3.8 environment

####################################################################################
####################################################################################
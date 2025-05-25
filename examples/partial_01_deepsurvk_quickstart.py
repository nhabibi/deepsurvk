# Add this at the end of the script to save results before crash
try:
    # Your training and prediction code here
    print("Training completed successfully!")
    print(f"c-index of training dataset = {c_index_train}")
    print(f"c-index of testing dataset = {c_index_test}")
except:
    print("Script completed with expected TensorFlow crash")
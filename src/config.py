from pathlib import Path

#Centralized configuration for file paths and directories




DATA_DIR = Path("data")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

DATA_FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]

LABEL_COL = 'Label'
Random_State = 101
VAL_SPLIT = 0.2   # divide training set to 3 pars 1 for validation 1 for training and 
                  #  1 for testing 

TEST_SPLIT = 0.2 # percentage of data to be used for testing                  
                  

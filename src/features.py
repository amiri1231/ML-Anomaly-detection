import pandas as pd
import numpy as np

#cloumn chekcer 
def _has_cols(X: pd.DataFrame, cols):
    for c in cols:
        if c not in X.columns:
            return False
    return True



def add_ratio_features(X: pd.DataFrame) -> pd.DataFrame:
 
    X = X.copy()
#adding breakers for my own organization and understanding    
#-------------------------------------------------------------------------------
    # 1) Forward/Backward packet ratio
    #for ddos attacks as they send a lot of forward packets and very few backward packets so they ratio will be
    #really higjh 

    if _has_cols(X, ["Total Fwd Packets", "Total Backward Packets"]):
        fwd_pkts = X["Total Fwd Packets"].astype(float)
        bwd_pkts = X["Total Backward Packets"].astype(float)
        # adding + 1 so it doesnt divbide by zero
        X["Pkt_Fwd_Bwd_Ratio"] = (fwd_pkts + 1.0) / (bwd_pkts + 1.0)  
#-------------------------------------------------------------------------------
    # 2) Forward/Backward BYTES ratio
    # similar to above but with bytes, cusing the total length of packets columns
    #  for example large downloads, backward bytes will be much higher than forward bytes
    if _has_cols(X, ["Total Length of Fwd Packets", "Total Length of Bwd Packets"]):
        fwd_bytes = X["Total Length of Fwd Packets"].astype(float)
        bwd_bytes = X["Total Length of Bwd Packets"].astype(float)
        X["Bytes_Fwd_Bwd_Ratio"] = (fwd_bytes + 1.0) / (bwd_bytes + 1.0)

    return X


def add_all_features(X: pd.DataFrame) -> pd.DataFrame:
   
    X = add_ratio_features(X)
    return X

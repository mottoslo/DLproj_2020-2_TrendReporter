def feature_select(str):
    if "111_click" in str:
        feature = [1,1,1]
    if "110_click" in str:
        feature = [1,1,0]
    if "101_click" in str:
        feature = [1,0,1]
    if "100_click" in str:
        feature = [1,0,0]
    if "011_click" in str:
        feature = [0,1,1]
    if "010_click" in str:
        feature = [0,1,0]
    if "001_click" in str:
        feature = [0,0,1]
    if "000_click" in str:
        feature = [0,0,0]

    return feature
import numpy as np

def extract_data(data):
    names = []
    features = []
    for k, v in data.items():
        name = k
        names.append(name)
        feature = v
        # print(v)
        # print(type(v))
        features.append(feature)
    features = np.asarray(features).astype(np.float32)
    features = np.expand_dims(features, 2)
    return names, features

def extract_proba(data):
    names = []
    features = []
    for k, v in data.items():
        name = k
        names.append(name)
        feature = v
        # print(v)
        # print(type(v))
        features.append(feature)
    features = np.asarray(features).astype(np.float32)
    return names, features
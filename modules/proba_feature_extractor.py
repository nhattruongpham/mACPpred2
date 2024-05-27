import joblib
import numpy as np
import tensorflow as tf
from utils import extract_data, extract_proba

def Process_Classifier(PKL, Prop):
	name = list(Prop.keys()); values = list(Prop.values())
	# with open(PKL, 'rb') as fid1:
	# 	ERT = pickle.load(fid1)
	ml_model = joblib.load(PKL)
	Res1 = ml_model.predict_proba(values)
	Res2 = list(ml_model.predict(values))
	result ={}
	lc = len(name)
	for i in range(lc):
		result[name[i]] = [Res1[i][1], Res2[i]]
	return result
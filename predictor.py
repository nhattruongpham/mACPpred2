import sys
sys.path.append("modules")
sys.path.append("models")
import argparse
import numpy as np
import pandas as pd
from feature_extractor import *
from proba_feature_extractor import *
from utils import *
import tensorflow as tf

def get_90_proba_features(F3, F4, F5, F6, F8, F9, F10, F11, F12, F13, F15, F17, F18, F19, F25, F26):
	# NLP-based:
	# F1 = Bepler
	# F2 = CPCProt
	# F3 =  ESB
	# F4 = ESV
	# F5 = ESM
	# F6 = FastText (FT)
	# F7 = GloVe
	# F8 = PLUSRNN (PRNN)
	# F9 = PTAB
	# F10 = PTBB
	# F11 = PTB
	# F12 = PTU
	# F13 = PTL
	# F14 = PTN
	# F15 = S2V 
	# F16 = Word2Vec (W2V)
	
	# AB
	# AB_F1 = Process_Classifier('models/baseline_models/AB_Bepler.pkl', F1)
	# AB_F2 = Process_Classifier('models/baseline_models/AB_CPCProt.pkl', F2)
	AB_F3 = Process_Classifier('models/baseline_models/AB_ESB.pkl', F3)
	AB_F4 = Process_Classifier('models/baseline_models/AB_ESV.pkl', F4)
	AB_F5 = Process_Classifier('models/baseline_models/AB_ESM.pkl', F5)
	AB_F6 = Process_Classifier('models/baseline_models/AB_FT.pkl', F6)
	# AB_F7 = Process_Classifier('models/baseline_models/AB_GloVe.pkl', F7)
	AB_F8 = Process_Classifier('models/baseline_models/AB_PLUSRNN.pkl', F8)
	AB_F9 = Process_Classifier('models/baseline_models/AB_PTAB.pkl', F9)
	AB_F10 = Process_Classifier('models/baseline_models/AB_PTBB.pkl', F10)
	AB_F11 = Process_Classifier('models/baseline_models/AB_PTB.pkl', F11)
	AB_F12 = Process_Classifier('models/baseline_models/AB_PTU.pkl', F12)
	AB_F13 = Process_Classifier('models/baseline_models/AB_PTL.pkl', F13)
	# AB_F14 = Process_Classifier('models/baseline_models/AB_PTN.pkl', F14)
	AB_F15 = Process_Classifier('models/baseline_models/AB_S2V.pkl', F15)
	# AB_F16 = Process_Classifier('models/baseline_models/AB_W2V.pkl', F16)

	# ERT
	# ERT_F1 = Process_Classifier('models/baseline_models/ERT_Bepler.pkl', F1)
	# ERT_F2 = Process_Classifier('models/baseline_models/ERT_CPCProt.pkl', F2)
	ERT_F3 = Process_Classifier('models/baseline_models/ERT_ESB.pkl', F3)
	ERT_F4 = Process_Classifier('models/baseline_models/ERT_ESV.pkl', F4)
	ERT_F5 = Process_Classifier('models/baseline_models/ERT_ESM.pkl', F5)
	# ERT_F6 = Process_Classifier('models/baseline_models/ERT_FT.pkl', F6)
	# ERT_F7 = Process_Classifier('models/baseline_models/ERT_GloVe.pkl', F7)
	# ERT_F8 = Process_Classifier('models/baseline_models/ERT_PLUSRNN.pkl', F8)
	# ERT_F9 = Process_Classifier('models/baseline_models/ERT_PTAB.pkl', F9)
	# ERT_F10 = Process_Classifier('models/baseline_models/ERT_PTBB.pkl', F10)
	ERT_F11 = Process_Classifier('models/baseline_models/ERT_PTB.pkl', F11)
	ERT_F12 = Process_Classifier('models/baseline_models/ERT_PTU.pkl', F12)
	ERT_F13 = Process_Classifier('models/baseline_models/ERT_PTL.pkl', F13)
	# ERT_F14 = Process_Classifier('models/baseline_models/ERT_PTN.pkl', F14)
	ERT_F15 = Process_Classifier('models/baseline_models/ERT_S2V.pkl', F15)
	# ERT_F16 = Process_Classifier('models/baseline_models/ERT_W2V.pkl', F16)

	# GB
	# GB_F1 = Process_Classifier('models/baseline_models/GB_Bepler.pkl', F1)
	# GB_F2 = Process_Classifier('models/baseline_models/GB_CPCProt.pkl', F2)
	GB_F3 = Process_Classifier('models/baseline_models/GB_ESB.pkl', F3)
	GB_F4 = Process_Classifier('models/baseline_models/GB_ESV.pkl', F4)
	GB_F5 = Process_Classifier('models/baseline_models/GB_ESM.pkl', F5)
	# GB_F6 = Process_Classifier('models/baseline_models/GB_FT.pkl', F6)
	# GB_F7 = Process_Classifier('models/baseline_models/GB_GloVe.pkl', F7)
	# GB_F8 = Process_Classifier('models/baseline_models/GB_PLUSRNN.pkl', F8)
	GB_F9 = Process_Classifier('models/baseline_models/GB_PTAB.pkl', F9)
	# GB_F10 = Process_Classifier('models/baseline_models/GB_PTBB.pkl', F10)
	GB_F11 = Process_Classifier('models/baseline_models/GB_PTB.pkl', F11)
	GB_F12 = Process_Classifier('models/baseline_models/GB_PTU.pkl', F12)
	GB_F13 = Process_Classifier('models/baseline_models/GB_PTL.pkl', F13)
	# GB_F14 = Process_Classifier('models/baseline_models/GB_PTN.pkl', F14)
	GB_F15 = Process_Classifier('models/baseline_models/GB_S2V.pkl', F15)
	# GB_F16 = Process_Classifier('models/baseline_models/GB_W2V.pkl', F16)

	# LRT
	# LRT_F1 = Process_Classifier('models/baseline_models/LRT_Bepler.pkl', F1)
	# LRT_F2 = Process_Classifier('models/baseline_models/LRT_CPCProt.pkl', F2)
	LRT_F3 = Process_Classifier('models/baseline_models/LRT_ESB.pkl', F3)
	LRT_F4 = Process_Classifier('models/baseline_models/LRT_ESV.pkl', F4)
	LRT_F5 = Process_Classifier('models/baseline_models/LRT_ESM.pkl', F5)
	# LRT_F6 = Process_Classifier('models/baseline_models/LRT_FT.pkl', F6)
	# LRT_F7 = Process_Classifier('models/baseline_models/LRT_GloVe.pkl', F7)
	# LRT_F8 = Process_Classifier('models/baseline_models/LRT_PLUSRNN.pkl', F8)
	LRT_F9 = Process_Classifier('models/baseline_models/LRT_PTAB.pkl', F9)
	# LRT_F10 = Process_Classifier('models/baseline_models/LRT_PTBB.pkl', F10)
	LRT_F11 = Process_Classifier('models/baseline_models/LRT_PTB.pkl', F11)
	LRT_F12 = Process_Classifier('models/baseline_models/LRT_PTU.pkl', F12)
	LRT_F13 = Process_Classifier('models/baseline_models/LRT_PTL.pkl', F13)
	# LRT_F14 = Process_Classifier('models/baseline_models/LRT_PTN.pkl', F14)
	LRT_F15 = Process_Classifier('models/baseline_models/LRT_S2V.pkl', F15)
	# LRT_F16 = Process_Classifier('models/baseline_models/LRT_W2V.pkl', F16)

	# RF
	# RF_F1 = Process_Classifier('models/baseline_models/RF_Bepler.pkl', F1)
	# RF_F2 = Process_Classifier('models/baseline_models/RF_CPCProt.pkl', F2)
	RF_F3 = Process_Classifier('models/baseline_models/RF_ESB.pkl', F3)
	RF_F4 = Process_Classifier('models/baseline_models/RF_ESV.pkl', F4)
	RF_F5 = Process_Classifier('models/baseline_models/RF_ESM.pkl', F5)
	# RF_F6 = Process_Classifier('models/baseline_models/RF_FT.pkl', F6)
	# RF_F7 = Process_Classifier('models/baseline_models/RF_GloVe.pkl', F7)
	# RF_F8 = Process_Classifier('models/baseline_models/RF_PLUSRNN.pkl', F8)
	# RF_F9 = Process_Classifier('models/baseline_models/RF_PTAB.pkl', F9)
	# RF_F10 = Process_Classifier('models/baseline_models/RF_PTBB.pkl', F10)
	RF_F11 = Process_Classifier('models/baseline_models/RF_PTB.pkl', F11)
	RF_F12 = Process_Classifier('models/baseline_models/RF_PTU.pkl', F12)
	RF_F13 = Process_Classifier('models/baseline_models/RF_PTL.pkl', F13)
	# RF_F14 = Process_Classifier('models/baseline_models/RF_PTN.pkl', F14)
	RF_F15 = Process_Classifier('models/baseline_models/RF_S2V.pkl', F15)
	# RF_F16 = Process_Classifier('models/baseline_models/RF_W2V.pkl', F16)

	# XGB
	# XGB_F1 = Process_Classifier('models/baseline_models/XGB_Bepler.pkl', F1)
	# XGB_F2 = Process_Classifier('models/baseline_models/XGB_CPCProt.pkl', F2)
	XGB_F3 = Process_Classifier('models/baseline_models/XGB_ESB.pkl', F3)
	XGB_F4 = Process_Classifier('models/baseline_models/XGB_ESV.pkl', F4)
	XGB_F5 = Process_Classifier('models/baseline_models/XGB_ESM.pkl', F5)
	# XGB_F6 = Process_Classifier('models/baseline_models/XGB_FT.pkl', F6)
	# XGB_F7 = Process_Classifier('models/baseline_models/XGB_GloVe.pkl', F7)
	XGB_F8 = Process_Classifier('models/baseline_models/XGB_PLUSRNN.pkl', F8)
	XGB_F9 = Process_Classifier('models/baseline_models/XGB_PTAB.pkl', F9)
	XGB_F10 = Process_Classifier('models/baseline_models/XGB_PTBB.pkl', F10)
	XGB_F11 = Process_Classifier('models/baseline_models/XGB_PTB.pkl', F11)
	XGB_F12 = Process_Classifier('models/baseline_models/XGB_PTU.pkl', F12)
	XGB_F13 = Process_Classifier('models/baseline_models/XGB_PTL.pkl', F13)
	# XGB_F14 = Process_Classifier('models/baseline_models/XGB_PTN.pkl', F14)
	XGB_F15 = Process_Classifier('models/baseline_models/XGB_S2V.pkl', F15)
	# XGB_F16 = Process_Classifier('models/baseline_models/XGB_W2V.pkl', F16)

	# CB
	# CB_F1 = Process_Classifier('models/baseline_models/CB_Bepler.pkl', F1)
	# CB_F2 = Process_Classifier('models/baseline_models/CB_CPCProt.pkl', F2)
	CB_F3 = Process_Classifier('models/baseline_models/CB_ESB.pkl', F3)
	CB_F4 = Process_Classifier('models/baseline_models/CB_ESV.pkl', F4)
	CB_F5 = Process_Classifier('models/baseline_models/CB_ESM.pkl', F5)
	# CB_F6 = Process_Classifier('models/baseline_models/CB_FT.pkl', F6)
	# CB_F7 = Process_Classifier('models/baseline_models/CB_GloVe.pkl', F7)
	CB_F8 = Process_Classifier('models/baseline_models/CB_PLUSRNN.pkl', F8)
	# CB_F9 = Process_Classifier('models/baseline_models/CB_PTAB.pkl', F9)
	CB_F10 = Process_Classifier('models/baseline_models/CB_PTBB.pkl', F10)
	CB_F11 = Process_Classifier('models/baseline_models/CB_PTB.pkl', F11)
	CB_F12 = Process_Classifier('models/baseline_models/CB_PTU.pkl', F12)
	CB_F13 = Process_Classifier('models/baseline_models/CB_PTL.pkl', F13)
	# CB_F14 = Process_Classifier('models/baseline_models/CB_PTN.pkl', F14)
	CB_F15 = Process_Classifier('models/baseline_models/CB_S2V.pkl', F15)
	# CB_F16 = Process_Classifier('models/baseline_models/CB_W2V.pkl', F16)

	# LGB
	# LGB_F1 = Process_Classifier('models/baseline_models/LGB_Bepler.pkl', F1)
	# LGB_F2 = Process_Classifier('models/baseline_models/LGB_CPCProt.pkl', F2)
	LGB_F3 = Process_Classifier('models/baseline_models/LGB_ESB.pkl', F3)
	LGB_F4 = Process_Classifier('models/baseline_models/LGB_ESV.pkl', F4)
	LGB_F5 = Process_Classifier('models/baseline_models/LGB_ESM.pkl', F5)
	# LGB_F6 = Process_Classifier('models/baseline_models/LGB_FT.pkl', F6)
	# LGB_F7 = Process_Classifier('models/baseline_models/LGB_GloVe.pkl', F7)
	# LGB_F8 = Process_Classifier('models/baseline_models/LGB_PLUSRNN.pkl', F8)
	# LGB_F9 = Process_Classifier('models/baseline_models/LGB_PTAB.pkl', F9)
	# LGB_F10 = Process_Classifier('models/baseline_models/LGB_PTBB.pkl', F10)
	LGB_F11 = Process_Classifier('models/baseline_models/LGB_PTB.pkl', F11)
	LGB_F12 = Process_Classifier('models/baseline_models/LGB_PTU.pkl', F12)
	LGB_F13 = Process_Classifier('models/baseline_models/LGB_PTL.pkl', F13)
	# LGB_F14 = Process_Classifier('models/baseline_models/LGB_PTN.pkl', F14)
	LGB_F15 = Process_Classifier('models/baseline_models/LGB_S2V.pkl', F15)
	# LGB_F16 = Process_Classifier('models/baseline_models/LGB_W2V.pkl', F16)
	
    # ANN
	ANN_F3 = Process_Classifier('models/baseline_models/ANN_ESB.pkl', F3)
	ANN_F4 = Process_Classifier('models/baseline_models/ANN_ESV.pkl', F4)
	ANN_F5 = Process_Classifier('models/baseline_models/ANN_ESM.pkl', F5)
	ANN_F9 = Process_Classifier('models/baseline_models/ANN_PTAB.pkl', F9)
	ANN_F11 = Process_Classifier('models/baseline_models/ANN_PTB.pkl', F11)
	ANN_F12 = Process_Classifier('models/baseline_models/ANN_PTU.pkl', F12)
	ANN_F15 = Process_Classifier('models/baseline_models/ANN_S2V.pkl', F15)

	# Conventional-based:
	# F17 = AAC
	# F18 = APAAC
	# F19 = PDE
	# F20 = CKSAAGP
	# F21 = CTDC F22, F23, F24, F25, F26
	# AB
	# AB_F17 = Process_Classifier('models/baseline_models/AB_AAC.pkl', F17)
	AB_F18 = Process_Classifier('models/baseline_models/AB_APAAC.pkl', F18)
	AB_F19 = Process_Classifier('models/baseline_models/AB_PDE.pkl', F19)
	# AB_F20 = Process_Classifier('models/baseline_models/AB_CKSAAGP.pkl', F20)
	# AB_F21 = Process_Classifier('models/baseline_models/AB_CTDC.pkl', F21)
	# AB_F22 = Process_Classifier('models/baseline_models/AB_CTDD.pkl', F22)
	# AB_F23 = Process_Classifier('models/baseline_models/AB_DDE.pkl', F23)
	# AB_F24 = Process_Classifier('models/baseline_models/AB_DPC.pkl', F24)
	AB_F25 = Process_Classifier('models/baseline_models/AB_PAAC.pkl', F25)
	# AB_F26 = Process_Classifier('models/baseline_models/AB_QSOrder.pkl', F26)

	# ERT
	ERT_F17 = Process_Classifier('models/baseline_models/ERT_AAC.pkl', F17)
	ERT_F18 = Process_Classifier('models/baseline_models/ERT_APAAC.pkl', F18)
	ERT_F19 = Process_Classifier('models/baseline_models/ERT_PDE.pkl', F19)
	ERT_F25 = Process_Classifier('models/baseline_models/ERT_PAAC.pkl', F25)
	ERT_F26 = Process_Classifier('models/baseline_models/ERT_QSOrder.pkl', F26)

	# GB
	# GB_F17 = Process_Classifier('models/baseline_models/GB_AAC.pkl', F17)
	GB_F18 = Process_Classifier('models/baseline_models/GB_APAAC.pkl', F18)
	# GB_F19 = Process_Classifier('models/baseline_models/GB_PDE.pkl', F19)
	# GB_F25 = Process_Classifier('models/baseline_models/GB_PAAC.pkl', F25)
	GB_F26 = Process_Classifier('models/baseline_models/GB_QSOrder.pkl', F26)

	# RF
	# RF_F17 = Process_Classifier('models/baseline_models/RF_AAC.pkl', F17)
	RF_F18 = Process_Classifier('models/baseline_models/RF_APAAC.pkl', F18)
	# RF_F19 = Process_Classifier('models/baseline_models/RF_PDE.pkl', F19)
	# RF_F25 = Process_Classifier('models/baseline_models/RF_PAAC.pkl', F25)
	RF_F26 = Process_Classifier('models/baseline_models/RF_QSOrder.pkl', F26)

	# XGB
	# XGB_F17 = Process_Classifier('models/baseline_models/XGB_AAC.pkl', F17)
	XGB_F18 = Process_Classifier('models/baseline_models/XGB_APAAC.pkl', F18)
	XGB_F19 = Process_Classifier('models/baseline_models/XGB_PDE.pkl', F19)
	# XGB_F25 = Process_Classifier('models/baseline_models/XGB_PAAC.pkl', F25)
	# XGB_F26 = Process_Classifier('models/baseline_models/XGB_QSOrder.pkl', F26)

	# CB
	# CB_F17 = Process_Classifier('models/baseline_models/CB_AAC.pkl', F17)
	CB_F18 = Process_Classifier('models/baseline_models/CB_APAAC.pkl', F18)
	# CB_F19 = Process_Classifier('models/baseline_models/CB_PDE.pkl', F19)
	# CB_F25 = Process_Classifier('models/baseline_models/CB_PAAC.pkl', F25)
	CB_F26 = Process_Classifier('models/baseline_models/CB_QSOrder.pkl', F26)

	# # LGB
	# LGB_F17 = Process_Classifier('models/baseline_models/LGB_AAC.pkl', F17)
	# LGB_F18 = Process_Classifier('models/baseline_models/LGB_APAAC.pkl', F18)
	# LGB_F19 = Process_Classifier('models/baseline_models/LGB_PDE.pkl', F19)
	# LGB_F25 = Process_Classifier('models/baseline_models/LGB_PAAC.pkl', F25)
	# LGB_F26 = Process_Classifier('models/baseline_models/LGB_QSOrder.pkl', F26)
	
	# Get probabilistic features
	PF = {}
	for k, v in CB_F12.items():
		PF[k] = [CB_F12[k][0], AB_F3[k][0], GB_F3[k][0], XGB_F3[k][0], LRT_F15[k][0], AB_F15[k][0], AB_F4[k][0], AB_F13[k][0], XGB_F12[k][0], RF_F3[k][0], AB_F5[k][0], LRT_F4[k][0], ERT_F3[k][0], AB_F12[k][0], XGB_F11[k][0],
				GB_F5[k][0], XGB_F5[k][0], LGB_F3[k][0], AB_F11[k][0], GB_F12[k][0], XGB_F4[k][0], LGB_F12[k][0], GB_F15[k][0], CB_F13[k][0], LRT_F5[k][0], GB_F13[k][0], XGB_F13[k][0], RF_F5[k][0], LGB_F5[k][0], GB_F11[k][0],
				GB_F4[k][0], AB_F9[k][0], LGB_F13[k][0], CB_F3[k][0], LGB_F11[k][0], CB_F15[k][0], RF_F12[k][0], ERT_F5[k][0], RF_F4[k][0], ANN_F11[k][0], LGB_F4[k][0], XGB_F15[k][0], RF_F13[k][0], RF_F15[k][0], LRT_F11[k][0],
				ERT_F4[k][0], ERT_F15[k][0], CB_F11[k][0], ERT_F13[k][0], ERT_F12[k][0], LGB_F15[k][0], ERT_F18[k][0], CB_F5[k][0], LRT_F3[k][0], RF_F11[k][0], ANN_F5[k][0], XGB_F19[k][0], ERT_F26[k][0], ANN_F15[k][0], ERT_F11[k][0],
				AB_F18[k][0], ERT_F19[k][0], CB_F4[k][0], CB_F18[k][0], AB_F8[k][0], ANN_F3[k][0], CB_F26[k][0], ERT_F17[k][0], LRT_F13[k][0], ERT_F25[k][0], LRT_F12[k][0], ANN_F4[k][0], ANN_F12[k][0], XGB_F9[k][0], RF_F18[k][0],
				GB_F9[k][0], CB_F8[k][0], AB_F10[k][0], AB_F19[k][0], GB_F18[k][0], XGB_F10[k][0], XGB_F18[k][0], RF_F26[k][0], GB_F26[k][0], LRT_F9[k][0], AB_F6[k][0], AB_F25[k][0], CB_F10[k][0], ANN_F9[k][0], XGB_F8[k][0]
		]

	return PF


def model_predict(list_test_df_1d, list_test_proba):
    X_test_1d_ls = []
    input_shape_1d_ls = []
    for i in range(len(list_test_df_1d)):
        # print(list_test_df_1d[i])
        names, features = extract_data(list_test_df_1d[i])
        input_shape = features[0].shape
        X_test_1d_ls.append(features)
        input_shape_1d_ls.append(input_shape)
	
    _, proba_features = extract_proba(list_test_proba)

    pre_pro_testing = []

    sets = ['model_f1', 'model_f2', 'model_f3', 'model_f4', 'model_f5']
	
    for se in sets:
        model = tf.keras.models.load_model("models/final_models/" + se + ".h5")

        # # Calculate scores during testing
        y_pred_testing = model.predict([X_test_1d_ls, proba_features])
        pre_pro_testing.append(y_pred_testing)

    pre_prob_testing = np.mean(pre_pro_testing, axis=0)
    y_pred_testing_cls = (pre_prob_testing > 0.5).astype(np.float32)
    
    Res1 = list(pre_prob_testing)
    Res2 = list(y_pred_testing_cls)
    result ={}
    lc = len(names)
    for i in range(lc):
        result[names[i]] = [float(np.round(Res1[i], 3)), Res2[i]]
    del model
    return result


def get_predictions(data):
	Ndata = []
	for k, v in data.items():
		Ndata.append([k, v])
	list_test_1d = []

	PTL_e =  extract_PTL(Ndata) #F13
	ESB_e = extract_ESB(Ndata) #F3
	PTB_e = extract_PTB(Ndata) #F11
	ESM_e = extract_ESM(Ndata) #F5
	ESV_e = extract_ESV(Ndata) #F4
	PTU_e = extract_PTU(Ndata) #F12
	PTBB_e = extract_PTBB(Ndata) #F10
	PTAB_e = extract_PTAB(Ndata) #F9
	S2V_e = extract_S2V(Ndata) #F15
	# Bepler_e = extract_Bepler(Ndata) #F1
	# CPCProt_e = extract_CPCProt(Ndata) #F2
	FT_e = extract_FT(Ndata) #F6
	# GloVe_e = extract_GloVe(Ndata) #F7
	# PTN_e = extract_PTN(Ndata) #F14
	PLUSRNN_e = extract_PLUSRNN(Ndata) #F8
	# W2V_e = extract_W2V(Ndata) #F16

	F_AAC = AAC(Ndata) #F17
	F_PAAC = Paac(Ndata) #F25
	F_APAAC = APaac(Ndata) #F18
	F_PDE = PDE(Ndata) #F19
	# F_CKSAAGP = CKSAAGP(Ndata) #F20
	# F_CTDC = CTDC(Ndata) #F21
	# F_CTDD = CTDD(Ndata) #F22
	# F_DDE = DDE(Ndata) #F23
	# F_DPC = DPC(Ndata) #F24
	F_QSOrder = QSOrder(Ndata) #F26

	list_test_1d.append(PTL_e)
	list_test_1d.append(ESB_e)
	list_test_1d.append(PTB_e)
	list_test_1d.append(ESM_e)
	list_test_1d.append(ESV_e)
	list_test_1d.append(PTU_e)
	list_test_1d.append(S2V_e)


	# list_test_proba = get_90_proba_features(Bepler_e, CPCProt_e, ESB_e, ESV_e, ESM_e, FT_e, GloVe_e, PLUSRNN_e, PTAB_e,
	# 										  	PTBB_e, PTB_e, PTU_e, PTL_e, PTN_e, S2V_e, W2V_e,
	# 											F_AAC, F_APAAC, F_PDE, F_CKSAAGP, F_CTDC, F_CTDD, F_DDE, F_DPC, F_PAAC, F_QSOrder
	# 										  )
	list_test_proba = get_90_proba_features(ESB_e, ESV_e, ESM_e, FT_e, PLUSRNN_e, PTAB_e,
											  	PTBB_e, PTB_e, PTU_e, PTL_e, S2V_e, 
												F_AAC, F_APAAC, F_PDE, F_PAAC, F_QSOrder
											  )

	res = model_predict(list_test_1d, list_test_proba)

	return res

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_file', type=str, required=True)
	parser.add_argument('--output_file', type=str, required=True)
	args = parser.parse_args()
	data = {}
	with open(args.input_file) as fh:
		for line in fh:
			if line.startswith('>'):
				header = line.strip()
				data[header] = ''
			else:
				data[header] += line.strip()
	res = get_predictions(data)
	result = []
	for k, v in res.items():
		name = k.replace('>', '')
		if v[1] == 1:
			loc = 'ACP'
		else:
			loc = 'Non-ACP'
		result.append([name, loc, float(round(v[0], 3))])
	pd_res = [[res[0], res[1], res[2]] for res in result]
	df = pd.DataFrame(pd_res, columns=['Name', 'Class', 'Probability'])
	df.to_csv(args.output_file, index=False)
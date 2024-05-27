import torch
import re, os, platform
from collections import Counter
import peptidy
import numpy as np
import math
from protlearn.features import paac, apaac
from bio_embeddings.embed import ESM1vEmbedder, ESM1bEmbedder, ProtTransT5XLU50Embedder, ProtTransT5UniRef50Embedder, \
	ProtTransXLNetUniRef100Embedder, ProtTransBertBFDEmbedder, ProtTransAlbertBFDEmbedder, SeqVecEmbedder, PLUSRNNEmbedder, \
    BeplerEmbedder, Word2VecEmbedder, GloveEmbedder, FastTextEmbedder, CPCProtEmbedder, ESMEmbedder, ProtTransT5BFDEmbedder

device = "cpu"

# ESV = ESM1vEmbedder(ensemble_id=1, device='cuda')
# ESB = ESM1bEmbedder(device='cuda')
# PTU = ProtTransT5UniRef50Embedder(device='cuda')
# PTL = ProtTransT5XLU50Embedder(device='cuda')
# PTN = ProtTransXLNetUniRef100Embedder(device='cuda')
# PTBB = ProtTransBertBFDEmbedder(device='cuda')
# PTAB = ProtTransAlbertBFDEmbedder(device='cuda')
# S2V = SeqVecEmbedder(device=device)
# PLUSRNN = PLUSRNNEmbedder(device=device)
# Bepler = BeplerEmbedder(device=device)
# W2V = Word2VecEmbedder(device=device)
# GloVe = GloveEmbedder(device=device)
# FT = FastTextEmbedder(device=device)
# CPCProt = CPCProtEmbedder(device=device)
# ESM = ESMEmbedder(device=device)
# PTB = ProtTransT5BFDEmbedder(device=device)

ESM1v_MODEL_PATH = "/media/nhattruongpham/DATA/DjangoProjects/DjangoProjects/BALALAB/utils/pretrained_models/esm1v_t33_650M_UR90S_1.pt"
ESM1b_MODEL_PATH = "/media/nhattruongpham/DATA/DjangoProjects/DjangoProjects/BALALAB/utils/pretrained_models/esm1b_t33_650M_UR50S.pt"

# call class -> functions
ESV = ESM1vEmbedder(model_file=ESM1v_MODEL_PATH, ensemble_id=1, device='cuda')
ESB = ESM1bEmbedder(model_file=ESM1b_MODEL_PATH, device='cuda')
PTU = ProtTransT5UniRef50Embedder(model_directory="/home/nhattruongpham/CBBL_SKKU_Projs/Pretrained_Model_Files/bio_embeddings/ProtTransT5UniRef50", device='cuda')
PTL = ProtTransT5XLU50Embedder(model_directory="/home/nhattruongpham/CBBL_SKKU_Projs/Pretrained_Model_Files/bio_embeddings/ProtTransT5XLU50", device='cuda')
PTN = ProtTransXLNetUniRef100Embedder(model_directory="/home/nhattruongpham/CBBL_SKKU_Projs/Pretrained_Model_Files/bio_embeddings/ProtTransXLNetUniRef100", device='cuda')
PTBB = ProtTransBertBFDEmbedder(model_directory="/home/nhattruongpham/CBBL_SKKU_Projs/Pretrained_Model_Files/bio_embeddings/ProtTransBertBFD", device='cuda')
PTAB = ProtTransAlbertBFDEmbedder(model_directory="/home/nhattruongpham/CBBL_SKKU_Projs/Pretrained_Model_Files/bio_embeddings/ProtTransAlbertBFD", device='cuda')
S2V = SeqVecEmbedder(weights_file="/home/nhattruongpham/CBBL_SKKU_Projs/Pretrained_Model_Files/bio_embeddings/Seq2Vec/weights_file", options_file="/home/nhattruongpham/CBBL_SKKU_Projs/Pretrained_Model_Files/bio_embeddings/Seq2Vec/options_file", device=device)
PLUSRNN = PLUSRNNEmbedder(model_file="/home/nhattruongpham/CBBL_SKKU_Projs/Pretrained_Model_Files/bio_embeddings/PLUSRNN/model_file", device=device)
Bepler = BeplerEmbedder(model_file="/home/nhattruongpham/CBBL_SKKU_Projs/Pretrained_Model_Files/bio_embeddings/Bepler/model_file", device=device)
W2V = Word2VecEmbedder(model_file="/home/nhattruongpham/CBBL_SKKU_Projs/Pretrained_Model_Files/bio_embeddings/Word2Vec/model_file", device=device)
GloVe = GloveEmbedder(model_file="/home/nhattruongpham/CBBL_SKKU_Projs/Pretrained_Model_Files/bio_embeddings/GloVe/model_file", device=device)
FT = FastTextEmbedder(model_file="/home/nhattruongpham/CBBL_SKKU_Projs/Pretrained_Model_Files/bio_embeddings/FastText/model_file", device=device)
CPCProt = CPCProtEmbedder(model_file="/home/nhattruongpham/CBBL_SKKU_Projs/Pretrained_Model_Files/bio_embeddings/CPCProt/model_file", device=device)
ESM = ESMEmbedder(model_file="/home/nhattruongpham/CBBL_SKKU_Projs/Pretrained_Model_Files/bio_embeddings/ESM/esm1_t34_670M_UR50S.pt", device=device)
PTB = ProtTransT5BFDEmbedder(model_directory="/home/nhattruongpham/CBBL_SKKU_Projs/Pretrained_Model_Files/bio_embeddings/ProtTransT5BFD", device=device)

# Extract 16 NLP-based features
def extract_Bepler(fastas):
	Names = []
	Seqs = []
	n_embeddings = {}
	model = Bepler
	for fasta in fastas:
		Names.append(fasta[0])
		Seqs.append(fasta[1])
	with torch.no_grad():
		for i in range(len(Seqs)):
			name = Names[i]
			seq = Seqs[i]
			embedding = model.embed(seq)
			bepler_1d_embed = model.reduce_per_protein(embedding)
			n_embeddings[name] = bepler_1d_embed
	del model, embedding, bepler_1d_embed
	return n_embeddings

def extract_CPCProt(fastas):
	Names = []
	Seqs = []
	n_embeddings = {}
	model = CPCProt
	for fasta in fastas:
		Names.append(fasta[0])
		Seqs.append(fasta[1])
	with torch.no_grad():
		for i in range(len(Seqs)):
			name = Names[i]
			seq = Seqs[i]
			embedding = model.embed(seq)
			cpcprot_1d_embed = model.reduce_per_protein(embedding)
			n_embeddings[name] = cpcprot_1d_embed
	del model, embedding, cpcprot_1d_embed
	return n_embeddings

def extract_ESV(fastas):
	Names = []
	Seqs = []
	n_embeddings = {}
	model = ESV
	for fasta in fastas:
		Names.append(fasta[0])
		Seqs.append(fasta[1])
	with torch.no_grad():
		for i in range(len(Seqs)):
			name = Names[i]
			seq = Seqs[i]
			embedding = model.embed(seq)
			esm1v_1d_embed = model.reduce_per_protein(embedding)
			n_embeddings[name] = esm1v_1d_embed
	del model, embedding, esm1v_1d_embed
	return n_embeddings

def extract_ESB(fastas):
	Names = []
	Seqs = []
	n_embeddings = {}
	model = ESB
	for fasta in fastas:
		Names.append(fasta[0])
		Seqs.append(fasta[1])
	with torch.no_grad():
		for i in range(len(Seqs)):
			name = Names[i]
			seq = Seqs[i]
			embedding = model.embed(seq)
			esm1b_1d_embed = model.reduce_per_protein(embedding)
			n_embeddings[name] = esm1b_1d_embed
	del model, embedding, esm1b_1d_embed
	return n_embeddings

def extract_ESM(fastas):
	Names = []
	Seqs = []
	n_embeddings = {}
	model = ESM
	for fasta in fastas:
		Names.append(fasta[0])
		Seqs.append(fasta[1])
	with torch.no_grad():
		for i in range(len(Seqs)):
			name = Names[i]
			seq = Seqs[i]
			embedding = model.embed(seq)
			esm_1d_embed = model.reduce_per_protein(embedding)
			n_embeddings[name] = esm_1d_embed
	del model, embedding, esm_1d_embed
	return n_embeddings

def extract_FT(fastas):
	Names = []
	Seqs = []
	n_embeddings = {}
	model = FT
	for fasta in fastas:
		Names.append(fasta[0])
		Seqs.append(fasta[1])
	with torch.no_grad():
		for i in range(len(Seqs)):
			name = Names[i]
			seq = Seqs[i]
			embedding = model.embed(seq)
			ft_1d_embed = model.reduce_per_protein(embedding)
			n_embeddings[name] = ft_1d_embed
	del model, embedding, ft_1d_embed
	return n_embeddings

def extract_GloVe(fastas):
	Names = []
	Seqs = []
	n_embeddings = {}
	model = GloVe
	for fasta in fastas:
		Names.append(fasta[0])
		Seqs.append(fasta[1])
	with torch.no_grad():
		for i in range(len(Seqs)):
			name = Names[i]
			seq = Seqs[i]
			embedding = model.embed(seq)
			glove_1d_embed = model.reduce_per_protein(embedding)
			n_embeddings[name] = glove_1d_embed
	del model, embedding, glove_1d_embed
	return n_embeddings

def extract_PTU(fastas):
	Names = []
	Seqs = []
	n_embeddings = {}
	model = PTU
	for fasta in fastas:
		Names.append(fasta[0])
		Seqs.append(fasta[1])
	with torch.no_grad():
		for i in range(len(Seqs)):
			name = Names[i]
			seq = Seqs[i]
			embedding = model.embed(seq)
			pt5ur50_1d_embed = model.reduce_per_protein(embedding)
			n_embeddings[name] = pt5ur50_1d_embed
	del model, embedding, pt5ur50_1d_embed
	return n_embeddings

def extract_PTL(fastas):
	Names = []
	Seqs = []
	n_embeddings = {}
	model = PTL
	for fasta in fastas:
		Names.append(fasta[0])
		Seqs.append(fasta[1])
	with torch.no_grad():
		for i in range(len(Seqs)):
			name = Names[i]
			seq = Seqs[i]
			print(seq)
			print(type(seq))
			embedding = model.embed(seq)
			pt5xlu50_1d_embed = model.reduce_per_protein(embedding)
			n_embeddings[name] = pt5xlu50_1d_embed
	del model, embedding, pt5xlu50_1d_embed
	return n_embeddings

def extract_PTBB(fastas):
	Names = []
	Seqs = []
	n_embeddings = {}
	model = PTBB
	for fasta in fastas:
		Names.append(fasta[0])
		Seqs.append(fasta[1])
	with torch.no_grad():
		for i in range(len(Seqs)):
			name = Names[i]
			seq = Seqs[i]
			embedding = model.embed(seq)
			ptbb_1d_embed = model.reduce_per_protein(embedding)
			n_embeddings[name] = ptbb_1d_embed
	del model, embedding, ptbb_1d_embed
	return n_embeddings

def extract_PTAB(fastas):
	Names = []
	Seqs = []
	n_embeddings = {}
	model = PTAB
	for fasta in fastas:
		Names.append(fasta[0])
		Seqs.append(fasta[1])
	with torch.no_grad():
		for i in range(len(Seqs)):
			name = Names[i]
			seq = Seqs[i]
			embedding = model.embed(seq)
			ptab_1d_embed = model.reduce_per_protein(embedding)
			n_embeddings[name] = ptab_1d_embed
	del model, embedding, ptab_1d_embed
	return n_embeddings

def extract_PTB(fastas):
	Names = []
	Seqs = []
	n_embeddings = {}
	model = PTB
	for fasta in fastas:
		Names.append(fasta[0])
		Seqs.append(fasta[1])
	with torch.no_grad():
		for i in range(len(Seqs)):
			name = Names[i]
			seq = Seqs[i]
			embedding = model.embed(seq)
			ptb_1d_embed = model.reduce_per_protein(embedding)
			n_embeddings[name] = ptb_1d_embed
	del model, embedding, ptb_1d_embed
	return n_embeddings

def extract_PTN(fastas):
	Names = []
	Seqs = []
	n_embeddings = {}
	model = PTN
	for fasta in fastas:
		Names.append(fasta[0])
		Seqs.append(fasta[1])
	with torch.no_grad():
		for i in range(len(Seqs)):
			name = Names[i]
			seq = Seqs[i]
			embedding = model.embed(seq)
			ptn_1d_embed = model.reduce_per_protein(embedding)
			n_embeddings[name] = ptn_1d_embed
	del model, embedding, ptn_1d_embed
	return n_embeddings

def extract_S2V(fastas):
	Names = []
	Seqs = []
	n_embeddings = {}
	model = S2V
	for fasta in fastas:
		Names.append(fasta[0])
		Seqs.append(fasta[1])
	with torch.no_grad():
		for i in range(len(Seqs)):
			name = Names[i]
			seq = Seqs[i]
			embedding = model.embed(seq)
			seq2vec_1d_embed = model.reduce_per_protein(embedding)
			n_embeddings[name] = seq2vec_1d_embed
	del model, embedding, seq2vec_1d_embed
	return n_embeddings

def extract_W2V(fastas):
	Names = []
	Seqs = []
	n_embeddings = {}
	model = W2V
	for fasta in fastas:
		Names.append(fasta[0])
		Seqs.append(fasta[1])
	with torch.no_grad():
		for i in range(len(Seqs)):
			name = Names[i]
			seq = Seqs[i]
			embedding = model.embed(seq)
			word2vec_1d_embed = model.reduce_per_protein(embedding)
			n_embeddings[name] = word2vec_1d_embed
	del model, embedding, word2vec_1d_embed
	return n_embeddings

def extract_PLUSRNN(fastas):
	Names = []
	Seqs = []
	n_embeddings = {}
	model = PLUSRNN
	for fasta in fastas:
		Names.append(fasta[0])
		Seqs.append(fasta[1])
	with torch.no_grad():
		for i in range(len(Seqs)):
			name = Names[i]
			seq = Seqs[i]
			embedding = model.embed(seq)
			plusrnn_1d_embed = model.reduce_per_protein(embedding)
			n_embeddings[name] = plusrnn_1d_embed
	del model, embedding, plusrnn_1d_embed
	return n_embeddings

# Extract ten conventional feature descriptors
def AAC(fastas, **kw):
	AA = 'ARNDCQEGHILKMFPSTWYV'
	encodings = []
	header = ['#']
	for i in AA:
		header.append(i)
	encodings.append(header)

	for i in fastas:
		name, sequence = i[0],  i[1]
		count = Counter(sequence)
		for key in count:
			count[key] = count[key]/len(sequence)
		code = [name]
		for aa in AA:
			code.append(count[aa])
		encodings.append(code)

	nencodings = {}
	for val in encodings:
		if val[0].startswith('#'): continue
		nencodings[val[0]] = val[1:]
	
	return nencodings

def QSOrder(fastas, nlag=3, w=0.05, **kw):

	dataFile = re.sub('codes$', '', os.path.split(os.path.realpath(__file__))[0]) + r'\codes\Schneider-Wrede.txt' if platform.system() == 'Windows' else re.sub('codes$', '', os.path.split(os.path.realpath(__file__))[0]) + '/codes/Schneider-Wrede.txt'
	dataFile1 = re.sub('codes$', '', os.path.split(os.path.realpath(__file__))[0]) + r'\codes\Grantham.txt' if platform.system() == 'Windows' else re.sub('codes$', '', os.path.split(os.path.realpath(__file__))[0]) + '/codes/Grantham.txt'

	AA = 'ACDEFGHIKLMNPQRSTVWY'
	AA1 = 'ARNDCQEGHILKMFPSTWYV'

	DictAA = {}
	for i in range(len(AA)):
		DictAA[AA[i]] = i

	DictAA1 = {}
	for i in range(len(AA1)):
		DictAA1[AA1[i]] = i

	with open(dataFile) as f:
		records = f.readlines()[1:]
	AADistance = []
	for i in records:
		array = i.rstrip().split()[1:] if i.rstrip() != '' else None
		AADistance.append(array)
	AADistance = np.array(
		[float(AADistance[i][j]) for i in range(len(AADistance)) for j in range(len(AADistance[i]))]).reshape((20, 20))

	with open(dataFile1) as f:
		records = f.readlines()[1:]
	AADistance1 = []
	for i in records:
		array = i.rstrip().split()[1:] if i.rstrip() != '' else None
		AADistance1.append(array)
	AADistance1 = np.array(
		[float(AADistance1[i][j]) for i in range(len(AADistance1)) for j in range(len(AADistance1[i]))]).reshape(
		(20, 20))

	encodings = []
	header = ['#']
	for aa in AA1:
		header.append('Schneider.Xr.' + aa)
	for aa in AA1:
		header.append('Grantham.Xr.' + aa)
	for n in range(1, nlag + 1):
		header.append('Schneider.Xd.' + str(n))
	for n in range(1, nlag + 1):
		header.append('Grantham.Xd.' + str(n))
	encodings.append(header)

	for i in fastas:
		name, sequence = i[0], i[1]
		code = [name]
		arraySW = []
		arrayGM = []
		for n in range(1, nlag + 1):
			arraySW.append(
				sum([AADistance[DictAA[sequence[j]]][DictAA[sequence[j + n]]] ** 2 for j in range(len(sequence) - n)]))
			arrayGM.append(sum(
				[AADistance1[DictAA1[sequence[j]]][DictAA1[sequence[j + n]]] ** 2 for j in range(len(sequence) - n)]))
		myDict = {}
		for aa in AA1:
			myDict[aa] = sequence.count(aa)
		for aa in AA1:
			code.append(myDict[aa] / (1 + w * sum(arraySW)))
		for aa in AA1:
			code.append(myDict[aa] / (1 + w * sum(arrayGM)))
		for num in arraySW:
			code.append((w * num) / (1 + w * sum(arraySW)))
		for num in arrayGM:
			code.append((w * num) / (1 + w * sum(arrayGM)))
		encodings.append(code)
	nencodings = {}
	for val in encodings:
		if val[0].startswith('#'): continue
		nencodings[val[0]] = val[1:]
	return nencodings

def Paac(fastas):
	Name = []; Seq =[]
	for val in fastas:
		Name.append(val[0]); Seq.append(val[1])
	paac_comp, desc = paac(Seq, lambda_=3, w=0.05, remove_zero_cols=True, start=1, end=None)
	nencodings ={}
	for k, v in enumerate(paac_comp):
		paac_features = paac_comp[k]
		nencodings[Name[k]] = paac_features
	return nencodings

def APaac(fastas):
	Name = []; Seq =[]
	for val in fastas:
		Name.append(val[0]); Seq.append(val[1])
	apaac_comp, desc = apaac(Seq, lambda_=3, w=0.05, remove_zero_cols=True)
	nencodings ={}
	for k, v in enumerate(apaac_comp):
		apaac_features = apaac_comp[k]
		nencodings[Name[k]] = apaac_features
	return nencodings

def PDE(peptides):
    """
    Function to extract Peptide Descriptor Encoding (PDE) from a list of peptides.
    
    Args:
        peptides (list): List of tuples [(header, sequence), ...]
    
    Returns:
        dict: Dictionary with headers as keys and encoded feature arrays as values.
    """
    pde_features = {}
    for header, sequence in peptides:
        encoded_peptide = peptidy.encoding.peptide_descriptor_encoding(sequence)
        pde_features[header] = np.array(encoded_peptide).flatten()
    return pde_features

def DDE(fastas, **kw):
	AA = 'ARNDCQEGHILKMFPSTWYV'

	myCodons = {
		'A': 4,
		'C': 2,
		'D': 2,
		'E': 2,
		'F': 2,
		'G': 4,
		'H': 2,
		'I': 3,
		'K': 2,
		'L': 6,
		'M': 1,
		'N': 2,
		'P': 4,
		'Q': 2,
		'R': 6,
		'S': 6,
		'T': 4,
		'V': 4,
		'W': 1,
		'Y': 2
	}

	encodings = []
	diPeptides = [aa1 + aa2 for aa1 in AA for aa2 in AA]
	header = ['#'] + diPeptides
	encodings.append(header)

	myTM = []
	for pair in diPeptides:
		myTM.append((myCodons[pair[0]] / 61) * (myCodons[pair[1]] / 61))

	AADict = {}
	for i in range(len(AA)):
		AADict[AA[i]] = i

	for i in fastas:
		name, sequence = i[0],  i[1]
		code = [name]
		tmpCode = [0] * 400
		for j in range(len(sequence) - 2 + 1):
			tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j+1]]] = tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j+1]]] +1
		if sum(tmpCode) != 0:
			tmpCode = [i/sum(tmpCode) for i in tmpCode]

		myTV = []
		for j in range(len(myTM)):
			myTV.append(myTM[j] * (1-myTM[j]) / (len(sequence) - 1))

		for j in range(len(tmpCode)):
			tmpCode[j] = (tmpCode[j] - myTM[j]) / math.sqrt(myTV[j])

		code = code + tmpCode
		encodings.append(code)

	nencodings = {}
	for val in encodings:
		if val[0].startswith('#'): continue
		nencodings[val[0]] = val[1:]
	
	return nencodings

def Count(seq1, seq2):
	sum = 0
	for aa in seq1:
		sum = sum + seq2.count(aa)
	return sum


def CTDC(fastas, **kw):
	group1 = {
		'hydrophobicity_PRAM900101': 'RKEDQN',
		'hydrophobicity_ARGP820101': 'QSTNGDE',
		'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
		'hydrophobicity_PONP930101': 'KPDESNQT',
		'hydrophobicity_CASG920101': 'KDEQPSRNTG',
		'hydrophobicity_ENGD860101': 'RDKENQHYP',
		'hydrophobicity_FASG890101': 'KERSQD',
		'normwaalsvolume': 'GASTPDC',
		'polarity':        'LIFWCMVY',
		'polarizability':  'GASDT',
		'charge':          'KR',
		'secondarystruct': 'EALMQKRH',
		'solventaccess':   'ALFCGIVW'
	}
	group2 = {
		'hydrophobicity_PRAM900101': 'GASTPHY',
		'hydrophobicity_ARGP820101': 'RAHCKMV',
		'hydrophobicity_ZIMJ680101': 'HMCKV',
		'hydrophobicity_PONP930101': 'GRHA',
		'hydrophobicity_CASG920101': 'AHYMLV',
		'hydrophobicity_ENGD860101': 'SGTAW',
		'hydrophobicity_FASG890101': 'NTPG',
		'normwaalsvolume': 'NVEQIL',
		'polarity':        'PATGS',
		'polarizability':  'CPNVEQIL',
		'charge':          'ANCQGHILMFPSTWYV',
		'secondarystruct': 'VIYCWFT',
		'solventaccess':   'RKQEND'
	}
	group3 = {
		'hydrophobicity_PRAM900101': 'CLVIMFW',
		'hydrophobicity_ARGP820101': 'LYPFIW',
		'hydrophobicity_ZIMJ680101': 'LPFYI',
		'hydrophobicity_PONP930101': 'YMFWLCVI',
		'hydrophobicity_CASG920101': 'FIWC',
		'hydrophobicity_ENGD860101': 'CVLIMF',
		'hydrophobicity_FASG890101': 'AYHWVMFLIC',
		'normwaalsvolume': 'MHKFRYW',
		'polarity':        'HQRKNED',
		'polarizability':  'KMHFRYW',
		'charge':          'DE',
		'secondarystruct': 'GNPSD',
		'solventaccess':   'MSPTHY'
	}

	groups = [group1, group2, group3]
	property = (
	'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101', 'hydrophobicity_PONP930101',
	'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 'hydrophobicity_FASG890101', 'normwaalsvolume',
	'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess')

	encodings = []
	header = ['#']
	for p in property:
		for g in range(1, len(groups) + 1):
			header.append(p + '.G' + str(g))
	encodings.append(header)
	for v in fastas:
		name, sequence = v[0], v[1]
		code = [name]
		for p in property:
			c1 = Count(group1[p], sequence) / len(sequence)
			c2 = Count(group2[p], sequence) / len(sequence)
			c3 = 1 - c1 - c2
			code = code + [c1, c2, c3]
		encodings.append(code)
	nencodings = {}
	for val in encodings:
		if val[0].startswith('#'): continue
		nencodings[val[0]] = val[1:]
	return nencodings

def Count1(aaSet, sequence):
	number = 0
	for aa in sequence:
		if aa in aaSet:
			number = number + 1
	cutoffNums = [1, math.floor(0.25 * number), math.floor(0.50 * number), math.floor(0.75 * number), number]
	cutoffNums = [i if i >=1 else 1 for i in cutoffNums]

	code = []
	for cutoff in cutoffNums:
		myCount = 0
		for i in range(len(sequence)):
			if sequence[i] in aaSet:
				myCount += 1
				if myCount == cutoff:
					code.append((i + 1) / len(sequence) * 100)
					break
		if myCount == 0:
			code.append(0)
	return code


def CTDD(fastas, **kw):
	group1 = {
		'hydrophobicity_PRAM900101': 'RKEDQN',
		'hydrophobicity_ARGP820101': 'QSTNGDE',
		'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
		'hydrophobicity_PONP930101': 'KPDESNQT',
		'hydrophobicity_CASG920101': 'KDEQPSRNTG',
		'hydrophobicity_ENGD860101': 'RDKENQHYP',
		'hydrophobicity_FASG890101': 'KERSQD',
		'normwaalsvolume': 'GASTPDC',
		'polarity':        'LIFWCMVY',
		'polarizability':  'GASDT',
		'charge':          'KR',
		'secondarystruct': 'EALMQKRH',
		'solventaccess':   'ALFCGIVW'
	}
	group2 = {
		'hydrophobicity_PRAM900101': 'GASTPHY',
		'hydrophobicity_ARGP820101': 'RAHCKMV',
		'hydrophobicity_ZIMJ680101': 'HMCKV',
		'hydrophobicity_PONP930101': 'GRHA',
		'hydrophobicity_CASG920101': 'AHYMLV',
		'hydrophobicity_ENGD860101': 'SGTAW',
		'hydrophobicity_FASG890101': 'NTPG',
		'normwaalsvolume': 'NVEQIL',
		'polarity':        'PATGS',
		'polarizability':  'CPNVEQIL',
		'charge':          'ANCQGHILMFPSTWYV',
		'secondarystruct': 'VIYCWFT',
		'solventaccess':   'RKQEND'
	}
	group3 = {
		'hydrophobicity_PRAM900101': 'CLVIMFW',
		'hydrophobicity_ARGP820101': 'LYPFIW',
		'hydrophobicity_ZIMJ680101': 'LPFYI',
		'hydrophobicity_PONP930101': 'YMFWLCVI',
		'hydrophobicity_CASG920101': 'FIWC',
		'hydrophobicity_ENGD860101': 'CVLIMF',
		'hydrophobicity_FASG890101': 'AYHWVMFLIC',
		'normwaalsvolume': 'MHKFRYW',
		'polarity':        'HQRKNED',
		'polarizability':  'KMHFRYW',
		'charge':          'DE',
		'secondarystruct': 'GNPSD',
		'solventaccess':   'MSPTHY'
	}

	groups = [group1, group2, group3]
	property = (
	'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101', 'hydrophobicity_PONP930101',
	'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 'hydrophobicity_FASG890101', 'normwaalsvolume',
	'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess')

	encodings = []
	header = ['#']
	for p in property:
		for g in ('1', '2', '3'):
			for d in ['0', '25', '50', '75', '100']:
				header.append(p + '.' + g + '.residue' + d)
	encodings.append(header)

	for v in fastas:
		name, sequence = v[0], v[1]
		code = [name]
		for p in property:
			code = code + Count1(group1[p], sequence) + Count1(group2[p], sequence) + Count1(group3[p], sequence)
		encodings.append(code)
	nencodings = {}
	for val in encodings:
		if val[0].startswith('#'): continue
		nencodings[val[0]] = val[1:]
	return nencodings

def DPC(fastas, **kw):
	AA = 'ARNDCQEGHILKMFPSTWYV'
	#AA = kw['order'] if kw['order'] != None else 'ACDEFGHIKLMNPQRSTVWY'
	encodings = []
	diPeptides = [aa1 + aa2 for aa1 in AA for aa2 in AA]
	header = ['#'] + diPeptides
	encodings.append(header)

	AADict = {}
	for i in range(len(AA)):
		AADict[AA[i]] = i

	for i in fastas:
		name, sequence = i[0], i[1]
		code = [name]
		tmpCode = [0] * 400
		for j in range(len(sequence) - 2 + 1):
			tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j+1]]] = tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j+1]]] +1
		if sum(tmpCode) != 0:
			tmpCode = [i/sum(tmpCode) for i in tmpCode]
		code = code + tmpCode
		encodings.append(code)
	nencodings = {}
	for val in encodings:
		if val[0].startswith('#'): continue
		nencodings[val[0]] = val[1:]
	
	return nencodings

def generateGroupPairs(groupKey):
        gPair = {}
        for key1 in groupKey:
                for key2 in groupKey:
                        gPair[key1+'.'+key2] = 0.0
        return gPair

def CKSAAGP(fastas, gap = 3, **kw):


	group = {
		'alphaticr': 'GAVLMI',
		'aromatic': 'FYW',
		'postivecharger': 'KRH',
		'negativecharger': 'DE',
		'uncharger': 'STCPNQ'
	}

	AA = 'ARNDCQEGHILKMFPSTWYV'

	groupKey = group.keys()

	index = {}
	for key in groupKey:
		for aa in group[key]:
			index[aa] = key

	gPairIndex = []
	for key1 in groupKey:
		for key2 in groupKey:
			gPairIndex.append(key1+'.'+key2)

	encodings = []
	header = ['#']
	for g in range(gap + 1):
		for p in gPairIndex:
			header.append(p+'.gap'+str(g))
	encodings.append(header)

	for v in fastas:
		name, sequence = v[0], v[1]
		code = [name]
		for g in range(gap + 1):
			gPair = generateGroupPairs(groupKey)
			sum = 0
			for p1 in range(len(sequence)):
				p2 = p1 + g + 1
				if p2 < len(sequence) and sequence[p1] in AA and sequence[p2] in AA:
					gPair[index[sequence[p1]]+'.'+index[sequence[p2]]] = gPair[index[sequence[p1]]+'.'+index[sequence[p2]]] + 1
					sum = sum + 1

			if sum == 0:
				for gp in gPairIndex:
					code.append(0)
			else:
				for gp in gPairIndex:
					code.append(gPair[gp] / sum)

		encodings.append(code)
	nencodings = {}
	for val in encodings:
		if val[0].startswith('#'): continue
		nencodings[val[0]] = val[1:]
	
	return nencodings
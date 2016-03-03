# from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize import word_tokenize
import os
import xml.etree.ElementTree as ET
import re
import tidylib

os.chdir('..') # punc-rnn root directory

ENG_SOURCES = {
'NYT_NYT':True,
'APW_ENG':True,
'CNN_HDL':True,
'ABC_WNT':True,
'NBC_NNW':True,
'MNB_NBW':True,
'PRI_TWD':True,
'VOA_ENG':True }

DATA_DIR = 'TDT/tdt3_em/tkn_sgm/'
OUTFILE_PATH = 'data/corpus.txt'

outfile = open(OUTFILE_PATH, 'w+')

for filename in os.listdir( DATA_DIR ):
	orig_source = filename[-15:-8]
	if orig_source not in ENG_SOURCES:
		continue
	print DATA_DIR + filename 
	with open(DATA_DIR + filename, 'r') as f:
		data = '<ROOT>' + f.read() + '</ROOT>'
		data = tidylib.tidy_document(data, {"input_xml": True})[0]
		root = ET.fromstring(data)
		for doc in root.findall('DOC'):
			text = doc.find('TEXT').text
			if not text:
				print "empty text"
				continue
			if '\n' in text:
				print "WARNING: NEWLINE FOUND"
			# print text
			tokens = word_tokenize(text)
			tokens_no_caps = [token.lower() for token in tokens] # no caps
			outfile.write(' '.join(tokens_no_caps)) 
			outfile.write('\n')
	
outfile.close()
###############################################################################
# Author     : Abhirup Bhattacharya
# Created on : 31st March, 2022
###############################################################################

import glob
import re
import string
import time
import sys
from sys import getsizeof
import os
import operator
from operator import itemgetter
from audioop import reverse
import xml.dom.minidom as minidom
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer
from itertools import islice
import ast

#Initialize the count variables below
docCount  = 0
tokCount  = 0
wordCount = 0
stemCount = 0

#Inilitialize the dictionary for the tokenized and the stmmed words
wordDict = {}

path = sys.argv[1]+"/"

stop_words = []
with open(path+"stopwords") as file:
    stop_words = file.read().split()

#######################################
#Funcion to tokenize the input sentence
#Input  -> text to be tokenized
#Output -> List of tokens
######################################

def tokenize(text):

	#print("text1" + text)
	text = text.replace("\'s","")
	#print("text3"+ text)
	words = re.sub('[^a-zA-Z]+',' ',text).split()
	return [word.strip() for word in words if word.strip() != '']

#######################################
#Funcion to parse file and read tokens
#Input  -> None
#Output -> Raw token list and stop words count
######################################

	
def dictBuild():

	file_ptr = minidom.parse(file)
	data = file_ptr.getElementsByTagName('TEXT')[0]
	text = data.firstChild.data

	sent = str(text).lower()

	inp_sent = re.sub('<[^>]*>','',sent)
	tok_list = tokenize(inp_sent)

	for tok in tok_list:
		if tok not in stop_words:
			raw_tokens.append(tok)
		else:
			stop_word_cnt += 1

	return raw_tokens,stop_word_cnt


##################################################################
#Funcion to perform stemming and build dictionary of stemmed words
#Input  -> Token List
#Output -> Dictionary of stems
##################################################################

ps = PorterStemmer()
def stemming(tok_list):

	stemDict = {}

	for tok in tok_list:
		#stemCount += 1
		stemWord = ps.stem(tok)
		if stemWord in stemDict:
			stemDict[stemWord] += 1
		else:
			stemDict[stemWord] = 1

	return stemDict


##################################################################
#Funcion to perform lemmatization and build dictionary of lemmas
#Input  -> Token List
#Output -> Dictionary of lemmas
##################################################################


lemmatizer = WordNetLemmatizer()

def lemmatize(tok_list):

	lemmaDict = {}
	for tok in tok_list:
		#stemCount += 1
		lemmaTok = lemmatizer.lemmatize(tok)

		if lemmaTok in lemmaDict:
			lemmaDict[lemmaTok] += 1
		else:
			lemmaDict[lemmaTok] = 1

	return lemmaDict


#Intialize the dictionaries and variables used for creating the postings
stem_dict = {}
lemma_dict = {}
stem_index_tot = 0
lemma_index_tot = 0

fileno = 1

#List of file names of the block files generated that will later be used for merging
file_list_lemma = []
file_list_stem = []

file_path_lemma = "lemma_posting_"+str(fileno)
file_path_stem = "stem_posting_"+str(fileno)
file_list_lemma.append(file_path_lemma)
file_list_stem.append(file_path_stem)
f1 = open(file_path_lemma,"w")
f2 = open(file_path_stem,"w")

#=========================================
#           Index Construction           #
#=========================================

start_time = time.time()
path1 = path + "../Cranfield/*"

files = glob.glob(path1)

#defining blocksize as 20000
block_size = 20000
tok_stream_cnt = 0

for file in files:

	if (tok_stream_cnt > block_size):
		if(not (f1.closed and f2.closed)):
			lemma_index_start = time.time()
			Index_Version1_Uncompress = dict(sorted(lemma_dict.items(), key=lambda x: x[0]))
			lemma_index_tot += time.time() - lemma_index_start

			
			stem_index_start = time.time()
			Index_Version2_Uncompress = dict(sorted(stem_dict.items(), key=lambda x: x[0]))
			stem_index_start += time.time() - stem_index_start

			lemma_index_start = time.time()
			for tok,posting in Index_Version1_Uncompress.items():
				value = ', '.join(map(str, posting))
				f1.write(str(tok)+" : "+value.replace('[','').replace(']','')+"\n")
			lemma_index_tot += time.time() - lemma_index_start

			stem_index_start = time.time()
			for tok,posting in Index_Version2_Uncompress.items():
				value = ', '.join(map(str, posting))
				f2.write(str(tok)+" : "+value.replace('[','').replace(']','')+"\n")

			stem_index_start += time.time() - stem_index_start

			f1.close()
			f2.close()

			#open the next file for writing only if there are more files from cranfield collection remaining to be processed
			

			if file:
				lemma_dict = {}
				stem_dict = {}
				fileno += 1
				tok_stream_cnt = 0
				file_path_lemma = "lemma_posting_"+str(fileno)
				f1 = open(file_path_lemma,"w")
				file_list_lemma.append(file_path_lemma)
				file_path_stem = "stem_posting_"+str(fileno)
				f2 = open(file_path_stem,"w")
				file_list_stem.append(file_path_stem)



	tok_list = []
	raw_tokens = []
	stop_word_cnt = 0
	
	file_ptr = minidom.parse(file)
	data = file_ptr.getElementsByTagName('TEXT')[0]
	text = data.firstChild.data

	sent = str(text).lower()

	inp_sent = re.sub('<[^>]*>','',sent)
	tok_list = tokenize(inp_sent)

	for tok in tok_list:
		tok_stream_cnt += 1
		if tok not in stop_words:
			raw_tokens.append(tok)
		else:
			stop_word_cnt += 1

	doc_len = len(raw_tokens) + stop_word_cnt

	stem_index_start = time.time()

	stem_toks = stemming(raw_tokens)


	for tok in stem_toks:
		doc_ids = []

		if tok not in stem_dict.keys():
			doc_id = int(os.path.basename(file).replace('cranfield',''))  # to find the document ID from the file name
			tf = stem_toks[tok]  # the term frequency of the stemmed tok in the current document
			max_tf = max(stem_toks.items(), key=operator.itemgetter(1))[1]  # find the term with highest frequency in the the stem dictionary of the doc
			df = 1  # initial df will be 1 since the term has been encountered for the first time

			tuple = (doc_id,tf,max_tf,doc_len)
			posting_list = [tuple]
			value = (df,posting_list)

		else:
			doc_id = int(os.path.basename(file).replace('cranfield',''))
			tf = stem_toks[tok]
			max_tf = max(stem_toks.items(), key=operator.itemgetter(1))[1]
			df = stem_dict[tok][0] + 1 # current df will be the original df in the posting entry incremented by 1

			tuple = [(doc_id,tf,max_tf,doc_len)]
			new_list = stem_dict[tok][1] + tuple # append the current tuple to the list of existing tuples
			value = (df,new_list)


		stem_dict[tok] = value # value will contain the updated or newly created posting list from the above if/else condition
		#f1.write(str(tok)+":"+str(value)+"\n")

	stem_index_tot += time.time() - stem_index_start

	lemma_index_start = time.time()

	lemma_toks = lemmatize(raw_tokens)

	
	# for tok in dict(sorted(lemma_toks.items(), key=lambda x: x[0])):
	for tok in lemma_toks:
		doc_ids = []

		if tok not in lemma_dict.keys():
			doc_id = int(os.path.basename(file).replace('cranfield','')) # to find the document ID from the file name
			tf = lemma_toks[tok] # the term frequency of the stemmed tok in the current document
			max_tf = max(lemma_toks.items(), key=operator.itemgetter(1))[1] # find the term with highest frequency in the the stem dictionary of the doc
			df = 1 # initial df will be 1 since the term has been encountered for the first time

			tuple = (doc_id,tf,max_tf,doc_len)
			posting_list = [tuple]
			value = (df,posting_list)

		else:
			doc_id = int(os.path.basename(file).replace('cranfield',''))
			tf = lemma_toks[tok]
			max_tf = max(lemma_toks.items(), key=operator.itemgetter(1))[1]
			df = lemma_dict[tok][0] + 1 # current df will be the original df in the posting entry incremented by 1

			tuple = [(doc_id,tf,max_tf,doc_len)]
			new_list = lemma_dict[tok][1] + tuple # append the current tuple to the list of existing tuples
			value = (df,new_list)


		lemma_dict[tok] = value # value will contain the updated or newly created posting list from the above if/else condition

	lemma_index_tot += time.time() - lemma_index_start



# Write the contents of the last posting file


lemma_index_start = time.time()
Index_Version1_Uncompress = dict(sorted(lemma_dict.items(), key=lambda x: x[0]))
lemma_index_tot += time.time() - lemma_index_start

stem_index_start = time.time()
Index_Version2_Uncompress = dict(sorted(stem_dict.items(), key=lambda x: x[0]))
stem_index_tot += time.time() - stem_index_start

lemma_index_start = time.time()
for tok,posting in Index_Version1_Uncompress.items():
	value = ', '.join(map(str, posting))
	f1.write(str(tok)+" : "+value.replace('[','').replace(']','')+"\n")

lemma_index_tot += time.time() - lemma_index_start


stem_index_start = time.time()
for tok,posting in Index_Version2_Uncompress.items():
	value = ', '.join(map(str, posting))
	f2.write(str(tok)+" : "+value.replace('[','').replace(']','')+"\n")
stem_index_tot += time.time() - stem_index_start

f1.close()
f2.close()

#######################################################################
#Funcion to merge the individual block files into a single posting file
#Input  -> File type -> 1: Index_Version1, 2: Index_version2
#Output -> Term Dictionary
#######################################################################

def merge(file_type):

	file_dict = {}
	
	output_file_path = "Index_Version"+str(file_type)+".uncompress"
	#debug_file_path = "debug.txt"
	output_file = open(output_file_path,"w")
	#dbg = open(debug_file_path,"w")
	
	if file_type == 1:
		file_handles = [open(f,'r') for f in file_list_lemma]
	else:
		file_handles = [open(f,'r') for f in file_list_stem]

	
	line_dict = {}
	
	for file in file_handles:
		line_dict[file] = file.readline()
	
	
	next_line_dict = {}
	for i in range(0,len(file_handles)):
		next_line_dict[i+1] = 0
	
	
	while line_dict:
		next_line_to_write_term = None
		next_line_to_write_val  = None
	
		next_line_file_index = []
	
		for file in line_dict:
	
				
			line_arr  = line_dict[file].split(':')
			#dbg.write("file : "+ str(file))
			#dbg.write("line_arr : "+ str(line_arr))
			line_term = line_arr[0]
			line_val  = line_arr[1]
			next_file_files = []
	
	
			if next_line_to_write_term is None:
				
				next_line_to_write_term = line_term
				next_line_to_write_val  = line_val

				next_line_file_index.clear()
				next_line_file_index.append(file)

	
			elif line_term == next_line_to_write_term:
				line_val_arr = line_val.split(',',1)
				line_val_df = line_val_arr[0]
				next_line_to_write_val_arr = next_line_to_write_val.split(',',1)
				next_line_to_write_val_df = int(next_line_to_write_val_arr[0]) + int(line_val_df)
				line_val_posting = line_val_arr[1][:-1]
				next_line_to_write_val_posting = next_line_to_write_val_arr[1][:-1] + ','+line_val_posting
				next_line_to_write_val  = str(next_line_to_write_val_df) + ',' + str(next_line_to_write_val_posting) + '\n'
				next_line_file_index.append(file)

	
			
			elif line_term < next_line_to_write_term:

				next_line_to_write_term = line_term
				next_line_to_write_val = line_val
				next_line_file_index.clear()
				next_line_file_index.append(file)


	
		line_pointer = output_file.tell()
		file_dict[next_line_to_write_term.strip()] = (int(next_line_to_write_val.split(',')[0]),line_pointer)
		output_file.write(next_line_to_write_term.strip()+":"+next_line_to_write_val)
	
	
		
	
		for next_file in next_line_file_index:
			#dbg.write("next_file : "+str(next_file))
			new_line = next_file.readline() 
			if not new_line:
				#dbg.write("in new_line none condition")
				line_dict.pop(next_file)
				next_file.close()
			else:
				#dbg.write("in new_line else condition")
				line_dict[next_file] = new_line
	
	
	
	output_file.close()
	return file_dict

#Call the merge() function to create the final Index_Version1.uncompress posting file
lemma_index_start = time.time()
version1_dict = merge(1)
lemma_index_tot += time.time() - lemma_index_start

#Call the merge() function to create the final Index_Version2.uncompress posting file
stem_index_start = time.time()
version2_dict = merge(2)
stem_index_tot += time.time() - stem_index_start


#######################################################################
#Funcion to obtain the unary code of an integer
#Input  -> Integer
#Output -> Unary Code
#######################################################################
def get_unary(number):
    a = ['1' for i in range(number)] + ['0']
    unar = ""
    return unar.join(a)


#######################################################################
#Funcion to obtain the gamma code of an integer
#Input  -> Integer
#Output -> Gamma Code
#######################################################################
def get_gamma_code(number):
    offset = str(bin(number)).split('b')[1][1:]
    unary = get_unary(len(offset))
    return (unary + offset)


#######################################################################
#Funcion to obtain the gamma codes of the list of gaps
#Input  -> Integer List
#Output -> Gamma Code List
#######################################################################
def get_gamma_codes(gaps):
    gamma_code = []
    for number in gaps:
        gamma_code.append(get_gamma_code(number))
    return gamma_code

#######################################################################
#Funcion to obtain the delta codes of the list of gaps
#Input  -> Integer List
#Output -> Delta Code List
#######################################################################

def get_delta_code(gaps):
    delta_codes = []
    for number in gaps:
        binary = str(bin(number)).split('b')[1]
        gamma = get_gamma_code(len(binary))
        offset = binary[1:]
        delta_codes.append(gamma + offset)
    return delta_codes


#######################################################################
#Funcion to obtain the Common Prefix in a list of words
#Input  -> Word List
#Output -> Common Prefix
#######################################################################

def get_common_prefix(input_arr):

    input_arr.sort(reverse=False)
    str1 = input_arr[0]
    str2 = input_arr[len(input_arr) - 1]
    len1 = len(str1)
    len2 = len(str2)
    prefix = ""
    i = 0
    j = 0
    while i < len1 and j < len2:
        if str1[i] != str2[j]:
            break
        prefix += str1[i]
        i += 1
        j += 1
    return prefix



#######################################################################
#Funcion to Perform compression of dictionary and posting files
#Input  -> File Type -> 1: Index_Version1, 2: Index_Version2
#Output -> None
#######################################################################


def compress_files(filetype):	
	index_ver1 = "Index_Version"+str(filetype)+".uncompress"
	index_ver_compress = "Index_Version"+str(filetype)+".compress"
	dict_ver_compress = "Dict_Version"+str(filetype)+".compress"
	key_compress = "Key_ver"+str(filetype)+".compress"
	index_ver_file = open(index_ver1,"r")
	output_ver_file = open(index_ver_compress,"wb")
	dict_ver_file = open(dict_ver_compress,"w")
	key_ver_file = open(key_compress,"w")
	
	next_line = index_ver_file.readline()
	
	count = 0
	key_str = ''
	comp_dict_ver1 = []
	comp_dict_ver2 = []
	front_coding_list = []
	count1 = 0
	front_encoding_compressed_terms = []
	first_flag = 1

	while next_line:
		line_arr  = next_line.split(':')
		line_term = line_arr[0]
		line_val  = line_arr[1]

	
		if filetype == 1:
			if count == 0:
				
				dict_tuple = (version1_dict[line_term][0],version1_dict[line_term][1],len(key_str))
				comp_dict_ver1.append(dict_tuple)
				dict_ver_file.write(str(dict_tuple)+'\n')
				key_str += str(len(line_term)) + line_term
				count += 1
	
			else:
				if count < 4:
					key_str += str(len(line_term)) + line_term
					dict_tuple = (version1_dict[line_term][0],version1_dict[line_term][1])
					comp_dict_ver1.append(dict_tuple)
					dict_ver_file.write(str(dict_tuple)+'\n')
					count += 1
					if (count == 4):
						count = 0

		else:

			if count1 == 0:
				if first_flag == 1:
					#print("in first flag 1 condition")
					front_coding_list.append(line_term)
					count1 += 1
					dict_tuple = (version2_dict[line_term][0],version2_dict[line_term][1],len(key_str))
					comp_dict_ver1.append(dict_tuple)
					dict_ver_file.write(str(dict_tuple)+'\n')
					count += 1
					first_flag = 0
				else:
					#print("in first flag 0 condition")
					front_coding_list.append(line_term)
					count1 += 1
					prefix = get_common_prefix(front_coding_list)
					front_encoding_compressed_terms.append(str(len(front_coding_list[0])))
					front_encoding_compressed_terms.append(prefix)
					front_encoding_compressed_terms.append('*')
					front_encoding_compressed_terms.append(str(front_coding_list[0][len(prefix):]))

					for i in range(1,len(front_coding_list)):
						front_encoding_compressed_terms.append(str(len(front_coding_list[i]) - len(prefix)))
						front_encoding_compressed_terms.append('$')
						front_encoding_compressed_terms.append(front_coding_list[i][len(prefix):])
				
					key_str += "".join(front_encoding_compressed_terms)
					#print(key_str)
					dict_tuple = (version2_dict[line_term][0],version2_dict[line_term][1],len(key_str))
					comp_dict_ver2.append(dict_tuple)
					dict_ver_file.write(str(dict_tuple)+'\n')
					count1 += 1

					front_encoding_compressed_terms = []
					front_coding_list = []

			else:
				if count < 5:
					front_coding_list.append(line_term)
					dict_tuple = (version2_dict[line_term][0],version2_dict[line_term][1])
					comp_dict_ver2.append(dict_tuple)
					dict_ver_file.write(str(dict_tuple)+'\n')
					count1 += 1

					if count1 == 5:
						count1 = 0


		posting_list = line_val.split(',',1)[1]
		posting_arr = posting_list.replace('(','').split('),')
		doc_id = []
	
		for posting in posting_arr:
			doc_id.append(int(posting.split(',')[0]))
	

		gaps = []
		gaps.append(doc_id[0])
	
		for i in range(1,len(doc_id)):
			gaps.append(doc_id[i] - doc_id[i-1])
	
		if filetype == 1:
			posting_compress = line_val.split(',',1)[0] + ',' + str(get_gamma_codes(gaps))
		
		else:
			posting_compress = line_val.split(',',1)[0] + ',' + str(get_delta_code(gaps))
	
		output_ver_file.write(bytearray(line_term.strip() + ":" + posting_compress.strip() + '\n','utf8'))
	
		next_line = index_ver_file.readline()

        
	output_ver_file.write(bytearray(key_str,'utf8'))
	key_ver_file.close()
	output_ver_file.close()
	dict_ver_file.close()

#Call the compress_files() Function to create the Index_Version1.compress and Index_Version2.compress files
lemma_index_comp_start = time.time()
compress_files(1)
lemma_index_comp_tot = time.time() - lemma_index_comp_start

#Call the compress_files() Function to create the Index_Version2.compress and Index_Version2.compress files
stem_index_comp_start = time.time()
compress_files(2)
stem_index_comp_tot = time.time() - stem_index_comp_start



ver1_file_path = "Index_Version1.uncompress"
ver1_file = open(ver1_file_path,"r")


print("1. Time taken for building indices:")
print("  1a. Time taken for building Index_Version1.uncompress : " +str(lemma_index_tot))
print("  1b. Time taken for building Index_Version2.uncompress : " +str(stem_index_tot))
print("  1c. Time taken for building Index_Version1.compress   : " +str(lemma_index_comp_tot))
print("  1d. Time taken for building Index_Version2.compress   : " +str(stem_index_comp_tot))
print("2. Size of Index_Version1.uncompress",os.path.getsize("Index_Version1.uncompress"))
print("3. Size of Index_Version2.uncompress",os.path.getsize("Index_Version2.uncompress"))
print("4. Size of Index_Version1.compress",os.path.getsize("Index_Version1.compress"))
print("5. Size of Index_Version2.compress",os.path.getsize("Index_Version2.compress"))
print("6. Number of Postings in each Version: ")

def _count_generator(reader):
    b = reader(1024 * 1024)
    while b:
        yield b
        b = reader(1024 * 1024)

with open(r'Index_Version1.uncompress', 'rb') as fp:
    c_generator = _count_generator(fp.raw.read)
    # count each \n
    count1 = sum(buffer.count(b'\n') for buffer in c_generator)
    print(' 6a. Number of Postings in Version1 Index:', count1 + 1)

with open(r'Index_Version2.uncompress', 'rb') as fp:
    c_generator = _count_generator(fp.raw.read)
    # count each \n
    count2 = sum(buffer.count(b'\n') for buffer in c_generator)
    print(' 6b. Number of Postings in Version2 Index:', count2 + 1)

###############################################
#    Finding postings for the given terms     #
###############################################

ver1_file_path = "Index_Version1.uncompress"
ver1_file = open(ver1_file_path,"r")

print("\n7. Postings for given list of words")
query_terms = ["Reynolds", "NASA", "Prandtl", "flow", "pressure", "boundary", "shock"]

for terms in query_terms:

	term = terms.lower()

	ver1_file.seek(version1_dict[term][1])

	line = ver1_file.readline()

	term_value = line.split(":")[1].split(",",1)
	term_df = term_value[0]

	print("\nTERM : " + term)
	print(" Df : " + term_df)
	term_value_posting = term_value[1].replace("(","").split("),")
	inv_list_size = getsizeof(term_value_posting)

	tf_sum = 0

	for i in range(1,len(term_value_posting)):

		term_value_posting_entries = term_value_posting[i].split(",")

		term_tf = term_value_posting_entries[1]
		#term_doclen = term_value_posting_entries[3]
		#term_maxtf = term_value_posting_entries[2]

		#print(" Entry "+str(i) +" values :")
		#print("  TF : " + term_tf)
		tf_sum += int(term_tf)

	print(" Total TF : "+ str(tf_sum))
	print(" Inverted Posting Lentgh (in Bytes) : "+ str(inv_list_size))


##########################################
#     Find DF,TF, Postings for NASA      #
##########################################

print("\n8. DF,TF and Postings for NASA")

query_term = "NASA"
ver1_file.seek(version1_dict[query_term.lower()][1])

line = ver1_file.readline()

term_value = line.split(":")[1].split(",",1)
term_df = term_value[0]

print("  DF for NASA : "+term_df)

term_value_posting = term_value[1].replace("(","").split("),")

#Find the posting entries of the first 3 entries in the posting list obtained

for i in range(1,4):
	term_value_posting_entries = term_value_posting[i].split(",")
	term_tf = term_value_posting_entries[1]
	term_doclen = term_value_posting_entries[3]
	term_maxtf = term_value_posting_entries[2]

	print("\n   Entry "+str(i) +" values :")
	print("    TF     : " + term_tf)
	print("    Doclen : " + term_doclen)
	print("    MaxTF  : " + term_maxtf)


###############################################
#   Finding Largest DF for Version 1 Index    #
###############################################

print("\n9. Dictionary Term(s) with Largest and Lowest DFs in Index Version 1")

max_df = 0
min_df = 10000000
max_df_term = []
min_df_term  = []

#Iterate through the Version 1 dictionary to find the terms with the maximum and minimum DFs
for key,value in version1_dict.items():
	df = value[0]

	if df > max_df:
		max_df = df
		max_df_term.clear()
		max_df_term.append(key)

	elif df == max_df:
		max_df_term.append(key)

	if (df < min_df):
		min_df = df
		min_df_term.clear()
		min_df_term.append(key)

	elif (df == min_df):
		min_df_term.append(key)

print("\n  VERSION 1 MAX and MIN DFs :")
print("  -----------------------------")
print("  Max DF Terms(s) : " + str(max_df_term))
print("  Min DF Terms(s) : " + str(min_df_term))

print("\n10. Dictionary Term(s) with Largest and Lowest DFs in Index Version 2")

max_df = 0
min_df = 10000000
max_df_term = []
min_df_term  = []

#Iterate through the Version 2 dictionary to find the terms with the maximum and minimum DFs
for key,value in version2_dict.items():
	df = value[0]

	if df > max_df:
		max_df = df
		max_df_term.clear()
		max_df_term.append(key)

	elif df == max_df:
		max_df_term.append(key)

	if (df < min_df):
		min_df = df
		min_df_term.clear()
		min_df_term.append(key)

	elif (df == min_df):
		min_df_term.append(key)

print("\n  VERSION 2 MAX and MIN DFs :")
print("  -----------------------------")
print("  Max DF Terms(s) : " + str(max_df_term))
print("  Min DF Terms(s) : " + str(min_df_term))

#####################################################
#    Finding Docs with Largest MAX_TF and DOCLEN    #
#####################################################

print("\n11. Documents with largest doclen and maxtf in the collection")

ver1_file.seek(0)
line = ver1_file.readline()
max_tf = 0
max_doc_len = 0
visited_docs = set() #set is used to add the documents for which the doclen has already been found to avoid revisiting the same docs

#Read the entire file to find the document with the largest doclen
while line:
	term_value = line.split(":")[1].split(",",1)
	term_df = term_value[0]

	#print("\nTERM : " + term)
	#print(" Df : " + term_df)
	term_value_posting = term_value[1].replace("(","").split("),")

	
	#iterate through each entry in the posting list for the term in the posting file
	for i in range(1,len(term_value_posting)):

		term_value_posting_entries = term_value_posting[i].replace(')\n',"").split(",")

		term_doc = int(term_value_posting_entries[0].strip())

		if term_doc not in visited_docs:
			#print(term_doc)
			visited_docs.add(term_doc)
			term_doclen = int(term_value_posting_entries[3])
			term_maxtf = int(term_value_posting_entries[2])

			if term_doclen > max_doc_len:
				max_doc_len = term_doclen
				max_doc_len_id = term_doc

			if term_maxtf > max_tf:
				max_tf = term_maxtf
				max_tf_doc = term_doc
	
	line = ver1_file.readline()

print("\n  Document with Max Doc Length : "+str(max_doc_len_id)+" with doc len = "+str(max_doc_len))
print("  Document with largest MaxTF  : "+str(max_tf_doc)+" with Max TF = "+str(max_tf))




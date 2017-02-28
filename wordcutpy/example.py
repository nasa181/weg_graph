#! -*- coding: UTF8 -*-
import requests
import json
import PyICU
import word2vec
import gensim
from wordcut import Wordcut
if __name__ == '__main__':
    with open('bigthai.txt') as dict_file:
        word_list = list(set([w.rstrip() for w in dict_file.readlines()]))
        word_list.sort()
        wordcut = Wordcut(word_list)
r = requests.get("http://localhost:3000/get_data")
r = r.json()
# # text = ''
count = 0
# f = open('word3', 'w')
for i in range(len(r)):
	tmp = ''
	idx = ''
	typ = ''
	if 'message' in r[i]:
		count += 1
		idx = str(r[i]['_id'])
		typ = str(r[i]['type'])
		tmp = str(r[i]['message'])
		f = open('word4','w')
		f.write(tmp)
		f.close()
		# word2vec.word2phrase('word4','word-phrase5',verbose=True)
		# word2vec.word2vec('word-phrase5','word5.bin',size=100,verbose=True)
		word2vec.word2vec('word4','word5.bin',size=100,verbose=True)
		# vec = f.read().decode('utf-8')
		with open('word5.bin', "r",encoding='utf-8', errors='ignore') as fdata:
			vec = ["{:02x}".format(ord(c)) for c in fdata.read()]
			# vec = fdata.read()
		re = requests.post("http://localhost:3000/get_data/edit_data", data={'id':idx, 'type':typ, 'vec':vec})
# 		tmp = tmp.replace("\n"," ")
# 		# text += tmp
# 		tmp = wordcut.tokenize(tmp)
# 		for word in tmp:
# 			word = word + " "
# 			f.write(word)
# 		# f.write(tmp)
# 		f.write("\n\n\n")
# 		print(">>> ",tmp,"\n")
# f.close()
print(count)
# model = unicode(word2vec.load('word3.bin'), errors='ignore')
# print(model.vectors)
# word2vec.word2phrase('word3','word-phrase4',verbose=True)
# word2vec.word2vec('word-phrase4','word4.bin',size=200,verbose=True)
# word2vec.word2clusters('word2','word-clusters2',100,verbose=True)
# text = text.replace("\n", " ")
# text = wordcut.tokenize(text)
# print(text)
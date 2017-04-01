#! -*- coding: UTF8 -*-
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from wordcut import Wordcut
if __name__ == '__main__':
    with open('bigthai.txt') as dict_file:
        word_list = list(set([w.rstrip() for w in dict_file.readlines()]))
        word_list.sort()
        wordcut = Wordcut(word_list)





# ========================================================================================================================
# 
# 	Preprocess read data from database 
# 	write to file .txt
# 
# ========================================================================================================================
# import requests
# import json

# r = requests.get("http://localhost:3000/get_data")
# r = r.json()
# count = 0
# f = open('word2', 'w')
# for i in range(len(r)):
# 	if 'message' in r[i]:
# 		count += 1
# 		tmp = str(r[i]['message'])
# 		idx = str(r[i]['_id'])
# 		if tmp is None:
# 			continue
# 		tmp = tmp.lower()
# 		tmp = tmp.replace("\t","")
# 		tmp = tmp.replace("\n","")
# 		shap_idx = tmp.find("#")
# 		while shap_idx != -1:
# 			space_idx = tmp[shap_idx:].find(" ")
# 			if space_idx == -1:
# 				tmp = tmp[:shap_idx]
# 				break
# 			tmp = tmp[:shap_idx]+tmp[shap_idx+space_idx:]
# 			shap_idx = tmp.find("#")
# 		if len(tmp) == 0 or tmp == None:	# clear status that only has a hashtag
# 			re = requests.post("http://localhost:3000/get_data/delete_data", data={'id':idx})
# 			continue
# 		print(count)
# 		f.write(tmp)
# 		f.write("\n")
# f.close()
# =========================================================================================================================





# =========================================================================================================================
# 
# 	yielding text data
# 	training word2vec
# 
# =========================================================================================================================
# import gensim
# import numpy as np

# class MySentences(object):
# 	def __init__(self, dirname):
# 		self.dirname = dirname

# 	def __iter__(self):
# 		for line in open(self.dirname):
# 			line = line.lower()
# 			line = wordcut.tokenize(line)
# 			yield line
# text = MySentences('word2') # a memory-friendly iterator

# model = gensim.models.Word2Vec(text,min_count=10,size=300,sg=1)
# model.save('fifth_model')
# =========================================================================================================================






# =========================================================================================================================
#
# 	checking key in model 
#
# =========================================================================================================================
# import gensim
# model = gensim.models.Word2Vec.load('fifth_model')
# print(model.vocab.keys())
# f = open('test_vocab','w')
# for word in model.vocab.keys():
# 	f.write(word + "\n")
# f.close()
# =========================================================================================================================






# =========================================================================================================================
# 
# 	load word2vec model 
# 	create vector of word
# 	append vector to database
# 
# =========================================================================================================================
# import gensim
# import numpy as np
# import requests
# import json

# model = gensim.models.Word2Vec.load('fifth_model')
# r = requests.get("http://localhost:3000/get_data")
# r = r.json()
# count = 0
# for i in range(len(r)):
# 	vec = np.array([0.0 for i in range(300)])
# 	if 'message' in r[i]:
# 		count += 1
# 		tmp = str(r[i]['message'])
# 		_id = str(r[i]['_id'])
# 		typ = str(r[i]['type'])
# 		tmp = tmp.lower()
# 		tmp = tmp.replace("\n","")
# 		idx = tmp.find("#")
# 		if idx != -1 :
# 			tmp = tmp[:idx]
# 		tmp = wordcut.tokenize(tmp)
# 		if tmp is None:
# 			continue
# 		# num_tokenize = len(tmp)
# 		for word in tmp:
# 			if word in model.wv.vocab:
# 				a = np.array(model[word])
# 				vec += a
# 		# for j in range(100):
# 		# 	vec[j] = vec[j]/num_tokenize
# 		print("message : ",tmp," >>>  id : ",_id," >>> type : ",typ," >>> Avg vec : ",vec)
# 		re = requests.post("http://localhost:3000/get_data/edit_data", data={'id':_id, 'type':typ, 'vec':vec})
# 		print("count >>> ",count)
# =========================================================================================================================






# import gensim
# import numpy as np
# from sklearn.cluster import KMeans

# model = gensim.models.Word2Vec.load('fifth_model')
# word_vectors = model.wv.syn0

# kmeans_clustering = KMeans( n_clusters = 4)
# idx = kmeans_clustering.fit_predict( word_vectors )
# word_centroid_map = dict(zip( model.index2word, idx ))

# for cluster in range(4):
#     #
#     # Print the cluster number  
#     print("\nCluster %d" % cluster)
#     #
#     # Find all of the words for that cluster number, and print them out
#     words = []
#     for i in word_centroid_map.values():
#         if( word_centroid_map.values()[i] == cluster ):
#             words.append(word_centroid_map.keys()[i])
#     print(words)






# =========================================================================================================================
# 
# 	clustering k-mean (k == 4) 
# 	save some example output to text file
# 	assign temporary to each status in database
# 
# =========================================================================================================================
# import requests
# import json
# import numpy as np
# import numpy.random as npr
# import math
# from sklearn import cluster

# def split_data(data,train_split=0.8):
#     data = np.array(data)
#     num_train = int(data.shape[0] * train_split)
#     # npr.shuffle(data)
    
#     return (data[:num_train],data[num_train:])


# r = requests.get("http://localhost:3000/get_data")
# r = r.json()
# count = 0
# data = []
# data_id = []
# data_message = []
# data_type = []
# advertisement = 0
# news = 0
# event = 0
# review = 0
# f = open('result_k-mean_output','w')
# for i in range(len(r)):
# 	if 'vec' in r[i]:
# 		count += 1
# 		# print(count)
# 		vec = r[i]['vec']
# 		vec = vec.split(",")
# 		vec = np.array(vec, dtype=np.float64)
# 		vec = vec.astype(np.float)
# 		data.append(vec)
# 		typ = r[i]['type']
# 		if typ == 'news':
# 			news += 1
# 		elif typ == 'advertisement':
# 			advertisement += 1
# 		elif typ == 'review':
# 			review += 1
# 		elif typ == 'event':
# 			event += 1
# 		# how to know the owner of the vector (which sentence is for which vector)
# 		data_id.append(r[i]['_id'])
# 		data_message.append(r[i]['message'])
# 		data_type.append(typ)
# training_data,test_data = split_data(data)

# kmeans = cluster.KMeans(n_clusters=4)
# kmeans.fit(data)
# results = kmeans.predict(data)
# group_count = [0,0,0,0]
# group_list = [[],[],[],[]]
# for i in range(len(results)):
# 	group_count[results[i]] += 1
# 	group_list[results[i]].append(i)

# print(group_count)
# print("news : ",news," >>> advertisement : ",advertisement," >>> review : ",review," >>> event : ",event)
# news_array = [0,0,0,0]
# news_position = 0
# advertisement_array = [0,0,0,0]
# advertisement_position = 0
# review_array = [0,0,0,0]
# review_position = 0
# event_array = [0,0,0,0]
# event_position = 0

# for i in range(len(data_type)):
# 	if data_type[i] == 'news':
# 		news_array[results[i]] += 1
# 		news_position = results[i]
# 	elif data_type[i] == 'advertisement':
# 		advertisement_array[results[i]] += 1
# 		advertisement_position = results[i]
# 	elif data_type[i] == 'review':
# 		review_array[results[i]] += 1
# 		review_position = results[i]
# 	elif data_type[i] == 'event':
# 		event_array[results[i]] += 1
# 		event_position = results[i]

# # ================================================================================
# # tmp_type = {}
# # for i in range(4):
# # 	mx = max(news_array[i],advertisement_array[i],review_array[i],event_array[i])
# # 	if mx == news_array[i]:
# # 		tmp_type[news_position] = 'news'
# # 	elif mx == advertisement_array[i]:
# # 		tmp_type[advertisement_position] = 'advertisement'
# # 	elif mx == review_array[i]:
# # 		tmp_type[review_position] = 'review'
# # 	elif mx == event_array[i]:
# # 		tmp_type[event_position] = 'event'
# # 	# print(tmp_type[results[i]])
# # print(tmp_type[0])	
# # ================================================================================

# print("news in 0 : ",news_array[0]," >>> advertisement in 0 : ",advertisement_array[0]," >>> review in 0 : ",review_array[0]," >>> event in 0 : ",event_array[0])
# print('\n')
# print("news in 1 : ",news_array[1]," >>> advertisement in 1 : ",advertisement_array[1]," >>> review in 1 : ",review_array[1]," >>> event in 1 : ",event_array[1])
# print('\n')
# print("news in 2 : ",news_array[2]," >>> advertisement in 2 : ",advertisement_array[2]," >>> review in 2 : ",review_array[2]," >>> event in 2 : ",event_array[2])
# print('\n')
# print("news in 3 : ",news_array[3]," >>> advertisement in 3 : ",advertisement_array[3]," >>> review in 3 : ",review_array[3]," >>> event in 3 : ",event_array[3])
# print('\n')

# for i in range(len(results)):
# 	print("results ",i," : ",results[i]," >>> type : ",data_type[i]," >>> message : ",data_message[i],"\n\n\n")
# 	f.write("results " + str(i) + " : " + str(results[i]) + " >>> type : " + data_type[i] + " >>> message : " + data_message[i] + "\n\n\n")
# # 	re = requests.post("http://localhost:3000/get_data/edit_data", data={'id':data_id[i], 'type':data_type[i], 'vec':data[i], 'tmp_type':tmp_type[results[i]]})
# f.close()

# print("kmeans.predict(test_data) : ",kmeans.predict(test_data))
# =========================================================================================================================


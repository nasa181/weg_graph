import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

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
# 	yielding document data
# 	training doc2vec
# 
# =========================================================================================================================
# import gensim
# import numpy as np
# import smart_open

# def read_corpus(fname, tokens_only=False):
#     with smart_open.smart_open(fname, encoding="iso-8859-1") as f:
#         for i, line in enumerate(f):
#             if tokens_only:
#                 yield gensim.utils.simple_preprocess(line)
#             else:
#                 # For training data, add tags
#                 yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])

# text = list(read_corpus('word2'))

# model = gensim.models.Doc2Vec(min_count=20,size=300,iter=30,alpha=0.05, min_alpha=0.05)
# model.build_vocab(text)
# model.train(text)
# model.save('first_model')
# =========================================================================================================================





# =========================================================================================================================
# 
# 	load doc2vec model 
# 	get vector of sentense from doc2vec model
# 	append vector to database
# 
# =========================================================================================================================
# import gensim
# import numpy as np
# import requests
# import json
# from wordcut import Wordcut
# if __name__ == '__main__':
#     with open('bigthai.txt') as dict_file:
#         word_list = list(set([w.rstrip() for w in dict_file.readlines()]))
#         word_list.sort()
#         wordcut = Wordcut(word_list)

# model = gensim.models.Doc2Vec.load('first_model')
# # model = gensim.models.Doc2Vec.load('second_model')
#f = open('vector_of_sentense','w')
# r = requests.get("http://localhost:3000/get_data")
# r = r.json()
# count = 0
# for i in range(len(r)):
# 	if 'message' in r[i]:
# 		count += 1
# 		tmp = str(r[i]['message'])
# 		_id = str(r[i]['_id'])
# 		typ = str(r[i]['type'])
# 		tmp = tmp.lower()
# 		tmp = tmp.replace("\n","")
# 		idx = tmp.find("#")
# 		shap_idx = tmp.find("#")
# 		while shap_idx != -1:
# 			space_idx = tmp[shap_idx:].find(" ")
# 			if space_idx == -1:
# 				tmp = tmp[:shap_idx]
# 				break
# 			tmp = tmp[:shap_idx]+tmp[shap_idx+space_idx:]
# 			shap_idx = tmp.find("#")
# 		tmp = wordcut.tokenize(tmp)
# 		if tmp is None:
# 			continue
# 		vec = model.infer_vector(tmp)
# 		vec = vec * 100
# 		# print("message : ",tmp," >>>  id : ",_id," >>> type : ",typ," >>> vec of sen : ",vec)
#		f.write(_id+"\t"+vec)
# 		re = requests.post("http://localhost:3000/get_data/edit_data", data={'id':_id, 'type':typ, 'vec':vec})
# 		print("count >>> ",count)
# f.close()
# =========================================================================================================================





# =========================================================================================================================
# 
# 	clustering k-mean (k == 4) k do not have to be 4(can be more than 4)
# 	save some example output to text file
# 	assign temporary to each status in database
# 
# =========================================================================================================================
import requests
import json
import numpy as np
import numpy.random as npr
import math
from sklearn import cluster

def split_data(data,train_split=0.8):
    data = np.array(data)
    num_train = int(data.shape[0] * train_split)
    # npr.shuffle(data)
    
    return (data[:num_train],data[num_train:])


r = requests.get("http://localhost:3000/get_data")
r = r.json()
count = 0
data = []
data_id = []
data_message = []
data_type = []
advertisement = 0
news = 0
event = 0
review = 0

for i in range(len(r)):
	if 'vec' in r[i]:
		count += 1
		# print(count)
		vec = r[i]['vec']
		vec = vec.split(",")
		vec = np.array(vec, dtype=np.float64)
		vec = vec.astype(np.float)
		data.append(vec)
		typ = r[i]['type']
		if typ == 'news':
			news += 1
		elif typ == 'advertisement':
			advertisement += 1
		elif typ == 'review':
			review += 1
		elif typ == 'event':
			event += 1
		# how to know the owner of the vector (which sentence is for which vector)
		data_id.append(r[i]['_id'])
		data_message.append(r[i]['message'])
		data_type.append(typ)
training_data,test_data = split_data(data)

# noc = []# number of cluster (106 ~ 146)
# for i in range(21):
# 	noc.append(106+i)
# for i in range(21):
kmeans = cluster.KMeans(init='k-means++',n_clusters=4,random_state=1)
kmeans.fit(data)
results = kmeans.predict(data)
group_count = [0 for x in range(147)]
for j in range(len(results)):
	group_count[results[i]] += 1

print("\n\n\n")
print(group_count)
print("news : ",news," >>> advertisement : ",advertisement," >>> review : ",review," >>> event : ",event)
news_array = [0 for x in range(147)]
advertisement_array = [0 for x in range(147)]
review_array = [0 for x in range(147)]
event_array = [0 for x in range(147)]

for j in range(len(data_type)):
	if data_type[j] == 'news':
		news_array[results[j]] += 1
	elif data_type[j] == 'advertisement':
		advertisement_array[results[j]] += 1
	elif data_type[j] == 'review':
		review_array[results[j]] += 1
	elif data_type[j] == 'event':
		event_array[results[j]] += 1
for j in range(len(results)):
	print("news in ", j," : ",news_array[j]," >>> advertisement in ", j," : ",advertisement_array[j]," >>> review in ",j," : ",review_array[j]," >>> event in ",j," : ",event_array[j])
	print('\n')

# for i in range(len(results)):
# 	if results[i] == 1:
# 		print("results ",i," : ",results[i]," >>> type : ",data_type[i]," >>> message : ",data_message[i],"\n\n\n")
# # 	re = requests.post("http://localhost:3000/get_data/edit_data", data={'id':data_id[i], 'type':data_type[i], 'vec':data[i], 'tmp_type':tmp_type[results[i]]})

# print("kmeans.predict(test_data) : ",kmeans.predict(test_data))
# =========================================================================================================================






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

# model = gensim.models.Doc2Vec(min_count=15,size=300,iter=30,alpha=0.05, min_alpha=0.05)
# model.build_vocab(text)
# model.train(text)
# model.save('second_model')
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

# # model = gensim.models.Doc2Vec.load('first_model')
# model = gensim.models.Doc2Vec.load('second_model')
# f = open('vector_of_sentense','w')
# r = requests.get("http://localhost:3000/get_data")
# r = r.json()
# count = 0
# for i in range(len(r)):
# 	if 'message' in r[i]:
# 		count += 1
# 		tmp = str(r[i]['message'])
# 		_id = str(r[i]['_id'])
# 		typ = str(r[i]['type'])
# 		tmp_type = str(r[i]['tmp_type'])
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
# 		f.write(_id+"\t")
# 		str_vec = ''
# 		for v in vec:
# 			str_vec += (str(v)) + ','
# 		if len(str_vec) > 0 :
# 			str_vec = str_vec[:-2]
# 			f.write(str_vec)
# 			f.write("\n")
# 		re = requests.post("http://localhost:3000/get_data/edit_data", data={'id':_id, 'type':typ, 'vec':vec, 'tmp_type':tmp_type})
# 		print("count >>> ",count)
# f.close()
# # =========================================================================================================================






# =========================================================================================================================
# 
# 	Elbow method
# 	find suitable number of cluster (ploting graph)
# 	from result k ~ 40 - 100
# 
# =========================================================================================================================
# import requests
# import json
# import numpy as np
# import math
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
# data = []
# r = requests.get("http://localhost:3000/get_data")
# r = r.json()
# for i in range(len(r)):
# 	if 'vec' in r[i]:
# 		vec = r[i]['vec']
# 		vec = vec.split(",")
# 		vec = np.array(vec, dtype=np.float64)
# 		vec = vec.astype(np.float)
# 		data.append(vec)

# sse = []
# for n_cluster in range(4, 120):
#     tmp = 0
#     print("n_cluster",n_cluster)
#     kmeans = KMeans(n_clusters=n_cluster,max_iter=1000).fit(data)
#     inter = kmeans.inertia_
#     sse.append(np.power(inter,2))
# plt.plot(sse)
# plt.ylabel('sse')
# plt.show()
# =========================================================================================================================






# =========================================================================================================================
# 
# 	clustering k-mean and k do not have to be 4(can be more than 4)
# 	try to find which k is suitable by running for loop
# 	save some example output to text file
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
# f = open('result_k-mean_output_after_plot2','w')
# noc = []# number of cluster (40 ~ 100)
# for i in range(31):
# 	noc.append(50+i)
# 	print(noc[i])
# for i in range(31):
# 	kmeans = cluster.KMeans(init='k-means++',n_clusters=noc[i],random_state=1,max_iter=1000)
# 	print("going to cluster data : ",i)
# 	kmeans.fit(data)
# 	print("clustering data is complete")
# 	results = kmeans.predict(data)
# 	group_count = [0 for x in range(noc[i])]
# 	for j in range(len(results)):
# 		group_count[results[j]] += 1

# 	print("\n\n\n")
# 	f.write("\n\n\n")
# 	f.write(str(i) + "\n")
# 	print(group_count)
# 	str_group_count = ''
# 	for j in range(len(group_count)):
# 		str_group_count += str(group_count[j]) + ','
# 	f.write(str_group_count)
# 	print("news : ",news," >>> advertisement : ",advertisement," >>> review : ",review," >>> event : ",event)
# 	f.write("news : "+str(news)+" >>> advertisement : "+str(advertisement)+" >>> review : "+str(review)+" >>> event : "+str(event))
# 	news_array = [0 for x in range(noc[i])]
# 	advertisement_array = [0 for x in range(noc[i])]
# 	review_array = [0 for x in range(noc[i])]
# 	event_array = [0 for x in range(noc[i])]

# 	for j in range(len(data_type)):
# 		if data_type[j] == 'news':
# 			news_array[results[j]] += 1
# 		elif data_type[j] == 'advertisement':
# 			advertisement_array[results[j]] += 1
# 		elif data_type[j] == 'review':
# 			review_array[results[j]] += 1
# 		elif data_type[j] == 'event':
# 			event_array[results[j]] += 1
# 	for j in range(len(news_array)):
# 		print("news in ", j," : ",news_array[j]," >>> advertisement in ", j," : ",advertisement_array[j]," >>> review in ",j," : ",review_array[j]," >>> event in ",j," : ",event_array[j])
# 		f.write("news in "+str(j)+" : "+str(news_array[j])+" >>> advertisement in "+str(j)+" : "+str(advertisement_array[j])+" >>> review in "+str(j)+" : "+str(review_array[j])+" >>> event in "+str(j)+" : "+str(event_array[j]))
# 		print('\n')
# 		f.write("\n")

# f.close()
# =========================================================================================================================





# =========================================================================================================================
# 
# 	clustering k-mean
# 	k = 78
# 	assign temporary group of cluster to each sentense
# 	For the unlabelable data, KNN will be used to find the nearest group
# 	In this case, there are 21 unlabelable groups
# 	record in database
# 
# =========================================================================================================================
# import requests
# import json
# import numpy as np
# import numpy.random as npr
# from sklearn import cluster
# from sklearn.neighbors import KNeighborsClassifier

# r = requests.get("http://localhost:3000/get_data")
# r = r.json()
# data = []
# data_id = []
# data_message = []
# data_type = []
# advertisement = 0
# news = 0
# event = 0
# review = 0

# for i in range(len(r)):
# 	if 'vec' in r[i]:
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
# 		data_id.append(r[i]['_id'])
# 		data_message.append(r[i]['message'])
# 		data_type.append(typ)

# n_cluster = 78
# kmeans = cluster.KMeans(init='k-means++',n_clusters=n_cluster,random_state=1,max_iter=1000)
# km = kmeans.fit(data)
# results = km.labels_
# group_list = [[] for i in range(n_cluster)]
# group_count = [0 for x in range(n_cluster)]
# news_array = [0 for x in range(n_cluster)]
# advertisement_array = [0 for x in range(n_cluster)]
# review_array = [0 for x in range(n_cluster)]
# event_array = [0 for x in range(n_cluster)]
# for j in range(len(results)):
# 	group_count[results[j]] += 1
# 	group_list[results[j]].append(j)
# 	if data_type[j] == 'news':
# 		news_array[results[j]] += 1
# 	elif data_type[j] == 'advertisement':
# 		advertisement_array[results[j]] += 1
# 	elif data_type[j] == 'review':
# 		review_array[results[j]] += 1
# 	elif data_type[j] == 'event':
# 		event_array[results[j]] += 1

# label_group = []
# n_no_group = 0
# for i in range(n_cluster):
# 	my_list = []
# 	my_list.extend([news_array[i],advertisement_array[i],review_array[i],event_array[i]])
# 	max_value = max(my_list)
# 	indices = [index for index, val in enumerate(my_list) if val == max_value]
# 	if len(indices) != 1:
# 		label_group.append('no_group')
# 		n_no_group += 1
# 	else:
# 		if max_value == my_list[0]:
# 			label_group.append('news')
# 		elif max_value == my_list[1]:
# 			label_group.append('advertisement')
# 		elif max_value == my_list[2]:
# 			label_group.append('review')
# 		elif max_value == my_list[3]:
# 			label_group.append('event')

# print(n_no_group)
# data_list_of_no_group = []
# data_id_list_of_no_group = []
# data_type_list_of_no_group = []
# grouped_data = []
# grouped_result = []
# count = 0
# for i in range(len(data)):
# 	print("group : ",label_group[results[i]])
# 	if label_group[results[i]] == 'no_group':
# 		data_list_of_no_group.append(data[i])
# 		data_id_list_of_no_group.append(data_id[i])
# 		data_type_list_of_no_group.append(data_type[i])
# 		count += 1
# 		print("count value : ",count)
# 	else:
# 		grouped_data.append(data[i])
# 		grouped_result.append(results[i])
# 		re = requests.post("http://localhost:3000/get_data/edit_data", data={'id':data_id[i], 'type':data_type[i], 'vec':data[i], 'tmp_type':label_group[results[i]]})
# num_of_no_group = n_no_group # number of no_group
# n_neighbor = n_cluster - num_of_no_group
# neigh = KNeighborsClassifier(n_neighbors=n_neighbor) # using KNN to find the nearest group
# neigh.fit(grouped_data, grouped_result)
# neigh_results = neigh.predict(data_list_of_no_group)
# for i in range(len(neigh_results)):
# 	print(neigh_results[i]," >>> ",label_group[neigh_results[i]],"\n")
# 	re = requests.post("http://localhost:3000/get_data/edit_data", data={'id':data_id_list_of_no_group[i], 'type':data_type_list_of_no_group[i], 'vec':data_list_of_no_group[i], 'tmp_type':label_group[neigh_results[i]]})
# =========================================================================================================================






# =========================================================================================================================
# 
# 	training neural network
# 	for Doc2Vec, Normal neural network with Backprop will be used
# 
# =========================================================================================================================
# import requests
# import json
# import numpy as np
# import numpy.random as npr
# import math
# from sklearn.neural_network import MLPClassifier
# from sklearn.neural_network import MLPRegressor
# from sklearn.externals import joblib

# def split_data(data,train_split=0.8):
#     data = np.array(data)
#     num_train = int(data.shape[0] * train_split)
#     # npr.shuffle(data)
    
#     return (data[:num_train],data[num_train:])

# r = requests.get("http://localhost:3000/get_data")
# r = r.json()
# data = []
# data_type = []
# count = 0
# for i in range(len(r)):
# 	if 'vec' in r[i]:
# 		vec = r[i]['vec']
# 		vec = vec.split(",")
# 		vec = np.array(vec, dtype=np.float64)
# 		vec = vec.astype(np.float)
# 		data.append(vec)
# 		typ = r[i]['type']
# 		if typ == 'tmp':	
# 			typ = r[i]['tmp_type']
# 		if typ == 'review':
# 			count +=1
# 		data_type.append(typ)

# print(count)
# training_data,test_data = split_data(data)
# training_target,test_target = split_data(data_type)
# clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(300, 100), random_state=1,activation='logistic',max_iter=1000)
# # clf = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(300, 100), random_state=1,activation='logistic',max_iter=1000)
# clf.fit(training_data, training_target)
# joblib.dump(clf, 'MLP_model.pkl')
# # joblib.dump(clf, 'MLRegress_model.pkl')
# results = clf.predict(test_data)
# f = open('result_categorise_output','w')
# for i in range(len(results)):
# 	print(results[i]," >>> ",test_target[i])
# 	f.write(str(results[i])+" >>> "+str(test_target[i])+"\n")
# f.close()
# =========================================================================================================================






# =============================================== Base Line ==========================================================================







# =========================================================================================================================
# 
# 	Create Base Line Matrix
# 	Matrix [num of data x num of word in model]
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

# f = open('thai_stop_word.txt','r')
# list_of_word_in_model = []
# for line in f:
# 	line = line.replace("\n","")
# 	list_of_word_in_model += [line]
# f.close()
# print(list_of_word_in_model)
# num_of_word_in_model = len(list_of_word_in_model)
# r = requests.get("http://localhost:3000/get_data")
# r = r.json()
# num_of_data = len(r)
# vec_from_base_line = []
# data_id = []
# count = 0
# f = open('base_line_id_and_vector.txt','w')
# for i in range(num_of_data):
# 	if 'vec' in r[i] and 'vec_from_word' in r[i]:
# 		count += 1
# 		tmp = []
# 		message = str(r[i]['message'])
# 		message = wordcut.tokenize(message)
# 		for j in range(num_of_word_in_model):
# 			if list_of_word_in_model[j] in message:
# 				# print(list_of_word_in_model[j])
# 				tmp.append(1)
# 			else:
# 				tmp.append(0)
# 		vec_from_base_line.append(tmp)
# 		data_id.append(str(r[i]['_id']))
# 		f.write(str(r[i]['_id']) + "\t" + str(tmp) + "\n")
# 		print(count)
# f.close()
# count = 0
# for i in range(len(data_id)):
# 	count += 1
# 	print(count)
# 	re = requests.post("http://localhost:3000/get_data/edit_data", data={'id':data_id[i], 'vec_from_base_line':vec_from_base_line[i]})
#	# re = requests.post("http://localhost:3000/get_data/edit_data", data={'id':data_id[i], 'type':typ[i], 'vec':vec[i], 'vec_from_word':vec_from_word[i], 'vec_from_base_line':vec_from_base_line[i], 'tmp_type':tmp_type[i], 'tmp_type_from_word':tmp_type_from_word[i]})	
# =========================================================================================================================






# =========================================================================================================================
# 
# 	Elbow method
# 	CLustering K mean on base lie matrix
# 
# =========================================================================================================================
import requests
import json
import numpy as np
import math
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
data = []
r = requests.get("http://localhost:3000/get_data")
r = r.json()
for i in range(len(r)):
	if 'vec_from_base_line' in r[i]:
		vec = str(r[i]['vec_from_base_line'])
		vec = vec.split(",")
		vec = np.array(vec, dtype=np.float64)
		vec = vec.astype(np.float)
		data.append(vec)

sse = []
for n_cluster in range(4, 150):
    tmp = 0
    print("n_cluster",n_cluster)
    kmeans = KMeans(n_clusters=n_cluster,max_iter=1000).fit(data)
    inter = kmeans.inertia_
    sse.append(np.power(inter,2))
plt.plot(sse)
plt.ylabel('sse')
plt.show()
# =========================================================================================================================






# =========================================================================================================================
# 
# 	Base Line
# 	clustering k-mean and k do not have to be 4(can be more than 4)
# 	try to find which k is suitable by running for loop
# 	save some example output to text file
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

# for i in range(len(r)):
# 	if 'vec_from_base_line' in r[i]:
# 		count += 1
# 		# print(count)
# 		vec = r[i]['vec_from_base_line']
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
# f = open('result_k-mean_output_after_plot2','w')
# noc = []# number of cluster (40 ~ 100)
# for i in range(31):
# 	noc.append(50+i)
# 	print(noc[i])
# for i in range(31):
# 	kmeans = cluster.KMeans(init='k-means++',n_clusters=noc[i],random_state=1,max_iter=1000)
# 	print("going to cluster data : ",i)
# 	kmeans.fit(data)
# 	print("clustering data is complete")
# 	results = kmeans.predict(data)
# 	group_count = [0 for x in range(noc[i])]
# 	for j in range(len(results)):
# 		group_count[results[j]] += 1

# 	print("\n\n\n")
# 	f.write("\n\n\n")
# 	f.write(str(noc[i]) + "\n")
# 	print(group_count)
# 	str_group_count = ''
# 	for j in range(len(group_count)):
# 		str_group_count += str(group_count[j]) + ','
# 	f.write(str_group_count)
# 	print("news : ",news," >>> advertisement : ",advertisement," >>> review : ",review," >>> event : ",event)
# 	f.write("news : "+str(news)+" >>> advertisement : "+str(advertisement)+" >>> review : "+str(review)+" >>> event : "+str(event))
# 	news_array = [0 for x in range(noc[i])]
# 	advertisement_array = [0 for x in range(noc[i])]
# 	review_array = [0 for x in range(noc[i])]
# 	event_array = [0 for x in range(noc[i])]

# 	for j in range(len(data_type)):
# 		if data_type[j] == 'news':
# 			news_array[results[j]] += 1
# 		elif data_type[j] == 'advertisement':
# 			advertisement_array[results[j]] += 1
# 		elif data_type[j] == 'review':
# 			review_array[results[j]] += 1
# 		elif data_type[j] == 'event':
# 			event_array[results[j]] += 1
# 	for j in range(len(news_array)):
# 		print("news in ", j," : ",news_array[j]," >>> advertisement in ", j," : ",advertisement_array[j]," >>> review in ",j," : ",review_array[j]," >>> event in ",j," : ",event_array[j])
# 		f.write("news in "+str(j)+" : "+str(news_array[j])+" >>> advertisement in "+str(j)+" : "+str(advertisement_array[j])+" >>> review in "+str(j)+" : "+str(review_array[j])+" >>> event in "+str(j)+" : "+str(event_array[j]))
# 		print('\n')
# 		f.write("\n")

# f.close()
# =========================================================================================================================


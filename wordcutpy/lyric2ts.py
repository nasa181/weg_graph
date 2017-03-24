import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from cdtw import pydtw

# !!!! Please download dataset from kaggle () and put it to './songlyrics' folder. !!!!

#######################################################################
#                   Libraries for New-running
#######################################################################
# import gensim
# from sklearn.manifold import TSNE
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords as stw
# from string import punctuation

# Load Dataset
df = pd.read_csv('songlyrics/songdata.csv')

#######################################################################
#                       Preprocess Text (SWR, PR)
#######################################################################
stopwords = set(stw.words('english'))
lyrics = np.asarray([[w for w in word_tokenize(l) if w not in stopwords and w not in punctuation] for l in df.text.str.lower().values])
# np.save('songlyrics/lyrics.npy', lyrics)
#######################################################################
# lyrics = np.load('songlyrics/lyrics.npy')


#######################################################################
#                           Word2Vec Model
#######################################################################
# model = gensim.models.Word2Vec(lyrics, size=50, min_count=0, sg=1, iter=20)
# word = np.asarray([v for v in model.vocab])
# word_count = np.asarray([model.vocab[v].count for v in word])
# word_vector = np.asarray([model[v] for v in word])
#
# np.save('songlyrics/word.npy', word)
# np.save('songlyrics/word_count.npy', word_count)
# np.save('songlyrics/word_vector.npy', word_vector)
#######################################################################
word = np.load('songlyrics/word.npy')
word_count = np.load('songlyrics/word_count.npy')
word_vector = np.load('songlyrics/word_vector.npy')

word_filter = word_count > 10
word_idx = dict()
for i, v in enumerate(word[word_filter]):
    word_idx[v] = i

#######################################################################
#                t-SNE (Large Memory usage warnning!!)
#######################################################################
# tsne = TSNE()
# transform_vectors = tsne.fit_transform(word_vector[word_filter])
# plt.scatter(transform_vectors[:, 0], transform_vectors[:, 1])
#
# np.save('songlyrics/transform_vectors.npy', transform_vectors)
#######################################################################
transform_vectors = np.load('songlyrics/transform_vectors_mt_10.npy')


#######################################################################
#                   Word Vector to t-SNE space
#######################################################################
def transform(lyric):
    return np.asarray([transform_vectors[word_idx[l]] for l in lyric if l in word_idx])
def transform_idx(lyric):
    return np.asarray([word_idx[l] for l in lyric if l in word_idx])

index_lyrics = np.asarray([transform_idx(l) for l in lyrics])
transform_lyrics = np.asarray([transform(l) for l in lyrics])
artist_label = df.artist.values
artist_list = np.asarray(sorted(list(set(artist_label))))

# np.save('songlyrics/index_lyrics.npy', index_lyrics)
# np.save('songlyrics/transform_lyrics.npy', transform_lyrics)
# np.save('songlyrics/artist_label.npy', artist_label)
# np.save('songlyrics/artist_list.npy', artist_list)
#######################################################################
# index_lyrics = np.load('songlyrics/index_lyrics.npy')
# transform_lyrics = np.load('songlyrics/transform_lyrics.npy')
# artist_label = np.load('songlyrics/artist_label.npy')
# artist_list = np.load('songlyrics/artist_list.npy')


#######################################################################
#                            Utilities Method
#######################################################################
def dist_from_org(vectors):
    return np.asarray([(v**2).sum()**0.5 for v in vectors])

def angle_from_org(vectors):
    hpi = np.arccos(0)
    return -np.abs(-(np.abs(np.asarray([np.arctan2(v[1], v[0]) for v in vectors])+hpi)-2*hpi))+hpi

def dist_change(vectors):
    return ((vectors[1:]-vectors[:-1])**2).sum(axis=1)**0.5

def angle_change(vectors):
    from_org = angle_from_org(vectors)
    return abs(from_org[:-1] - from_org[1:]) % (2*np.arccos(0))

def to_time_series(vectors):
    xy_ts = np.column_stack([np.cumsum(dist_from_org(vectors)), angle_from_org(vectors)])
    # xy_ts = np.column_stack([np.cumsum(dist_change(vectors)), np.cumsum(angle_change(vectors))])
    return np.interp(list(range(int(xy_ts[-1, 0]))), xy_ts[:, 0], xy_ts[:, 1])

def plot_cost(ts1, ts2, d):
    ils = []
    jls = []
    for i, j in d.get_path():
        ils.append(i), jls.append(j)
    fig = plt.figure()
    axl = fig.add_subplot(121)
    axr = fig.add_subplot(122)
    axl.plot(jls,ils, 'k', lw = 1.5)
    axr.plot(jls,ils, 'k', lw = 1.5)
    axl.imshow(d.get_cost(), interpolation = 'nearest', cmap = cm.autumn) #plot global cost matrix
    dist_matrix = np.abs(ts1[..., np.newaxis] - ts2) #local cost matrix
    axr.imshow(dist_matrix, interpolation = 'nearest', cmap = cm.autumn)

#######################################################################
#                     Lyrics to Time Series
#######################################################################
ts_lyrics = np.asarray([to_time_series(transformed) for transformed in transform_lyrics])
# np.save('songlyrics/ts_lyrics.npy', ts_lyrics)
#######################################################################
# ts_lyrics = np.load('songlyrics/ts_lyrics.npy')

lyrics_mean = np.asarray([ts.mean() for ts in ts_lyrics])
np.argmin(lyrics_mean)
np.argmax(lyrics_mean)
plt.hist(lyrics_mean, bins=100)

dtwSetting = pydtw.Settings(dist='euclid', step = 'dp1', window = 'palival_mod',  param = 0.1, norm = False, compute_path = True)

# Songs list by artist name
# print df.song[artist_label == 'Bruno Mars']

# Songs index by name
# print np.argwhere(df.song.str.lower().values == 'we wish you a merry christmas'.lower())

from_i = 24220
print 'From Song:', df.artist[from_i], '-', df.song[from_i]
distances = np.asarray([pydtw.dtw(ts_lyrics[from_i], ts, dtwSetting).get_dist() for ts in ts_lyrics])
min_i = np.argsort(distances)[1]
print 'Nearest Song:', df.artist[min_i], '-', df.song[min_i], distances[min_i]
print 'Top 10:'
for rank, i in enumerate(np.argsort(distances)[1:11]):
    print '#%d' % rank, 'Song(%5d): %50s' % (i, df.artist[i] + ' - ' + df.song[i]), '\t| Distance:', distances[i]

#########################################
#        Distance Histogram
#########################################
plt.figure('Histogram')
_ = plt.hist(distances, bins=100)

#########################################
#   Scatter word in lyrics (tsne-space)
#########################################
plt.figure('Scatter')
plt.scatter(transform_lyrics[from_i][:, 0], transform_lyrics[from_i][:, 1], c='blue', s=200, alpha=0.2)
plt.scatter(transform_lyrics[min_i][:, 0], transform_lyrics[min_i][:, 1], c='red', s=100, alpha=0.2)

###################################
#      Plot with alignment
###################################
plt.figure('Alignment')
pydtw.dtw(ts_lyrics[from_i], ts_lyrics[min_i], dtwSetting).plot_alignment()

###################################
#         Print lyrics
###################################
print '####################'
print df.text[from_i]
print '####################'
print df.text[min_i]
print '####################'

###################################
#       Plot dtw heat-map
###################################
# d = pydtw.dtw(ts_lyrics[from_i], ts_lyrics[min_i], dtwSetting)
# plot_cost(ts_lyrics[from_i], ts_lyrics[min_i], d)


#####################################
# Plot 2 TS without alignment lines
#####################################
def shifted_y(ts, val):
    ts = ts.copy()
    ts += val
    return ts
plt.figure('Time Series')
plt.plot(ts_lyrics[from_i], c='blue')
plt.plot(shifted_y(ts_lyrics[min_i], 3), c='red')

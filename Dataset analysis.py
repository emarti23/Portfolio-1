import numpy as np
import pandas as pd
import statistics
import matplotlib.pyplot as plt
import seaborn as sns

############# 

dataset = pd.read_csv('.../dataset_youtube_8M.csv', engine='python')
dataset = dataset.iloc[:,1:] #drop first index column
dataset = dataset.drop('category_id', axis=1)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
print(dataset.head(20))
print(dataset.shape)

title_length = []
for sentence in dataset['title']:
    words = len(sentence.split())
    title_length.append(words)
    
dataset.insert(2, 'title_length', title_length)

print(pd.core.dtypes.common.is_datetime_or_timedelta_dtype(dataset['published']))


dataset['duration'] = dataset['duration'].str[2:-1].str.replace('M','.') 
dataset['published'] = dataset['published'].str[0:10]


dataset['collection date'] = pd.Timestamp('2022-09-08')


print(dataset.head(20))
print(dataset.shape)
print(dataset.info())


#count zeroes in input variables

countTags = (dataset['tag_count']==0).sum()
countLikes = (dataset['like_count']==0).sum()
countComment = (dataset['comment_count']==0).sum()
print(countTags)
print(countLikes)
print(countComment)


#######save datasets to csv

dataset.to_csv(r'../final_dataset.csv', index=False)



############# view range

#reduce dataset according to view range 1000 / 1000000
dataset = dataset[ (dataset['view_count']>= 1000) & (dataset['view_count'] <= 1000000)]
print(dataset.head(20))
print(dataset.shape)
print(dataset['view_count'].min())
print(dataset['view_count'].max())



############ histograms and data analysis

sc = pd.plotting.scatter_matrix(dataset)
[s.xaxis.label.set_rotation(20) for s in sc.reshape(-1)]
[s.yaxis.label.set_rotation(20) for s in sc.reshape(-1)]
[s.set_xticks(()) for s in sc.reshape(-1)]
[s.set_yticks(()) for s in sc.reshape(-1)]
plt.show()

#views histogram (result is skewed)
plt.hist(dataset['view_count'], bins=50,range= [0, 1000000], alpha=0.45, color='red')
plt.title('view count')
plt.xlabel('number of views')
plt.ylabel('number of videos')
plt.show()

views_skew_meter = dataset['view_count'].skew()
views_kurtosis_meter = dataset['view_count'].kurtosis()
print('Skewness is', views_skew_meter)
print('Kurtosis is', views_kurtosis_meter)
print('Views mean is', dataset['view_count'].mean())
print('Views median is', dataset['view_count'].median())
print('Views mode is', dataset['view_count'].mode()[0])
print('Views variance is', dataset['view_count'].var())
print('Views sd is', dataset['view_count'].std())



#title length histogram 
plt.hist(dataset['title_length'], bins=50, alpha=0.45, color='red')
plt.title('title length')
plt.xlabel('titles')
plt.ylabel('number of videos')
plt.show()

title_skew_meter = dataset['title_length'].skew()
title_kurtosis_meter = dataset['title_length'].kurtosis()
print('Skewness is', title_skew_meter)
print('Kurtosis is', title_kurtosis_meter)
print('Length title mean is', dataset['title_length'].mean())
print('Length title median is', dataset['title_length'].median())
print('Length title mode is', dataset['title_length'].mode()[0])
print('Length title variance is', dataset['title_length'].var())
print('Length title sd is', dataset['title_length'].std())


#tags count histogram 
plt.hist(dataset['tag_count'], bins=50, alpha=0.45, color='red')
plt.title('tags count')
plt.xlabel('number of tags')
plt.ylabel('number of videos')
plt.show()

tags_skew_meter = dataset['tag_count'].skew()
tags_kurtosis_meter = dataset['tag_count'].kurtosis()
print('Skewness is', tags_skew_meter)
print('Kurtosis is', tags_kurtosis_meter)
print('Tags mean is', dataset['tag_count'].mean())
print('Tags median is', dataset['tag_count'].median())
print('Tags mode is', dataset['tag_count'].mode()[0])
print('Tags variance is', dataset['tag_count'].var())
print('Tags sd is', dataset['tag_count'].std())


#likes histogram 
plt.hist(dataset['like_count'], bins=50, alpha=0.45, color='red')
plt.title('likes count')
plt.xlabel('likes')
plt.ylabel('number of videos')
plt.show()

likes_skew_meter = dataset['like_count'].skew()
likes_kurtosis_meter = dataset['like_count'].kurtosis()
print('Skewness is', likes_skew_meter)
print('Kurtosis is', likes_kurtosis_meter)
print('Likes mean is', dataset['like_count'].mean())
print('Likes median is', dataset['like_count'].median())
print('Likes mode is', dataset['like_count'].mode()[0])
print('Likes variance is', dataset['like_count'].var())
print('Likes sd is', dataset['like_count'].std())


#comment count histogram 
plt.hist(dataset['comment_count'], bins=50, alpha=0.45, color='red')
plt.title('comment count')
plt.xlabel('number of comments')
plt.ylabel('number of videos')
plt.show()

comment_skew_meter = dataset['comment_count'].skew()
comment_kurtosis_meter = dataset['comment_count'].kurtosis()
print('Skewness is', comment_skew_meter)
print('Kurtosis is', comment_kurtosis_meter)
print('Comments mean is', dataset['comment_count'].mean())
print('Comments median is', dataset['comment_count'].median())
print('Comments mode is', dataset['comment_count'].mode()[0])
print('Comments variance is', dataset['comment_count'].var())
print('Comments sd is', dataset['comment_count'].std())


########## further analysis

#correlation heatmap (green positive, red negative)

corr = dataset.corr()
map_ = sns.heatmap(corr,
                   vmin=-1, vmax=1, center=0,
                   cmap=sns.diverging_palette(20, 220, n=200),
                   square=True)
map_.set_xticklabels(map_.get_xticklabels(),
                     rotation=30,
                     horizontalalignment='right')
plt.show()

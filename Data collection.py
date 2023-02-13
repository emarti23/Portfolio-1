import numpy as np 
import pandas as pd 
import urllib.request 
import os
import tensorflow as tf


video_lvl_record= os.scandir("../Folder of tfrecords")

vid_ids = []
labels = []

for record in video_lvl_record:
    try:
        raw_record = os.path.abspath(record)
        raw_dataset = tf.data.TFRecordDataset(raw_record)  
        for raw_example in raw_dataset:
            example = tf.train.Example()
            example.ParseFromString(raw_example.numpy())
            vid_ids.append(example.features.feature['id'].bytes_list.value[0].decode(encoding='UTF-8'))
            labels.append(example.features.feature['labels'].int64_list.value) 
    except Exception:
        continue

  

vid_dict = dict(zip(vid_ids, labels))


print('Number of videos in this tfrecord: ',len(vid_ids))

#Vocabulary list by category arts & entertainment


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
df_vocabulary = pd.read_csv('*vocabulary.csv', engine = 'python')

df_voc = df_vocabulary[['Index', 'Name', 'Vertical1']]
vocabulary = df_voc.loc[df_voc['Vertical1'] == 'Arts & Entertainment']
print(vocabulary[['Index', 'Name']])


#####filter videos by labels

categories = { 3:'Musician', 14:'Music video'}


filtered_vid = set()

for i in vid_dict:
    for l in categories:
        if l in vid_dict[i]:
            filtered_vid.add(i)
            
#print(filtered_vid)
print(len(filtered_vid))


#####obtain real video ID

initial_url = 'http://data.yt8m.org/2/j/i/'
current_url = ''
url_dataset= set()
real_video_ids = []

for ids in filtered_vid:
    current_url = initial_url + ids[0:2] + '/' + ids + '.js'
    url_dataset.add(current_url)
    #print(url_dataset)
    
  
for url in url_dataset:
    try:
        f = urllib.request.urlopen(url)
        myfile = f.read().decode("utf8")
        real_video_ids.append(myfile[10:21])
        print(myfile)
        #print(myfile[10:21])
    except urllib.error.HTTPError:
        continue
ids_dataframe = pd.DataFrame(real_video_ids)  
ids_dataframe.to_csv(r'../ids_list.csv')


#####import from youtube (through API key)

import googleapiclient.discovery 

api_key = 'API-KEY'

youtube = googleapiclient.discovery.build('youtube', 'v3', developerKey = api_key)



def get_video_stats(youtube, video_ids):
    
    videos_statistics = []
    for i in range(0, len(video_ids), 50):
        request = youtube.videos().list(
            part= 'snippet, contentDetails, statistics',
            id= ','.join(video_ids[i:i+50]))
        response = request.execute()
        
  
        for video in response['items']:
            video_stats = dict(ID = video['id'],
                               title = video['snippet']['title'],
                               published = video['snippet']['publishedAt'],
                               tags = ','.join(video['snippet'].get('tags', [])),
                               tag_count = len(video['snippet'].get('tags', [])),
                               category_id = video['snippet']['categoryId'],
                               duration = video['contentDetails']['duration'],
                               view_count = video["statistics"].get('viewCount', 0),
                               like_count = video['statistics'].get('likeCount', 0),
                               comment_count = video['statistics'].get('commentCount', 0)
                               )
            videos_statistics.append(video_stats)
            

    return videos_statistics


video_statistics = get_video_stats(youtube, real_video_ids)


#####create pandas dataframe and save csv

videos_dataframe = pd.DataFrame.from_dict(video_statistics)
pd.set_option("display.max_columns", None)
print(videos_dataframe.head())

videos_dataframe.to_csv(r'../dataset_youtube_8M.csv')


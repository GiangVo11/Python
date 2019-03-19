
# coding: utf-8

# In[1]:


from PIL import Image


# In[19]:


from fuzzywuzzy import fuzz


# In[2]:


from PIL import ImageOps


# In[32]:


from PIL import Image, ImageEnhance, ImageFilter


# In[3]:


import numpy as np


# In[6]:


import os


# In[5]:


import cv2


# In[4]:


from pytesseract import image_to_string


# In[20]:


from matplotlib import pyplot as plt


# In[131]:


import difflib


# In[ ]:


from fuzzywuzzy import fuzz
from fuzzywuzzy import process


# In[ ]:


import math


# In[23]:


os.chdir(r'C:\Program Files (x86)\Tesseract-OCR')


# **How to**
# 
# 1. Open Video
# 2. Extract frames (via OpenCV)
# 3. Use PIL/ Pillow package to extract only Headline (with or without 'upper part')
# 4. Use tesseract to get text (frame -> time)
# 5. Match with caption and topic (label)
# 

# In[24]:


os.chdir(r'F:\Project')


# In[ ]:


#### EXTRACT FRAME FROM VIDEO( VIA OPEN CV)


# In[213]:


#OPTION 1: ONE FRAME PER SECOND , DONT RETURN THE 0-BYTE lAST FRAME 

def video_to_frames(input_loc, output_loc):
    try:
        os.mkdir(output_loc)
    except OSError:
        pass
    # Start capturing the feed
    cap = cv2.VideoCapture(input_loc)
    frameRate = cap.get(5)
    count = 0
    # Start converting the video
    while cap.isOpened():
        frameId = cap.get(1) #current frame number
        ret, frame = cap.read()
        if (ret != True):
            break
        if (frameId % math.floor(frameRate) == 0):
            cv2.imwrite(output_loc +"/%#05d.jpg" %count, frame)
        count = count + 1
        # If there are no more frames left
        if cv2.waitKey(1) & 0xFF == ord('q'):
             break
input_loc = 'tagesschau 2000 Uhr 05122012.mp4'
output_loc = 'data7'
video_to_frames(input_loc, output_loc)


# In[48]:


def video_to_frames(input_loc, output_loc):
    try:
        os.mkdir(output_loc)
    except OSError:
        pass
    # Start capturing the feed
    cap = cv2.VideoCapture(input_loc)
    frameRate = cap.get(5)
    count = 1
    # Start converting the video
    while cap.isOpened():
        frameId = cap.get(1) #current frame number
        ret, frame = cap.read()
        if (ret != True):
            break
        if (frameId % math.floor(frameRate) == 0):
            cv2.imwrite(output_loc + "/%#05d.jpg" %count, frame)
        count = count + 1
        # If there are no more frames left
        if cv2.waitKey(1) & 0xFF == ord('q'):
             break
input_loc = 'sample.mp4'
output_loc = 'data6'
video_to_frames(input_loc, output_loc)


# In[49]:


image1=Image.open(r'F:\Project\data5\09600.jpg')
crop2=ImageOps.crop(image1,(15,25,230,200))
crop2
crop2.save('cropped3.jpg')


# In[53]:


# open image
im = Image.open('cropped1.jpg')

# preprocessing
im = im.convert('L')  # grayscale
print(im)
im = im.filter(ImageFilter.MedianFilter())# a little blur
print(im)
im = im.point(lambda x: 0 if x < 150 else 250,'1')   # threshold (binarize)
print(im)

text = image_to_string(im)           # pass preprocessed image to tesseract
print(text)                                      # print 


# In[ ]:


### USE PILLOW & tesseract


# In[124]:


import glob
image_list = []
for filename in glob.glob('data5/*.jpg'):
    img = Image.open(filename)
    crop = ImageOps.crop(img, (15, 25, 230, 200))
    crop.save(os.path.join('data6', filename.split('/')[-1]))
    
   
 


# In[13]:


import glob
image_list = []
for filename in glob.glob('data6/data5/*.jpg'):
     img=Image.open(filename)
     b=image_to_string(img, lang = 'deu' )
     image_list.append(b)
   


# In[14]:


df11= pd.read_excel('df11.xlsx',sheetname="Sheet1")


# In[87]:


dfs = pd.read_excel('Df18.xlsx',sheetname="Sheet1")


# In[249]:


a=df11.iloc[0]['Topic']
b,c=a.split(":")
d=c.split(",")
dftopic = pd.DataFrame({'Topic':d})
dftopic


# In[312]:


Time=list(range(0,len(image_list)))


# In[426]:


import pandas as pd
df2 = pd.DataFrame({'Headline':image_list,'Time': Time})
df2


# In[434]:


import re
def clean ( row ):
    return re.sub(r'[^a-zA-Z0-9 ]',r' ', row['Headline'])


# In[435]:


df2['Headline'] = df2.apply( clean ,axis = 1 )


# In[436]:


df2['Headline'].replace('', np.nan, inplace=True)
df3=df2.dropna()
    


# In[437]:


def format_duration(row):
    seconds = row.Time 
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return '{:02d}:{:02d}:{:02d}'.format(hours, minutes, seconds)
df3['Time2'] = df3.apply(format_duration, axis=1)
df3[1:100]


# In[438]:


def syllable_count(word):
  
    count = 0
    vowels = "aeiouäöü"
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1 
    if count == 0:
        count += 1
    return count
df3["Syllabe"] = df3["Headline"].apply(syllable_count)


# In[439]:


df3['word_count'] = df3['Headline'].str.split().map(len)


# In[440]:


df4=(df3[df3["Syllabe"] >= 2])
df4


# In[450]:


CH = df4.set_index("Time2").Headline.to_dict()


# In[451]:


res = [(lookup,) + item for lookup in d for item in process.extract(lookup, CH, scorer=fuzz.ratio,limit=40)]
df = pd.DataFrame(res, columns=["lookup", "matched", "score", "Time2"])
dfNew=(df[df["score"] >= 50])


# In[452]:


dfNew[29:40]


# In[1]:


Min = dfNew.groupby(['lookup'])['Time2'].min()
Min


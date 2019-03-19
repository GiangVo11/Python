
# coding: utf-8

# In[1]:


import os
import time


# In[2]:


from selenium import webdriver


# In[3]:


from bs4 import BeautifulSoup
import re


# In[4]:


import numpy as np
import pandas as pd


# In[5]:


import datetime as dt


# In[6]:


import pysrt


# In[7]:


from pytube import YouTube


# In[13]:


import os


# In[9]:


import requests


# In[10]:


from string import punctuation


# In[12]:


import nltk
#nltk.download()
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk.data


# In[12]:


from hyphen import Hyphenator


# In[13]:


from nltk import punkt


# In[35]:


from PIL import Image
import cv2


# In[36]:


from pytesseract import image_to_string


# In[14]:


video_link = 'https://www.youtube.com/user/TagesschauBackup/videos'


# In[16]:


browser = webdriver.Chrome(executable_path=r'C:\webdriver\chromedriver.exe')


# In[18]:


browser.get( video_link )


# In[19]:


browser.execute_script("window.scroll(1, 500);")


# In[20]:


for i in range(0, 4):
       
    if i == 0:
         browser.execute_script("window.scroll(1, 10000);")
            
    elif i == 1:
         browser.execute_script("window.scroll(1, 20000);")   
                    
    elif i == 2:
         browser.execute_script("window.scroll(1, 30000);") 
        
    elif i == 3:
         browser.execute_script("window.scroll(1, 40000);")
    time.sleep(7)
    print(i)


# In[21]:


page       = browser.page_source
page_html  = BeautifulSoup(page, 'html.parser')


# In[22]:


vid = page_html.findAll('a',attrs={'ytd-grid-video-renderer'})


# In[22]:


len(vid)


# In[23]:


video=[]
link=[]
date=[]
a=[]
b=[]
for x in range(len(vid)):
    title1 = vid[x].get('title') 
    href  = 'https://www.youtube.com'+ vid[x].get('href')
    if re.match('tagesschau', title1):
          video.append(title1)
          link.append(href)
          a,b=title1.split(",")
          date.append(b.strip())


# In[24]:


df6 = pd.DataFrame({'$Name':link, '$Link': video, '$Date': date})
df6.columns = ['Date','Name', 'Link',]
df6.set_index('Date')
df6['Date'] = pd.to_datetime(df6['Date'],format="%d.%m.%Y")
df7=df6.drop_duplicates(['Date'], keep='last')
df7['year'], df7['month'],df7['day'] = df7['Date'].apply(lambda x: x.year), df7['Date'].apply(lambda x: x.month),df7['Date'].apply(lambda x: x.day)


# In[25]:


df8=df7.sort_values('Date')
df9=df8.set_index('Date')
df10=df9[::-1]


# In[26]:


for row in range(len(df10)):
    r = requests.get(df10.iloc[row]['Link'])
    soup = BeautifulSoup(r.text, 'html.parser')
    topic= soup.find(id="eow-description")
    df10.set_value(df10.index[row],'Topic',topic.text)


# In[27]:


for row in range(0,4):
   try:
    yt = YouTube(df10.iloc[row]['Link'])
    caption = yt.captions.get_by_language_code('de')
    df10.set_value(df10.index[row],'Caption',caption.generate_srt_captions()) 
   except:
        print('No caption')


# In[28]:


df11=df10.dropna()
df11


# In[29]:


#Caption 
os.chdir(r'F:\Channel4')


# In[33]:


for row in range(0,4):
     path=os.path.join('Tagesschauchannel','tagesschau' + '_' + str(df10.iloc[row]['year']) + '.' + str(df10.iloc[row]['month'])+'.' + str(df10.iloc[row]['day']))
     os.makedirs(path)
     yt = YouTube(df10.iloc[row]['Link'])
     yt.streams.filter(progressive = True, file_extension = 'mp4').order_by('resolution').desc().first().download(path)
     path1=os.path.join(path, "raw_data.srt")
     f = open(path1,'w')
     caption = yt.captions.get_by_language_code('de')
     f.write(caption.generate_srt_captions())
     f.close() 


# In[42]:


os.chdir(r'F:\Channel4')


# In[39]:


import cv2
print(cv2.__version__)
vidcap = cv2.VideoCapture('tagesschau 2000 Uhr 05122012.mp4')
success,image = vidcap.read()
count = 0
success = True
while success:
  cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
  success,image = vidcap.read()
  print ('Read a new frame: ', success)
  count += 1


# In[41]:


def frames_to_video(inputpath,outputpath,fps):
   image_array = []
   files = [f for f in os.listdir(inputpath) if isfile(join(inputpath, f))]
   files.sort(key = lambda x: int(x[5:-4]))
   for i in range(len(files)):
       img = cv2.imread(inputpath + files[i])
       size =  (img.shape[1],img.shape[0])
       img = cv2.resize(img,size)
       image_array.append(img)
   fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
   out = cv2.VideoWriter(outputpath,fourcc, fps, size)
   for i in range(len(image_array)):
       out.write(image_array[i])
   out.release()


inputpath = r'F:\Channel4\Tagesschauchannel\tagesschau_2012.12.5'
outpath =  r'F:\Channel4\Tagesschauchannel\tagesschau_2012.12.5\'tagesschau 2000 Uhr 05122012.mp4'
fps = 29
frames_to_video(inputpath,outpath,fps)


# In[44]:


# create a folder to store extracted images
import os
folder = 'test'  
os.mkdir(folder)
# use opencv to do the job
import cv2
print(cv2.__version__)  # my version is 3.1.0
vidcap = cv2.VideoCapture(r'F:\Channel4\Tagesschauchannel\tagesschau_2012.12.5\'tagesschau 2000 Uhr 05122012.mp4')
count = 0
while True:
    success,image = vidcap.read()
    if not success:
        break
    cv2.imwrite(os.path.join(folder,"frame{:d}.jpg".format(count)), image)     # save frame as JPEG file
    count += 1
print("{} images are extacted in {}.".format(count,folder))


# In[47]:


cap = cv2.VideoCapture(r'F:\Channel4\Tagesschauchannel\tagesschau_2012.12.5\'tagesschau 2000 Uhr 05122012.mp4')


# In[46]:


def video_to_frames(input_loc, output_loc):
    """Function to extract frames from input video file
    and save them as separate frames in an output directory.
    Args:
        input_loc: Input video file.
        output_loc: Output directory to save the frames.
    Returns:
        None
    """
    try:
        os.mkdir(output_loc)
    except OSError:
        pass
    # Log the time
    time_start = time.time()
    # Start capturing the feed
    cap = cv2.VideoCapture(input_loc)
    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print ("Number of frames: ", video_length)
    count = 0
    print ("Converting video..\n")
    # Start converting the video
    while cap.isOpened():
        # Extract the frame
        ret, frame = cap.read()
        # Write the results back to output location.
        cv2.imwrite(output_loc + "/%#05d.jpg" % (count+1), frame)
        count = count + 1
        # If there are no more frames left
        if (count > (video_length-1)):
            # Log the time again
            time_end = time.time()
            # Release the feed
            cap.release()
            # Print stats
            print ("Done extracting frames.\n%d frames extracted" % count)
            print ("It took %d seconds forconversion." % (time_end-time_start))
            break

input_loc = r'F:\Channel4\Tagesschauchannel\tagesschau_2012.12.5\'tagesschau 2000 Uhr 05122012.mp4'
output_loc = r'F:\Channel4\test'
video_to_frames(input_loc, output_loc)


# In[49]:


import cv2
import numpy as np

def extract_image_one_fps(video_source_path):

    vidcap = cv2.VideoCapture(video_source_path)
    count = 0
    success = True
    while success:
      vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))      
      success,image = vidcap.read()

      ## Stop when last frame is identified
      image_last = cv2.imread("frame{}.png".format(count-1))
      if np.array_equal(image,image_last):
          break

      cv2.imwrite("frame%d.png" % count, image)     # save frame as PNG file
      print ('{}.sec reading a new frame: {} '.format(count,success))
      count += 1


# In[57]:


'''''Using OpenCV takes a mp4 video and produces a number of images.
Requirements
----
You require OpenCV 3.2 to be installed.
Run
----
Open the main.py and edit the path to the video. Then run:
$ python main.py
Which will produce a folder called data with the images. There will be 2000+ images for example.mp4.
'''


# Playing video from file:
cap = cv2.VideoCapture(r'F:\Channel4\Tagesschauchannel\tagesschau_2012.12.5\'tagesschau 2000 Uhr 05122012.mp4')

try:
    if not os.path.exists('data2'):
        os.makedirs('data2')
except OSError:
    print ('Error: Creating directory of data')


# In[56]:


currentFrame = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Saves image of the current frame in jpg file
    name = './data1/frame' + str(currentFrame) + '.png'
    print ('Creating...' + name)
    cv2.imwrite(name, frame)

    # To stop duplicate images
    currentFrame += 1

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


# In[70]:


def extractFrames(pathIn, pathOut):
    os.mkdir(pathOut)
    cap = cv2.VideoCapture(pathIn)
    count = 0
    while (cap.isOpened()):
 
        # Capture frame-by-frame
        ret, frame = cap.read()
 
        if ret == True:
            print('Read %d frame: ' % count, ret)
            cv2.imwrite(os.path.join(pathOut, "frame{:d}.jpg".format(count)), frame)  # save frame as JPEG file
            count += 1
        else:
            break
 
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
 


# In[71]:


def main():
    extractFrames(r'F:\Channel4\Tagesschauchannel\tagesschau_2012.12.5\'tagesschau 2000 Uhr 05122012.mp4','data4')
 
if __name__=="__main__":
    main()


# In[72]:


def extractFrames(pathIn, pathOut):
    os.mkdir(pathOut)
 
    cap = cv2.VideoCapture(pathIn)
    count = 0
 
    while (cap.isOpened()):
 
        # Capture frame-by-frame
        ret, frame = cap.read()
 
        if ret == True:
            print('Read %d frame: ' % count, ret)
            cv2.imwrite(os.path.join(pathOut, "frame{:d}.jpg".format(count)), frame)  # save frame as JPEG file
            count += 1
        else:
            break
 
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
 
def main():
    extractFrames(r'F:\Channel4\Tagesschauchannel\tagesschau_2012.12.5\'tagesschau 2000 Uhr 05122012.mp4','data5')
 
if __name__=="__main__":
    main()


# In[ ]:


import cv2
 
def play_videoFile(filePath,mirror=False):
 
    cap = cv2.VideoCapture(filePath)
    cv2.namedWindow('Video Life2Coding',cv2.WINDOW_AUTOSIZE)
    while True:
        ret_val, frame = cap.read()
 
        if mirror:
            frame = cv2.flip(frame, 1)
 
        cv2.imshow('Video Life2Coding', frame)
 
        if cv2.waitKey(1) == 27:
            break  # esc to quit
 
    cv2.destroyAllWindows()
 
def main():
    play_videoFile('bigbuckbunny720p5mb.mp4',mirror=False)
 
if __name__ == '__main__':
    main()


# In[68]:


extractFrames(r'F:\Channel4\Tagesschauchannel\tagesschau_2012.12.5\'tagesschau 2000 Uhr 05122012.mp4','data2')
 


# In[73]:


import sys
import argparse

import cv2
print(cv2.__version__)

def extractImages(pathIn, pathOut):
    count = 0
    vidcap = cv2.VideoCapture(pathIn)
    success,image = vidcap.read()
    success = True
    while success:
      vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))    # added this line 
      success,image = vidcap.read()
      print ('Read a new frame: ', success)
      cv2.imwrite( pathOut + "\\frame%d.jpg" % count, image)     # save frame as JPEG file
      count = count + 1

if __name__=="__main__":
    print("aba")
    a = argparse.ArgumentParser()
    a.add_argument("--pathIn", help="path to video")
    a.add_argument("--pathOut", help="path to images")
    args = a.parse_args()
    print(args)
    extractImages(args.pathIn, args.pathOut)


# In[76]:


def main():
    extractImages('bigbuckbunny720p5mb.mp4',data6)
 


# In[ ]:


def extract_image_one_fps(video_source_path):

    vidcap = cv2.VideoCapture(video_source_path)
    count = 0
    success = True
    while success:
      vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))      
      success,image = vidcap.read()

      ## Stop when last frame is identified
      image_last = cv2.imread("frame{}.png".format(count-1))
      if np.array_equal(image,image_last):
          break

      cv2.imwrite("frame%d.png" % count, image)     # save frame as PNG file
      print '{}.sec reading a new frame: {} '.format(count,success)
      count += 1


# In[30]:


def clean_subtitles( subtitles_pc):
    
    
    # Helper Functions
    # Remove <font color = "#ff0000"> (start) tag
    def remove_start_tag( row ):
        
        return re.sub( '<font color="#.{6}">', ' ', row['Text'] )
    
    
    # Remove </font> (end) tag    
    def remove_end_tag( row ):
        
        return re.sub( '</font>', ' ', row['Text'] )
    
    
    def remove_white_spaces( row ):
    
        return ' '.join( row['Text'].split() )

    
    
    # Create data frame (to store result)
    #
    subtitles_pc_clean = pd.DataFrame( columns = ['Date','N_Gram' ,'Number','Text', 'Time start', 'Time end'] )
    
    
    # Split column 'Time'
    #
    subtitles_pc_clean['Time start'] = subtitles_pc['Time'].str.split( '-->', expand = True )[0]
    subtitles_pc_clean['Time end']   = subtitles_pc['Time'].str.split( '-->', expand = True )[1]
    
    
    # Strip white spaces and newlines
    #
    subtitles_pc_clean['Number']     = subtitles_pc['Number'].str.strip()
    subtitles_pc_clean['Text']       = subtitles_pc['Text'].str.strip()
    
    subtitles_pc_clean['Time start'] = subtitles_pc_clean['Time start'].str.strip()
    subtitles_pc_clean['Time end']   = subtitles_pc_clean['Time end'].str.strip()
    
    
    # Format the number
    #
    subtitles_pc_clean['Number'] = pd.to_numeric( subtitles_pc_clean['Number'] )
    
    
    # Format the time
    #
    subtitles_pc_clean['Time start'] = pd.to_datetime( subtitles_pc_clean['Time start'], format = '0%H:0%M:%S,%f').dt.time
    subtitles_pc_clean['Time end']   = pd.to_datetime( subtitles_pc_clean['Time end'],   format = '0%H:0%M:%S,%f').dt.time
    
    
    # Remove html tags
    #
    subtitles_pc_clean['Text'] = subtitles_pc_clean.apply( remove_start_tag, axis = 1 )
    subtitles_pc_clean['Text'] = subtitles_pc_clean.apply( remove_end_tag,   axis = 1 )
    
    
    # Remove duplicated white spaces (inside the sentence)
    #
    subtitles_pc_clean['Text'] = subtitles_pc_clean.apply( remove_white_spaces, axis = 1 )
    
    # Set ECB PC date (as index)
    return subtitles_pc_clean


# In[31]:


def parse_subtitles( lines ):
    
    # Create data frame (to store result)
    #
    subtitles_pc = pd.DataFrame( columns = ['Date','N_Gram','Number','Time', 'Text'] )
    
    
    # Initialize variables
    index = 0
    i     = 0

    # Run loop
    while i < len(lines):
    
        if re.match( '[1]?\d{1,6}\n', lines[i] ):                   # E.g. 1,   max subtitle number = 1164, year = 2013 (e.g.)
            subtitles_pc.set_value(index, 'Number', lines[i]) 

        elif re.match( '\d{3}:\d{2}', lines[i] ):                  # E.g. 000:00
            subtitles_pc.set_value(index, 'Time', lines[i]) 
    
        elif re.match( '.', lines[i] ):                            # E.g. conference before ...
            text = ''
            text = text + ' ' + lines[i]
 
            subtitles_pc.set_value(index, 'Text', text)
        
        elif re.match( '\n', lines[i] ):
            index = index + 1
        
        i = i + 1

        
    return subtitles_pc


# In[32]:


lines = []
for row in range(0,len(df11)):
    path=os.path.join('Tagesschauchannel','tagesschau' + '_' + str(df11.iloc[row]['year']) + '.' + str(df11.iloc[row]['month'])+'.' + str(df11.iloc[row]['day']))
    path1= os.path.join(path, "raw_data.srt") 
    with open( path1) as f:
                for line in f:
                      lines.append( line )


# In[33]:


subtitles_pc=parse_subtitles( lines )
subtitles_pc


# In[35]:


subtitles_pc1 = subtitles_pc[subtitles_pc['Text'].notnull()]
subtitles_pc1 


# In[36]:


subtitles_pc_clean1=clean_subtitles(subtitles_pc1)
subtitles_pc_clean1


# In[37]:


A=list(range(1,209))
A


# In[38]:


i=0
for row1 in range(0,len(subtitles_pc_clean1)):
        if  subtitles_pc_clean1.iloc[row1]['Number']==1:
            subtitles_pc_clean1.set_value(subtitles_pc_clean1.index[row1], 'Date',date[i])
            subtitles_pc_clean1.set_value(subtitles_pc_clean1.index[row1], 'N_Gram',A[i])
            i=i+1


# In[45]:


subtitles_pc_clean1[420:440]


# subtitles_pc_clean4=subtitles_pc_clean2.fillna(method='ffill')

# In[39]:


subtitles_pc_clean2=subtitles_pc_clean1.fillna(method='ffill')
subtitles_pc_clean2


# In[40]:


subtitles_pc_clean2["Token"] = subtitles_pc_clean2["Text"].apply(nltk.word_tokenize)


# In[41]:


subtitles_pc_clean2


# In[42]:


a=subtitles_pc_clean2.groupby('N_Gram')['Text'].apply(' '.join).reset_index()
a


# In[91]:


#### Split topic


# In[43]:


df16=df11.drop(df11.columns[[2,3,4]], axis=1)


# In[44]:


df17= df16.reset_index()


# In[45]:


df17['Caption'] = a['Text']


# In[46]:


df17


# In[47]:


def tidy_split(df, column, sep='|', keep=False):
    """
    Split the values of a column and expand so the new DataFrame has one split
    value per row. Filters rows where the column is missing.

    Params
    ------
    df : pandas.DataFrame
        dataframe with the column to split and expand
    column : str
        the column to split and expand
    sep : str
        the string used to split the column's values
    keep : bool
        whether to retain the presplit value as it's own row

    Returns
    -------
    pandas.DataFrame
        Returns a dataframe with the same columns as `df`.
    """
    indexes = list()
    new_values = list()
    df = df.dropna(subset=[column])
    for i, presplit in enumerate(df[column].astype(str)):
        values = presplit.split(sep)
        if keep and len(values) > 1:
            indexes.append(i)
            new_values.append(presplit)
        for value in values:
            indexes.append(i)
            new_values.append(value)
    new_df = df.iloc[indexes, :].copy()
    new_df[column] = new_values
    return new_df


# In[48]:


df18=tidy_split(df17, 'Topic',',')


# In[72]:


df18["Token"] = df18["Caption"].apply(nltk.word_tokenize)


# In[73]:


df18


# In[50]:


import pyphen
dic = pyphen.Pyphen(lang='de')


# In[51]:


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


# In[1]:


subtitles_pc_clean2


# In[50]:


def syllable_count(word):
    word = word.lower()
    count = 0
    vowels = "aeiouy"
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
            if word.endswith("e"):
                count -= 1
    if count == 0:
        count += 1
    return count


# In[52]:


syllable_count('castle')


# In[56]:


def syllables(word):
    count = 0
    vowels = 'aeiouy'
    word = word.lower().strip(".:;?!")
    if word[0] in vowels:
        count +=1
    for index in range(1,len(word)):
        if word[index] in vowels and word[index-1] not in vowels:
            count +=1
    if word.endswith('e'):
        count -= 1
    if word.endswith('le'):
        count+=1
    if count == 0:
        count +=1
    return count


# In[61]:


def sylco(word) :

    word = word.lower()

    # exception_add are words that need extra syllables
    # exception_del are words that need less syllables

    exception_add = ['serious','crucial']
    exception_del = ['fortunately','unfortunately']

    co_one = ['cool','coach','coat','coal','count','coin','coarse','coup','coif','cook','coign','coiffe','coof','court']
    co_two = ['coapt','coed','coinci']

    pre_one = ['preach']

    syls = 0 #added syllable number
    disc = 0 #discarded syllable number

    #1) if letters < 3 : return 1
    if len(word) <= 3 :
        syls = 1
        return syls

    #2) if doesn't end with "ted" or "tes" or "ses" or "ied" or "ies", discard "es" and "ed" at the end.
    # if it has only 1 vowel or 1 set of consecutive vowels, discard. (like "speed", "fled" etc.)

    if word[-2:] == "es" or word[-2:] == "ed" :
        doubleAndtripple_1 = len(re.findall(r'[eaoui][eaoui]',word))
        if doubleAndtripple_1 > 1 or len(re.findall(r'[eaoui][^eaoui]',word)) > 1 :
            if word[-3:] == "ted" or word[-3:] == "tes" or word[-3:] == "ses" or word[-3:] == "ied" or word[-3:] == "ies" :
                pass
            else :
                disc+=1

    #3) discard trailing "e", except where ending is "le"  

    le_except = ['whole','mobile','pole','male','female','hale','pale','tale','sale','aisle','whale','while']

    if word[-1:] == "e" :
        if word[-2:] == "le" and word not in le_except :
            pass

        else :
            disc+=1

    #4) check if consecutive vowels exists, triplets or pairs, count them as one.

    doubleAndtripple = len(re.findall(r'[eaoui][eaoui]',word))
    tripple = len(re.findall(r'[eaoui][eaoui][eaoui]',word))
    disc+=doubleAndtripple + tripple

    #5) count remaining vowels in word.
    numVowels = len(re.findall(r'[eaoui]',word))

    #6) add one if starts with "mc"
    if word[:2] == "mc" :
        syls+=1

    #7) add one if ends with "y" but is not surrouned by vowel
    if word[-1:] == "y" and word[-2] not in "aeoui" :
        syls +=1

    #8) add one if "y" is surrounded by non-vowels and is not in the last word.

    for i,j in enumerate(word) :
        if j == "y" :
            if (i != 0) and (i != len(word)-1) :
                if word[i-1] not in "aeoui" and word[i+1] not in "aeoui" :
                    syls+=1

    #9) if starts with "tri-" or "bi-" and is followed by a vowel, add one.

    if word[:3] == "tri" and word[3] in "aeoui" :
        syls+=1

    if word[:2] == "bi" and word[2] in "aeoui" :
        syls+=1

    #10) if ends with "-ian", should be counted as two syllables, except for "-tian" and "-cian"

    if word[-3:] == "ian" : 
    #and (word[-4:] != "cian" or word[-4:] != "tian") :
        if word[-4:] == "cian" or word[-4:] == "tian" :
            pass
        else :
            syls+=1

    #11) if starts with "co-" and is followed by a vowel, check if exists in the double syllable dictionary, if not, check if in single dictionary and act accordingly.

    if word[:2] == "co" and word[2] in 'eaoui' :

        if word[:4] in co_two or word[:5] in co_two or word[:6] in co_two :
            syls+=1
        elif word[:4] in co_one or word[:5] in co_one or word[:6] in co_one :
            pass
        else :
            syls+=1

    #12) if starts with "pre-" and is followed by a vowel, check if exists in the double syllable dictionary, if not, check if in single dictionary and act accordingly.

    if word[:3] == "pre" and word[3] in 'eaoui' :
        if word[:6] in pre_one :
            pass
        else :
            syls+=1

    #13) check for "-n't" and cross match with dictionary to add syllable.

    negative = ["doesn't", "isn't", "shouldn't", "couldn't","wouldn't"]

    if word[-3:] == "n't" :
        if word in negative :
            syls+=1
        else :
            pass   

    #14) Handling the exceptional words.

    if word in exception_del :
        disc+=1

    if word in exception_add :
        syls+=1     

    # calculate the output
    return numVowels - disc + syls


# In[68]:


sylco('speed')


# In[74]:


subtitles_pc_clean3=subtitles_pc_clean2[0:3]
subtitles_pc_clean3


# In[78]:


for i in range(0,len(subtitles_pc_clean3)):
   subtitles_pc_clean3.iloc[i]['Syllable']=syllable_count(subtitles_pc_clean3.iloc[i]['Text'])


# In[52]:


subtitles_pc_clean2["Syllabe1"] = subtitles_pc_clean2["Text"].apply(syllable_count)


# In[53]:


subtitles_pc_clean2


# In[54]:


subtitles_pc_clean2.to_excel()


# In[55]:


writer = pd.ExcelWriter('output.xlsx')


# In[57]:


subtitles_pc_clean2.to_excel(writer,'Sheet1')


# In[76]:


writer2.save()


# In[ ]:


### IMAGE


# In[59]:


from PIL import Image


# In[ ]:


import cv2


# In[61]:


from pytesseract import image_to_string


# In[74]:


writer2 = pd.ExcelWriter('Df18.xlsx')


# In[75]:


df18.to_excel(writer2,'Sheet1')



# coding: utf-8

# In[ ]:


##SETUP


# In[1]:


import os
import numpy as np
import pandas as pd
import pysrt


# In[2]:


import datetime as dt
import time


# In[3]:


from bs4 import BeautifulSoup
import re
from selenium import webdriver
import requests


# In[4]:


from pytube import YouTube
import cv2


# In[5]:


import nltk
#nltk.download()
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk.data
from nltk import punkt


# In[6]:


from string import punctuation


# In[4]:


os.chdir(r'C:\Program Files (x86)\Tesseract-OCR')


# In[5]:


video_link = 'https://www.youtube.com/user/TagesschauBackup/videos'


# In[6]:


browser = webdriver.Chrome()


# In[7]:


browser.get( video_link )


# In[8]:


browser.execute_script("window.scroll(1, 500);")


# In[9]:


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


# In[13]:


page       = browser.page_source
page_html  = BeautifulSoup(page, 'html.parser')


# In[14]:


vid = page_html.findAll('a',attrs={'ytd-grid-video-renderer'})


# In[15]:


vid


# In[14]:


len(vid)


# In[15]:


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


# In[16]:


df6 = pd.DataFrame({'$Name':link, '$Link': video, '$Date': date})
df6.columns = ['Date','Name', 'Link',]
df6.set_index('Date')
df6['Date'] = pd.to_datetime(df6['Date'],format="%d.%m.%Y")
df7=df6.drop_duplicates(['Date'], keep='last')
df7['year'], df7['month'],df7['day'] = df7['Date'].apply(lambda x: x.year), df7['Date'].apply(lambda x: x.month),df7['Date'].apply(lambda x: x.day)


# In[17]:


df8=df7.sort_values('Date')
df9=df8.set_index('Date')
df10=df9[::-1]


# In[18]:


for row in range(len(df10)):
    r = requests.get(df10.iloc[row]['Link'])
    soup = BeautifulSoup(r.text, 'html.parser')
    topic= soup.find(id="eow-description")
    a,b,*rest=topic.text.split(':')
    df10.set_value(df10.index[row],'Topic',b)


# In[20]:


df10


# In[21]:


# Delete videos without captions
for row in range(0,len(df10)):
   try:
    yt = YouTube(df10.iloc[row]['Link'])
    caption = yt.captions.get_by_language_code('de')
    df10.set_value(df10.index[row],'Caption',caption.generate_srt_captions()) 
   except:
        print('No caption')


# In[22]:


df11=df10.dropna()
df12= df11.reset_index()
date=df12.Date
df11


# In[27]:


#Caption 
os.chdir(r'F:\Project')


# In[61]:


for row in range(0,len(df11)):
     path=os.path.join('Tagesschauchannel','tagesschau' + '_' + str(df11.iloc[row]['year']) + '.' + str(df11.iloc[row]['month'])+'.' + str(df11.iloc[row]['day']))
     os.makedirs(path)


# In[31]:


# Create folder
for row in range(0,len(df11)):
     path=os.path.join('Gemarn-Tagesschauchannel','tagesschau' + '_' + str(df11.iloc[row]['year']) + '.' + str(df11.iloc[row]['month'])+'.' + str(df11.iloc[row]['day']))
     os.makedirs(path)
     yt = YouTube(df11.iloc[row]['Link'])
     path1=os.path.join(path, "Gesubtitle.srt")
     f = open(path1,'w')
     caption = yt.captions.get_by_language_code('de')
     f.write(caption.generate_srt_captions())
     f.close() 


# In[ ]:


### SUBTITLE


# In[30]:


## download German captions from website downsub.com
Gelines = []
for row in range(0,len(df11)):
    path=os.path.join('Tagesschauchannel','tagesschau' + '_' + str(df11.iloc[row]['year']) + '.' + str(df11.iloc[row]['month'])+'.' + str(df11.iloc[row]['day']))
    path1= os.path.join(path, "Ge.srt") 
    with open( path1) as f:
                for line in f:
                      Gelines.append( line )


# In[243]:


# Download German captions automatically by python
Gemarnlines = []
for row in range(0,len(df11)):
    path=os.path.join('Gemarn-Tagesschauchannel','tagesschau' + '_' + str(df11.iloc[row]['year']) + '.' + str(df11.iloc[row]['month'])+'.' + str(df11.iloc[row]['day']))
    path1= os.path.join(path, "Gesubtitle.srt") 
    with open( path1) as f:
                for line in f:
                      Gemarnlines.append( line )


# In[ ]:


##SUBTITLES


# In[ ]:


# German ,use python


# In[ ]:


def clean_subtitles( subtitles_pc):
    
    
    # Helper Functions
    #
    
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
    subtitles_pc_clean = pd.DataFrame( columns = ['Date', 'Number', 'N_Gram' 'Text', 'Time start', 'Time end'] )
    
    
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
    #
    
    
    return subtitles_pc_clean
    
    


# In[ ]:


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


# In[249]:


Gemarnsubtitles_pc=parse_subtitles( Gemarnlines )


# In[250]:


Gemarnsubtitles_pc


# In[262]:


Gemarnsubtitles_pc1 = Gemarnsubtitles_pc[Gemarnsubtitles_pc['Text'].notnull()]
Gemarnsubtitles_pc1


# In[268]:


Gemarnsubtitles_pc_clean=clean_subtitles(Gemarnsubtitles_pc1)
Gemarnsubtitles_pc_clean


# In[ ]:


i=0
for row1 in range(0,len(Gemarnsubtitles_pc_clean)):
        if  Gemarnsubtitles_pc_clean.iloc[row1]['Number']==1:
            Gemarnsubtitles_pc_clean.set_value(Gemarnsubtitles_pc_clean.index[row1], 'Date',Date[i])
            Gemarnsubtitles_pc_clean.set_value(Gemarnsubtitles_pc_clean.index[row1], 'N_GramText',A[i])
            i=i+1


# In[286]:


Gemarnsubtitles_pc_clean=Gemarnsubtitles_pc_clean.fillna(method='ffill')
Gemarnsubtitles_pc_clean


# In[285]:


### Down Manually


# In[289]:


def clean_subtitles1( subtitles_pc):
    
    
    # Helper Functions
    #
    
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
    subtitles_pc_clean = pd.DataFrame( columns = ['Date', 'Number', 'N_Gram' 'Text', 'Time start', 'Time end'] )
    
    
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
    subtitles_pc_clean['Time start'] = pd.to_datetime( subtitles_pc_clean['Time start'], format = '%H:%M:%S,%f').dt.time
    subtitles_pc_clean['Time end']   = pd.to_datetime( subtitles_pc_clean['Time end'],   format = '%H:%M:%S,%f').dt.time
    
    
    # Remove html tags
    #
    subtitles_pc_clean['Text'] = subtitles_pc_clean.apply( remove_start_tag, axis = 1 )
    subtitles_pc_clean['Text'] = subtitles_pc_clean.apply( remove_end_tag,   axis = 1 )
    
    
    # Remove duplicated white spaces (inside the sentence)
    #
    subtitles_pc_clean['Text'] = subtitles_pc_clean.apply( remove_white_spaces, axis = 1 )
    
    
    # Set ECB PC date (as index)
    #
    
    
    return subtitles_pc_clean
    
    


# In[288]:


def parse_subtitles1( lines ):
    
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


# In[290]:


Gesubtitles_pc=parse_subtitles1( Gelines )
Gesubtitles_pc


# In[254]:


Gesubtitles_pc1 = Gesubtitles_pc[Gesubtitles_pc['Text'].notnull()]
Gesubtitles_pc1


# In[ ]:


Gesubtitles_pc_clean=clean_subtitles1(Gesubtitles_pc1)
Gesubtitles_pc_clean


# In[ ]:


i=0
for row1 in range(0,len(Gesubtitles_pc_clean)):
        if  Gesubtitles_pc_clean.iloc[row1]['Number']==1:
            Gesubtitles_pc_clean.set_value(Gesubtitles_pc_clean.index[row1], 'Date',Date[i])
            Gesubtitles_pc_clean.set_value(Gesubtitles_pc_clean.index[row1], 'N_GramText',A[i])
            i=i+1


# In[ ]:


Gesubtitles_pc_clean=Gesubtitles_pc_clean.fillna(method='ffill')
Gesubtitles_pc_clean


# In[ ]:


subtitles_pc=parse_subtitles1(lines)


# In[133]:


subtitles_pc1 = subtitles_pc[subtitles_pc['Text'].notnull()]
subtitles_pc1


# In[216]:


subtitles_pc_clean2=clean_subtitles1(subtitles_pc1)


# In[215]:


A=list(range(1,209))


# In[ ]:


Date=[]
for i in range(0,len(df12)):
    date=df12.Date[i]
    Date.append(date)


# In[179]:


i=0
for row1 in range(0,len(subtitles_pc_clean2)):
        if  subtitles_pc_clean2.iloc[row1]['Number']==1:
            subtitles_pc_clean2.set_value(subtitles_pc_clean2.index[row1], 'Date',Date[i])
            subtitles_pc_clean2.set_value(subtitles_pc_clean2.index[row1], 'N_GramText',A[i])
            i=i+1


# In[279]:


Gesubtitles_pc_clean=Gesubtitles_pc_clean.fillna(method='ffill')
Gesubtitles_pc_clean


# subtitles_pc_clean4=subtitles_pc_clean2.fillna(method='ffill')

# In[40]:


##TOKEN


# In[ ]:


subtitles_pc_clean2["Token"] = subtitles_pc_clean2["Text"].apply(nltk.word_tokenize)


# In[41]:


subtitles_pc_clean2


# In[42]:


a=subtitles_pc_clean2.groupby('N_Gram')['Text'].apply(' '.join).reset_index()
a


# In[91]:


#### Split topic


# In[32]:


df16=df11.drop(df11.columns[[2,3,4]], axis=1)


# In[33]:


df17= df16.reset_index()


# In[31]:


df17


# In[35]:


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


# In[36]:


df18=tidy_split(df17, 'Topic',',')


# In[40]:


df18["Token"] = df18["Caption"].apply(nltk.word_tokenize)


# In[54]:


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


# In[78]:


for i in range(0,len(subtitles_pc_clean3)):
   subtitles_pc_clean3.iloc[i]['Syllable']=syllable_count(subtitles_pc_clean3.iloc[i]['Text'])


# In[52]:


subtitles_pc_clean2["Syllabe1"] = subtitles_pc_clean2["Text"].apply(syllable_count)


# In[53]:


subtitles_pc_clean2


# In[57]:


subtitles_pc_clean2.to_excel(writer,'Sheet1')


# In[ ]:


### IMAGE


# In[282]:


Ge1 = pd.ExcelWriter('Gesubtitles_pc_clean.xlsx')
Gesubtitles_pc_clean.to_excel(Ge1,'Sheet1')


# In[283]:


Gesubtitles_pc_clean.to_excel(Ge1,'Sheet1')


# In[272]:


os.chdir(r'F:\Project')


# In[ ]:


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


# In[ ]:


# create a folder to store extracted images
import os
folder = 'test1'  
os.mkdir(folder)
# use opencv to do the job
import cv2
print(cv2.__version__)  # my version is 3.1.0
vidcap = cv2.VideoCapture('tagesschau 2000 Uhr 05122012.mp4')
while True:
    success,image = vidcap.read()
    if not success:
        break
    cv2.imwrite(os.path.join(folder,"frame{:d}.jpg".format(count)), image)     # save frame as JPEG file
    count += 1
print("{} images are extacted in {}.".format(count,folder))


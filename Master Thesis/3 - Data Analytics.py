
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs
from sklearn import feature_extraction
import scipy
from scipy import stats


# In[3]:


from bs4 import BeautifulSoup   


# In[4]:


from nltk.tokenize import WordPunctTokenizer 


# In[5]:


from sklearn.cluster import KMeans


# In[6]:


from nltk.tokenize import sent_tokenize


# In[7]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from __future__ import print_function


# In[8]:


import mpld3


# In[9]:


from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")


# In[10]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[11]:


stopwords = nltk.corpus.stopwords.words('German')
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")


# In[12]:


os.chdir(r'F:\Project')


# In[13]:


DAX= pd.read_excel('DAX.xlsx',sheetname="DAX")
DAX['pct_change'] = DAX['Adj Close'].pct_change()
DAX['log_ret'] = np.log(DAX['Adj Close']) - np.log(DAX['Adj Close'].shift(1))
in_range_df = DAX[DAX["Date"].isin(pd.date_range("2010-09-03", "2012-12-05"))]
in_range_df


# In[14]:


Caption1= pd.read_excel('Topic and Caption1.xlsx',sheet_name="Topic and Caption")


# In[ ]:


def Clean_text(text):
    from nltk.tokenize import word_tokenize
    tokens = word_tokenize(text)
# convert to lower case
    tokens = [w.lower() for w in tokens]
# remove punctuation from each word
    import string
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
# remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    return(words)


# In[ ]:


def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens


# In[ ]:


Caption1["Token"] =Caption1["English"].apply(tokenize_only)
Caption1['Token1'] = pd.DataFrame([str(line).strip('[').strip(']').replace("'","") for line in Caption1['Token']])


# In[ ]:


Wordlist= pd.read_excel('List.xls')
Wordlist1=Wordlist.loc[Wordlist['Positiv'] == 'Positiv']
PositveWordlist =Wordlist1.loc[:,["Entry","Positiv"]]
PositveWordlist["clean"] =PositveWordlist["Entry"].apply(Clean_text)
PositveWordlist = PositveWordlist[~PositveWordlist['clean'].apply(tuple).duplicated()]
PositveWordlist['clean'] = PositveWordlist['clean'].str.get(0)
Positive=PositveWordlist.clean


# In[17]:


Caption1['Count-Positive']=Caption1['Token1'].str.split(',\s?', expand=True).stack().isin(Positive.tolist()).groupby(level=0).sum()


# In[18]:


Wordlist1= pd.read_excel('List.xls')
Wordlist2=Wordlist1.loc[Wordlist['Negativ'] == 'Negativ']
NegativeWordlist =Wordlist2.loc[:,["Entry","Negativ"]]
NegativeWordlist['Entry']=NegativeWordlist['Entry'].str.replace('#', '')
NegativeWordlist['Entry'] = NegativeWordlist['Entry'].str.replace('\d+', '')
NegativeWordlist['Entry']=NegativeWordlist.apply(lambda col: col.str.lower())
Negative=NegativeWordlist.drop_duplicates(subset=None, keep='first', inplace=False).Entry


# In[19]:


Caption1['Count-Negative']=Caption1['Token1'].str.split(',\s?', expand=True).stack().isin(Negative.tolist()).groupby(level=0).sum()


# In[20]:


Caption1['totalwords'] = Caption1['Token1'].str.split(",").str.len()


# In[21]:


Caption1['totalwords-token'] = Caption1['Token1'].str.split(",").str.len()


# In[22]:


Caption1[['Positive tone','Negative tone']] = Caption1[['Count-Positive','Count-Negative']].div(Caption1['totalwords'].values,axis=0)


# In[23]:


Caption1.to_excel('Caption1.xlsx', sheet_name='Sheet1', engine='xlsxwriter')


# In[15]:


Caption2= pd.read_excel('Caption1.xlsx',sheetname="Sheet1")


# In[16]:


Caption2


# In[17]:


a= pd.read_excel('Resultnew.xlsx',sheetname="New")
a


# In[18]:


Caption2['Topic'] = Caption2['Topic'].str.strip()
a['Topic'] = a['Topic'].str.strip()
b=(Caption2.merge(a[["Date","Topic","log_ret1"]], 
           how="right", 
           right_on=["Date","Topic"], 
           left_on=["Date","Topic"]))
new=b.drop_duplicates(subset=None, keep='first', inplace=False)
new


# In[19]:


new["P-N"]=(new["Count-Positive"]-new["Count-Negative"])/new["totalwords"]


# In[24]:


del new['totalwords-token']


# In[25]:


del new['Link']


# In[27]:


del new['German']


# In[ ]:


del new['To']


# In[46]:


new.to_excel('Data.xlsx', sheet_name='Sheet1', engine='xlsxwriter')


# In[38]:


from nltk.corpus import stopwords # Import the stop word list
print (stopwords.words("english") )


# In[32]:


def review_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review).get_text() 
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))   


# In[39]:


# Get the number of reviews based on the dataframe column size
num_Text = new["English"].size

# Initialize an empty list to hold the clean reviews
clean_text = []

# Loop over each review; create an index i that goes from 0 to the length
# of the movie review list 
for i in range( 0, num_Text ):
    # Call our function for each one, and add the result to the list of
    # clean reviews
    clean_text.append( review_to_words( new["English"][i] ) )


# In[42]:


clean_text


# In[50]:


print ("Creating the bag of words...\n")
from sklearn.feature_extraction.text import CountVectorizer

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.  
vectorizer = CountVectorizer(analyzer = "word",                                tokenizer = None,                                 preprocessor = None,                              stop_words = None,                                max_features = 5000) 

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of 
# strings.
train_data_features = vectorizer.fit_transform(clean_text)

# Numpy arrays are easy to work with, so convert the result to an 
# array
train_data_features = train_data_features.toarray()


# In[49]:


train_data_features


# In[43]:


print (train_data_features.shape)


# In[44]:


vocab = vectorizer.get_feature_names()
print (vocab)


# In[45]:


import numpy as np

# Sum up the counts of each vocabulary word
dist = np.sum(train_data_features, axis=0)

# For each, print the vocabulary word and the number of times it 
# appears in the training set
for tag, count in zip(vocab, dist):
    print (count, tag)


# In[28]:


Text2=new.English
Topic2=new.Topic


# In[2]:


text= 'aadaaf  fdfdff 100 ffff rrg!!!'


# In[5]:


text= 'aadaaf  fdfdff 100 ffff rrg!!!'
tex1=tokenize_and_stem(text)


# In[4]:


def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        words = [word for word in tokens if word.isalpha()]
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


# In[30]:


def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens


# In[68]:


#not super pythonic, no, not at all.
#use extend so it's a big flat list of vocab
totalvocab_stemmed = []
totalvocab_tokenized = []
for i in Text2:
    allwords_stemmed = tokenize_and_stem(i) #for each item in 'Text, tokenize/stem
    totalvocab_stemmed.extend(allwords_stemmed) #extend the 'totalvocab_stemmed' list
    
    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)


# In[69]:


vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
print( 'there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')


# In[107]:


from sklearn.metrics.pairwise import cosine_similarity
dist = 1 - cosine_similarity(tfidf_matrix)
print
print


# In[31]:


stopwords2=['i',"'s",'also','one','second','today','years','want','first','years','zu','guttenberg', 'bundeswehr',
 'me',
 'my',
 'myself',
 'we',
 'our',
 'ours',
 'ourselves',
 'you',
 "you're",
 "you've",
 "you'll",
 "you'd",
 'your',
 'yours',
 'yourself',
 'yourselves',
 'he',
 'him',
 'his',
 'himself',
 'she',
 "she's",
 'her',
 'hers',
 'herself',
 'it',
 "it's",
 'its',
 'itself',
 'they',
 'them',
 'their',
 'theirs',
 'themselves',
 'what',
 'which',
 'who',
 'whom',
 'this',
 'that',
 "that'll",
 'these',
 'those',
 'am',
 'is',
 'are',
 'was',
 'were',
 'be',
 'been',
 'being',
 'have',
 'has',
 'had',
 'having',
 'do',
 'does',
 'did',
 'doing',
 'a',
 'an',
 'the',
 'and',
 'but',
 'if',
 'or',
 'because',
 'as',
 'until',
 'while',
 'of',
 'at',
 'by',
 'for',
 'with',
 'about',
 'against',
 'between',
 'into',
 'through',
 'during',
 'before',
 'after',
 'above',
 'below',
 'to',
 'from',
 'up',
 'down',
 'in',
 'out',
 'on',
 'off',
 'over',
 'under',
 'again',
 'further',
 'then',
 'once',
 'here',
 'there',
 'when',
 'where',
 'why',
 'how',
 'all',
 'any',
 'both',
 'each',
 'few',
 'more',
 'most',
 'other',
 'some',
 'such',
 'no',
 'nor',
 'not',
 'only',
 'own',
 'same',
 'so',
 'than',
 'too',
 'very',
 's',
 't',
 'can',
 'will',
 'just',
 'don',
 "don't",
 'should',
 "should've",
 'now',
 'd',
 'll',
 'm',
 'o',
 're',
 've',
 'y',
 'ain',
 'aren',
 "aren't",
 'couldn',
 "couldn't",
 'didn',
 "didn't",
 'doesn',
 "doesn't",
 'hadn',
 "hadn't",
 'hasn',
 "hasn't",
 'haven',
 "haven't",
 'isn',
 "isn't",
 'ma',
 'mightn',
 "mightn't",
 'mustn',
 "mustn't",
 'needn',
 "needn't",
 'shan',
 "shan't",
 'shouldn',
 "shouldn't",
 'wasn',
 "wasn't",
 'weren',
 "weren't",
 'won',
 "won't",
 'wouldn',
 "wouldn't"]


# In[72]:


from sklearn.feature_extraction.text import TfidfVectorizer

#define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords2,
                                 use_idf=True, tokenizer=tokenize_and_stem)
tfidf_matrix = tfidf_vectorizer.fit_transform(Text2) #fit the vectorizer to synopses
terms = tfidf_vectorizer.get_feature_names()


# In[73]:


km = KMeans(n_clusters=8)
km.fit(tfidf_matrix)
clusters = km.labels_.tolist()


# In[74]:


frame1 = pd.DataFrame({'Title1':Topic2, 'Text1': Text2, 'Cluster': clusters})
frame1.set_index('Cluster', inplace=True)


# In[53]:


from sklearn.feature_extraction.text import TfidfVectorizer

#define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords2,
                                 use_idf=True, tokenizer=tokenize_and_stem)
tfidf_matrix = tfidf_vectorizer.fit_transform(Text2) #fit the vectorizer to synopses
terms = tfidf_vectorizer.get_feature_names()


# In[54]:


for k in range (8,9):
    km = KMeans(n_clusters=k)
    km.fit(tfidf_matrix)
    clusters = km.labels_.tolist()
    print("k= %d" % k, end='')
    print() #add whitespace
    print("Top terms per cluster:")
    print()
#sort cluster centers by proximity to centroid
    order_centroids = km.cluster_centers_.argsort()[:, ::-1] 
    for i in range(k):
        print("Cluster %d words:" % i, end='')
        for ind in order_centroids[i, :10]: #replace 6 with n words per cluster
            print(' %s' % vocab_frame.loc[terms[ind].split(' ')].values.tolist()[0][0], end=',')
        print() #add whitespace
        print() #add whitespace
        print("Cluster %d titles:" % i, end='')
        for title in frame1.loc[i]['Title1']:
            print(' %s,' % title, end='')
        print() #add whitespace
        print() #add whitespace
print()
print()


# In[20]:


ETopic1= pd.read_excel('Resultnew.xlsx',sheetname="new 3")


# In[21]:


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


# In[22]:


SETopic1=tidy_split(ETopic1, 'Topic',',')


# In[23]:


SETopic1


# In[ ]:


### For DAX30 INDEX


# In[24]:


SETopic1['Topic'] = SETopic1['Topic'].str.strip()
Q=(SETopic1.merge(a[["Topic","log_ret1"]], 
           how="right", 
           right_on=["Topic"], 
           left_on=["Topic"]))
Q1=Q.drop_duplicates(subset=['Topic', 'log_ret1'], keep='first')
Q1


# In[25]:


Join=(SETopic1.merge(a[["Topic", "log_ret1","Date"]], 
           how="right", 
           right_on=["Topic"], 
           left_on=["Topic"]))
Join1=Join.drop_duplicates(subset=['Topic', 'log_ret1'], keep='first')
Join1


# In[26]:


Join1.to_excel('Cluster.xlsx', sheet_name='Sheet1', engine='xlsxwriter')


# In[21]:


Join2=(Join1.merge(new[["Topic","Positive tone","Negative tone","P-N","Date"]], 
           how="right", 
           right_on=["Topic","Date"], 
           left_on=["Topic","Date"]))
Join3=Join2.drop_duplicates(subset=['Topic', 'log_ret1'], keep='first')
Join3


# In[101]:


my_cols_list=['Cluster1','Cluster2','Cluster3','Cluster4','Cluster5','Cluster6','Cluster7','Cluster8']


# In[44]:


my_cols_list2=['A','B','C','D','E','F','G','H']


# In[41]:


F = pd.get_dummies(Join3, columns=['Cluster'])
F


# In[42]:


F.rename(columns=dict(zip(F.columns[6:], my_cols_list2)),inplace=True)
F


# In[44]:


from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf


# In[41]:


robust_ols1 = smf.ols(formula='log_ret1 ~ B', data=F).fit(cov_type='HAC', cov_kwds={'maxlags':5})
robust_ols1.summary()


# In[45]:


robust_ols = smf.ols(formula='log_ret1 ~ A', data=F).fit().get_robustcov_results(cov_type='HAC', maxlags=5)
robust_ols.summary()


# In[621]:


robust_ols = smf.ols(formula='log_ret1 ~ B', data=F).fit().get_robustcov_results(cov_type='HAC', maxlags=5)
robust_ols.summary()


# In[622]:


robust_ols = smf.ols(formula='log_ret1 ~ C', data=F).fit().get_robustcov_results(cov_type='HAC', maxlags=5)
robust_ols.summary()


# In[109]:


robust_ols = smf.ols(formula='log_ret1 ~ D', data=F).fit().get_robustcov_results(cov_type='HAC', maxlags=5)
robust_ols.summary()


# In[110]:


robust_ols = smf.ols(formula='log_ret1 ~ E', data=F).fit().get_robustcov_results(cov_type='HAC', maxlags=5)
robust_ols.summary()


# In[623]:


robust_ols = smf.ols(formula='log_ret1 ~ F', data=F).fit().get_robustcov_results(cov_type='HAC', maxlags=5)
robust_ols.summary()


# In[624]:


robust_ols = smf.ols(formula='log_ret1 ~ G', data=F).fit().get_robustcov_results(cov_type='HAC', maxlags=5)
robust_ols.summary()


# In[625]:


robust_ols = smf.ols(formula='log_ret1 ~ H', data=F).fit().get_robustcov_results(cov_type='HAC', maxlags=5)
robust_ols.summary()


# In[47]:


d = dict(tuple(Join3.groupby('Cluster')))


# In[46]:


Join3


# In[ ]:


# Sentiment Analysis


# In[48]:


Cluster1=d['A']
Cluster2=d['B']
Cluster3=d['C']
Cluster4=d['D']
Cluster5=d['E']
Cluster6=d['F']
Cluster7=d['G']
Cluster8=d['H']


# In[49]:


#Positive tone
def function(df):
    df['decile'] = pd.qcut(df['Positive tone'], 10, labels=False,duplicates='drop')
    df['quintile'] = pd.qcut(df['Positive tone'], 5, labels=False,duplicates='drop')
    df1=df[df['Positive tone'] >=df['Positive tone'].quantile(.90)]
    Mean1=df1["log_ret1"].mean()
    df2=df[df['Positive tone'] >= df['Positive tone'].quantile(.00)][df['Positive tone'] <= df['Positive tone'].quantile(.10)]
    Mean2=df2["log_ret1"].mean()
    Test=scipy.stats.ttest_ind(df1["log_ret1"],df2["log_ret1"], equal_var=False)
    print("For positive tone:")
    print("Mean-top quantile:",Mean1)
    print("Mean-bottom quantile:", Mean2)
    print("Test result:",Test)


# In[61]:


Rtotal=function(Join3)


# In[65]:


R1=function(Cluster1)


# In[66]:


R2=function(Cluster2)


# In[67]:


R3=function(Cluster3)


# In[170]:


R4=function(Cluster4)


# In[68]:


R5=function(Cluster5)


# In[69]:


R6=function(Cluster6)


# In[70]:


R7=function(Cluster7)


# In[71]:


R8=function(Cluster8)


# In[62]:


# Negative tone
def function2(df):
    df['decile'] = pd.qcut(df['Negative tone'], 10, labels=False,duplicates='drop')
    df['quintile'] = pd.qcut(df['Negative tone'], 5, labels=False,duplicates='drop')
    df1=df[df['Negative tone'] >=df['Negative tone'].quantile(.90)]
    Mean1=df1["log_ret1"].mean()
    df2=df[df['Negative tone'] >= df['Negative tone'].quantile(.00)][df['Negative tone'] <= df['Negative tone'].quantile(.10)]
    Mean2=df2["log_ret1"].mean()
    Test=scipy.stats.ttest_ind(df1["log_ret1"],df2["log_ret1"], equal_var=False)
    print("For negative tone:")
    print("Mean-top quantile:",Mean1)
    print("Mean-bottom quantile:", Mean2)
    print("Test result:",Test)


# In[60]:


Ntotal=function2(Join3)


# In[72]:


N1=function2(Cluster1)


# In[147]:


N2=function2(Cluster2)


# In[73]:


N3=function2(Cluster3)


# In[150]:


N4=function2(Cluster4)


# In[151]:


N5=function2(Cluster5)


# In[152]:


N6=function2(Cluster6)


# In[154]:


N7=function2(Cluster7)


# In[155]:


N8=function2(Cluster8)


# In[63]:


# Optimism tone
def function3(df):
    df['decile'] = pd.qcut(df['P-N'], 10, labels=False,duplicates='drop')
    df['quintile'] = pd.qcut(df['P-N'], 5, labels=False,duplicates='drop')
    df1=df[df['P-N'] >=df['P-N'].quantile(.90)]
    Mean1=df1["log_ret1"].mean()
    df2=df[df['P-N'] >= df['N-P'].quantile(.00)][df['P-N'] <= df['P-N'].quantile(.10)]
    Mean2=df2["log_ret1"].mean()
    Test=scipy.stats.ttest_ind(df1["log_ret1"],df2["log_ret1"], equal_var=False)
    print("For P-N tone:")
    print("Mean-top quantile:",Mean1)
    print("Mean-bottom quantile:", Mean2)
    print("Test result:",Test)


# In[62]:


Ktotal=function3(Join3)


# In[83]:


K1=function3(Cluster1)


# In[84]:


K2=function3(Cluster2)


# In[85]:


K3=function3(Cluster3)


# In[86]:


K4=function3(Cluster4)


# In[87]:


K5=function3(Cluster5)


# In[88]:


K6=function3(Cluster6)


# In[89]:


K7=function3(Cluster7)


# In[90]:


K8=function3(Cluster8)


# In[52]:


#Time plot for Group A
Join4=Join3.groupby(['Date', 'Cluster'], sort=True).agg(
                     {'Topic': ', '.join, 'log_ret1': 'first'})
Join4.to_excel('Join4.xlsx', sheet_name='Sheet1', engine='xlsxwriter')
Join41= pd.read_excel('Join4.xlsx',sheetname="Sheet1")
Join41=Join41.loc[Join41['Cluster'] == "A"]
Join5 = Join41.sort_values('log_ret1',ascending = False).groupby('Date').head(500)
Join5


# In[75]:


Q2 = Join3.filter(['Date','Topic','log_ret1','Cluster'], axis=1)


# In[54]:


import plotly
plotly.tools.set_credentials_file(username='makeawonder51', api_key='VbUVP3DKQ47zZc15wrZr')


# In[55]:


import plotly.plotly as py


# In[67]:


#Calculate volatility for DAX30 
Join3['Volatility'] = Join3['log_ret1']**2
Join3 


# In[68]:


Join3_volatility=Join3.groupby(['Cluster'])['Volatility'].mean().to_frame()
Join3_volatility_sorted     = Join3_volatility.sort_values( by = ['Volatility'], ascending = False)
Join3_volatility_sorted


# In[33]:


#Bank data
# Read-in bank data
Bank                 = pd.read_table(r'F:\Project\es_banks_201012', sep = ';' )
Bank['pct_change'] = Bank['lastprice'].pct_change()
Bank['log_ret_Bank'] = np.log(Bank['lastprice']) - np.log(Bank['lastprice'].shift(1))
Bank.loc[:,'log_ret_Bank'] *= 100
Bank['loctimestamp'] = Bank['loctimestamp'].str.strip()
Bank.to_excel('Bank.xlsx', sheet_name='Sheet1', engine='xlsxwriter')


# In[45]:


Join3['Date1']=Join3['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
Q4=(Join3.merge(Bank[["loctimestamp","log_ret_Bank"]], 
           how="right", 
           right_on=["loctimestamp"], 
           left_on=["Date1"]))
Q5 = Q4.dropna()
Q5sort     = Q5.sort_values( by = ['Date1'], ascending = True )
Q5sort


# In[46]:


Join_Bank=(SETopic1.merge(Q5sort[["Topic", "log_ret_Bank","Date"]], 
           how="right", 
           right_on=["Topic"], 
           left_on=["Topic"]))
Join_Bank1=Join_Bank.drop_duplicates(subset=['Topic', "log_ret_Bank"], keep='first')
Join_Bank1


# In[47]:


Join_Bank2=(Join_Bank1.merge(new[["Topic","Positive tone","Negative tone","P-N","Date"]], 
           how="right", 
           right_on=["Topic","Date"], 
           left_on=["Topic","Date"]))
Join_Bank3=Join_Bank2.drop_duplicates(subset=['Topic', "log_ret_Bank","Date"], keep='first')
 


# In[48]:


Join_Bank3


# In[49]:


F1 = pd.get_dummies(Join_Bank3, columns=['Cluster'])
F1.rename(columns=dict(zip(F1.columns[6:], my_cols_list2)),inplace=True)
F1


# In[72]:


robust_ols = smf.ols(formula='log_ret_Bank ~ A', data=F1).fit().get_robustcov_results(cov_type='HAC', maxlags=5)
robust_ols.summary()


# In[459]:


robust_ols = smf.ols(formula='log_ret_Bank ~ B', data=F1).fit().get_robustcov_results(cov_type='HAC', maxlags=5)
robust_ols.summary()


# In[427]:


robust_ols = smf.ols(formula='log_ret_Bank ~ C', data=F1).fit().get_robustcov_results(cov_type='HAC', maxlags=5)
robust_ols.summary()


# In[63]:


robust_ols = smf.ols(formula='log_ret_Bank ~ D', data=F1).fit().get_robustcov_results(cov_type='HAC', maxlags=5)
robust_ols.summary()


# In[429]:


robust_ols = smf.ols(formula='log_ret_Bank ~ E', data=F1).fit().get_robustcov_results(cov_type='HAC', maxlags=5)
robust_ols.summary()


# In[430]:


robust_ols = smf.ols(formula='log_ret_Bank ~ F', data=F1).fit().get_robustcov_results(cov_type='HAC', maxlags=5)
robust_ols.summary()


# In[458]:


robust_ols = smf.ols(formula='log_ret_Bank ~ G', data=F1).fit().get_robustcov_results(cov_type='HAC', maxlags=5)
robust_ols.summary()


# In[435]:


robust_ols = smf.ols(formula='log_ret_Bank ~ H', data=F1).fit().get_robustcov_results(cov_type='HAC', maxlags=5)
robust_ols.summary()


# In[66]:


def functionb(df):
    df['decile'] = pd.qcut(df['Positive tone'], 10, labels=False,duplicates='drop')
    df['quintile'] = pd.qcut(df['Positive tone'], 5, labels=False,duplicates='drop')
    df1=df[df['Positive tone'] >=df['Positive tone'].quantile(.85)]
    Mean1=df1["log_ret_Bank"].mean()
    df2=df[df['Positive tone'] >= df['Positive tone'].quantile(.00)][df['Positive tone'] <= df['Positive tone'].quantile(.15)]
    Mean2=df2["log_ret_Bank"].mean()
    Test=scipy.stats.ttest_ind(df1["log_ret_Bank"],df2["log_ret_Bank"], equal_var=False)
    print("For positive tone:")
    print("Mean-top decile:",Mean1)
    print("Mean-bottom declie:", Mean2)
    print("Test result:",Test)


# In[175]:


Rtotal=functionb(Join_Bank3)


# In[156]:


R1=functionb(Cluster1b)


# In[157]:


R2=functionb(Cluster2b)


# In[158]:


R3=functionb(Cluster3b)


# In[159]:


R4=functionb(Cluster4b)


# In[160]:


R5=functionb(Cluster5b)


# In[161]:


R6=functionb(Cluster6b)


# In[162]:


R7=functionb(Cluster7b)


# In[154]:


R8=functionb(Cluster8b)


# In[67]:


def function2b(df):
    df['decile'] = pd.qcut(df['Negative tone'], 10, labels=False,duplicates='drop')
    df['quintile'] = pd.qcut(df['Negative tone'], 5, labels=False,duplicates='drop')
    df1=df[df['Negative tone'] >=df['Negative tone'].quantile(.90)]
    Mean1=df1["log_ret_Bank"].mean()
    df2=df[df['Negative tone'] >= df['Negative tone'].quantile(.00)][df['Negative tone'] <= df['Negative tone'].quantile(.10)]
    Mean2=df2["log_ret_Bank"].mean()
    Test=scipy.stats.ttest_ind(df1["log_ret_Bank"],df2["log_ret_Bank"], equal_var=False)
    print("For negative tone:")
    print("Mean-top decile:",Mean1)
    print("Mean-bottom declie:", Mean2)
    print("Test result:",Test)


# In[173]:


Ntotal=function2b(Join_Bank3)


# In[135]:


N1=function2b(Cluster1b)


# In[136]:


N2=function2b(Cluster2b)


# In[137]:


N3=function2b(Cluster3b)


# In[138]:


N4=function2b(Cluster4b)


# In[139]:


N5=function2b(Cluster5b)


# In[141]:


N6=function2b(Cluster6b)


# In[142]:


N7=function2b(Cluster7b)


# In[143]:


N8=function2b(Cluster8b)


# In[163]:


def function3b(df):
    df['decile'] = pd.qcut(df['P-N'], 10, labels=False,duplicates='drop')
    df['quintile'] = pd.qcut(df['P-N'], 5, labels=False,duplicates='drop')
    df1=df[df['N-P'] >=df['P-N'].quantile(.95)]
    Mean1=df1["log_ret_Bank"].mean()
    df2=df[df['P-N'] >= df['P-N'].quantile(.00)][df['P-N'] <= df['P-N'].quantile(.05)]
    Mean2=df2["log_ret_Bank"].mean()
    Test=scipy.stats.ttest_ind(df1["log_ret_Bank"],df2["log_ret_Bank"], equal_var=False)
    print("For negative tone:")
    print("Mean-top decile:",Mean1)
    print("Mean-bottom declie:", Mean2)
    print("Test result:",Test)


# In[172]:


K1total=function3b(Join_Bank3)


# In[164]:


K1=function3b(Cluster1b)


# In[165]:


K2=function3b(Cluster2b)


# In[166]:


K3=function3b(Cluster3b)


# In[167]:


K4=function3b(Cluster4b)


# In[168]:


K5=function3b(Cluster5b)


# In[169]:


K6=function3b(Cluster6b)


# In[170]:


K7=function3b(Cluster7b)


# In[171]:


K8=function3b(Cluster8b)


# In[ ]:


Date=Join41.Date
Date=Date.tolist()
Return=Join41.log_ret1
Return=Return.tolist()
trace = go.Scatter(
    x = Date,
    y = Return,
    mode = 'lines')
data = [trace]
py.iplot(data, filename='Topic plot')


# In[ ]:


# Group data for only group A
Join4=Join3.groupby(['Date', 'Cluster'], sort=True).agg(
                     {'Topic': ', '.join, 'log_ret1': 'first'})
Join4.to_excel('Join4.xlsx', sheet_name='Sheet1', engine='xlsxwriter')
Join41= pd.read_excel('Join4.xlsx',sheetname="Sheet1")
Join41=Join41.loc[Join41['Cluster'] == "A"]
Join5 = Join41.sort_values('log_ret1',ascending = False).groupby('Date').head(500)
Join5


# In[57]:


Join_Bank4=Join_Bank3.groupby(['Date', 'Cluster'], sort=True).agg(
                     {'Topic': ', '.join, 'log_ret_Bank': 'first'})
Join_Bank4.to_excel('Join_Bank4.xlsx', sheet_name='Sheet1', engine='xlsxwriter')
Join_Bank41= pd.read_excel('Join_Bank4.xlsx',sheetname="Sheet1")
Join_Bank41=Join_Bank41.loc[Join_Bank41['Cluster'] == "A"]
Join_Bank5 = Join_Bank41.sort_values('log_ret_Bank',ascending = False).groupby('Date').head(500)
Join_Bank5


# In[59]:


Join_Bank5


# In[545]:



F1 = pd.get_dummies(Q6, columns=['Cluster'])
Q7.replace(np.nan,"0")
Q8=pd.crosstab(Q7.Date1,Q7.Cluster)
Q8
Q10=Q9.drop_duplicates(['Date1'], keep='first')
A_bank = Q10[Q10['A'] != 0]
A_bank['Date1']           = pd.to_datetime( A_bank['Date1'], format = '%Y-%m-%d %H:%M:%S' )
A_bank
y_bank=A_bank.plot(x='Date1', y='log_ret_Bank')


# In[69]:


Join_Bank3['Volatility'] = Join_Bank3['log_ret_Bank']**2
Join_Bank3


# In[70]:


Join_Bank3_volatility=Join_Bank3.groupby(['Cluster'])['Volatility'].mean().to_frame()
Join_Bank3_volatility_sorted     = Join_Bank3_volatility.sort_values( by = ['Volatility'], ascending = False)
Join_Bank3_volatility_sorted


# In[ ]:


## DAX FUTURE 


# In[538]:



# Read-in stock data
Dax_2010                   = pd.read_table(r'F:\dataej\dax_future_trades_2010', sep = ';' )
Dax_2010                   =Dax_2010 .rename( columns = {'price' : 'Dax30_fut', 'loctimestamp' : 'Date'} )
# Delete column
del Dax_2010['instrumentid']
# Format date and set index
Dax_2010['Date']           = pd.to_datetime( Dax_2010['Date'], format = '%Y-%m-%d %H:%M:%S' )
Dax_2010                 = Dax_2010.set_index( 'Date' )
# Show first observations
Dax_2010.head(2)


# In[539]:


# Read-in stock data
Dax_2011                   = pd.read_table(r'F:\dataej\dax_future_trades_2011', sep = ';' )
Dax_2011                  =Dax_2011.rename( columns = {'price' : 'Dax30_fut', 'loctimestamp' : 'Date'} )
# Delete column
del Dax_2011['instrumentid']
# Format date and set index
Dax_2011['Date']           = pd.to_datetime( Dax_2011['Date'], format = '%Y-%m-%d %H:%M:%S' )
Dax_2011                 = Dax_2011.set_index( 'Date' )
# Show first observations
Dax_2011.head(2)


# In[540]:


# Read-in stock data
Dax_2012                   = pd.read_table(r'F:\dataej\dax_future_trades_2012', sep = ';' )
Dax_2012                  =Dax_2012 .rename( columns = {'price' : 'Dax30_fut', 'loctimestamp' : 'Date'} )
# Delete column
del Dax_2012['instrumentid']
# Format date and set index
Dax_2012['Date']           = pd.to_datetime( Dax_2012['Date'], format = '%Y-%m-%d %H:%M:%S' )
Dax_2012                 = Dax_2012.set_index( 'Date' )
# Show first observations
Dax_2012.head(2)


# In[541]:


DAX30_all = Dax_2010.append(Dax_2011)
DAX30_all = DAX30_all.append(Dax_2012)


# In[705]:


DAX30_all


# In[706]:


# First, based on trading time
DAX30_all_sub = DAX30_all.between_time( '17:30:00', '22:15:00' )


# In[707]:


# Sort data frame
DAX30_all_sub['Date only'] = DAX30_all_sub.index.date
DAX30_all_sub['DateTime']  = DAX30_all_sub.index
DAX30_all_sub_sorted       = DAX30_all_sub.sort_values( by = ['DateTime', 'daystomaturity'], ascending = True )


# In[708]:


DAX30_all_sub_sorted 


# In[709]:


# Select shortest maturity
#
DAX30_all_sub_shortest = pd.DataFrame()

for date in DAX30_all_sub_sorted['Date only'].unique():
    
    print( date )
    
    DAX30_all_sub_date          = DAX30_all_sub_sorted[DAX30_all_sub_sorted['Date only'] == date].copy()
    
    shortest_maturity           = DAX30_all_sub_date.iloc[0]['daystomaturity']
    
    DAX30_all_sub_date_shortest = DAX30_all_sub_date[DAX30_all_sub_date['daystomaturity'] == shortest_maturity].copy()
    
    DAX30_all_sub_shortest      = DAX30_all_sub_shortest.append( DAX30_all_sub_date_shortest )

    
DAX30_all_sub_shortest.head(2)


# In[710]:


del DAX30_all_sub_shortest['daystomaturity']
del DAX30_all_sub_shortest['volume']
del DAX30_all_sub_shortest['underlyingprice']
del DAX30_all_sub_shortest['Date only']
del DAX30_all_sub_shortest['DateTime']


# In[711]:


del DAX30_all_sub_shortest['expirationdate']


# In[712]:


DAX30_all_sub_shortest


# In[713]:


# Final sort
DAX30_all_sub_shortest['Date']       = DAX30_all_sub_shortest.index
DAX30_all_sub_shortest_sorted        = DAX30_all_sub_shortest.sort_values( by = 'Date' )
DAX30_all_sub_shortest_sorted.head()


# In[714]:


import pickle
pickle.dump( DAX30_all_sub_shortest_sorted, open( r'F:\dataej\DAX30_all_sub.p', 'wb' ) )


# In[74]:


import pickle


# In[547]:


import pickle
DAX30_all_sub_shortest_sorted = pickle.load( open(r'F:\dataej\DAX30_all_sub.p', 'rb' ) )
DAX30_all_sub_shortest_sorted.head()


# In[664]:


new1=new
new1['Date']           = pd.to_datetime( new1['Date'], format = '%Y-%m-%d %H:%M:%S' )
new1                   = new1.set_index( 'Date' )
new1['DateTime']  = new1.index
new1.insert(12, 'Start', '20:00:00')
new1


# In[636]:


new1['Startdatetime'] = new1.DateTime + pd.Timedelta('20 hours')


# In[647]:


def first(text): 
        first = ' '.join(re.split(r'\n', text)[:1])
        return first


# In[648]:


def last(text):
   return ([i for i in text.split('\n') if i != ''][-1])


# In[665]:


new1['English'] = new1['English'].str.strip()
new1['First sentence'] = new1['English'].apply(first)
new1['Last sentence'] = new1['English'].apply(last)


# In[666]:


new2 = new1.filter(['Date','Topic','English','Start','Startdatetime','First sentence','Last sentence'], axis=1)
new2


# In[637]:


Caption_caption= pd.read_excel('Topic and Caption1.xlsx',sheetname="Caption")


# In[667]:


subset=Caption_caption.filter(["Date",'Time start','Time end','EngText'])
subset               = subset.set_index( 'Date' )
subset1=subset.dropna()


# In[668]:


new2['Last sentence'] = new2['Last sentence'].str.strip()
new2['First sentence'] = new2['First sentence'].str.strip()
subset1  ['EngText']=subset1  ['EngText'].str.strip()


# In[670]:


merge_start=(new2.merge(subset1[["EngText","Time start"]], 
           how="right", 
           right_on=["EngText",'Date'], 
           left_on=["First sentence","Date"]))
merge_start=merge_start.dropna()


# In[671]:


merge_start_end=(merge_start.merge(subset1[["EngText","Time end"]], 
           how="right", 
           right_on=["EngText",'Date'], 
           left_on=["Last sentence","Date"]))
merge_start_end=merge_start_end.dropna()
merge_start_end
merge_start_end=merge_start_end.drop_duplicates(subset=['First sentence','Last sentence','Time start'],keep='first')


# In[673]:


merge_start_end.to_excel('merge_start_end.xlsx', sheet_name='Sheet1', engine='xlsxwriter')


# In[674]:


merge_start_end1= pd.read_excel('merge_start_end.xlsx',sheetname="Sheet1")


# In[698]:


mnew4=merge_start_end1


# In[699]:


mnew4['Start'] = pd.to_timedelta(mnew4['Start'])
mnew4['Time start'] = pd.to_timedelta(mnew4['Time start'])
mnew4['Time end'] = pd.to_timedelta(mnew4['Time end'])
mnew4['Topic Time Start']=mnew4['Start']+mnew4['Time start']
mnew4['Topic Time End']=mnew4['Start']+mnew4['Time end']
mnew4['DateTime']  = mnew4.index
mnew4['Topic Start datetime'] = mnew4.DateTime + mnew4['Topic Time Start']
mnew4['Topic End datetime'] = mnew4.DateTime + mnew4['Topic Time End']


# In[700]:


mnew4_sort     = mnew4.sort_values( by = ['Topic Start datetime'] )
mnew4_sort


# In[716]:


DAX30_all_sub_shortest


# In[717]:


mnew4_return=  pd.merge_asof( mnew4_sort, DAX30_all_sub_shortest, 
                                              left_on = 'Topic Start datetime', 
                                              right_on = 'Date', 
                                              direction = 'forward' )
mnew4_return


# In[702]:


mnew4_return.to_excel('mnew4_return.xlsx', sheet_name='Sheet1', engine='xlsxwriter')


# In[703]:


DAX30_all_sub_shortest.to_excel('DAX30_all_sub_shortest.xlsx', sheet_name='Sheet1', engine='xlsxwriter')


# In[718]:


mnew4_returns_sorted = mnew4_return.sort_values( by = 'Topic End datetime' )


# In[719]:


mnew4_return = pd.merge_asof( mnew4_returns_sorted, DAX30_all_sub_shortest, 
                                              left_on = 'Topic End datetime', 
                                              right_on = 'Date', 
                                              direction = 'forward', 
                                              suffixes = ['', '_end'] )


# In[720]:


mnew4_return


# In[721]:


mnew4_return.set_index('DateTime', inplace = True, drop = False )


# In[722]:


#Calculate returns
mnew4_return['Returns'] = np.log( mnew4_return['Dax30_fut_end'] / 
                                                  mnew4_return['Dax30_fut'] ) * 100


# In[723]:


mnew4_return


# In[724]:





# In[725]:



del mnew4_return['Date']
del mnew4_return['Date_end']


# In[729]:


del mnew4_return['EngText_x']
del mnew4_return['EngText_y']


# In[730]:


mnew4_return


# In[73]:


import pickle
mnew4_return = pickle.load( open(r'F:\dataej\mnew4_return.p', 'rb' ) )


# # 

# In[76]:


z=(mnew4_return.merge(Q2[["Topic","Cluster","Date"]], 
           how="right", 
           right_on=["Topic","Date"], 
           left_on=["Topic","DateTime"]))
z=z.dropna()


# In[86]:


subset1 =z.filter(["Date",'Topic','Returns','Cluster'])
subset1


# In[88]:


Subset2=(subset1.merge(new[["Topic","Positive tone","Negative tone","P-N","Date"]], 
           how="right", 
           right_on=["Topic","Date"], 
           left_on=["Topic","Date"]))
Subset2


# In[96]:


F_instra = pd.get_dummies(Subset2, columns=['Cluster'])
F_instra.rename(columns=dict(zip(F_instra.columns[6:], my_cols_list2)),inplace=True)
F_instra


# In[98]:


robust_ols = smf.ols(formula='Returns ~ A', data=F_instra).fit().get_robustcov_results(cov_type='HAC', maxlags=5)
robust_ols.summary()


# In[99]:


robust_ols = smf.ols(formula='Returns ~ B', data=F_instra).fit().get_robustcov_results(cov_type='HAC', maxlags=5)
robust_ols.summary()


# In[100]:


robust_ols = smf.ols(formula='Returns ~ C', data=F_instra).fit().get_robustcov_results(cov_type='HAC', maxlags=5)
robust_ols.summary()


# In[101]:


robust_ols = smf.ols(formula='Returns ~ D', data=F_instra).fit().get_robustcov_results(cov_type='HAC', maxlags=5)
robust_ols.summary()


# In[102]:


robust_ols = smf.ols(formula='Returns ~ E', data=F_instra).fit().get_robustcov_results(cov_type='HAC', maxlags=5)
robust_ols.summary()


# In[103]:


robust_ols = smf.ols(formula='Returns ~ F', data=F_instra).fit().get_robustcov_results(cov_type='HAC', maxlags=5)
robust_ols.summary()


# In[106]:


robust_ols = smf.ols(formula='Returns ~ G', data=F_instra).fit().get_robustcov_results(cov_type='HAC', maxlags=5)
robust_ols.summary()


# In[190]:


robust_ols = smf.ols(formula='Returns ~ H', data=F_instra).fit().get_robustcov_results(cov_type='HAC', maxlags=5)
robust_ols.summary()


# In[194]:


def function_instra(df):
    df['decile'] = pd.qcut(df['Positive tone'], 10, labels=False,duplicates='drop')
    df['quintile'] = pd.qcut(df['Positive tone'], 5, labels=False,duplicates='drop')
    df1=df[df['Positive tone'] >=df['Positive tone'].quantile(.90)]
    Mean1=df1["Returns"].mean()
    df2=df[df['Positive tone'] >= df['Positive tone'].quantile(.00)][df['Positive tone'] <= df['Positive tone'].quantile(.10)]
    Mean2=df2["Returns"].mean()
    Test=scipy.stats.ttest_ind(df1["Returns"],df2["Returns"], equal_var=False)
    print("For positive tone:")
    print("Mean-top quantile:",Mean1)
    print("Mean-bottom quantile:", Mean2)
    print("Test result:",Test)


# In[207]:


def function2_instra(df):
    df['decile'] = pd.qcut(df['Negative tone'], 10, labels=False,duplicates='drop')
    df['quintile'] = pd.qcut(df['Negative tone'], 5, labels=False,duplicates='drop')
    df1=df[df['Negative tone'] >=df['Negative tone'].quantile(.90)]
    Mean1=df1["Returns"].mean()
    df2=df[df['Negative tone'] >= df['Negative tone'].quantile(.00)][df['Negative tone'] <= df['Negative tone'].quantile(.10)]
    Mean2=df2["Returns"].mean()
    Test=scipy.stats.ttest_ind(df1["Returns"],df2["Returns"], equal_var=False)
    print("For negative tone:")
    print("Mean-top quantile:",Mean1)
    print("Mean-bottom quantile:", Mean2)
    print("Test result:",Test)
#


# In[220]:


def function3_instra(df):
    df['decile'] = pd.qcut(df['P-N'], 10, labels=False,duplicates='drop')
    df['quintile'] = pd.qcut(df['P-N'], 5, labels=False,duplicates='drop')
    df1=df[df['P-N'] >=df['P-N'].quantile(.91)]
    Mean1=df1["Returns"].mean()
    df2=df[df['P-N'] >= df['P-N'].quantile(.00)][df['P-N'] <= df['P-N'].quantile(.09)]
    Mean2=df2["Returns"].mean()
    Test=scipy.stats.ttest_ind(df1["Returns"],df2["Returns"], equal_var=False)
    print("For P-N tone:")
    print("Mean-top quantile:",Mean1)
    print("Mean-bottom quantile:", Mean2)
    print("Test result:",Test)


# In[ ]:


# SENTiMENT ANALYTICS


# In[192]:


d_id = dict(tuple(Subset2.groupby('Cluster')))
Cluster1_id=d_id['A']
Cluster2_id=d_id['B']
Cluster3_id=d_id['C']
Cluster4_id=d_id['D']
Cluster5_id=d_id['E']
Cluster6_id=d_id['F']
Cluster7_id=d_id['G']
Cluster8_id=d_id['H']


# In[197]:


Rtotal=function_instra(Subset2)


# In[198]:


R1=function_instra(Cluster1_id)


# In[199]:


R2=function_instra(Cluster2_id)


# In[200]:


R3=function_instra(Cluster3_id)


# In[201]:


R4=function_instra(Cluster4_id)


# In[202]:


R5=function_instra(Cluster5_id)


# In[203]:


R6=function_instra(Cluster6_id)


# In[204]:


R7=function_instra(Cluster7_id)


# In[205]:


R8=function_instra(Cluster8_id)


# In[209]:


Ntotal=function2_instra(Subset2)


# In[210]:


N1=function2_instra(Cluster1_id)


# In[211]:


N2=function2_instra(Cluster2_id)


# In[212]:


N3=function2_instra(Cluster3_id)


# In[213]:


N4=function2_instra(Cluster4_id)


# In[214]:


N5=function2_instra(Cluster5_id)


# In[215]:


N6=function2_instra(Cluster6_id)


# In[216]:


N7=function2_instra(Cluster7_id)


# In[217]:


N8=function2_instra(Cluster8_id)


# In[221]:


Ktotal=function3_instra(Subset2)


# In[222]:


K1=function3_instra(Cluster1_id)


# In[224]:


K2=function3_instra(Cluster2_id)


# In[230]:


K3=function3_instra(Cluster3_id)


# In[225]:


K4=function3_instra(Cluster4_id)


# In[226]:


K5=function3_instra(Cluster5_id)


# In[227]:


K6=function3_instra(Cluster6_id)


# In[228]:


K7=function3_instra(Cluster7_id)


# In[229]:


K8=function3_instra(Cluster8_id)


# In[77]:


#Calculate volatility
mnew4_return['Volatility'] = mnew4_return['Returns']**2
mnew4_return


# In[78]:


z1=z.groupby(['Cluster'])['Volatility'].mean().to_frame()
z1_sorted     = z1.sort_values( by = ['Volatility'], ascending = False)
z1_sorted


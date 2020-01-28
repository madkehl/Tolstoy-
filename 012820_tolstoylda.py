import pandas as pd
import nltk

dnev = pd.read_csv('/users/madke/documents/TOLSTOY/diaries_to_madeline.csv', index_col = 0)
dnev = dnev.reset_index()

from gensim.models.wrappers import LdaMallet
from gensim.corpora import Dictionary
mallet_path = '/mallet/mallet-2.0.8/bin/mallet'

formallet = dnev['entry']
import re 
cleaned = []
for i in formallet:
    i = str(i)
    q = i.lower()
    for z in punct:
        q = q.replace(z, '')
    q = q.replace('\n', '')
    q = q.replace('\xa0', '')
    z = q.split()
    z = [morph.parse(i)[0].normal_form for i in z if morph.parse(i)[0].tag.POS not in ['PRED', 'PREP', 'CONJ', 'INTJ', 'PRCL']]  
    n = [re.sub(r'\d+', '', i) for i in z]
    cleaned.append(z)

data_long2 = []
for n in cleaned:
    if len(n) > 1:
        holder = []
        for i in n:
            if len(i) > 2: 
                holder.append(i)
    data_long2.append(holder)


dictionary = Dictionary(data_long2)
corpus1 = [dictionary.doc2bow(doc) for doc in data_long2]
#prefix (str, optional)

model = LdaMallet(mallet_path, corpus=corpus1, num_topics=6, id2word=dictionary, prefix = "TOLSTOY")

count = 0
catsss = []
for n in corpus1:
    catsss.append(model[n])
    print(count)
    count += 1

topic_0 = []
topic_1 = []
topic_2 = []
topic_3 = []
topic_4 = []
topic_5 = []



for i in catsss:
    topic_0.append(i[0][1])
    topic_1.append(i[1][1])
    topic_2.append(i[2][1])
    topic_3.append(i[3][1])
    topic_4.append(i[4][1])
    topic_5.append(i[5][1])


df_dnevsents = pd.DataFrame({
    
    'id': dnev['id'],
    'date':dnev['date'],
    'topic_0': topic_0,
    'topic_1': topic_1,
    'topic_2': topic_2,
    'topic_3': topic_3,
    'topic_4': topic_4,
    'topic_5': topic_5,


    #'text': dnevsents
    
})

y = dnev.drop_duplicates(subset='entry', keep='first', inplace=False)
result = pd.merge(y[['id']], df_dnevsents, on = 'id', how = 'left')
result.to_csv('/users/madke/012820_tolstoylda6cleaned.csv')



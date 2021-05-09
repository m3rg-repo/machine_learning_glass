import sample_Panini as Panini
import pandas as pd
import os

"""
Revised_Glass_Database has following information:
"PII,Title_x,Abstract_x,Query,DOI,Date,Journal,Keywords,Images,Sections"
"""
DF=pd.read_csv('sample_dataset.csv')
DF = DF.head(10)
D=Panini.Document(DF)

base = os.path.join(os.getcwd(),'data')
#for caption cluster plot
D.generate_Captions(PII='PII',Query='Query',base=base)

D.caption_DF

#for labelling captions
D.label_Captions()

#plotting CCP
D.plot_caption_cluster()

#Running LDA on abstracts for top 15 topics
num_topics = 3
D.run_LDA('Abstract_x',num_topics)

#Creating LDA plot
D.generate_LDA_plot('Abstract_x',num_topics)

#Saving the tsne data and LDA topic label and scores for corresponding abstracts
D.generate_LDA_topic_df(save=True)

#Identifying elements present in abstracts
D.extract_chemicals('Abstract_x')


D.create_chemical_dictionary()

#Creating elemental maps
D.generate_elemental_maps()

#generating records and saving all information
D.generate_records(num_topics)

D.save()





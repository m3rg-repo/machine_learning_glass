import Panini
import pandas as pd

"""
Revised_Glass_Database has following information:
"PII,Title_x,Abstract_x,Query,DOI,Date,Journal,Keywords,Images,Sections"
"""
DF=pd.read_csv('Revised_Glass_Database.csv')

#sampling 1000 random papers from the database
D=Panini.Document(DF.sample(1000))


#for caption cluster plot
D.generate_Captions(PII='PII',Query='Query')

D.caption_DF

#for labelling captions
D.label_Captions()

#plotting CCP
D.plot_caption_cluster()

#Running LDA on abstracts for top 15 topics
D.run_LDA('Abstract_x',15)

#Creating LDA plot
D.generate_LDA_plot('Abstract_x',15)

#Saving the tsne data and LDA topic label and scores for corresponding abstracts
D.generate_LDA_topic_df(save=True)

#Identifying elements present in abstracts
D.extract_chemicals('Abstract_x')


D.create_chemical_dictionary()

#Creating elemental maps
D.generate_elemental_maps()

#generating records and saving all information
D.generate_records()

D.save()





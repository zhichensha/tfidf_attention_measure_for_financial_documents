## This python script will generate list of keywords according to a modified version of tfidf algorithm from Karthik's Attention Cycles https://www.karthiksastry.com/files/AC_Feb07_Final.pdf
## p27 3.2 Measuring (Macroeconomic) Attention - Methodology
## tf(w)_{it}: term frequency for a word w in the filing of firm i at time t, measure as the proportion of total English words
## df(w): document frequency of a given word w among all observed regulartory filings, measured as a proprotion of total documents that use the word at least once
## For each word that appears in the 10-Q and
## 10-K corpus, we calculate the tf-idf using term frequencies in each of the three textbooks
## and (inverse) document frequencies among regulatory filings.
import utils
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
import json

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
wd = '/user/zs2613/Documents/Work/NLP Textual Analysis with Pari'

###########################################################################################################
## STEP 1: Data Extracting and Pre-Processing
###########################################################################################################

## For guideline documents and 10Ks

## Pre-process iea and mankiw guideline doc
iea_path = 'NetZeroby2050-ARoadmapfortheGlobalEnergySector_CORR.pdf'
iea_text = utils.extract_text_from_pdf(iea_path, page_nums=None)
iea_doc = utils.preprocess(iea_text)

mankiw_path = 'n.-gregory-mankiw-macroeconomics-7th-edition-2009.pdf'
mankiw_text = utils.extract_text_from_pdf(mankiw_path, page_nums=None)
mankiw_doc = utils.preprocess(mankiw_text)

combined_documents = utils.combine_documents(oil_gas=True)


###########################################################################################################
## STEP 2: Extract Top Keywords
###########################################################################################################

## Approach 1: Automatic Extraction 

### Calcualting TFIDF of each company's yearly 10K with regards to the Guideline docudment (IEA, Mankiw)
mankiw_tfidf = utils.calculate_tfidf(mankiw_doc, combined_documents)
iea_tfidf = utils.calculate_tfidf(iea_doc, combined_documents)
iea_tfidf_df = pd.DataFrame(iea_tfidf, columns=['Word', 'Weight'])

### Getting Top 200 Keywords after remvoing those also appearing in Mankiw
## Step 1: Generate top200 keywords from IEA X 10K
iea_10K_topwords = [word for (word, value) in iea_tfidf]
iea_10K_top200words = [word for (word, value) in iea_tfidf[:200]]


## Step 2:Filter the keywords by removing keywords appearing in Mankiw
mankiw_10K_top200words = [word for (word, value) in mankiw_tfidf[:200]]

filtered_iea_10K_topwords = [word for word in iea_10K_topwords if word not in mankiw_10K_top200words]
filtered_iea_10K_top200words = filtered_iea_10K_topwords[:200]

## Min-Max normalization
# filtered_iea_tfidf_df = iea_tfidf_df.loc[iea_tfidf_df['Word'].isin(filtered_iea_10K_top200words)]
# filtered_iea_tfidf_df['Weight_normalized'] = (filtered_iea_tfidf_df['Weight'] - filtered_iea_tfidf_df['Weight'].min()) / (filtered_iea_tfidf_df['Weight'].max() - filtered_iea_tfidf_df['Weight'].min())
iea_tfidf_df['Weight_normalized'] = (iea_tfidf_df['Weight'] - iea_tfidf_df['Weight'].min()) / (iea_tfidf_df['Weight'].max() - iea_tfidf_df['Weight'].min())

###########################################################################################################
## Approach 2: Manual Keywords
###########################################################################################################
anti_climate_keywords = [
    'exploration', 'new oilfield', 'New coal mine'
]
long_term_keywords =[
    "Grid infrastructure",
    "Modernization of electricity network",
    "Refinery closure",
    "Renewables R&D",
    "Green R&D",
    "Renewables Acquisition",
    "Diversified energy company",
    "Batteries",
    "Electrolysers",
    "Electrolyzers",
    "Sustainable aviation fuel",
    "Managed phaseout",
    "Transition finance",
    "Electric vehicle",
    "Hydrogen",
    "Hydrogen infrastructure",
    "CCUS",
    "Biofuel",
    "Bioenergy"
]
keywords_lists = [anti_climate_keywords, long_term_keywords]
given_keywords = []
for word_list in keywords_lists:
    for word in word_list:
        given_keywords.append(utils.preprocess(word))
given_keywords = list(set(given_keywords))


###########################################################################################################
## Part 3: Results Gathering
###########################################################################################################

## Automatic
attention_dict_automatic = utils.create_attention_dict(filtered_iea_10K_topwords, combined_documents)
final_df_automatic = utils.reshape_result_panel(attention_dict_automatic, combined_documents)
final_df_automatic = pd.merge(final_df_automatic, iea_tfidf_df, how = 'left', on = 'Word')

## Manual
attention_dict_manual = utils.create_attention_dict(given_keywords, combined_documents)
final_df_manual = utils.reshape_result_panel(attention_dict_manual, combined_documents)

# Display
utils.display_results(final_df_automatic, wd, oil_gas=True, manual=False)

utils.display_results(final_df_manual, wd, oil_gas=True, manual=True)



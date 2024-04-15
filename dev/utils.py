## Package Importing
import fitz
import nltk
import inflect
# Download wordnet, a lexical database for the English language, to support word lemmatization
nltk.download('wordnet')
from concurrent.futures import ProcessPoolExecutor

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import zipfile


def extract_text_from_pdf(pdf_path, page_nums=None):
    """
    Extracts text from specific pages of a PDF using PyMuPDF (fitz).

    Parameters:
    - pdf_path (str): Path to the PDF file.
    - page_nums (list, optional): List of page numbers to extract. If None, extracts all pages. 

    Returns:
    - str: Extracted text.
    """
    text = ''
    # Open the PDF file
    with fitz.open(pdf_path) as doc:
        # If no specific pages are provided, extract from all pages.
        if page_nums is None:
            page_nums = range(len(doc))
            
        for page_num in page_nums:
            # Get the page object
            page = doc.load_page(page_num)
            # Extract text from the current page
            text += page.get_text()

    return text

def preprocess(text):
    """
    Process texts extracted: lower case, tokenizing, convert to singulars, keep only alphabetic and lemmatize

    Parameters:
    - text (str): Text to be processed

    Returns:
    - str: Processed text.
    """
    # Convert all text to lowercase and tokenize it (split it into individual words)
    tokens = nltk.word_tokenize(text.lower())
    
    #  # Convert plurals to singulars using inflect
    # p = inflect.engine()
    # tokens = [p.singular_noun(word) or word for word in tokens]  # singular_noun returns False if it can't convert the word
    
    # Keep only alphabetic words (removing numbers, punctuation, etc.)
    words = [word for word in tokens if word.isalpha()]
    
    # Lemmatize words, converting them to their base or root form (e.g., "running" -> "run")
    lemmatizer = nltk.WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word, pos="n") for word in words]
    words = [lemmatizer.lemmatize(word, pos="v") for word in words]
    words = [lemmatizer.lemmatize(word, pos="a") for word in words]
    words = [lemmatizer.lemmatize(word, pos="r") for word in words]
    
    return ' '.join(words)

def process_once(args):
    """
    Helper function for processing, initialized for multi-processing

    Parameters:
    - args: a 3-element tuple to identify the company

    Returns:
    - documents: a dictionary of processed 10Ks for one company
    """
    zipname, key, CIKs = args
    documents = {key: {} for key in CIKs}
    with zipfile.ZipFile("./SEC10K_data/" + zipname, 'r') as zip_ref:
        for name in zip_ref.namelist():
            if CIKs[key] in name and "10-K" in name and "10-K-A" not in name:
                with zip_ref.open(name) as file:
                    print(key, name)
                    content = file.read().decode('utf-8')  # Convert byte data to string
                    soup = BeautifulSoup(content, 'html.parser')  # Parse the content with BeautifulSoup
                    year = soup.find("filename").text[:4]
                    # save data by company and year in a nested dictionary
                    if year in documents[key]:
                        documents[key][year].append(preprocess(soup.get_text()))
                    else:
                        documents[key][year] = [preprocess(soup.get_text())]
    return documents


def combine_documents(oil_gas = False):
    """
    Multi-Process all documents for text processing, and combine them into 1 large dictionary

    Parameters:
    - None
    
    Returns:
    - combined_documents: dictionaries of processed 10Ks for all company
    """
    zip_files = ['10-X_C_2011-2015.zip', '10-X_C_2016-2020.zip', '10-X_C_2021.zip', '10-X_C_2022.zip']
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    tables = pd.read_html(url)
    sp500_table = tables[0]

    #####################################################################################################
    ## Keep only Oil and Gas industry
    if oil_gas == True:
        sp500_table = sp500_table.loc[sp500_table['GICS Sub-Industry'].str.lower().str.contains('oil')]

    sp500_table['CIK'] = sp500_table['CIK'].apply(lambda x: f"_{x}_")
    CIKs = dict(zip(sp500_table['Symbol'], sp500_table['CIK']))

    # Use ProcessPoolExecutor to parallelize the task
    with ProcessPoolExecutor() as executor:
        # Prepare arguments for each task
        args = [(zipname, key, CIKs) for zipname in zip_files for key in CIKs.keys()]

        # Submit tasks to the executor
        futures = [executor.submit(process_once, arg) for arg in args]

        # Collect results as they complete
        results = [future.result() for future in futures]

    # Combine results from all processes
    combined_documents = {}
    for result in results:
        for key in result:
            if key in combined_documents:
                for year in result[key]:
                    if year in combined_documents[key]:
                        combined_documents[key][year].extend(result[key][year])
                    else:
                        combined_documents[key][year] = result[key][year]
            else:
                combined_documents[key] = result[key]
    return combined_documents


def calculate_tfidf(guideline_doc, combined_documents):
    """
    calculate tfidf scores for each term (unigrams & bigrams) according to TF of the guideline, and IDF of all 10Ks

    Parameters:
    - guideline_doc: guideline document
    - combined_documents: dictionaries of all processed 10K
    
    Returns:
    - sorted_tfidf: a list of tuples of all unigrams/bigrams with non-zero tfidf, sorted by their tfidf
    """
    ## Counting starts:
    ## Part 1: TF
    # Calculate TF using the guideline document
    ## Utilizing sklearn's CountVectorizer to count the number of occurances of all words, including unigram/bigram
    vectorizer_tf = CountVectorizer(ngram_range=(1, 2), stop_words="english")
    # term count matrix to count guideline_doc
    tf_matrix = vectorizer_tf.fit_transform([guideline_doc])
    tf_array = tf_matrix.toarray().flatten()# /np.sum(tc_matrix.toarray().flatten())
    # tf_df = pd.DataFrame(tf_matrix.toarray(), columns=vectorizer_tf.get_feature_names_out())

    ## Part 2: IDF
    # Calculate IDF using the 10K documents for S&P 500
    vectorizer_idf = CountVectorizer(ngram_range=(1, 2), stop_words="english")
    documents_list = [s for inner_dict in combined_documents.values() for lst in inner_dict.values() for s in lst]
    idf_matrix = vectorizer_idf.fit_transform(documents_list)
    # Convert idf_matrix to a binary form (terms present in a document are marked as 1)
    idf_array = np.sum(idf_matrix.toarray() > 0, axis=0)
    idf_array = np.log((len(documents_list) + 1) / (1 + idf_array)) + 1 ## smoothing verion
    # idf_array = np.log((len(documents_list)) / (idf_array)) + 1 ## non-smoothing verion
    # idf_df = pd.DataFrame(idf_matrix.toarray(), columns=vectorizer_idf.get_feature_names_out())

    ## Part 3: Vocabs Selecting
    # documents_list = []
    # for company_ticker in combined_documents.keys():
    #     print(company_ticker)
    #     if len(combined_documents[company_ticker]) > 0:
    #         string = [' '.join(lst) for lst in combined_documents[company_ticker].values()][0]
    #         documents_list.append(string)
    all_documents_list = [' '.join(documents_list)]

    ## TF * IDF
    # vectorizer_tf_all = CountVectorizer(ngram_range=(1,2),  stop_words="english")
    # tf_matrix_all = vectorizer_tf_all.fit_transform(all_documents_list)
    # Mapping the words of guideline TF to the words of all documents' IDF
    tfidf_values = []
    for word in vectorizer_tf.get_feature_names_out():
        if word in vectorizer_idf.vocabulary_:
            try:
                tfidf = tf_array[vectorizer_tf.vocabulary_[word]] * idf_array[vectorizer_idf.vocabulary_[word]]
                tfidf_values.append((word, tfidf))
            except:
                continue

    # Sort by descending order of tfidf values
    sorted_tfidf = sorted(tfidf_values, key=lambda x: x[1], reverse=True)
    
    return sorted_tfidf



def get_attention_score(company_ticker, documents, top_words):
    """
     calculate attention score for each company according to the generated top words

    Parameters:
    - company_ticker: guideline document
    - documents: dictionaries of all processed 10K
    - top_words: top words according to tfidf
    - top_dicts: top words with scores according to tfidf
    
    Returns:
    - tfidf_df['attention_score']: a pd Series of attention scores, by year
    """
    document_list = [' '.join(lst) for lst in documents[company_ticker].values()]
    ## STEP 4: Calculate attention scores
    # Compute TF-IDF scores for each 10-K document based on the extracted keywords
    vectorizer_tf = CountVectorizer(vocabulary=top_words, ngram_range=(1,2))
    # vectorizer = TfidfVectorizer(vocabulary=topwords, ngram_range=(1,2))

    tfidf_matrix = vectorizer_tf.fit_transform(document_list).toarray()
    tfidf_df = pd.DataFrame(tfidf_matrix, columns=vectorizer_tf.get_feature_names_out(), index=documents[company_ticker].keys())    

    # tfidf_weights = pd.Series(dict(top_dicts))
    # tfidf_weights = tfidf_weights/tfidf_weights.sum()
    # tfidf_weights = tfidf_weights.reindex(tfidf_df.columns)

    # tfidf_df['attention_score'] = tfidf_df.dot(tfidf_weights)
    return(tfidf_df)



def get_attention_score_cond(company_ticker, documents, top_words):
    """
     calculate attention score for each company according to the generated top words, conditioning on being with 6 windows

    Parameters:
    - company_ticker: guideline document
    - documents: dictionaries of all processed 10K
    - top_words: top words according to tfidf
    - top_dicts: top words with scores according to tfidf
    
    Returns:
    - tfidf_df: a dataframe of attention scores, by year
    """
    document_list = [' '.join(lst) for lst in documents[company_ticker].values()]

    ## STEP 4: Calculate attention scores
    # Initialize CountVectorizer
    vectorizer_tf = CountVectorizer(vocabulary=top_words, ngram_range=(1, 2))
    # Transform documents
    raw_counts = vectorizer_tf.fit_transform(document_list).toarray()
    tfidf_df = pd.DataFrame(raw_counts, columns=vectorizer_tf.get_feature_names_out(), index=documents[company_ticker].keys())

    # Define neighbor words and window size
    neighbor_words = ["invest", "acquire", "build"]
    window_size = 6

    # Process each document
    for i, doc in enumerate(document_list):
        words = doc.split()
        for top_word in top_words:
            if top_word in words:
                # Find the positions of the top_word and neighbor words
                top_word_positions = [i for i, word in enumerate(words) if word == top_word]
                neighbor_word_positions = [i for i, word in enumerate(words) if word in neighbor_words]
                
                # Check for any neighbor word within the window of each occurrence of top_word
                for pos in top_word_positions:
                    if any(abs(pos - npos) <= window_size for npos in neighbor_word_positions):
                        continue
                    else:
                        # If no neighbor word is within the window, set the count of this top_word in this document to 0
                        tfidf_df.at[tfidf_df.index[i], top_word] = 0
                        break

    # # Compute attention scores
    # tfidf_weights = pd.Series(dict(top_dicts))
    # tfidf_weights = tfidf_weights / tfidf_weights.sum()
    # tfidf_weights = tfidf_weights.reindex(tfidf_df.columns)

    # tfidf_df['attention_score'] = tfidf_df.dot(tfidf_weights)
    return tfidf_df


# ### Create a panel dataset with statistics for top keywords for each document

def calculate_word_count(documents):
    """
     calculate word count for each documents

    Parameters:
    - documents: dictionaries of all processed 10K
    
    Returns:
    - wordcount_df: a dataframe containing wordcount for each document
    """
    # Initialize an empty DataFrame to hold the word counts
    wordcount_df = pd.DataFrame(columns=['Company', 'Year', 'TotalCount'])

    for company_ticker in documents.keys():
        for year, docs in documents[company_ticker].items():
            # Assuming each 'documents' is a list of words for that year
            # Concatenate all document lists (if there are multiple) and get the total word count
            word_count = sum(len(document) for document in docs)

            # Append the result to the DataFrame
            temp_df = pd.DataFrame({
                'Company': [company_ticker],
                'Year': [year],
                'TotalCount': [word_count]
            })
            wordcount_df = pd.concat([wordcount_df, temp_df], ignore_index=True)

    return wordcount_df



def create_attention_dict(top_words, documents, get_attention_score = get_attention_score):
    """
     Create a dictionary of attention

    Parameters:
    - documents: dictionaries of all processed 10K
    
    Returns:
    - wordcount_df: a dataframe containing wordcount for each document
    """
    attention_dict = {}
    print("Calculating attention scores for each company")
    for company_ticker in tqdm(documents.keys()):
        attention_dict[company_ticker] = get_attention_score(company_ticker, documents, top_words)
    return attention_dict



def reshape_result_panel(attention_dict, documents):
    # Assuming your dictionary is named 'data_dict'
    final_df = pd.DataFrame()
    print("Reshaping the attention dataset")
    for ticker, df in attention_dict.items():
        # Reset index if the years are set as index
        df = df.reset_index()

        # Melt the dataframe to long format
        melted_df = df.melt(id_vars=[df.columns[0]], var_name='Word', value_name='Count')

        # Rename the first column to 'Year' if it's not already named
        melted_df = melted_df.rename(columns={melted_df.columns[0]: 'Year'})

        # Add the 'Company' column
        melted_df['Company'] = ticker

        # Append to the final dataframe
        final_df = pd.concat([final_df,  melted_df], ignore_index=True)

    # Reorder columns to match your requirement: ['Company', 'Year', 'Word', 'Frequency']
    final_df = final_df[['Company', 'Year', 'Word', 'Count']]

    # Optionally, you might want to sort the dataframe based on your preference
    final_df = final_df.sort_values(by=['Company', 'Year', 'Word']).reset_index(drop=True)

    # Get Word count and merge with the df
    final_df = pd.merge(final_df, calculate_word_count(documents), on = ['Company', 'Year'], how = 'outer', indicator = True)
    
    return final_df



def display_results(df, wd, oil_gas=False, manual=False):

    # Create "Results" directory inside the given working directory if it doesn't exist
    results_dir = os.path.join(wd, "Results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Calculate Frequency
    df['Frequency'] = df['Count'] / df['TotalCount']
    
    # Determine industry and selection method based on function parameters
    if oil_gas == False:
        industry = "Full S&P 500"
        industry_short = "full"
    elif oil_gas == True:
        industry = "Oil and Gas from S&P 500"
        industry_short = "oil_gas"
    
    if manual == True:
        selection_method = "Manual"
        selection_method_short = "manual"
    elif manual == False:
        selection_method = "Auto-Generated"
        selection_method_short = "auto"

        ## Also calculate using the normalized weights
        df['Frequency_n'] = df['Frequency'] * df['Weight_normalized']
        df['Count_n'] = df['Count'] * df['Weight_normalized']

        agg_df = df.groupby(['Company', 'Year'])[['Count_n', 'Frequency_n']].sum().reset_index().groupby(['Year'])[['Count_n', 'Frequency_n']].mean()

        
        # Plot and save the first graph (Average Count)
        plt.figure()
        agg_df['Count_n'].plot(kind='line', title='Average Normalized Count of {} Keywords in 10K across All Firms in {}'.format(selection_method, industry))
        plt.ylabel('Word Count Normalized')
        count_fig_path = os.path.join(results_dir, 'average_normalized_count_{}_{}_plot.png'.format(selection_method_short, industry_short))
        plt.savefig(count_fig_path)
        plt.show()
        
        # Plot and save the second graph (Average Frequency)
        plt.figure()
        agg_df['Frequency_n'].plot(kind='line', title='Average Normalized Frequency of {} Keywords in 10K across All Firms in {}'.format(selection_method, industry))
        plt.ylabel('Frequency Normalized')
        freq_fig_path = os.path.join(results_dir, 'average_normlaized_frequency_{}_{}_plot.png'.format(selection_method_short, industry_short))
        plt.savefig(freq_fig_path)
        plt.show()
        
        # Export aggregated dataframe and panel df to the "Results" folder
        # agg_df_path = os.path.join(results_dir, 'aggregated_normlaized_tf_{}_{}_data.csv'.format(selection_method_short, industry_short))
        # agg_df.to_csv(agg_df_path, index = False)

        df_path = os.path.join(results_dir, 'panel_normlaized_tf_{}_{}_data.csv'.format(selection_method_short, industry_short))
        df.groupby(['Company', 'Year'])[['Count_n', 'Frequency_n']].sum().reset_index().to_csv(df_path,index = False)


    # Aggregate data
    agg_df = df.groupby(['Company', 'Year'])[['Count', 'Frequency']].sum().reset_index().groupby(['Year'])[['Count', 'Frequency']].mean()
    
    # Plot and save the first graph (Average Count)
    plt.figure()
    agg_df['Count'].plot(kind='line', title='Average Count of {} Keywords in 10K across All Firms in {}'.format(selection_method, industry))
    plt.ylabel('Word Count')
    count_fig_path = os.path.join(results_dir, 'average_count_{}_{}_plot.png'.format(selection_method_short, industry_short))
    plt.savefig(count_fig_path)
    plt.show()
    
    # Plot and save the second graph (Average Frequency)
    plt.figure()
    agg_df['Frequency'].plot(kind='line', title='Average Frequency of {} Keywords in 10K across All Firms in {}'.format(selection_method, industry))
    plt.ylabel('Frequency')
    freq_fig_path = os.path.join(results_dir, 'average_frequency_{}_{}_plot.png'.format(selection_method_short, industry_short))
    plt.savefig(freq_fig_path)
    plt.show()
    
    # Export aggregated dataframe and panel df to the "Results" folder
    # agg_df_path = os.path.join(results_dir, 'aggregated_tf_{}_{}_data.csv'.format(selection_method_short, industry_short))
    # agg_df.to_csv(agg_df_path, index = False)

    df_path = os.path.join(results_dir, 'panel_tf_{}_{}_data.csv'.format(selection_method_short, industry_short))
    df.groupby(['Company', 'Year'])[['Count', 'Frequency']].sum().reset_index().to_csv(df_path,index = False)
    
    return
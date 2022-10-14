from io import StringIO
import pandas as pd
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import torch
from transformers import BertTokenizer, BertModel, AutoModel, AutoTokenizer
import unicodedata
import re
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import streamlit as st

def predictToolClass():

    #result_button = st.button('Predict the task of the tool', key=2)
    test_abstract = string_data

    # Lowercasing the text
    test_abstract = test_abstract.lower()
    # removing links
    regex_link = r"\bhttp[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+\b"
    test_abstract = test_abstract.replace(regex_link, "")
    # removing numbers
    regex_nums = r"\b[0-9][0-9]*\b"
    test_abstract = test_abstract.replace(regex_nums, "")
    # removing special characters
    special_character = list("←=()[]/‘’|><\\∼+%$&×–−-·")
    for spec_char in special_character:
        test_abstract = test_abstract.replace(spec_char, '')
    # removing punctuation
    punctuation_signs = list("?:!.,;")
    for punct_sign in punctuation_signs:
        test_abstract = test_abstract.replace(punct_sign, '')
    # removing strings with length 1-2
    regex_short = r"\b\w{0,2}\b"
    test_abstract = test_abstract.replace(regex_short, "")

    # removing strings starting with numbers
    regex_short = r"\b[0-9][0-9]*\w\b"
    test_abstract = test_abstract.replace(regex_short, "")
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    wordnet_lemmatizer = WordNetLemmatizer()
    # Iterating through every word to lemmatize
    lemmatized_text_list = []
    lemmatized_list = []
    text_words = test_abstract.split(" ")
    # Iterate through every word to lemmatize
    for word in text_words:
        lemmatized_list.append(wordnet_lemmatizer.lemmatize(word, pos="v"))

    # Join the list
    lemmatized_text = " ".join(lemmatized_list)

    # Append to the list containing the texts
    lemmatized_text_list.append(lemmatized_text)
    df = pd.DataFrame(lemmatized_text_list, columns=["text"])
    df["text"] = df["text"].replace("'s", "")

    # removing possessive pronoun terminations
    # lemmatized_text_list = lemmatized_text_list.replace("'s", "")
    # removing english stop words
    # Downloading the stop words list
    nltk.download('stopwords')
    # Loading the stop words in english
    stop_words = list(stopwords.words('english'))
    # looping through all stop words
    for stop_word in stop_words:
        regex_stopword = r"\b" + stop_word + r"\b"
        df["text"] = df["text"].replace(regex_stopword, '')

    #st.write("### Pre-processed text")
    #st.write(df["text"][0])


    # preprocessing:entry to tokens, map each token to integers
    checkpoint = "dmis-lab/biobert-base-cased-v1.2"

    model = BertModel.from_pretrained(checkpoint, output_hidden_states=True, )
    tokenizer = BertTokenizer.from_pretrained(checkpoint)


    def bert_text_preparation(text, tokenizer):
        """Preparing the input for BERT

        Takes a string argument and performs
        pre-processing like adding special tokens,
        tokenization, tokens to ids, and tokens to
        segment ids. All tokens are mapped to seg-
        ment id = 1.

        Args:
            text (str): Text to be converted
            tokenizer (obj): Tokenizer object
                to convert text into BERT-re-
                adable tokens and ids

        Returns:
            list: List of BERT-readable tokens
            obj: Torch tensor with token ids
            obj: Torch tensor segment ids


        """
        marked_text = "[CLS] " + text + " [SEP]"
        tokenized_text = tokenizer.tokenize(marked_text)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1] * len(indexed_tokens)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        return tokenized_text, tokens_tensor, segments_tensors


    def get_bert_embeddings(tokens_tensor, segments_tensors, model):
        """Get embeddings from an embedding model

        Args:
            tokens_tensor (obj): Torch tensor size [n_tokens]
                with token ids for each token in text
            segments_tensors (obj): Torch tensor size [n_tokens]
                with segment ids for each token in text
            model (obj): Embedding model to generate embeddings
                from token and segment ids

        Returns:
            list: List of list of floats of size
                [n_tokens, n_embedding_dimensions]
                containing embeddings for each token

        """

        # Gradient calculation id disabled
        # Model is in inference mode
        with torch.no_grad():
            outputs = model(tokens_tensor, segments_tensors)
            # Removing the first hidden state
            # The first state is the input state
            hidden_states = outputs[2][1:]

        # Getting embeddings from the final BERT layer
        token_embeddings = hidden_states[-1]
        # Collapsing the tensor into 1-dimension
        token_embeddings = torch.squeeze(token_embeddings, dim=0)
        # Converting torchtensors to lists
        list_token_embeddings = [token_embed.tolist() for token_embed in token_embeddings]

        return list_token_embeddings


    re.compile('<title>(.*)</title>')
    df['text'][0] = unicodedata.normalize('NFKD', df['text'][0]).encode('ascii', 'ignore').decode("utf-8")
    df['text'][0] = re.sub(r'[^\w]', ' ', df['text'][0])
    df['text'][0] = df['text'][0].encode("ascii", "ignore")
    df['text'][0] = df['text'][0].decode()

    # Getting embeddings for the target
    # word in all given contexts
    embeddings = []
    all_embeddings = []
    sentence_embedding = np.empty(768, dtype=object)
    c = 0

    tokenized_text, tokens_tensor, segments_tensors = bert_text_preparation(df['text'][0], tokenizer)
    list_token_embeddings = get_bert_embeddings(tokens_tensor, segments_tensors, model)
    embeddings.append(list_token_embeddings)

    for l in range(len(embeddings)):
        for v in embeddings[l]:
            sentence_embedding = np.vstack((sentence_embedding, v))
        sentence_embedding = np.delete(sentence_embedding, obj=0, axis=0)
        sentence_embedding = (np.mean(sentence_embedding, axis=0)).tolist()
        all_embeddings.append(sentence_embedding)

    all_embeddings = np.array(all_embeddings)
    all_embeddings = pd.DataFrame(all_embeddings)

    all_embeddings.insert(loc=0, column='text', value=df['text'])

    X = all_embeddings.drop(["text"], axis=1)

    category_codes = {
        'Sequence alignment': 0,
        'Taxonomic classification': 1,
        'Virus detection': 2,
        'Virus identification': 3,
        'Mapping': 4,
        'Sequence assembly': 5,
        'RNA-seq quantification for abundance estimation': 6,
        'Sequence trimming': 7,
        'Sequencing quality control': 8,
        'Sequence annotation': 9,
        'SNP-Discovery': 10,
        'Visualization': 11,
        'Sequence assembly validation': 12
    }
    res = loaded_model.predict(X)

    res_proba = loaded_model.predict_proba(X)
    tasks_proba = pd.DataFrame(res_proba, columns=category_codes)
    tasks_probaT= tasks_proba.transpose()
    tasks_probaT.columns = ['Prediction probability']

    value = [i for i in category_codes if category_codes[i] == res][0]
    if result_button:
        st.write(f"The predicted main task of this tool is: **{value}**")
        st.write("The tasks were predicted with the following probabilities:")
        #st.write(tasks_proba)
        #st.write(tasks_probaT)
        st.bar_chart(tasks_probaT)

#importing the best model
model_path = "Trained_Models/abstracts only/BIOBERTabs-nsl/best_BIOBERT-nsl_lrc.pickle"

loaded_model = pickle.load(open(model_path, 'rb'))


st.write("## Infer the main task of a metagenomics tool from its textual description")
st.write('This app demonstrates the main functionality presented in the paper [NLP-based classification of software tools for metagenomics sequencing data analysis into EDAMsemantic annotation](https://github.com/kaoutarDaoudHiri/NLP-based-classification-of-metagenomics-tools).')

tab1, tab2 = st.tabs(["Upload description file", "Input description text"])

with tab1:
    uploaded_file = st.file_uploader("Upload the text file containing the tool description")
    st.write("")
    st.write("")
    st.write('**Example text files:** [CDKAM.txt](https://raw.githubusercontent.com/kaoutarDaoudHiri/NLP-based-classification-of-metagenomics-tools/main/examples/CDKAM.txt "download"), [GAVISUNK.txt](https://raw.githubusercontent.com/kaoutarDaoudHiri/NLP-based-classification-of-metagenomics-tools/main/examples/GAVISUNK.txt "download")')
    st.write('Right click to save the files and then upload to the app.')

    if uploaded_file:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        st.write("### Tool description")
        string_data = stringio.read()
        st.write(string_data)
        result_button = st.button('Predict the task of the tool', key=1)
        predictToolClass()

with tab2:
    description = st.text_area('Write the tool description')
    st.write("")
    st.write("")
    st.write('**Example tool description:**')
    st.write('"*GAVISUNK: Genome assembly validation via inter-SUNK distances in Oxford Nanopore reads.*')
    st.write('*Highly contiguous de novo genome assemblies are now feasible for large numbers of species and individuals. Methods are needed to validate assembly accuracy and detect misassemblies with orthologous sequencing data to allow for confident downstream analyses. We developed GAVISUNK, an open-source pipeline that detects misassemblies and produces a set of reliable regions genome-wide by assessing concordance of distances between unique k-mers in Pacific Biosciences high-fidelity (HiFi) assemblies and raw Oxford Nanopore Technologies reads.*"')
    if description:
        stringio = description

        string_data = stringio

        result_button = st.button('Predict the task of the tool', key=2)
        predictToolClass()





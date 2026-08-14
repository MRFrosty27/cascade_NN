import re,pandas as pd
import numpy as np

prefix = ('un','dis','re','pre','mis','anti','sub','inter','intra','im','il','in','ir')
suffix = ('ful','less','able','ible','er','or','ly','ness','ment','ion','ist')

def tokenize_string(string,vocabulary):
    string = string.lower()
    text_clean = re.findall(r"[a-z]+", string)
    token_array = []

    for word in text_clean:

        for pf in prefix:
            if pf and word.startswith(prefix):
                word = word[len(prefix):]
                token_array.append(prefix)
            
        for sf in suffix:
            if sf and word.endswith(suffix):
                word = word[:-len(suffix)]
                token_array.append(suffix)

        for sub_string in vocabulary.keys():
            if sub_string in word: token_array.append(word.replace(sub_string, "",1))

    return token_array

def dataset_tokinization(dataset_path,vocabulary):
    ds = pd.read_csv(dataset_path)
    ds = ds.str.lower()
    vocabulary = []

    for row in ds.itertuples():
        for col in ds.columns:
            text_clean = re.findall(r"[a-z]+", getattr(row, col))
        
        for word in text_clean:

            for pf in prefix:
                if pf and word.startswith(prefix):
                    word = word[len(prefix):]
            
            for sf in suffix:
                if sf and word.endswith(suffix):
                    word = word[:-len(suffix)]

            for sub_string in vocabulary:
                if sub_string in word: vocabulary.append(word.replace(sub_string, "",1))

            if word not in vocabulary: vocabulary[word] = len(vocabulary) + 1

    return vocabulary

def vectorize_vocabulary(vocabulary,num_attributes):
    vectorize_vocabulary = dict()

    for token in vocabulary: 
        # initialize each token's vector as a random matrix
        vectorize_vocabulary[token] = np.random.rand(num_attributes, num_attributes)

    return vectorize_vocabulary

def vectorize_token(token,dataset_path,vectorize_vocabulary):
    ds = pd.read_csv(dataset_path)
    ds = ds.str.lower()
    array_of_strings = []
    vectorize_token_array = []

    #get all strings from the dataset that contain the token and calculate the vector representation of the token based on its context in those strings
    for row in ds.itertuples():
        for col in ds.columns:
            text_clean = re.findall(r"[a-z]+", getattr(row, col))
            if token in text_clean:
                text_clean.remove(token)#remove the token from the string to avoid it being counted in the context
                array_of_strings.append(text_clean)

    #tokenize the strings in the array_of_strings
    array_of_strings = [tokenize_string(string, vectorize_vocabulary) for string in array_of_strings]

    # Calculate the vector representation of the token based on its context
    for token in array_of_strings:
        if token in vectorize_vocabulary.keys():
            vectorize_token_array.append(vectorize_vocabulary[token])
    return vectorize_token_array

def calculate_token_vector(target_token,vectorize_token_array,vectorize_vocabulary):
    if len(vectorize_token_array) == 0:
        raise ValueError("arg 1 is empty")

    elif not isinstance(vectorize_token_array, list) and not all(isinstance(i, np.ndarray) for i in vectorize_token_array):
        raise TypeError("arg 1 must be a list of numpy arrays")

    else:# calculate the mean for target token vector representation based on its context in the dataset
        mean_token_sentence_array = []

        for tokenized_sentence in vectorize_token_array:
            if len(tokenized_sentence) > 0:
                mean_token_sentence_array.append(np.mean(tokenized_sentence, axis=0))
            else:
                pass#drop the empty tokenized sentences from the array

        #adjust token vector representation to converge target token vector representation to the mean of the context vectors for each sentence in the dataset
        if mean_token_sentence_array:
            return np.mean(mean_token_sentence_array, axis=0)
        else:
            raise ValueError("No valid context vectors found")
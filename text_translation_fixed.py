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
    """
    Returns a list of dictionaries mapping token names to their vectors for each sentence context.
    This maintains the token-to-vector relationship needed for proper mean calculation.
    """
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
    # Now we maintain token names with their vectors as dictionaries
    for tokenized_sentence in array_of_strings:
        sentence_token_vectors = {}
        for token_name in tokenized_sentence:
            if token_name in vectorize_vocabulary.keys():
                sentence_token_vectors[token_name] = vectorize_vocabulary[token_name]
        if sentence_token_vectors:  # Only add non-empty sentences
            vectorize_token_array.append(sentence_token_vectors)

    return vectorize_token_array

def calculate_token_vector(target_token, vectorize_token_array, vectorize_vocabulary):
    """
    Now receives a list of dictionaries where each dict maps token names to their vectors.
    This allows proper tracking of which vectors belong to which tokens and sentences.
    """
    if len(vectorize_token_array) == 0:
        raise ValueError("arg 1 is empty")

    elif not isinstance(vectorize_token_array, list) and not all(isinstance(i, dict) for i in vectorize_token_array):
        raise TypeError("arg 1 must be a list of dictionaries mapping tokens to numpy arrays")

    else:
        # calculate the mean for target token vector representation based on its context in the dataset
        mean_token_sentence_array = []

        for sentence_token_vectors in vectorize_token_array:
            # Extract just the vectors from this sentence's token dictionary
            if len(sentence_token_vectors) > 0:
                vectors = list(sentence_token_vectors.values())
                # Stack vectors and calculate mean across all tokens in this sentence
                sentence_vectors = np.array(vectors)
                mean_token_sentence_array.append(np.mean(sentence_vectors, axis=0))

        # adjust token vector representation to converge target token vector representation 
        # to the mean of the context vectors for each sentence in the dataset
        if mean_token_sentence_array:
            # Calculate overall mean across all sentences
            overall_mean = np.mean(np.array(mean_token_sentence_array), axis=0)
            return overall_mean
        else:
            raise ValueError("No valid context vectors found")

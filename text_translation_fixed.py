import re,pandas as pd
import numpy as np

prefix = ('un','dis','re','pre','mis','anti','sub','inter','intra','im','il','in','ir')
suffix = ('ful','less','able','ible','er','or','ly','ness','ment','ion','ist')

def tokenize_string(string,vocabulary):
    """
    Tokenizes a string into an array of tokens based on a given vocabulary. The function first converts the input string to lowercase and then uses regular expressions to extract words from the string. 
    It then checks each word for known prefixes and suffixes, removing them and adding them to the token array. Finally, it checks if any known words from the vocabulary are present in the remaining word and adds them to the token array as well.
    """
    if isinstance(string,str) == False: raise TypeError('arg 1 must be str'))
    if isinstance(vocabulary,list) == False: raise TypeError('arg 2 must be list')
    string = string.lower()
    text_clean = re.findall(r"[a-z]+", string)
    token_array = []

    for word in text_clean:

        for pf in prefix:
            if pf and word.startswith(pf):
                word = word[len(pf):]
                token_array.append(pf)

        for sf in suffix:
            if sf and word.endswith(sf):
                word = word[:-len(sf)]
                token_array.append(sf)

        for known_word in vocabulary:
            if known_word == word: token_array.append(word.replace(known_word, "",1))

    return token_array

def dataset_tokinization(dataset_path,vocabulary=None):
    """
    Tokenizes a dataset from a CSV file into a vocabulary of unique tokens. The function reads the dataset using pandas, converts all text to lowercase, and iterates through each row and column to extract words. 
    It checks for known prefixes and suffixes, removing them and adding them to the vocabulary. It also checks for known words in the vocabulary and adds any remaining words to the vocabulary as well.
    """
    ds = pd.read_csv(dataset_path)
    ds = ds.str.lower()
    if vocabulary is None:
        vocabulary = []

    for row in ds.itertuples():
        for col in ds.columns:
            text_clean = re.findall(r"[a-z]+", getattr(row, col))

        for word in text_clean:

            for pf in prefix:
                if pf and word.startswith(pf):
                    word = word[len(pf):]

            for sf in suffix:
                if sf and word.endswith(sf):
                    word = word[:-len(sf)]

            for known_word in vocabulary:
                if known_word in word: vocabulary.append(word.replace(known_word, "",1))

            if word not in vocabulary: vocabulary.append(word)

    return vocabulary

def vectorize_vocabulary(vocabulary,num_attributes):
    #initializes a dictionary mapping each token in the vocabulary to a random vector of size (num_attributes, num_attributes). Each token's vector is a random matrix that can be used for further processing in a neural network or other machine learning models.
    vocabulary_vectorized = dict()

    for token in vocabulary: 
        # initialize each token's vector as a random matrix
        vocabulary_vectorized[token] = np.random.rand(num_attributes, num_attributes)

    return vocabulary_vectorized

def vectorize_token(token_name,vocabulary_vectorized,token_description):
    """
    Returns a dictionary mapping token names to their vectors by relating each dimension of the token's vector to another token.
    for example, if the token is "cat" and the vector has 3 dimensions, the first dimension might relate to "dog", the second to "pet", and the third to "animal". The values in the vector would represent how strongly "cat" is associated with each of these related tokens based on their co-occurrence in the dataset.
    if a token is a dimension itself, the value in that dimension would be 1, indicating a perfect association with itself. If a token is not present in the vocabulary_vectorized, the function returns None.
    token_description is used to provide context for the token's vectorization, such as its meaning or usage in the dataset. This can help in understanding the relationships between tokens and their associated vectors.
    """

    vec = vocabulary_vectorized[token_name]
    if isinstance(token_description, str) == False: raise TypeError('token_description must be a string')

    # Use the token description to modify the vector
    tokenized_description = tokenize_string(token_description, list(vocabulary_vectorized.keys()))
    vectorized_description = [vocabulary_vectorized[word] for word in tokenized_description if word in vocabulary_vectorized]

    #the mean value of the token description is used as the tokens vector
    # If the token description contains words that are not in the vocabulary, we can ignore them or handle them as needed. For now, we'll ignore them.
    vocabulary_vectorized[token_name] = np.mean(vectorized_description, axis=0)

    return None

def set_token_as_attribute(token_name,vocabulary_vectorized,nth_attribute):
    """
    Sets the token as an attribute in the vocabulary_vectorized by setting its vector for the token with a value of 1 in the dimension corresponding to itself and 0 in all other dimensions. This indicates that the token is perfectly associated with itself and not associated with any other tokens.
    If the token is already present in the vocabulary_vectorized, it updates its vector to reflect this association. If the token is not present, it adds it to the vocabulary_vectorized with the new vector.
    only used to manually initialize the vocabulary
    """
    vec = vocabulary_vectorized[token_name]
    vec[nth_attribute,:] = 1#make row for token_name all 1s to indicate perfect association with itself
    #future update
    #nth_attribute automatically set to next available attribute if not provided

def set_token_as_related(target_token_name,token_attribute_name,vocabulary_vectorized,nth_attribute):
    """
    Sets the token as related to another token in the vocabulary_vectorized by setting its vector for the related token with a value of 1 in the dimension corresponding to the related token and 0 in all other dimensions. This indicates that the token is perfectly associated with the related token and not associated with any other tokens.
    Used for manual tuning only
    """
    vocabulary_vectorized[target_token_name] = vocabulary_vectorized[token_attribute_name]
    # future updates
    #target token can be partially related
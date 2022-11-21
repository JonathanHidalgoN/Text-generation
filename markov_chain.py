import numpy as np 

class MarkovChain :
    """
    The Markov Chain class
    Attributes:
        N: Matrix of transition probabilities.
        P: Matrix of normalized transition probabilities.
        special_character: Character used to indicate the beggining or end of a word.
    """
    def __init__(self, dataset_splited = True):
        """
        Initialize the Markov Chain
        """
        self.special_character = "<S>"
        self.N = None
        self.P = None
        self.dataset_splited = dataset_splited

    @staticmethod
    def _split_dataset(dataset):
        """
        Split the dataset into a list of words
        Args:
            dataset: The dataset to split.
        Returns:
            A list of words.
        """
        return " ".join(dataset).split()

    def compute_probabilities(self, X):
        """
        Compute the probabilities of the Markov Chain
        Args:
            X: list of strings.
        Returns:
            N : Matrix of transition probabilities.
        """
        if not self.dataset_splited:
            X = self._split_dataset(X)
        characters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]
        characters =  characters + [self.special_character]    
        #Create a lookup table for the characters
        stoi = {c:i for i,c in enumerate(characters)}
        #Define a matrix of zeros to store frequencies of transitions
        N = np.zeros((len(characters), len(characters)), dtype=np.int32)
        for word in X:
            #Add the beggining character to the beggining of the word, and the ending character to the end of the word
            extended_word = [self.special_character] + list(word) + [self.special_character]
            #Here we iterate over the characters of the word, and we update the matrix of frequencies
            for ch1, ch2 in zip(extended_word, extended_word[1:]): 
                #Get the index of the character in the lookup table
                idx1 = stoi[ch1]
                idx2 = stoi[ch2]
                #Increment the frequency of the transition
                N[idx1, idx2] += 1
        self.N = N
        return N

    @staticmethod
    def _check_normalization(matrix):
        """
        Check if the matrix is normalized
        Args:
            matrix: Matrix of transition probabilities.
        Raises:
            ValueError: If the matrix is not normalized.
        """
        matrix_sum = matrix.sum(axis=1)
        status =  np.allclose(matrix_sum, 1.0)
        if not status:
            raise ValueError("The matrix is not normalized")
        

    def normalize_probabilities(self, N, check_normalization=True):
        """
        Normalize the probabilities of the Markov Chain
        Args:
            N: Matrix of transition probabilities.
        Returns:
            N : Matrix of transition probabilities.
        """
        #Compute the sum of the frequencies of the transitions
        P = N.astype(np.float32)
        P_sum = P.sum(axis=1, keepdims=True)
        #Keep dims is used to keep the dimensions of the array, so braodcasting can be used
        #Divide each row by the sum of the frequencies
        P /= P_sum
        if check_normalization:
            self._check_normalization(P)
        self.P = P
        return P

if __name__ == "__main__":
    from utils.read_text import read_text
    X = read_text("cleaned_tweets.txt","|-|",300)
    mc = MarkovChain(dataset_splited=False)
    lk = mc.compute_probabilities(X)
    mc.normalize_probabilities(lk, check_normalization=True)
    print(lk)
    print(mc.P)
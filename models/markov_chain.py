import numpy as np 


class MarkovChain :

    def __init__(self):
        self.beggining_character = "<S>"
        self.ending_character = "<E>" 
        self.N = None


    def fit(self, X):
        """
        Fit the model to the data.
        Args:
            X: list of strings.
        Returns:
            N : Matrix of transition probabilities.
        """
        characters = [" ","a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]
        characters = [self.beggining_character] + characters + [self.ending_character]    
        #Create a lookup table for the characters
        stoi = {c:i for i,c in enumerate(characters)}
        #Define a matrix of zeros to store frequencies of transitions
        N = np.zeros((len(characters), len(characters)), dtype=np.int32)
        for word in X:
            #Add the beggining character to the beggining of the word, and the ending character to the end of the word
            extended_word = [self.beggining_character] + list(word) + [self.ending_character]
            #Here we iterate over the characters of the word, and we update the matrix of frequencies
            for ch1, ch2 in zip(extended_word, extended_word[1:]): 
                #Get the index of the character in the lookup table
                idx1 = stoi[ch1]
                idx2 = stoi[ch2]
                #Increment the frequency of the transition
                N[idx1, idx2] += 1
        self.N = N
        return N


    def compute_state_matrix(self, X):
        pass

if __name__ == "__main__":
    mc = MarkovChain()
    X = ["hello", "world", "goodbye", "world"]
    lk = mc.fit(X)
    print(lk)
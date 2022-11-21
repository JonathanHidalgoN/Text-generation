import numpy as np 


class MarkovChain :

    def __init__(self):
        self.beggining_character = "<START>"
        self.ending_character = "<END>" 


    def fit(self, X):
        pass

    def create_lookup_table(self, X):
        characters = [" ","a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]
        characters = [self.beggining_character] + characters + [self.ending_character]    
        stoi = {c:i for i,c in enumerate(characters)}
        N = np.zeros((len(characters), len(characters)), dtype=np.int32)
        for word in X:
            extended_word = self.beggining_character + word + self.ending_character
            for ch1, ch2 in zip(extended_word, extended_word[1:]):
                idx1 = stoi[ch1]
                idx2 = stoi[ch2]
                N[idx1, idx2] += 1
        return N


    def compute_state_matrix(self, X):
        pass

class MarkovGenerator :

    def __init__(self):
        pass


    def fit(self, X, generation_type = 'words', k = 2):
        self.k = k
        self.generation_type = generation_type
        prepared_data = self._organize_data(X, generation_type)


    @staticmethod
    def _organize_data(X, generation_type):
        if generation_type == 'words':
            return X.split(' ')
        elif generation_type == 'chars':
            return list(X)
        else:
            raise ValueError('Generation type not supported, try words or chars')

    def count_frequency(self,X):
        freq = {}
        for i in range(len(X) - self.k):
            X_part = " ".join(X[i:i+self.k])
            if X_part in freq:
                freq[X_part] += 1
            else:
                freq[X_part] = 1
        return freq
    


if __name__ == '__main__':
    X = 'hello world this is a test i am testing the markov generator' 
    markov = MarkovGenerator()
    markov.fit(X)
    print(markov.count_frequency(markov._organize_data(X, 'words')))
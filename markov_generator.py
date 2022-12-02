from utils.read_text import clean_text
import numpy as np


class MarkovGenerator:

    """
    A class to generate text using a Markov chain, character by character or word by word.
    Attr:
        k (int): The length of the k-gram.
        generation_type (str): The type of generation, either 'words' or 'chars'.
    """

    def __init__(self):
        self.k = 2
        self.generation_type = "words"

    def fit(self, X, generation_type="words", k=2):
        """
        Fit the Markov chain to the text.
        Args:
            X (list): The text to fit the Markov chain to.
            generation_type (str): The type of generation, either 'words' or 'chars'.
            k (int): The length of the k-gram.
        """
        self.k = k
        self.generation_type = generation_type
        prepared_data = self._organize_data(X, generation_type)
        freq = self.count_frequency(prepared_data, k)
        freq = self.normalize_frequency(freq)
        self.freq = freq

    @staticmethod
    def _organize_data(X, generation_type):
        """
        Organize the data into a list of words or characters.
        Args:
            X (str): The text to be organized.
            generation_type (str): The type of generation, either 'words' or 'chars'.
        Returns:
            A list of words or characters.
        Raises:
            ValueError: If the generation type is not 'words' or 'chars'.
        """
        cleaned_X = " ".join([clean_text(x) for x in X])
        if generation_type == "words":
            return cleaned_X.split(" ")
        elif generation_type == "chars":
            return list(cleaned_X)
        else:
            raise ValueError("Generation type not supported, try words or chars")

    @staticmethod
    def count_frequency(X, k):
        """
        Count the frequency of each k-gram in the data.
        Args:
            X (list): The data to be analyzed.
        Returns:
            A dictionary with the frequency of each k-gram.
        """
        freq = {}
        for i in range(len(X) - k):
            X_part = " ".join(X[i : i + k])
            if freq.get(X_part) is None:
                freq[X_part] = {}
                freq[X_part][X[i + k]] = 1
            else:
                if freq[X_part].get(X[i + k]) is None:
                    freq[X_part][X[i + k]] = 1
                else:
                    freq[X_part][X[i + k]] += 1
        return freq

    @staticmethod
    def normalize_frequency(freq):
        """
        Normalize the frequency of each k-gram.
        Args:
            freq (dict): The frequency of each k-gram.
        Returns:
            A dictionary with the normalized frequency of each k-gram.
        """
        for k_gram in freq:
            total = sum(freq[k_gram].values())
            for next_word in freq[k_gram]:
                freq[k_gram][next_word] /= total
        return freq

    @staticmethod
    def sample_next(k_gram, model):
        """
        Sample the next word or character from the model.
        Args:
            k_gram (str): The current k-gram.
            model (dict): The model to sample from.
        Returns:
            The next word or character.
        """
        try:
            probabilitites = list(model[k_gram].values())
            words = list(model[k_gram].keys())
            return np.random.choice(words, p=probabilitites)
        except KeyError:
            random_word = np.random.choice(list(model.keys()))
            return np.random.choice(list(model[random_word].keys()))

    def generate_text(self, length, seed):
        """
        Generate text using the Markov chain.
        Args:
            length (int): The length of the text to generate.
            seed (str): The seed to start the generation.
        Returns:
            The generated text.
        """
        if self.generation_type == "words":
            seed = seed.split(" ")
        else:
            seed = list(seed)
        generated_text = seed
        for _ in range(length - self.k):
            next_word = self.sample_next(" ".join(generated_text[-self.k :]), self.freq)
            generated_text.append(next_word)
        return " ".join(generated_text)


if __name__ == "__main__":
    from utils.read_text import read_text

    tweets = read_text(
        "/home/jonathan/proyects/proyectos_programacion/Text generation/tweets.txt",
        lines_to_return=1000,
        delimiter="|-|",
    )
    markov = MarkovGenerator()
    markov.fit(tweets, k=3, generation_type="chars")
    print(markov.generate_text(100, "I love"))

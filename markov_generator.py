from utils.read_text import clean_text


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
        self.k = k
        self.generation_type = generation_type
        prepared_data = self._organize_data(X, generation_type)
        freq = self.count_frequency(prepared_data)

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
        cleaned_X = [clean_text(x) for x in X]
        if generation_type == "words":
            return cleaned_X.split(" ")
        elif generation_type == "chars":
            return list(cleaned_X)
        else:
            raise ValueError("Generation type not supported, try words or chars")

    def count_frequency(self, X):
        """
        Count the frequency of each k-gram in the data.
        Args:
            X (list): The data to be analyzed.
        Returns:
            A dictionary with the frequency of each k-gram.
        """
        freq = {}
        for i in range(len(X) - self.k):
            X_part = " ".join(X[i : i + self.k])
            if X_part in freq:
                freq[X_part] += 1
            else:
                freq[X_part] = 1
        return freq


if __name__ == "__main__":
    X = "hello world this is a test i am testing the markov generator"
    markov = MarkovGenerator()
    markov.fit(X)
    print(markov.count_frequency(markov._organize_data(X, "words")))

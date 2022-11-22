import numpy as np
import random
from utils.read_text import read_text

class MarkovChainGenerator:
    def __init__(self):
        """
        Initialize MarkovChainGenerator.
        Args:
            None
        Parameters:
            text (str): text to generate markov chain
            k (int): number of words to generate markov chain
            number_words (int): number of words to generate
            model (dict): markov chain
            extra_text (str): generated text
        """
        self.text = None
        self.K = None
        self.number_words = None
        self.model = None
        self.extra_text = None


    def from_text_to_list(self, text):
        """
        Convert text to list.
        Args:
            text (str): text to convert
        Returns:
            text (list): converted text
        """
        text = text.split(" ")
        return text

    def generate_lookup_table(self, text, n=2):
        """
        Generate lookup table.
        Args:
            text (str): text to generate lookup table
            n (int): number of words to generate lookup table
        Returns:
            lookup_table (dict): lookup table
        """
        lookup_table = {}
        for i in range(len(text) - n):
            # Get the n-gram from the text
            word = text[i]
            if word == "" or word == " ":
                continue
            # Get the next letter from n-gram
            next_words = " ".join(text[i + 1 : i + n + 1])
            # check if the n-gram is in the lookup table
            if lookup_table.get(word) is None:
                # If not, add it to the lookup table
                lookup_table[word] = {}
                lookup_table[word][next_words] = 1
            else:
                # If it is, check if the next letter is in the lookup table
                if lookup_table[word].get(next_words) is None:
                    # If not, add it to the lookup table
                    lookup_table[word][next_words] = 1
                else:
                    # If it is, add 1 to the next letter
                    lookup_table[word][next_words] += 1
        return lookup_table

    def from_text_to_probability(self, lookup_table):
        """
        Convert lookup table to probability.
        Args:
            lookup_table (dict): lookup table
        Returns:
            lookup_table (dict): lookup table with probability
        """
        # For each word in the lookup table
        for word in lookup_table:
            # Get the total number of letters
            total = sum(lookup_table[word].values())
            # For each letter in the lookup table
            for next_letters in lookup_table[word]:
                # Convert the number of letters to probability by dividing by the total
                lookup_table[word][next_letters] /= total
        return lookup_table

    def markov_chain(self, text, n=2):
        """
        Generate markov chain.
        Args:
            text (str): text to generate markov chain
            n (int): number of words to generate markov chain
        Returns:
            lookup_table (dict): markov chain
        """
        lookup_table = self.generate_lookup_table(text, n)
        lookup_table = self.from_text_to_probability(lookup_table)
        return lookup_table

    def next_letter(self, text, model, k):
        """
        Get next letter.
        Args:
            text (str): text to generate markov chain
            model (dict): markov chain
            k (int): number of words to generate markov chain
        Returns:
            next_letter (str): next letter
        """
        last_str = "".join(text[-k:])
        if model.get(last_str) is None:
            _,values = random.choice(list(model.items()))
            #print(f"the word {last_str} is not in the model")
            return random.choice(list(values.keys()))
        possible_letters = list(model[last_str].keys())
        possible_values = list(model[last_str].values())

        return np.random.choice(possible_letters, p=possible_values)

    def generate_text(self, text, model, k, size=1000):
        """
        Generate text.
        Args:
            text (str): text to generate markov chain
            model (dict): markov chain
            k (int): number of words to generate markov chain
            size (int): number of words to generate
        Returns:
            text (str): generated text
        """
        empty_text = ""
        last_str = " ".join(text[-k:])
        if last_str == "" or last_str == " ":
            last_str = " ".join(text[-k - 1 :-k])
        for i in range(size):
            next_letter_ = self.next_letter(last_str, model, k)
            empty_text += next_letter_ + " "
            text += next_letter_ 
            last_str = next_letter_
        return empty_text

    def fit(self, text, k=2, number_words=1000):
        """
        Assign parameters.
        """
        self.text = text 
        self.K = k
        self.number_words = number_words
        self.model = self.markov_chain(self.text, k)

    def predict(self, k=None, number_words=None):
        """
        Predict text.
        Returns:
            extra_text (str): generated text
        """
        if k is None:
            k = self.K
        if number_words is None:
            number_words = self.number_words

        self.extra_text = self.generate_text(self.text, self.model, k, number_words)
        return self.extra_text

if __name__ == "__main__":
    text = read_text("tweets.txt", "|-|",1000)
    markov_chain_generator = MarkovChainGenerator()
    space =   "\n" +  "-"*100 + "\n"
    markov_chain_generator.fit(text, k=4, number_words=100)
    extra_text = markov_chain_generator.predict()
    pass
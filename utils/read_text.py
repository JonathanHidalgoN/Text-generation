from re import sub as resub

def clean_text(text):
    """
    Cleans a text by removing special characters and converting to lowercase.
    Args:
        text: The text to clean.
    Returns:
        A cleaned text.
    """
    text = text.lower()
    text = resub('[^a-zA-Z ]+', '', text)
    return text

def read_text(filename, delimiter=',', lines_to_return=1000):
    """
    Reads a text file and returns a list of lines.
    Args:
        filename: The name of the file to read.
        delimiter: The delimiter to use when reading the file.
        lines_to_return: The number of lines to return.
    Returns:
        A list of lines.
    """
    with open(filename, "r") as f:
        text = f.read()
    lines_to_return = text.split(delimiter)[:lines_to_return]
    return list(map(clean_text, lines_to_return))

if __name__ == "__main__":
    X = read_text("cleaned_tweets.txt","|-|",10)
    print(X)
    
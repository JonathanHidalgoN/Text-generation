
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
    return text.split(delimiter)[:lines_to_return]

if __name__ == "__main__":
    X = read_text("cleaned_tweets.txt","|-|",10)
    print(X)
    
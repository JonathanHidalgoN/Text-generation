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
    text_filter = "(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"
    text = resub(text_filter," ",text)
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
    text = "@midnight is the best show on @comedycentral. #midnight #comedycentral"
    print(clean_text(text))

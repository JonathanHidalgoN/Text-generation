import csv

def read_text(filename, delimiter=','):
    """
    Reads a text file and returns a list of lines.
    Args:
        filename: The name of the file to read.
        delimiter: The delimiter to use when reading the file.
    Returns:
        A list of lines.
    """
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=delimiter)
        text = [row[0] for row in reader]
    return text


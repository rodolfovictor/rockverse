import codecs

def load_text_file(filename, encoding=None):
    """
    Load text file content into memory with specified encoding.
    """

    # Check for valid inputs
    if not isinstance(filename, str):
        raise ValueError('Expected string for filename.')
    if encoding is not None and not isinstance(encoding, str):
        raise ValueError('Expected string for encoding variable.')

    # Try to load the whole file
    if encoding is not None:
        test = (encoding,)
    else:
        test = ('ascii', 'utf-8', 'latin-1')
    for enc in test:
        try:
            with codecs.open(filename, encoding=enc) as f:
                content = f.readlines()
                break
        except Exception:
            content = None
    if content is not None:
        return content

    # Failed to read, sweep the file to find the first invalid
    # line and raise the exception.
    k = 1
    with codecs.open(filename, encoding=test[-1]) as f:
        while True:
            try:
                _ = f.readline()
                k += 1
            except Exception:
                raise ValueError((
                    f"Invalid characters found in {filename}, line {k}, for "
                    "'ascii', 'utf-8', and 'latin-1' encodings. "
                    "Try to specify the correct file encoding when reading the file."))

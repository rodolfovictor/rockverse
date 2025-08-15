class LasImportError(Exception):

    def __init__(self, message, line_number=None):
        super().__init__(message)
        self.line_number = line_number
        self.message = message

    def __str__(self):
        if self.line_number is not None:
            return f"(File line {self.line_number+1}) {self.message}"
        return self.message
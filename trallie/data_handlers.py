import json
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader


class DataHandler:
    def __init__(self, document):
        self.document = document
        self.datatype = None  # Datatype will be set by the decorator
        self.length = None  # Length will be set by the decorator
        self.text = None  # Contains extracted text

    def infer_datatype(func):
        """
        Decorator to infer the datatype of the document and set it.
        """

        def wrapper(self, *args, **kwargs):
            self.datatype = self.document.split(".")[-1].lower()
            return func(self, *args, **kwargs)

        return wrapper

    def get_text_from_pdf(self):
        # Use a pdf extractor
        try:
            # Open and read the PDF file
            reader = PdfReader(self.document)
            text = ""
            # Iterate through all pages and extract text
            for page in reader.pages:
                text += page.extract_text()
            return text.strip() if text else "Error: No text found in the PDF"
        except FileNotFoundError:
            return "Error: File not found"
        except Exception as e:
            return f"Error: An unexpected error occurred: {str(e)}"
        return "Extracted text from PDF"

    def get_text_from_html(self):
        # Use an HTML parser
        try:
            with open(self.document, "r", encoding="utf-8") as file:
                soup = BeautifulSoup(file, "html.parser")
            # Extract and return plain text from the HTML
            return soup.get_text()
        except FileNotFoundError:
            return "Error: File not found"
        except Exception as e:
            return f"Error: An unexpected error occurred: {str(e)}"
        return "Extracted text from HTML"

    def get_text_from_json(self):
        # Extract text from JSON
        try:
            with open(self.document, "r", encoding="utf-8") as file:
                data = json.load(file)
            # Convert the JSON object to a pretty-printed string for readability
            return json.dumps(data, indent=4)
        except FileNotFoundError:
            return "Error: File not found"
        except json.JSONDecodeError:
            return "Error: Invalid JSON format"
        return "Extracted text from JSON"

    def get_text_from_txt(self):
        try:
            with open(self.document, "r") as file:
                return file.read()
        except FileNotFoundError:
            return "Error: File not found"
        except Exception as e:
            return f"Error: An unexpected error occurred: {str(e)}"

    def chunk_text(self, char_limit=20000):
        """
        Decorator to chunk longer document if it exceeds the word length.
        """
        if self.length > char_limit:
            self.text = self.text[:char_limit]
        return self.text

    @infer_datatype
    def get_text(self):
        """
        Get text from the document based on its datatype.
        """
        if self.datatype == "pdf":
            self.text = self.get_text_from_pdf()
        elif self.datatype in {"html", "htm"}:
            self.text = self.get_text_from_html()
        elif self.datatype == "json":
            self.text = self.get_text_from_json()
        elif self.datatype == "txt":
            self.text = self.get_text_from_txt()
        else:
            return "Unsupported file type"

        # Chunk text
        self.length = len(self.text)
        self.chunk_text()
        return self.text



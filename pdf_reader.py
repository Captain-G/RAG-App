import os
from PyPDF2 import PdfReader
from logging_info import logging_setup


logger = logging_setup()

def read_pdf(file):
    return PdfReader(file)

def extract_text_from_pdf(reader):
    try:
        text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text() or ""  # Extract text from each page
        return text
    except Exception as e:
        logger.error(f"Having following issue with extract_text_from_pdf - {e}")
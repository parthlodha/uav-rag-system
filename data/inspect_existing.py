import os
from pypdf import PdfReader

pdf_files = [f for f in os.listdir('.') if f.lower().endswith('.pdf')]

print(f"Found {len(pdf_files)} PDF files.")

for pdf_file in sorted(pdf_files):
    print("=" * 60)
    print(f"File: {pdf_file}")
    try:
        reader = PdfReader(pdf_file)
        # Metadata
        meta = reader.metadata
        title = meta.title if meta and meta.title else "Unknown Title"
        author = meta.author if meta and meta.author else "Unknown Author"
        print(f"Metadata Title: {title}")
        print(f"Metadata Author: {author}")
        print(f"Pages: {len(reader.pages)}")
        
        # Get first page text to see if we can find abstract/content summary
        if len(reader.pages) > 0:
            first_page_text = reader.pages[0].extract_text()
            print("--- First 400 characters of text ---")
            print(first_page_text[:400].strip())
            print("-----------------------------------")
    except Exception as e:
        print(f"Error reading file {pdf_file}: {e}")

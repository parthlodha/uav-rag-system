import os
import re
from pypdf import PdfReader
from fpdf import FPDF

DATA_DIR = "data"
OUTPUT_PDF = "PDF_Summaries_Report.pdf"

def extract_abstract(pdf_path):
    """Extracts text from the PDF and attempts to isolate the Abstract section."""
    try:
        reader = PdfReader(pdf_path)
        if not reader.pages:
            return "Empty PDF file."
        
        # Extract text from first 2 pages (abstracts are typically on page 1 or 2)
        text = ""
        for page in reader.pages[:2]:
            text += page.extract_text() + "\n"
        
        # Regex search for Abstract section
        abstract_match = re.search(r'(?i)abstract[:\s\n]+(.*?)(?=\n\s*\n\s*1|\n\s*\n\s*introduction|\Z)', text, re.DOTALL)
        
        if abstract_match:
            abstract = abstract_match.group(1).strip()
            # Clean up extra line breaks
            abstract = re.sub(r'\s+', ' ', abstract)
            return abstract[:1200] + ("..." if len(abstract) > 1200 else "")
        else:
            # Fallback: return the first 600 characters of text
            clean_text = re.sub(r'\s+', ' ', text).strip()
            return clean_text[:600] + "..." if len(clean_text) > 600 else clean_text
            
    except Exception as e:
        return f"Error reading PDF: {e}"

def generate_pdf_report(summaries):
    """Generates a structured PDF report containing Filenames and Abstracts."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Title
    pdf.set_font("Helvetica", style="B", size=16)
    pdf.cell(0, 10, "UAV Dataset Summary Report", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", style="I", size=10)
    pdf.cell(0, 6, f"Total Documents Evaluated: {len(summaries)}", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(8)
    
    for item in summaries:
        # Filename Header
        pdf.set_font("Helvetica", style="B", size=11)
        pdf.set_fill_color(230, 240, 250)
        pdf.cell(0, 8, f" File: {item['filename']}", new_x="LMARGIN", new_y="NEXT", fill=True)
        pdf.ln(2)
        
        # Abstract Content
        pdf.set_font("Helvetica", size=9)
        # Encode string to latin-1 to avoid standard FPDF character errors
        clean_abstract = item['abstract'].encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 5, clean_abstract)
        pdf.ln(6)
        
    pdf.output(OUTPUT_PDF)
    print(f"\n Summary report PDF generated successfully: '{OUTPUT_PDF}'")

def main():
    if not os.path.exists(DATA_DIR):
        print(f"Directory '{DATA_DIR}' does not exist.")
        return

    pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"No PDF files found in '{DATA_DIR}'.")
        return

    print(f"\nProcessing {len(pdf_files)} PDFs in '{DATA_DIR}'...\n")
    summaries = []

    for filename in sorted(pdf_files):
        filepath = os.path.join(DATA_DIR, filename)
        print(f" Reading: {filename}")
        abstract = extract_abstract(filepath)
        summaries.append({
            "filename": filename,
            "filepath": filepath,
            "abstract": abstract
        })

    # Generate output PDF summary file
    try:
        generate_pdf_report(summaries)
    except Exception as e:
        print(f"Note: Could not build PDF report (fpdf module missing). Run 'pip install fpdf2' if required. Error: {e}")

    # Interactive filtering terminal UI
    print("\n==========================================================")
    print("                PDF FILTERING & CLEANUP                    ")
    print("==========================================================\n")
    
    for i, item in enumerate(summaries):
        print(f"[{i+1}/{len(summaries)}] File: {item['filename']}")
        print(f"Abstract Preview:\n{item['abstract'][:300]}...\n")
        
        choice = input("Keep this file? (y/n, default='y'): ").strip().lower()
        if choice == 'n':
            os.remove(item['filepath'])
            print(f" Removed: {item['filename']}\n")
        else:
            print(f" Kept: {item['filename']}\n")
            
    remaining = [f for f in os.listdir(DATA_DIR) if f.endswith('.pdf')]
    print("==========================================================")
    print(f" Cleanup Complete! {len(remaining)} PDFs remain in '{DATA_DIR}/'.")
    print(" You can now safely run: python3 src/ingestion.py\n")

if __name__ == "__main__":
    main()

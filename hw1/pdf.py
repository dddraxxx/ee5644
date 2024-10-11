from nbconvert import PDFExporter
from PyPDF2 import PdfMerger
import nbformat

# Paths to the Jupyter notebooks
notebook_paths = ["q1.ipynb", "q2.ipynb", "q3.ipynb"]

# Create a PDF exporter
pdf_exporter = PDFExporter()
pdf_exporter.exclude_input = False  # Includes both input cells and output cells

pdf_files = []

# Convert each notebook to PDF
for notebook_path in notebook_paths:
    # Read the notebook
    with open(notebook_path) as f:
        notebook = nbformat.read(f, as_version=4)

    # Export the notebook to PDF format
    pdf_data, _ = pdf_exporter.from_notebook_node(notebook)

    # Save the PDF to a file
    pdf_output_path = notebook_path.replace('.ipynb', '.pdf')
    with open(pdf_output_path, 'wb') as pdf_file:
        pdf_file.write(pdf_data)

    pdf_files.append(pdf_output_path)

# Merge the generated PDFs using PdfMerger
pdf_merger = PdfMerger()

for pdf_file in pdf_files:
    pdf_merger.append(pdf_file)

# Save the merged PDF to a file
output_pdf = "combined_output.pdf"
pdf_merger.write(output_pdf)
pdf_merger.close()

print(f"PDFs have been merged and saved as {output_pdf}")

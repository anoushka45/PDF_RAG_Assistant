from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

def load_pdfs(pdf_dir):
    documents = []
    pdf_files = Path(pdf_dir).glob("*.pdf")

    for pdf in pdf_files:
        loader = PyPDFLoader(str(pdf))
        docs = loader.load()

        for d in docs:
            d.metadata["source"] = pdf.name

        documents.extend(docs)

    return documents

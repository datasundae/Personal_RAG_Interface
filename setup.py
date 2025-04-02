from setuptools import setup, find_packages

setup(
    name="personal_rag_interface",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'flask',
        'psycopg2-binary',
        'numpy',
        'sentence-transformers',
        'cryptography',
        'pytesseract',
        'pdf2image',
        'opencv-python',
        'PyMuPDF',
        'PyPDF2',
        'python-docx',
        'beautifulsoup4',
        'pandas',
        'tiktoken',
        'transformers'
    ],
    author="datasundae",
    description="A modular RAG system for personal document management",
    python_requires=">=3.8",
) 
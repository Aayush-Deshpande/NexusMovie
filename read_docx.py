import docx
import sys

def read_docx(file_path):
    try:
        doc = docx.Document(file_path)
        fullText = []
        for para in doc.paragraphs:
            fullText.append(para.text)
        return '\n'.join(fullText)
    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    text = read_docx("d:/Code/DAVL_Exam/📘 Functional Requirements Document.docx")
    with open("reqs.txt", "w", encoding="utf-8") as f:
        f.write(text)

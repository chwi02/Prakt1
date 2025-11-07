from PyPDF2 import PdfReader, PdfWriter
with open("outputs/aufgabe1.pdf", "rb") as f1, open("outputs/aufgabe2.pdf", "rb") as f2, open("outputs/aufgabe3.pdf", "rb") as f3:
    pdfOne = PdfReader(f1)
    pdfTwo = PdfReader(f2)
    pdfThree = PdfReader(f3)
    output = PdfWriter()
    output.add_page(pdfOne.pages[0])
    output.add_page(pdfTwo.pages[0])
    output.add_page(pdfThree.pages[0])
    with open("outputs/Lösungsergebnisse.pdf", "wb") as out:
        output.write(out)
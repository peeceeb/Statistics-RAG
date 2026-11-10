from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.lib.colors import HexColor
from xml.sax.saxutils import escape
import os

output_path = "/mnt/data/Enterprise_RAG_Sparse_Retrieval_Test_Document.pdf"

doc = SimpleDocTemplate(
    output_path,
    pagesize=A4,
    rightMargin=19*mm,
    leftMargin=19*mm,
    topMargin=18*mm,
    bottomMargin=18*mm,
    title="Enterprise RAG - Sparse Retrieval Test Document",
    author="Enterprise RAG Test Dataset",
)

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name="DocTitle", parent=styles["Title"], fontName="Helvetica-Bold",
    fontSize=19, leading=24, alignment=TA_CENTER, textColor=HexColor("#17365D"),
    spaceAfter=6*mm
))
styles.add(ParagraphStyle(
    name="Subtitle", parent=styles["Normal"], fontName="Helvetica",
    fontSize=9, leading=13, alignment=TA_CENTER, textColor=colors.grey,
    spaceAfter=8*mm
))
styles.add(ParagraphStyle(
    name="Section", parent=styles["Heading2"], fontName="Helvetica-Bold",
    fontSize=12, leading=15, textColor=HexColor("#1F4E79"),
    spaceBefore=4*mm, spaceAfter=2*mm
))
styles.add(ParagraphStyle(
    name="BodyCustom", parent=styles["BodyText"], fontName="Helvetica",
    fontSize=9.5, leading=14, spaceAfter=2*mm
))
styles.add(ParagraphStyle(
    name="BulletCustom", parent=styles["BodyText"], fontName="Helvetica",
    fontSize=9.5, leading=13, leftIndent=12, firstLineIndent=-7,
    spaceAfter=1.2*mm
))
styles.add(ParagraphStyle(
    name="Meta", parent=styles["BodyText"], fontName="Helvetica",
    fontSize=9, leading=13
))

story = []
story.append(Paragraph("Enterprise RAG – Sparse Retrieval Test Document", styles["DocTitle"]))
story.append(Paragraph(
    "Synthetic technical document for testing keyword-based sparse retrieval and hybrid search",
    styles["Subtitle"
]))
story.append(Paragraph("Document Information", styles["Section"]))

metadata = [
    [Paragraph("<b>Document ID</b>", styles["Meta"]), Paragraph("DOC-1001", styles["Meta"])],
    [Paragraph("<b>Document Version</b>", styles["Meta"]), Paragraph("1.0", styles["Meta"])],
    [Paragraph("<b>Category</b>", styles["Meta"]), Paragraph("Technical Documentation", styles["Meta"])],
]
meta_table = Table(metadata, colWidths=[42*mm, 125*mm], hAlign="LEFT")
meta_table.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (0,-1), HexColor("#EAF1F8")),
    ("GRID", (0,0), (-1,-1), 0.5, HexColor("#C7D3E0")),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("LEFTPADDING", (0,0), (-1,-1), 7),
    ("RIGHTPADDING", (0,0), (-1,-1), 7),
    ("TOPPADDING", (0,0), (-1,-1), 6),
    ("BOTTOMPADDING", (0,0), (-1,-1), 6),
]))
story.append(meta_table)
story.append(Spacer(1, 3*mm))

sections = [
("1. Document Ingestion", [
"The document ingestion pipeline extracts text from PDF documents and divides the extracted content into smaller chunks.",
"Each chunk is processed using two embedding approaches:",
"• Dense embeddings using OpenAI text-embedding-3-large.",
"• Sparse embeddings using a BM25-based sparse encoder.",
"The resulting vectors are stored in the Qdrant vector database."
]),
("2. Sparse Retrieval", [
"Sparse retrieval identifies relevant documents using keyword-level matching.",
"Important terminology:",
"• Sparse vectors",
"• BM25",
"• Term frequency",
"• Inverse document frequency",
"• Keyword matching",
"• Exact identifier matching",
"• Tokenization",
"The retrieval service uses sparse vectors to identify documents containing relevant technical terms."
]),
("3. Dense Retrieval", [
"Dense retrieval uses semantic embeddings to identify documents based on contextual similarity.",
"The embedding model is text-embedding-3-large, which generates 3072-dimensional vectors."
]),
("4. Hybrid Retrieval", [
"Hybrid retrieval combines dense and sparse search results.",
"The retrieval pipeline retrieves the top 20 candidate chunks and applies reranking to select the final top 5 results."
]),
("5. Error Codes and Identifiers", [
"• ERR-503: Retrieval service unavailable.",
"• ERR-401: Authentication failure.",
"• DOC-1001: Sparse retrieval documentation.",
"• MODEL-2026: Embedding model configuration."
]),
("6. System Components", [
"The Enterprise RAG application uses:",
"• Python for backend development.",
"• FastAPI for REST APIs.",
"• Streamlit for the user interface.",
"• LangGraph for agent orchestration.",
"• Qdrant for vector storage.",
"• FlashRank for reranking.",
"• PostgreSQL for conversation persistence."
]),
("7. Performance Configuration", [
"The retrieval configuration uses:",
"• RETRIEVAL_TOP_K = 20",
"• FINAL_TOP_K = 5",
"• EMBEDDING_DIMENSION = 3072",
"• RERANKER = ms-marco-MiniLM-L-12-v2"
]),
("8. Summary", [
"Sparse retrieval improves exact keyword matching. Dense retrieval improves semantic matching. Hybrid retrieval combines both approaches to improve document discovery."
])
]

for heading, paragraphs in sections:
    story.append(Paragraph(heading, styles["Section"]))
    for p in paragraphs:
        if p.startswith("•"):
            story.append(Paragraph(p, styles["BulletCustom"]))
        else:
            story.append(Paragraph(escape(p), styles["BodyCustom"]))

def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.grey)
    canvas.drawString(19*mm, 10*mm, "Enterprise RAG | Sparse Retrieval Test")
    canvas.drawRightString(A4[0] - 19*mm, 10*mm, f"Page {doc.page}")
    canvas.restoreState()

doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)

print(f"PDF created: {output_path}")
print(f"File size: {os.path.getsize(output_path):,} bytes")

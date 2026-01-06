import requests
import re
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from io import BytesIO
from config import CHUNK_SIZE, CHUNK_OVERLAP


HEADERS = {
    "User-Agent": "SEC-RAG-Project/1.0 (email@example.com)",
    "Accept-Encoding": "gzip, deflate",
    "Accept": "*/*"
}


class SECDocumentProcessor:

    def fetch_filing_text(self, url: str) -> str:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            if resp.status_code != 200:
                return ""

            content_type = resp.headers.get("Content-Type", "").lower()

            # ---------- PDF ----------
            if "pdf" in content_type or url.lower().endswith(".pdf"):
                return self._extract_pdf_text(resp.content)

            # ---------- HTML ----------
            soup = BeautifulSoup(resp.text, "html.parser")

            # 🔑 CASE 1: Index page → find real filing
            filing_link = self._extract_actual_filing_link(soup)
            if filing_link:
                return self.fetch_filing_text(filing_link)

            # 🔑 CASE 2: Actual filing page
            for tag in soup(["script", "style", "table", "noscript"]):
                tag.decompose()

            text = soup.get_text(separator=" ")
            text = re.sub(r"\s+", " ", text).strip()

            return text

        except Exception:
            return ""

    def _extract_actual_filing_link(self, soup):
        """
        From SEC index page, find the first real filing document
        """
        for link in soup.find_all("a", href=True):
            href = link["href"]

            if any(ext in href.lower() for ext in [".htm", ".html", ".pdf"]):
                if "Archives/edgar" in href:
                    if not href.startswith("http"):
                        return "https://www.sec.gov" + href
                    return href
        return None

    def _extract_pdf_text(self, content: bytes) -> str:
        reader = PdfReader(BytesIO(content))
        pages = []

        for page in reader.pages:
            text = page.extract_text()
            if text and len(text) > 50:
                pages.append(text)

        return "\n".join(pages)

    def chunk_document(self, text: str, metadata: dict):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

        return [
            Document(page_content=chunk, metadata=metadata)
            for chunk in splitter.split_text(text)
        ]

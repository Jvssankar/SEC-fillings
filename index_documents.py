# index_documents.py

import pandas as pd
from tqdm import tqdm
from document_processor import SECDocumentProcessor
from vector_store import SECVectorStore

# ---------------- CONFIG ----------------
CSV_PATH = "./data/sec_filings.csv"   # must exist
MAX_ROWS = 800
MIN_TEXT_LENGTH = 300
# ---------------------------------------


def normalize_sec_url(raw_url: str) -> str:
    """
    Fix malformed SEC URLs from CSV.
    """
    if not raw_url or raw_url.lower() == "nan":
        return ""

    raw_url = raw_url.strip()

    if raw_url.startswith("http"):
        return raw_url

    raw_url = raw_url.lstrip("/")

    if raw_url.startswith("www.sec.gov"):
        return "https://" + raw_url

    return "https://www.sec.gov/" + raw_url


def main():
    print("\n🚀 Starting SEC filings indexing pipeline...\n")

    # -------- Load CSV --------
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"❌ CSV file not found: {CSV_PATH}")
        return

    print(f"📊 Total rows available in CSV: {len(df)}")
    print(f"📌 Indexing first {MAX_ROWS} filings only.\n")

    # -------- Init components --------
    processor = SECDocumentProcessor()
    vector_store = SECVectorStore()

    filings_processed = 0
    chunks_added = 0
    skipped_rows = 0

    # -------- Iterate rows --------
    for idx, row in tqdm(df.head(MAX_ROWS).iterrows(), total=MAX_ROWS):

        try:
            raw_url = str(row["Filing URL"])
            url = normalize_sec_url(raw_url)

            if not url:
                skipped_rows += 1
                continue

            # -------- Fetch filing text --------
            text = processor.fetch_filing_text(url)

            if not text or len(text) < MIN_TEXT_LENGTH:
                skipped_rows += 1
                continue

            # -------- Metadata (MATCHES YOUR CSV EXACTLY) --------
            filed_at = str(row["Filed At"])
            year = filed_at[:4] if filed_at and filed_at.lower() != "nan" else "Unknown"

            metadata = {
                "accession_no": str(row["Accession No"]),
                "cik": str(row["CIK"]),
                "company": str(row["Company Name"]),
                "ticker": str(row["Ticker"]),
                "description": str(row["Description"]),
                "form_type": str(row["Form Type"]),
                "filing_type": str(row["Filing Type"]),
                "filed_at": filed_at,
                "year": year,
                "source_url": url,
            }

            # -------- Chunk --------
            documents = processor.chunk_document(text, metadata)

            if not documents:
                skipped_rows += 1
                continue

            # -------- Store --------
            vector_store.add_documents(documents)

            filings_processed += 1
            chunks_added += len(documents)

            if filings_processed % 10 == 0:
                print(
                    f"\n📄 Processed {filings_processed} filings | "
                    f"📦 Total chunks: {chunks_added}"
                )

        except Exception as e:
            skipped_rows += 1
            print(f"\n⚠️ Row {idx} skipped due to error: {e}")

    # -------- Final report --------
    print("\n✅ Indexing completed!")
    print(f"📄 Filings processed: {filings_processed}")
    print(f"📦 Chunks stored in vector DB: {chunks_added}")
    print(f"🚫 Rows skipped: {skipped_rows}")

    if chunks_added == 0:
        print("\n❌ NO CHUNKS CREATED")
        print("👉 Root causes are ONLY one of these:")
        print("   1. Filing URLs point to index pages (not actual docs)")
        print("   2. SEC blocked requests (User-Agent issue)")
        print("   3. Text extraction returned empty content")
    else:
        print("\n🎉 Vector DB is READY for querying!")


if __name__ == "__main__":
    main()

import io
import os
import csv
import gzip
import zipfile
import tarfile
import sqlite3
from io import BytesIO, StringIO
import tempfile
import json
import pandas as pd
from PIL import Image

import pdfplumber
# Optional helpers (all guarded; will degrade gracefully if missing)
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except Exception:
    PDFPLUMBER_AVAILABLE = False

import pandas as pd

try:
    import xmltodict
    XMLTODICT_AVAILABLE = True
except Exception:
    XMLTODICT_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except Exception:
    BS4_AVAILABLE = False

try:
    import docx
    DOXC_AVAILABLE = True
except Exception:
    DOXC_AVAILABLE = False

try:
    import chardet
    CHARDET_AVAILABLE = True
except Exception:
    CHARDET_AVAILABLE = False


# -----------------------------
# Universal file loader
# -----------------------------
MAX_ARCHIVE_FILES = int(os.getenv("MAX_ARCHIVE_FILES", 50))
MAX_FILE_BYTES    = int(os.getenv("MAX_FILE_BYTES", 25 * 1024 * 1024))  # 25MB per file in archives
MAX_PDF_PAGES     = int(os.getenv("MAX_PDF_PAGES", 100))

def _decode_text_bytes(b: bytes) -> str:
    # Try common encodings fast; fall back to chardet if available
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return b.decode(enc)
        except Exception:
            pass
    if CHARDET_AVAILABLE:
        try:
            guess = chardet.detect(b).get("encoding")
            if guess:
                return b.decode(guess, errors="replace")
        except Exception:
            pass
    return b.decode("utf-8", errors="replace")

def _sniff_csv_delimiter(text: str) -> str | None:
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample)
        return dialect.delimiter
    except Exception:
        # quick heuristics
        if "\t" in sample: return "\t"
        if ";" in sample:  return ";"
        if "|" in sample:  return "|"
        if "," in sample:  return ","
        return None

def _concat_with_origin(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    # unify string columns
    for f in frames:
        f.columns = f.columns.map(str)
    return pd.concat(frames, ignore_index=True, sort=True)

def _read_excel_all_sheets(content: bytes, filename: str) -> pd.DataFrame:
    try:
        sheets = pd.read_excel(BytesIO(content), sheet_name=None)
    except Exception:
        # fallback older engines
        xls = pd.ExcelFile(BytesIO(content))
        sheets = {name: xls.parse(name) for name in xls.sheet_names}
    frames = []
    for sname, df in sheets.items():
        df = pd.DataFrame(df)
        df["_sheet"] = str(sname)
        df["_source"] = filename
        frames.append(df)
    return _concat_with_origin(frames) if frames else pd.DataFrame()

def _read_pdf(content: bytes, filename: str) -> pd.DataFrame:
    frames = []

    # Prefer tables with pdfplumber
    if PDFPLUMBER_AVAILABLE:
        try:
            with pdfplumber.open(BytesIO(content)) as pdf:
                text_pages = []
                for i, page in enumerate(pdf.pages[:MAX_PDF_PAGES]):
                    try:
                        tables = page.extract_tables() or []
                    except Exception:
                        tables = []
                    if tables:
                        for t in tables:
                            df = pd.DataFrame(t)
                            df["_page"] = i + 1
                            df["_source"] = filename
                            frames.append(df)
                    else:
                        text = page.extract_text() or ""
                        text_pages.append({"page": i + 1, "text": text, "_source": filename})
                if frames:
                    return _concat_with_origin(frames)
                # no tables → return page-wise text
                return pd.DataFrame(text_pages)
        except Exception:
            pass


    # Last resort: raw bytes as a single row
    return pd.DataFrame({"text": ["<pdf content>"], "_source": [filename]})

def _read_sqlite(content: bytes, filename: str) -> pd.DataFrame:
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    try:
        tmp.write(content)
        tmp.flush()
        tmp.close()
        conn = sqlite3.connect(tmp.name)
        cur = conn.cursor()
        tbls = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        frames = []
        for t in tbls:
            try:
                df = pd.read_sql_query(f"SELECT * FROM '{t}'", conn)
                df["_table"] = t
                df["_source"] = filename
                frames.append(df)
            except Exception:
                continue
        conn.close()
        return _concat_with_origin(frames) if frames else pd.DataFrame({"_source": [filename]})
    finally:
        try: os.unlink(tmp.name)
        except Exception: pass

def _read_html_text(content: bytes, filename: str) -> pd.DataFrame:
    text = _decode_text_bytes(content)
    # Try tables
    try:
        tables = pd.read_html(StringIO(text))
        if tables:
            frames = []
            for i, df in enumerate(tables):
                df["_html_table_index"] = i
                df["_source"] = filename
                frames.append(df)
            return _concat_with_origin(frames)
    except Exception:
        pass
    # Fallback to page text via BeautifulSoup if available
    if BS4_AVAILABLE:
        try:
            soup = BeautifulSoup(text, "html.parser")
            page_text = soup.get_text(separator="\n", strip=True)
            return pd.DataFrame({"text": [page_text], "_source": [filename]})
        except Exception:
            pass
    return pd.DataFrame({"text": [text], "_source": [filename]})

def _read_xml(content: bytes, filename: str) -> pd.DataFrame:
    txt = _decode_text_bytes(content)
    if XMLTODICT_AVAILABLE:
        try:
            obj = xmltodict.parse(txt)
            df = pd.json_normalize(obj, max_level=2)
            df["_source"] = filename
            return df
        except Exception:
            pass
    # very rough fallback: plain text
    return pd.DataFrame({"text": [txt], "_source": [filename]})

def _read_docx(content: bytes, filename: str) -> pd.DataFrame:
    if not DOXC_AVAILABLE:
        return pd.DataFrame({"text": ["<docx content>"], "_source": [filename]})
    tmp = tempfile.NamedTemporaryFile(suffix=".docx", delete=False)
    try:
        tmp.write(content); tmp.flush(); tmp.close()
        d = docx.Document(tmp.name)
        paras = [p.text for p in d.paragraphs if p.text.strip()]
        return pd.DataFrame({"paragraph": paras, "_source": [filename]*len(paras)})
    finally:
        try: os.unlink(tmp.name)
        except Exception: pass

def _read_gzip_singlefile(content: bytes, inner_name_hint: str, filename: str) -> pd.DataFrame:
    try:
        raw = gzip.decompress(content)
    except Exception:
        return pd.DataFrame({"text": ["<gz content>"], "_source": [filename]})
    # try csv/json/text in that order
    try:
        return pd.read_csv(BytesIO(raw)).assign(_source=filename)
    except Exception:
        pass
    try:
        return pd.read_json(BytesIO(raw)).assign(_source=filename)
    except Exception:
        pass
    txt = _decode_text_bytes(raw)
    delim = _sniff_csv_delimiter(txt)
    if delim:
        try:
            return pd.read_csv(StringIO(txt), sep=delim).assign(_source=filename)
        except Exception:
            pass
    return pd.DataFrame({"text": [txt], "_source": [filename]})

def _read_archive(content: bytes, filename: str) -> pd.DataFrame:
    frames = []
    processed = 0
    lower = filename.lower()

    # ZIP
    if lower.endswith(".zip"):
        with zipfile.ZipFile(BytesIO(content)) as zf:
            for info in zf.infolist():
                if processed >= MAX_ARCHIVE_FILES: break
                if info.is_dir(): continue
                if info.file_size and info.file_size > MAX_FILE_BYTES: continue
                try:
                    inner = zf.read(info.filename)
                except Exception:
                    continue
                df = load_any_file_to_dataframe(inner, info.filename)
                if not df.empty:
                    df["_file"] = info.filename
                    df["_source"] = filename
                    frames.append(df)
                    processed += 1
        return _concat_with_origin(frames)

    # TAR / TGZ / TBZ2
    try:
        mode = "r:*" if any(lower.endswith(e) for e in (".tar", ".tgz", ".tar.gz", ".tbz2", ".tar.bz2", ".txz", ".tar.xz")) else None
        if mode:
            with tarfile.open(fileobj=BytesIO(content), mode=mode) as tf:
                for m in tf.getmembers():
                    if processed >= MAX_ARCHIVE_FILES: break
                    if not m.isfile(): continue
                    if m.size and m.size > MAX_FILE_BYTES: continue
                    f = tf.extractfile(m)
                    if not f: continue
                    inner = f.read()
                    df = load_any_file_to_dataframe(inner, m.name)
                    if not df.empty:
                        df["_file"] = m.name
                        df["_source"] = filename
                        frames.append(df)
                        processed += 1
            return _concat_with_origin(frames)
    except Exception:
        pass

    # Unknown archive type fallback
    return pd.DataFrame({"text": ["<archive content>"], "_source": [filename]})

def load_any_file_to_dataframe(content: bytes, filename: str) -> pd.DataFrame:
    """
    Best-effort loader for 'anything': returns a pandas DataFrame.
    Unknown/unsupported types fall back to a single 'text' column.
    """
    name = (filename or "").lower()

    # --- known table-ish formats ---
    if name.endswith((".csv",)):
        return pd.read_csv(BytesIO(content)).assign(_source=filename)

    if name.endswith((".tsv", ".tab", ".tsc")):  # treat .tsc as TSV
        return pd.read_csv(BytesIO(content), sep="\t").assign(_source=filename)

    if name.endswith((".xlsx", ".xls")):
        return _read_excel_all_sheets(content, filename)

    if name.endswith(".parquet"):
        return pd.read_parquet(BytesIO(content)).assign(_source=filename)

    if name.endswith(".json"):
        try:
            return pd.read_json(BytesIO(content)).assign(_source=filename)
        except ValueError:
            try:
                obj = json.loads(_decode_text_bytes(content))
                return pd.json_normalize(obj).assign(_source=filename)
            except Exception:
                return pd.DataFrame({"text": [_decode_text_bytes(content)], "_source": [filename]})

    # --- binary / structured ---
    if name.endswith((".pdf",)):
        return _read_pdf(content, filename)

    if name.endswith((".db", ".sqlite", ".sqlite3")):
        return _read_sqlite(content, filename)

    if name.endswith((".gz", ".gzip")):
        return _read_gzip_singlefile(content, filename[:-3], filename)

    if name.endswith(".zip") or any(name.endswith(e) for e in (".tar", ".tgz", ".tar.gz", ".tbz2", ".tar.bz2", ".txz", ".tar.xz")):
        return _read_archive(content, filename)

    if name.endswith((".html", ".htm")):
        return _read_html_text(content, filename)

    if name.endswith((".xml",)):
        return _read_xml(content, filename)

    if name.endswith((".docx",)):
        return _read_docx(content, filename)

    # --- images (keep your existing path using PIL above if needed) ---
    if name.endswith((".png", ".jpg", ".jpeg")):
            try:
                image = Image.open(BytesIO(content)).convert("RGB")
                return pd.DataFrame({"image": [image], "_source": [filename]})
            except Exception:
               return pd.DataFrame({"text": ["<image>"], "_source": [filename]})

    # --- plain text-ish: try parsing as delimited; fallback to free text ---
    if name.endswith((".txt", ".log", ".md", ".rst", "")):
        txt = _decode_text_bytes(content)
        delim = _sniff_csv_delimiter(txt)
        if delim:
            try:
                return pd.read_csv(StringIO(txt), sep=delim).assign(_source=filename)
            except Exception:
                pass
        return pd.DataFrame({"text": [txt], "_source": [filename]})

    # --- final fallback ---
    return pd.DataFrame({"text": [_decode_text_bytes(content)], "_source": [filename]})

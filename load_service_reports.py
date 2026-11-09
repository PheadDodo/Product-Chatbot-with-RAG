from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
SERVICE_DIR = PROJECT_DIR / "data" / "HistoricalServices"

def load_service_reports(folder=SERVICE_DIR):
    folder = Path(folder).resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"Service folder does not exist: {folder}")

    paths = sorted(path for path in folder.glob("FSR-*.txt") if path.is_file())
    if not paths:
        raise FileNotFoundError(f"No service reports found in: {folder}")

    records = []
    for path in paths:
        text = path.read_text(encoding="utf-8-sig")
        if not text.strip():
            raise ValueError(f"Service report is empty: {path.name}")

        records.append({
            "id": path.stem,
            "source": path.name,
            "source_path": str(path),
            "text": text,
            "lines": [
                {"line_number": number, "text": line} for number, line in enumerate(text.splitlines(), start=1)
            ]
        })
    return records

def main():
    records = load_service_reports()
    print(f"Loaded service reports: {len(records)}")

    record = records[0]
    print(f"\nID: {record['id']}")
    print(f"Source: {record['source']}")
    print(f"Lines: {len(record['lines'])}")
    print("Preview:")
    for line in record["lines"][:8]:
        print(f"{line['line_number']:>3}: {line['text']}")

if __name__ == "__main__":
    main()
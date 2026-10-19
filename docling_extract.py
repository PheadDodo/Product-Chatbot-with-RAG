from pathlib import Path
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import ImageRefMode

options = PdfPipelineOptions(
    do_ocr=True,
    do_table_structure=True,
    generate_page_images=True,
    images_scale=2.0,
)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=options)
    }
)

pdf_paths = sorted(Path("./data/brochures").glob("*.pdf"))

for index, pdf_path in enumerate(pdf_paths, start=1):
    print(f"\n[{index}/{len(pdf_paths)}] {pdf_path.name}", flush=True)

    document = converter.convert(pdf_path).document

    output_dir = (
        Path("./data/processed/docling") / pdf_path.stem
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    document.save_as_json(
        output_dir / "document.json",
        artifacts_dir=Path("images"),
        image_mode=ImageRefMode.REFERENCED,
        ensure_ascii=False,
    )

    document.save_as_markdown(
        output_dir / "document.md",
        artifacts_dir=Path("images"),
        image_mode=ImageRefMode.REFERENCED,
        page_break_placeholder="<!-- page break -->",
        traverse_pictures=True,
    )

    nonempty_tables = [
        table
        for table in document.tables
        if any(cell.text.strip() for cell in table.data.table_cells)
    ]

    print("Pages:", len(document.pages))
    print("Tables detected:", len(document.tables))
    print("Nonempty tables:", len(nonempty_tables))
    print("Pictures:", len(document.pictures))
    print("Saved to:", output_dir)

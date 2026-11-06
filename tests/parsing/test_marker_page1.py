"""Parse only the first page of test.pdf using marker and print markdown.

Run with:  conda run -n paperindex python test_marker_page1.py
"""
import os
from pathlib import Path

from marker.converters.pdf import PdfConverter
from marker.config.parser import ConfigParser
from marker.models import create_model_dict
from marker.output import text_from_rendered

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = str(PROJECT_ROOT / "test.pdf")


def main():
    assert Path(PDF_PATH).exists(), f"PDF not found: {PDF_PATH}"

    print("Loading marker models...")
    artifact_dict = create_model_dict()

    config = {
        "output_format": "markdown",
        "use_llm": False,
        "page_range": "2",
    }
    config_parser = ConfigParser(config)

    converter = PdfConverter(
        config=config_parser.generate_config_dict(),
        artifact_dict=artifact_dict,
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
    )

    print(f"Converting first page of {PDF_PATH} ...")
    rendered = converter(PDF_PATH)
    text, ext, images = text_from_rendered(rendered)

    output_dir = "extracted_images"
    os.makedirs(output_dir, exist_ok=True)

    for filename, image in images.items():
        save_path = os.path.join(output_dir, filename)
        image.save(save_path)
        print(f"Saved {save_path}")

    print(f"\n{'=' * 60}")
    print(f"Output format: {ext}")
    print(f"Images extracted: {len(images)}")
    print(images)
    print(f"{'=' * 60}\n")
    print(text)


if __name__ == "__main__":
    main()

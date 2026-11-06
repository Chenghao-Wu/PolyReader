"""PaperIndex robustness: image path security, page ranges, atomic save/load."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from polyreader.indexing import (
    IndexLoadError,
    PaperIndex,
    load_index,
    save_index,
    verify_index,
)
from polyreader.indexing.metadata.metadata_extractor import PaperMetadata
from polyreader.utils.json_utils import write_json_atomic

PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
    b"\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xf8\xcf\xc0\x00\x00\x03\x01\x01\x00\xc9\xfe\x92"
    b"\xef\x00\x00\x00\x00IEND\xaeB`\x82"
)


@pytest.fixture
def image_index(tmp_path: Path) -> PaperIndex:
    images = tmp_path / "images_paper"
    images.mkdir()
    (images / "_page_0_Figure_1.png").write_bytes(PNG)
    (images / "notes.txt").write_text("not an image")
    (images / "subdir").mkdir()
    (tmp_path / "secret.png").write_bytes(PNG)  # outside images_dir
    return PaperIndex(images_dir=str(images), page_texts=["p1"])


class TestImagePathSecurity:
    def test_plain_filename(self, image_index):
        assert image_index.get_page_image(img_path="_page_0_Figure_1.png")["success"] is True

    def test_markdown_reference_and_prefix(self, image_index):
        assert image_index.get_page_image(img_path="![](_page_0_Figure_1.png)")["success"] is True
        # LLM prepends the saved directory name
        assert image_index.get_page_image(img_path="images_paper/_page_0_Figure_1.png")["success"] is True

    @pytest.mark.parametrize("bad", ["../secret.png", "../../etc/passwd", "/etc/passwd", "subdir/../../secret.png"])
    def test_traversal_rejected(self, image_index, bad):
        result = image_index.get_page_image(img_path=bad)
        assert result["success"] is False
        assert "image_base64" not in result

    def test_absolute_path_inside_dir_still_served_by_name(self, image_index):
        full = str(Path(image_index.images_dir) / "_page_0_Figure_1.png")
        assert image_index.get_page_image(img_path=full)["success"] is True

    def test_symlink_escape_rejected(self, image_index, tmp_path):
        link = Path(image_index.images_dir) / "link.png"
        try:
            os.symlink(tmp_path / "secret.png", link)
        except OSError:
            pytest.skip("symlinks not supported")
        assert image_index.get_page_image(img_path="link.png")["success"] is False

    @pytest.mark.parametrize("bad", ["notes.txt", "subdir", "", "a\x00b.png", "missing.png"])
    def test_malformed_or_missing(self, image_index, bad):
        assert image_index.get_page_image(img_path=bad)["success"] is False

    def test_non_string_path(self, image_index):
        assert image_index.get_page_image(img_path=["x.png"])["success"] is False


class TestPageRanges:
    @pytest.fixture
    def index(self):
        return PaperIndex(page_texts=[f"page {i}" for i in range(1, 11)])

    def test_huge_range_is_clamped_quickly(self, index):
        index.get_page_content(pages="1")  # warm up the tokenizer (first use may download)
        start = time.perf_counter()
        result = index.get_page_content(pages="1-100000000")
        assert time.perf_counter() - start < 5.0  # expanding 10^8 pages would take far longer
        assert result["success"] is True
        assert result["returned_pages"] == "1-10"
        assert "out of range" in result["next_steps"]["summary"]

    def test_en_dash_and_reversed(self, index):
        assert index.get_page_content(pages="2–4")["returned_pages"] == "2-4"
        assert index.get_page_content(pages="4-2")["returned_pages"] == "2-4"

    def test_list_of_strings(self, index):
        assert index.get_page_content(pages=["1", "3"])["returned_pages"] == "1,3"

    def test_unchanged_formats(self, index):
        result = index.get_page_content(pages="3,1,2,12")
        assert result["requested_pages"] == "1-3,12"
        assert result["returned_pages"] == "1-3"
        assert "Pages 12 out of range (1-10)" in result["next_steps"]["summary"]

    @pytest.mark.parametrize("bad", ["S3", "abc", True, [], "", None])
    def test_invalid(self, index, bad):
        assert index.get_page_content(pages=bad)["success"] is False


def _sample_index(tmp_path: Path) -> PaperIndex:
    images = tmp_path / "parse" / "images"
    images.mkdir(parents=True)
    (images / "a.png").write_bytes(PNG)
    return PaperIndex(
        tree=[{"title": "Intro", "start_index": 1, "end_index": 2, "node_id": "0000"}],
        metadata=PaperMetadata(doi="10.1/x", title="T"),
        pdf_path="/tmp/x.pdf",
        images_dir=str(images),
        mode_used="mineru",
        doc_name="x.pdf",
        page_texts=["one ![](a.png)", "two"],
        parser_info={"backend": "mineru", "version": "4.1.4", "tier": "standard"},
    )


class TestSaveLoad:
    def test_roundtrip_records_parser(self, tmp_path):
        out = tmp_path / "out"
        save_index(_sample_index(tmp_path), str(out / "paper.json"))
        tree = json.loads((out / "paper_tree.json").read_text())
        assert tree["parser"]["backend"] == "mineru"
        assert "mode_used" not in tree
        loaded = load_index(str(out / "paper_tree.json"))
        assert loaded.mode_used == "mineru"
        assert loaded.parser_info["version"] == "4.1.4"
        assert loaded.get_page_content(pages="1")["content"][0]["text"].startswith("one")
        assert loaded.get_page_image(img_path="a.png")["success"] is True
        assert verify_index(str(out / "paper_tree.json")) == []
        assert not list(out.glob("*.tmp")) and not list(out.glob(".*.tmp"))

    def test_resave_replaces_images(self, tmp_path):
        out = tmp_path / "out"
        save_index(_sample_index(tmp_path), str(out / "paper.json"))
        second = _sample_index(tmp_path / "again")
        save_index(second, str(out / "paper.json"))
        assert (out / "images_paper" / "a.png").exists()
        assert not (out / "images_paper.incoming").exists()

    def test_missing_tree(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_index(str(tmp_path / "nope_tree.json"))
        assert verify_index(str(tmp_path / "nope_tree.json"))

    def test_corrupted_tree(self, tmp_path):
        bad = tmp_path / "bad_tree.json"
        bad.write_text('{"version": 2, "tree": [')
        with pytest.raises(IndexLoadError):
            load_index(str(bad))
        assert "Corrupted" in verify_index(str(bad))[0]

    def test_corrupted_pages_reported_not_raised(self, tmp_path):
        out = tmp_path / "out"
        save_index(_sample_index(tmp_path), str(out / "paper.json"))
        (out / "paper_page_contents.json").write_text("{trunc")
        loaded = load_index(str(out / "paper_tree.json"))
        result = loaded.get_page_content(pages="1")
        assert result["success"] is False and "Corrupted" in result["error"]
        assert verify_index(str(out / "paper_tree.json"))

    def test_page_count_mismatch_detected(self, tmp_path):
        out = tmp_path / "out"
        save_index(_sample_index(tmp_path), str(out / "paper.json"))
        write_json_atomic(out / "paper_page_contents.json",
                          {"total_pages": 1, "page_contents": [{"page": 1, "text": "x"}]})
        assert any("mismatch" in p for p in verify_index(str(out / "paper_tree.json")))

    def test_old_index_without_parser_still_loads(self, tmp_path):
        out = tmp_path / "out"
        save_index(_sample_index(tmp_path), str(out / "paper.json"))
        tree = json.loads((out / "paper_tree.json").read_text())
        del tree["parser"]
        (out / "paper_tree.json").write_text(json.dumps(tree))
        loaded = load_index(str(out / "paper_tree.json"))
        assert loaded.mode_used == "" and loaded.parser_info == {}


class TestAtomicWrite:
    def test_failed_write_keeps_previous_file(self, tmp_path):
        target = tmp_path / "data.json"
        write_json_atomic(target, {"ok": 1})
        with pytest.raises(TypeError):
            write_json_atomic(target, {"bad": object()})
        assert json.loads(target.read_text()) == {"ok": 1}
        assert [p.name for p in tmp_path.iterdir()] == ["data.json"]

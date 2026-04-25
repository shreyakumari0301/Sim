from __future__ import annotations

from pathlib import Path

from world_model.generate_dataset import load_drug_names_from_csv, load_drug_names_from_file


def test_load_drug_names_from_file_dedup_and_comments(tmp_path: Path) -> None:
    p = tmp_path / "drugs.txt"
    p.write_text("foo\n# ignored\nfoo\nbar\n", encoding="utf-8")
    assert load_drug_names_from_file(p) == ["foo", "bar"]
    assert load_drug_names_from_file(p, max_drugs=1) == ["foo"]


def test_load_drug_names_from_csv_order_and_unique(tmp_path: Path) -> None:
    p = tmp_path / "t.csv"
    p.write_text("name,dose\naspirin,100\nibuprofen,200\naspirin,50\n", encoding="utf-8")
    assert load_drug_names_from_csv(p, "name") == ["aspirin", "ibuprofen"]

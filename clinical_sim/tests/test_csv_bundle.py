from pathlib import Path

from csv_bundle import build_text_bundle, drugbank_text_for_drug


def test_build_text_bundle_matches_drug(tmp_path: Path) -> None:
    ncbi = tmp_path / "ncbi_data.csv"
    ncbi.write_text(
        "pmid,title,abstract,authors,journal,pub_date,doi,drug_name\n"
        "1,Aspirin study,Abstract about aspirin mechanism.,Smith J,J Clin Pharm,2020-01-01,10.1000/x,aspirin\n",
        encoding="utf-8",
    )
    openfda = tmp_path / "openfda_v1.csv"
    openfda.write_text(
        "drug_name_clean,warnings,mechanism_of_action\n"
        "aspirin,Do not co-administer with X.,COX inhibition.\n",
        encoding="utf-8",
    )
    db = tmp_path / "drugbank.csv"
    db.write_text(
        "drug_id,secondary_ids,name,type,description,indication,state,cas,status,targets,interactions\n"
        'DB00945,BTD1,"Aspirin, Acetylsalicylic acid",small molecule,Desc,Ind,liquid,50-78-2,approved,Targ,Int\n',
        encoding="utf-8",
    )

    p, o, d = build_text_bundle(
        "aspirin",
        openfda_csv=openfda,
        ncbi_csv=ncbi,
        drugbank_csv=db,
    )
    assert "Aspirin study" in p and "Smith J" in p
    assert "COX inhibition" in o
    assert "DB00945" in d and "BTD1" in d and "50-78-2" in d


def test_drugbank_aspirin_matches_inn_only(tmp_path: Path) -> None:
    """INN-only row (acetylsalicylic acid) matches query ``aspirin`` via synonyms."""
    db = tmp_path / "drugbank.csv"
    db.write_text(
        "drug_id,secondary_ids,name,type,description,indication,state,cas,status,targets,interactions\n"
        "DB1,,Acetylsalicylic acid,small molecule,,,,,,,\n",
        encoding="utf-8",
    )
    text = drugbank_text_for_drug(db, "aspirin")
    assert "DB1" in text and "Acetylsalicylic acid" in text


def test_drugbank_prefers_richest_row(tmp_path: Path) -> None:
    """When multiple rows match, use the one with more non-empty text."""
    db = tmp_path / "drugbank.csv"
    db.write_text(
        "drug_id,secondary_ids,name,type,description,indication,state,cas,status,targets,interactions\n"
        "DB0,,Aspirin,small molecule,,,,,,,\n"
        'DB9,,Aspirin,small molecule,Long description text here.,Indication text,,,,,\n',
        encoding="utf-8",
    )
    text = drugbank_text_for_drug(db, "aspirin")
    assert "DB9" in text and "Long description" in text


def test_drugbank_prefers_id_when_explicit(tmp_path: Path) -> None:
    """``drugbank_id=...`` picks that row so a sparse earlier ``name`` hit does not win."""
    db = tmp_path / "drugbank.csv"
    db.write_text(
        "drug_id,secondary_ids,name,type,description,indication,state,cas,status,targets,interactions\n"
        "DB0,,Aspirin,small molecule,,,,,,,\n"
        'DB00945,BTD1,"Aspirin, Acetylsalicylic acid",small molecule,Desc,Ind,liquid,50-78-2,approved,Targ,Int\n',
        encoding="utf-8",
    )
    text = drugbank_text_for_drug(db, "aspirin", drugbank_id="DB00945")
    assert "DB00945" in text and "Desc" in text and "BTD1" in text


def test_drugbank_utf8_bom_header(tmp_path: Path) -> None:
    """BOM before ``drug_id`` must not break ``DictReader`` keys (``utf-8-sig``)."""
    db = tmp_path / "drugbank.csv"
    db.write_text(
        "\ufeffdrug_id,secondary_ids,name,type,description,indication,state,cas,status,targets,interactions\n"
        "DB00945,,Aspirin,small molecule,Desc,,,,,,\n",
        encoding="utf-8",
    )
    text = drugbank_text_for_drug(db, "aspirin")
    assert "DB00945" in text and "Desc" in text


def test_drugbank_word_boundary_matches_salt_name(tmp_path: Path) -> None:
    """``ibuprofen`` matches ``Ibuprofen sodium`` (word boundary), not only exact ``name``."""
    db = tmp_path / "drugbank.csv"
    db.write_text(
        "drug_id,secondary_ids,name,type,description,indication,state,cas,status,targets,interactions\n"
        "DB01050,,Ibuprofen sodium,small molecule,COX inhibitor,,,,,,\n",
        encoding="utf-8",
    )
    text = drugbank_text_for_drug(db, "ibuprofen")
    assert "DB01050" in text and "Ibuprofen sodium" in text


def test_drugbank_word_boundary_no_false_positive_nitroaspirin(tmp_path: Path) -> None:
    """``aspirin`` must not match inside ``Nitroaspirin``."""
    db = tmp_path / "drugbank.csv"
    db.write_text(
        "drug_id,secondary_ids,name,type,description,indication,state,cas,status,targets,interactions\n"
        "DBX,,Nitroaspirin,small molecule,Some text,,,,,,\n",
        encoding="utf-8",
    )
    assert drugbank_text_for_drug(db, "aspirin") == ""


def test_drugbank_matches_description_when_name_empty(tmp_path: Path) -> None:
    db = tmp_path / "drugbank.csv"
    db.write_text(
        "drug_id,secondary_ids,name,type,description,indication,state,cas,status,targets,interactions\n"
        "DBX,,,small molecule,Ibuprofen is a nonsteroidal anti-inflammatory.,,,,,,\n",
        encoding="utf-8",
    )
    text = drugbank_text_for_drug(db, "ibuprofen")
    assert "DBX" in text and "nonsteroidal" in text


def test_drugbank_id_env_override(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DRUGBANK_ID", "DBX")
    db = tmp_path / "drugbank.csv"
    db.write_text(
        "drug_id,secondary_ids,name,type,description,indication,state,cas,status,targets,interactions\n"
        "DBX,,OtherDrug,small molecule,From env,,,,,,\n"
        "DB00945,,Aspirin,small molecule,Wrong pick,,,,,,\n",
        encoding="utf-8",
    )
    text = drugbank_text_for_drug(db, "metformin")  # name would not match; ID drives selection
    assert "DBX" in text and "From env" in text
    monkeypatch.delenv("DRUGBANK_ID", raising=False)

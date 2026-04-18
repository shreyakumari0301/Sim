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

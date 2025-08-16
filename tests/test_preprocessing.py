from src.preprocessing import clean_text, split_multilabel

def test_clean_text():
    t = "EGFR+ Stage IV NSCLC"
    out = clean_text(t)
    assert "egfr" in out and "stage iv" in out and "non small cell lung cancer" in out

def test_split_multilabel():
    assert split_multilabel("A;B; C") == ["A","B","C"]

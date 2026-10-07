import pytest

from src.preprocessing import InvalidInput, detect_script, normalize_text
from src.langid import detect_language


def test_normalize_keeps_code_mixing_digits_and_scripts():
    s = normalize_text("  Bhai   road ki light 3 din se band hai \n")
    assert s == "Bhai road ki light 3 din se band hai"
    assert normalize_text("सुबह से पानी नहीं आया है।") == "सुबह से पानी नहीं आया है।"
    assert "3" in s


def test_normalize_unicode_nfc():
    decomposed = "é"  # e + combining acute
    assert normalize_text(decomposed + "x") == "éx"


@pytest.mark.parametrize("bad", ["", "   ", "\n\t", "!!!", None, 5])
def test_invalid_inputs_raise(bad):
    with pytest.raises(InvalidInput):
        normalize_text(bad)


def test_truncates_very_long_input():
    assert len(normalize_text("a " * 5000, max_chars=100)) == 100


def test_zwj_preserved():
    assert "‍" in normalize_text("क‍ष")


@pytest.mark.parametrize("text,script", [
    ("Subah se paani nahi aaya", "Latn"), ("सुबह से पानी नहीं आया है", "Deva"), ("मेरा card block हो गया", "Deva+Latn"),
    ("ನನ್ನ ಕಾರ್ಡ್", "Knda"), ("12345", "unknown"), ("தமிழ் text", "mixed")])
def test_detect_script(text, script):
    assert detect_script(text) == script


def test_detect_language_non_latin_rules():
    assert detect_language("सुबह से पानी नहीं आया है")[0] == "hi-Deva"
    assert detect_language("ನನ್ನ ಕಾರ್ಡ್ ಬಂದಿಲ್ಲ")[0] == "kn-Knda"
    assert detect_language("मेरा card block हो गया")[0] == "hi-mixed"
    assert detect_language("hello world")[0] in {"en-Latn", "hi-Latn", "und-Latn"}

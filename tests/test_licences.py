import pytest

from jeff.licences import LicenceError, check, is_allowed


def test_permissive_licences_allowed():
    for lic in ["apache-2.0", "mit", "cc0-1.0", "cc-by-4.0", "cc-by-sa-3.0", "oanc-mixed"]:
        assert is_allowed(lic)


def test_noncommercial_and_unknown_refused():
    for lic in ["cc-by-nc-4.0", "cc-by-nc-sa-4.0", "gpl-3.0", "other", "unknown", ""]:
        assert not is_allowed(lic)


def test_check_passes_clean_train_split():
    check("legacy-datasets/banking77", "cc-by-4.0")


@pytest.mark.parametrize(
    "hf_id", ["facebook/anli", "choyiny/decidebench", "karanxa/agent-action-safety-dataset", "SetFit/sst5"]
)
def test_forbidden_datasets_refused_even_with_ok_licence(hf_id):
    with pytest.raises(LicenceError, match="forbidden"):
        check(hf_id, "mit")


def test_forbidden_config_refused():
    with pytest.raises(LicenceError, match="forbidden"):
        check("montehoover/DynaBench", "mit", config="DynaBenchSafetyMix")
    check("montehoover/DynaBench", "mit", config="DynaBenchTrain")


@pytest.mark.parametrize("split", ["test", "validation", "dev", "validation_matched", "test[:100]"])
def test_non_train_splits_refused(split):
    with pytest.raises(LicenceError, match="split"):
        check("legacy-datasets/banking77", "cc-by-4.0", split=split)


def test_sliced_train_split_allowed():
    check("google/civil_comments", "cc0-1.0", split="train[:300000]")


def test_bad_licence_refused():
    with pytest.raises(LicenceError, match="licence"):
        check("some/dataset", "cc-by-nc-4.0")

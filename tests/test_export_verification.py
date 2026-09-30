from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

from finsight_ashare.common import digest

spec = importlib.util.spec_from_file_location("verify_server_export", Path(__file__).resolve().parents[1] / "scripts/verify_server_export.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_provenance_legacy_reconstruction_requires_exact_original_hash():
    base = {"save_steps": 50}
    experiment = {"training": deepcopy(base), "pilot": True}
    receipt = {"experiment_digest": digest(experiment)}
    provenance = {**experiment, **receipt, "status": "running", "created": "test"}
    runtime = {"save_steps": 20, "max_steps": 20}
    assert module.verify_provenance(provenance, receipt, base, runtime) == "exact"
    provenance["training"] = deepcopy(runtime)
    assert module.verify_provenance(provenance, receipt, base, runtime).startswith("legacy_")
    with pytest.raises(ValueError, match="cannot reproduce"):
        module.verify_provenance(provenance, receipt, {"save_steps": 49}, runtime)
    provenance["training"]["max_steps"] = 21
    with pytest.raises(ValueError, match="Unrecognized"):
        module.verify_provenance(provenance, receipt, base, runtime)

from types import SimpleNamespace
import sys

import pytest

from finsight_ashare import server
from finsight_ashare.common import digest, read_json, write_json
from finsight_ashare.dataset import build


def test_pilot_cannot_start_full_training(project):
    with pytest.raises(ValueError, match="only allowed with --probe"):
        server.train(project, pilot=True)


def install_template(monkeypatch, tokens):
    template = SimpleNamespace(fix_special_tokens=lambda t: None,
                               encode_multiturn=lambda *a, **k: [([1] * tokens, [2])])
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: object())))
    monkeypatch.setitem(sys.modules, "llamafactory.data.template", SimpleNamespace(TEMPLATES={"qwen3_5_nothink": template}))


def test_pilot_token_limit_checked_before_gpu_launch(project, monkeypatch):
    build(project, pilot=True)
    install_template(monkeypatch, 4096)
    with pytest.raises(ValueError, match="refusing training-time truncation"):
        server.train(project, probe=True, pilot=True)
    assert not (project / "outputs/training").exists()


@pytest.mark.parametrize("step,loss", [(20, 1.0), (19, 1.0), (20, float("nan"))])
def test_pilot_probe_separate_dataset_and_receipt(project, monkeypatch, step, loss):
    build(project, pilot=True)
    install_template(monkeypatch, 200)
    write_json(project / "configs/server-profile.json", {})
    (project / "configs/lora.yaml").write_text("num_train_epochs: 2\ncutoff_len: 4096\n", encoding="utf-8")
    write_json(project / "models/snapshots.json", {})
    monkeypatch.setattr(server.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="4\n"))
    monkeypatch.setitem(sys.modules, "finsight_ashare.models", SimpleNamespace(
        LocalModel=lambda *a: SimpleNamespace(generate=lambda *a: "reload response")))
    def mock_train(command, root, env):
        import yaml
        generated = yaml.safe_load(command[-1].read_text(encoding="utf-8"))
        assert generated["dataset_dir"] == str(project / "data/pilot")
        assert generated["max_steps"] == 20
        assert env["NPROC_PER_NODE"] == "4"
        output = command[-1].parent
        write_json(output / "adapter_config.json", {})
        (output / "adapter_model.safetensors").write_bytes(b"mock adapter")
        write_json(output / "trainer_state.json", {"global_step": step, "log_history": [{"loss": loss}]})
    monkeypatch.setattr(server, "run_command", mock_train)
    if step != 20 or loss != loss:
        with pytest.raises(ValueError, match="optimizer steps|non-finite"):
            server.train(project, probe=True, pilot=True)
        assert not (project / "outputs/training/pilot-probe.json").exists()
        return
    server.train(project, probe=True, pilot=True)
    receipt = read_json(project / "outputs/training/pilot-probe.json")
    assert receipt["pilot"] is True and receipt["global_step"] == 20
    provenance = read_json(project / receipt["adapter"] / "provenance.json")
    experiment = {k: v for k, v in provenance.items() if k not in {"experiment_digest", "status", "created"}}
    assert digest(experiment) == receipt["experiment_digest"] == provenance["experiment_digest"]
    assert "max_steps" not in experiment["training"]
    assert not (project / "outputs/training/probe.json").exists()
    assert not (project / "outputs/training/production.json").exists()
    with pytest.raises(FileNotFoundError):
        server.train(project)


def test_pilot_receipt_does_not_unlock_formal_training(project, monkeypatch):
    from finsight_ashare import dataset
    monkeypatch.setattr(dataset, "validate_freeze", lambda *a, **k: {"pilot": False})
    write_json(project / "configs/server-profile.json", {})
    (project / "configs/lora.yaml").write_text("num_train_epochs: 2\ncutoff_len: 4096\n", encoding="utf-8")
    write_json(project / "models/snapshots.json", {})
    write_json(project / "outputs/training/pilot-probe.json", {"status": "completed", "pilot": True})
    with pytest.raises(ValueError, match="Run train --probe first"):
        server.train(project)

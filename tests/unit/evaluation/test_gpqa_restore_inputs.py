"""Restore safety and original-identity checks with an explicit fake Drive SDK."""
from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation.supplemental_gpqa import SOURCE_REPOSITORY, SOURCE_REVISION, prepare_gpqa_panel
from operations.drive import DriveStore, INTENDED_ACCOUNT, PROJECT_TITLE
from shared_procedures.identity import canonical_json, content_hash
from shared_procedures.storage import read_json
from training.sft.individual_fit.sft_fit import TARGET_MODULES

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/restore_gpqa_sft_inputs.py"
SPEC = importlib.util.spec_from_file_location("restore_gpqa_sft_inputs_fixture", SCRIPT)
restore = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(restore)


def sealed(value, field):
    value[field] = content_hash({key: item for key, item in value.items() if key != field})
    return value


class Request:
    def __init__(self, value):
        self.value = value

    def execute(self):
        return deepcopy(self.value())


class FakeFiles:
    def __init__(self, fixture):
        self.fixture = fixture

    def get(self, *, fileId, fields):
        return Request(lambda: self.fixture.remote[fileId])

    def list(self, *, q, **kwargs):
        def matching():
            records = [record for record in self.fixture.remote.values()
                       if record.get("appProperties", {}).get("artifact_id")
                       and record["appProperties"]["artifact_id"] in q
                       and record["appProperties"]["project_id"] in q]
            return {"files": records}
        return Request(matching)


class FakeStore(DriveStore):
    def __init__(self, fixture, state_root):
        self.fixture = fixture
        service = SimpleNamespace(files=lambda: FakeFiles(fixture),
                                  about=lambda: SimpleNamespace(get=lambda **kwargs: Request(
                                      lambda: {"user": {"emailAddress": INTENDED_ACCOUNT}, "storageQuota": {}})))
        super().__init__(service, restore.PROJECT_FOLDER_ID, state_root=state_root)

    def download_artifact(self, artifact_id, destination, *, config_hash=None):
        self.fixture.downloaded.append(artifact_id)
        remote = self.resolve_artifact_id(artifact_id)
        assert remote["appProperties"]["config_hash"] == config_hash
        Path(destination).write_bytes(self.fixture.payloads[artifact_id])
        if self.fixture.after_download:
            self.fixture.after_download(remote["id"])
        return Path(destination)


class Fixture:
    def __init__(self, tmp_path):
        self.tmp_path, self.payloads, self.remote, self.downloaded = tmp_path, {}, {}, []
        self.after_download = None
        self.remote[restore.PROJECT_FOLDER_ID] = {
            "id": restore.PROJECT_FOLDER_ID, "name": PROJECT_TITLE,
            "mimeType": "application/vnd.google-apps.folder", "parents": ["root"], "trashed": False}
        self.remote["artifact_folder"] = {
            "id": "artifact_folder", "mimeType": "application/vnd.google-apps.folder",
            "parents": [restore.PROJECT_FOLDER_ID], "trashed": False}
        self.inventory = {"version": restore.VERSION, "project_folder_id": restore.PROJECT_FOLDER_ID,
                          "family": "rot13", "model": "14b", "generation_profile": "long_32000",
                          "artifacts": []}
        for level in range(1, 11):
            prefix = f"training/sft/rot13/14b/level_{level:02d}"
            configuration_hash = content_hash({"original_training_configuration": "old" if level == 10 else "new"})
            identity = {"family": "rot13", "model_id": "14b", "level": level,
                        "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
                        "model_revision": restore.MODEL_REVISION, "tokenizer_revision": restore.MODEL_REVISION,
                        "configuration_hash": configuration_hash, "execution_mode": "production",
                        "source_code_commit": "a" * 40 if level == 10 else "b" * 40,
                        "original_recipe": "preserved"}
            config = {"r": 8, "lora_alpha": 16, "lora_dropout": .05,
                      "target_modules": list(TARGET_MODULES), "base_model_name_or_path": identity["model_name"]}
            payloads = {"adapter_config.json": canonical_json(config) + b"\n",
                        "adapter_model.safetensors": b"CPU-only fake weights, never scientific results",
                        "README.md": b"CPU fixture adapter"}
            manifest = sealed({"identity": identity, "checkpoint_hash": content_hash({"level": level}),
                               "files": {name: hashlib.sha256(value).hexdigest()
                                         for name, value in payloads.items()}}, "manifest_hash")
            payloads["manifest.json"] = canonical_json(manifest) + b"\n"
            for name, payload in payloads.items():
                self.add(prefix + "/final_adapter/" + name, payload, configuration_hash)
            fit = {"status": "complete", "identity": identity, "adapter_manifest_hash": manifest["manifest_hash"],
                   "checkpoint_hash": manifest["checkpoint_hash"]}
            self.add(prefix + "/fit_result.json", canonical_json(fit) + b"\n", configuration_hash)
        rows = [{"Question": f"CPU fixture question {index}", "Correct Answer": f"correct {index}",
                 "Incorrect Answer 1": "wrong one", "Incorrect Answer 2": "wrong two",
                 "Incorrect Answer 3": "wrong three"} for index in range(198)]
        panel = prepare_gpqa_panel(rows, variant="gpqa_diamond", source={
            "repo_id": SOURCE_REPOSITORY, "revision": SOURCE_REVISION, "filename": "gpqa_diamond.csv",
            "csv_sha256": "c" * 64, "asset_manifest_hash": "d" * 64})
        self.add(restore.PANEL_PATH, canonical_json(panel) + b"\n", "e" * 64)
        sealed(self.inventory, "inventory_hash")

    def add(self, path, payload, configuration_hash):
        artifact_id = content_hash({"path": path})
        file_id = "file_" + str(len(self.inventory["artifacts"]))
        record = {"relative_path": path, "artifact_id": artifact_id, "configuration_hash": configuration_hash,
                  "sha256": hashlib.sha256(payload).hexdigest(), "md5": hashlib.md5(payload).hexdigest(),
                  "size": len(payload), "drive_file_id": file_id, "drive_folder_id": "artifact_folder"}
        self.inventory["artifacts"].append(record)
        self.payloads[artifact_id] = payload
        self.remote[file_id] = {"id": file_id, "size": str(len(payload)), "md5Checksum": record["md5"],
                                "parents": ["artifact_folder"], "trashed": False,
                                "appProperties": {"artifact_id": artifact_id, "sha256": record["sha256"],
                                                  "config_hash": configuration_hash, "upload_status": "verified",
                                                  "project_id": restore.PROJECT_FOLDER_ID}}

    def factory(self):
        return FakeStore(self, self.tmp_path / "fake_drive_state")

    def write_local(self, destination):
        for record in self.inventory["artifacts"]:
            path = destination / record["relative_path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(self.payloads[record["artifact_id"]])


@pytest.fixture
def fixture(tmp_path):
    return Fixture(tmp_path)


def test_restore_and_resume_preserve_original_training_identities(fixture, tmp_path):
    root = tmp_path / "restored"
    result = restore.restore_inventory(fixture.inventory, root, store_factory=fixture.factory, workers=4)
    assert result["status"] == "VERIFIED" and result["downloaded"] == 51 and result["reused"] == 0
    manifest = read_json(root / "gpqa_sft_adapters_manifest.json")
    assert [entry["level"] for entry in manifest["adapters"]] == list(range(1, 11))
    level10 = root / "training/sft/rot13/14b/level_10/final_adapter/manifest.json"
    assert read_json(level10)["identity"]["source_code_commit"] == "a" * 40
    assert read_json(level10)["identity"]["configuration_hash"] != read_json(
        root / "training/sft/rot13/14b/level_01/final_adapter/manifest.json")["identity"]["configuration_hash"]
    before = (root / "gpqa_sft_restore_receipt.json").read_bytes()
    second = restore.restore_inventory(fixture.inventory, root, store_factory=fixture.factory)
    assert second["downloaded"] == 0 and second["reused"] == 51 and len(fixture.downloaded) == 51
    assert (root / "gpqa_sft_restore_receipt.json").read_bytes() == before


def test_validate_only_uses_no_drive_or_writes(fixture, tmp_path):
    root = tmp_path / "local"
    fixture.write_local(root)
    result = restore.restore_inventory(fixture.inventory, root, validate_only=True,
                                      store_factory=lambda: pytest.fail("No Drive contact expected"))
    assert result["status"] == "VALIDATED"
    assert not (root / "gpqa_sft_restore_receipt.json").exists()
    assert not (root / "gpqa_sft_adapters_manifest.json").exists()


@pytest.mark.parametrize("path", ["../escape", "/absolute", "evaluation/../gpqa_diamond_panel.json",
                                 "training/sft/kazitext/14b/level_01/fit_result.json",
                                 "training/sft/rot13/14b/level_11/fit_result.json",
                                 "training/sft/rot13/14b/level_01/checkpoint_000625/full_state.pt",
                                 "training/sft/rot13/14b/level_01/final_adapter/optimizer.pt",
                                 "training/sft/rot13/14b/level_01/final_adapter/nested/file.json"])
def test_unsafe_or_other_state_paths_rejected_before_drive(fixture, tmp_path, path):
    inventory = deepcopy(fixture.inventory)
    inventory["artifacts"][0]["relative_path"] = path
    sealed(inventory, "inventory_hash")
    with pytest.raises(ValueError, match="allowlist"):
        restore.restore_inventory(inventory, tmp_path / "dest", store_factory=fixture.factory)
    assert not fixture.downloaded


def test_wrong_project_folder_rejected_before_network(fixture, tmp_path):
    inventory = deepcopy(fixture.inventory)
    inventory["project_folder_id"] = "unrelated_drive_folder"
    sealed(inventory, "inventory_hash")
    with pytest.raises(ValueError, match="identity or seal"):
        restore.restore_inventory(inventory, tmp_path / "dest", store_factory=fixture.factory)
    assert not fixture.downloaded


def test_wrong_remote_folder_ancestry_rejected(fixture, tmp_path):
    fixture.remote["artifact_folder"]["parents"] = ["root"]
    with pytest.raises(RuntimeError, match="outside the canonical project"):
        restore.restore_inventory(fixture.inventory, tmp_path / "dest", store_factory=fixture.factory)
    assert not fixture.downloaded


@pytest.mark.parametrize("key,value", [("upload_status", "pending"), ("sha256", "f" * 64),
                                       ("config_hash", "f" * 64), ("project_id", "unrelated"),
                                       ("artifact_id", "f" * 64)])
def test_remote_publication_and_provenance_tamper_rejected(fixture, tmp_path, key, value):
    first = fixture.inventory["artifacts"][0]
    fixture.remote[first["drive_file_id"]]["appProperties"][key] = value
    with pytest.raises((RuntimeError, ValueError)):
        restore.restore_inventory(fixture.inventory, tmp_path / "dest", store_factory=fixture.factory)
    assert not fixture.downloaded


def test_remote_tamper_during_transfer_never_publishes(fixture, tmp_path):
    first = fixture.inventory["artifacts"][0]
    fixture.after_download = lambda file_id: fixture.remote[file_id]["appProperties"].update(sha256="f" * 64)
    root = tmp_path / "dest"
    with pytest.raises(RuntimeError, match="integrity/provenance"):
        restore.restore_inventory(fixture.inventory, root, store_factory=fixture.factory)
    assert not (root / first["relative_path"]).exists()


@pytest.mark.parametrize("mode", ["root", "ancestor", "file"])
def test_symlink_paths_rejected_without_overwrite(fixture, tmp_path, mode):
    root = tmp_path / "dest"
    outside = tmp_path / "outside"
    outside.mkdir()
    if mode == "root":
        root.symlink_to(outside, target_is_directory=True)
    elif mode == "ancestor":
        root.mkdir()
        (root / "training").symlink_to(outside, target_is_directory=True)
    else:
        path = root / fixture.inventory["artifacts"][0]["relative_path"]
        path.parent.mkdir(parents=True)
        (outside / "victim").write_bytes(b"unchanged")
        path.symlink_to(outside / "victim")
    with pytest.raises(ValueError, match="symbolic links"):
        restore.restore_inventory(fixture.inventory, root, store_factory=fixture.factory)
    assert not fixture.downloaded
    if mode == "file":
        assert (outside / "victim").read_bytes() == b"unchanged"


def test_different_existing_file_is_preserved_before_any_network(fixture, tmp_path):
    root = tmp_path / "dest"
    path = root / fixture.inventory["artifacts"][0]["relative_path"]
    path.parent.mkdir(parents=True)
    path.write_bytes(b"existing unrelated bytes")
    with pytest.raises(ValueError, match="integrity"):
        restore.restore_inventory(fixture.inventory, root, store_factory=fixture.factory)
    assert path.read_bytes() == b"existing unrelated bytes" and not fixture.downloaded


def test_inventory_tamper_duplicate_and_incomplete_matrix_rejected(fixture):
    inventory = deepcopy(fixture.inventory)
    inventory["artifacts"][0]["size"] += 1
    with pytest.raises(ValueError, match="seal"):
        restore.validate_inventory(inventory)
    inventory = deepcopy(fixture.inventory)
    inventory["artifacts"].append(inventory["artifacts"][0])
    sealed(inventory, "inventory_hash")
    with pytest.raises(ValueError, match="duplicate"):
        restore.validate_inventory(inventory)
    inventory = deepcopy(fixture.inventory)
    inventory["artifacts"].pop()
    sealed(inventory, "inventory_hash")
    with pytest.raises(ValueError, match="all ten"):
        restore.validate_inventory(inventory)


def test_completed_fit_pairing_checked_after_exact_bytes(fixture, tmp_path):
    root = tmp_path / "dest"
    fixture.write_local(root)
    record = next(r for r in fixture.inventory["artifacts"] if r["relative_path"].endswith("level_10/fit_result.json"))
    path = root / record["relative_path"]
    result = read_json(path)
    result["status"] = "paused"
    payload = canonical_json(result) + b"\n"
    path.write_bytes(payload)
    record.update(size=len(payload), sha256=hashlib.sha256(payload).hexdigest(), md5=hashlib.md5(payload).hexdigest())
    sealed(fixture.inventory, "inventory_hash")
    with pytest.raises(ValueError, match="matching completed-fit"):
        restore.restore_inventory(fixture.inventory, root, validate_only=True)


def test_uninventoried_adapter_file_rejected(fixture, tmp_path):
    root = tmp_path / "dest"
    fixture.write_local(root)
    (root / "training/sft/rot13/14b/level_01/final_adapter/extra.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="uninventoried"):
        restore.restore_inventory(fixture.inventory, root, validate_only=True)

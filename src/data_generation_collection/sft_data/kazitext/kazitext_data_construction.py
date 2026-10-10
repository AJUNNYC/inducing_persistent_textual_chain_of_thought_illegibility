"""Pinned official KaziText with an exact numeric/mathematical protection layer.

Protected spans are copied untouched; official noising runs on each intervening
prose segment. This avoids corruptible sentinel tokens. RNG is isolated in a
PYTHONHASHSEED=0 worker, derived from global 42 and immutable model/example/level.
The same request produces identical acceptance/final-construction text even when
other candidates are rejected. All effective settings and asset hashes are kept.
"""
from __future__ import annotations

import atexit
import difflib
import hashlib
import json
import os
import re
import selectors
import subprocess
import sys
import tempfile
import threading
import unicodedata
from pathlib import Path

from shared_procedures.identity import content_hash, derive_seed
from shared_procedures.storage import atomic_write_json, notify_artifact

REVISION = "435f68748fdded6d79446b8ab6bdf76fbde88818"
PROFILE_PATH = "profiles/dev/en_clean_second_learners.json"
PROFILE_SHA256 = "8c723dd5925fc872b4ee8a5191f585dee95d1146e849fc80f5e84f3cd0f62d73"
UDPIPE_MODEL = "english-ewt-ud-2.5-191206.udpipe"
UDPIPE_SHA256 = "784bd0fa85e3d831fd02a55290d0acfd05c953159dc38cc33d52e1b28add9957"
UDPIPE_MD5 = "4284eb5c16348e716d739b8ecec7de6d"
UDPIPE_BYTES = 16309608
UDPIPE_METADATA_URL = "https://lindat.mff.cuni.cz/repository/server/api/core/bitstreams/8d51e3d3-1f87-47fe-8975-c197ddf7907b"
UDPIPE_URL = UDPIPE_METADATA_URL + "/content"
ALPHAS = ("0.200", "0.301", "0.454", "0.684", "1.031", "1.553", "2.339", "3.524", "5.310", "8.000")
SETTINGS = {"lang": "en", "alpha_std": 0, "alpha_uniformity_prob": 0, "beta": 0,
            "no_word_order": True, "no_error_sentence_boost": 0, "spelling_detailed_ratio": 0.3,
            "strip_all_diacritics": False, "no_diacritics": False, "no_spelling": False,
            "no_casing": False, "no_whitespace": False, "no_punctuation": False,
            "no_suffix_prefix": False, "no_common_other": False}
PROTECTION_VERSION = "math_prose_segments_v2"
_NUMBER = re.compile(r"(?<!\w)[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?%?(?!\w)")
_PATTERNS = [re.compile(p, re.DOTALL) for p in (
    r"\$\$.*?\$\$", r"(?<!\\)\$[^\n$]*?(?<!\\)\$", r"\\\(.*?\\\)", r"\\\[.*?\\\]",
    r"```(?:math|latex)\s.*?```", r"\\(?:begin\{[^}]+\}.*?\\end\{[^}]+\}|[A-Za-z]+(?:\{[^{}]*\})*)",
    r"\b(?:[A-Za-z]\w*|\d+(?:\.\d+)?)(?:\s*[=+*/^<>≤≥≠×÷−-]\s*(?:[A-Za-z]\w*|\d+(?:\.\d+)?))+",
    r"[=+*/^<>≤≥≠×÷−∑∏√∞∫%]", r"[+-]?(?:\d[\d,]*(?:\.\d+)?)(?:[eE][+-]?\d+)?",
)]


def protected_spans(text: str) -> list[tuple[int, int]]:
    spans = []
    for pattern in _PATTERNS:
        for match in pattern.finditer(text):
            start, end = match.span()
            # Boundary punctuation/space belongs to the protected literal too:
            # deleting '. ' after '12' must not produce the lexical token
            # '12the'. Internal prose punctuation/whitespace remains mutable.
            def delimiter(character):
                return character.isspace() or unicodedata.category(character).startswith("P")
            while start and delimiter(text[start - 1]):
                start -= 1
            while end < len(text) and delimiter(text[end]):
                end += 1
            spans.append((start, end))
    spans.sort()
    merged = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def split_protected(text: str) -> list[dict]:
    segments, pos = [], 0
    for start, end in protected_spans(text):
        if pos < start:
            segments.append({"protected": False, "text": text[pos:start]})
        segments.append({"protected": True, "text": text[start:end]})
        pos = end
    if pos < len(text):
        segments.append({"protected": False, "text": text[pos:]})
    return segments


def changed_word_fraction(original: str, changed: str) -> float:
    """Levenshtein-style changed original-word fraction, capped at one.

    Whitespace-delimited original words are the denominator; replacements and
    deletions count their source span and insertions count their inserted span.
    This is a logged manipulation measurement, never a selection-quality score.
    """
    a, b = original.split(), changed.split()
    if not a:
        return 0.0
    edits = sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if tag != "equal")
    return min(1.0, edits / len(a))


def _verify_udpipe_binary(path: Path) -> dict:
    """Check pinned authoritative bytes and the actual native model loader."""
    binary = path.read_bytes()
    if (len(binary) != UDPIPE_BYTES or content_hash(binary) != UDPIPE_SHA256
            or hashlib.md5(binary).hexdigest() != UDPIPE_MD5):
        raise ValueError("Pinned official English-EWT UDPipe identity mismatch")
    from ufal.udpipe import Model
    model = Model.load(str(path))
    if model is None or model.newTokenizer(Model.DEFAULT) is None:
        raise ValueError("Official UDPipe binary cannot create its native tokenizer")
    return {"name": UDPIPE_MODEL, "sha256": UDPIPE_SHA256, "md5": UDPIPE_MD5,
            "size_bytes": UDPIPE_BYTES, "native_load_passed": True}


def provision_udpipe_model(destination) -> dict:
    """Download the individual public LINDAT model with TLS/checksum checks.

    Does not submit license acceptance or use a mirror. An access denial or a
    changed upstream identity propagates to the shared recovery procedure.
    """
    import requests
    path = Path(destination).resolve()
    if path.name != UDPIPE_MODEL:
        raise ValueError("UDPipe destination must retain the official model name")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        metadata_response = requests.get(UDPIPE_METADATA_URL, timeout=(20, 90))
        metadata_response.raise_for_status()
        metadata = metadata_response.json()
        checksum = metadata.get("checkSum", {})
        if (metadata.get("name") != UDPIPE_MODEL or metadata.get("sizeBytes") != UDPIPE_BYTES
                or checksum.get("checkSumAlgorithm") != "MD5" or checksum.get("value") != UDPIPE_MD5
                or metadata.get("_links", {}).get("content", {}).get("href") != UDPIPE_URL):
            raise ValueError("Authoritative LINDAT UDPipe metadata changed")
        temporary = None
        try:
            with requests.get(UDPIPE_URL, stream=True, timeout=(20, 90)) as response:
                response.raise_for_status()
                if response.url != UDPIPE_URL:
                    raise ValueError("Unexpected UDPipe download redirect; access must be reviewed")
                with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".udpipe-", delete=False) as handle:
                    temporary = Path(handle.name)
                    size = 0
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            size += len(chunk)
                            if size > UDPIPE_BYTES:
                                raise ValueError("UDPipe download exceeds pinned byte count")
                            handle.write(chunk)
                    handle.flush()
                    os.fsync(handle.fileno())
            _verify_udpipe_binary(temporary)
            temporary.replace(path)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    identity = _verify_udpipe_binary(path)
    result = {"path": str(path), **identity, "source_url": UDPIPE_URL,
              "metadata_url": UDPIPE_METADATA_URL, "license": "CC-BY-NC-SA-4.0",
              "status": "verified", "simulation": False}
    record_path = path.with_suffix(path.suffix + ".provenance.json")
    atomic_write_json(record_path, result)
    notify_artifact(record_path, {"artifact_type": "auxiliary_provenance", "simulation": False})
    return result


def discover_aspell_dictionary() -> dict:
    """Resolve the actual official English speller's recursive dictionary files.

    Uses the native binding's effective configuration, including optional local
    dictionaries if present. Only names/hashes and non-path settings define the
    immutable identity; filesystem locations are returned separately.
    """
    import aspell
    speller = aspell.Speller("lang", "en")
    effective = {key: value[1] for key, value in speller.ConfigKeys().items()}
    roots = [Path(effective[key]).resolve() for key in ("actual-dict-dir", "dict-dir", "data-dir", "local-data-dir")]
    files: set[Path] = set()

    def resolve(name: str, relative_to: Path | None = None) -> Path:
        requested = Path(name)
        options = [requested] if requested.is_absolute() else [root / requested for root in ([relative_to] if relative_to else []) + roots]
        matches = [option.resolve() for option in options if option.is_file()]
        if not matches:
            raise FileNotFoundError("English Aspell dependency is absent: " + name)
        return matches[0]

    def visit(path: Path):
        if path in files:
            return
        files.add(path)
        if path.suffix in {".multi", ".alias"}:
            for line in path.read_text(encoding="latin-1").splitlines():
                line = line.partition("#")[0].strip()
                if line.startswith("add "):
                    visit(resolve(line[4:].strip(), path.parent))

    visit(resolve(effective["master-path"]))
    lang_path = resolve(effective["actual-lang"] + ".dat")
    visit(lang_path)
    for line in lang_path.read_text(encoding="latin-1").splitlines():
        fields = line.partition("#")[0].split()
        if len(fields) < 2:
            continue
        key, value = fields[:2]
        if key in {"affix", "soundslike", "repl-table"}:
            suffix = {"affix": "_affix.dat", "soundslike": "_phonet.dat", "repl-table": ""}[key]
            if value != "none":
                visit(resolve(value + suffix))
        elif key == "charset":
            charset = re.sub(r"^iso(\d+)-", r"iso-\1-", value)
            for suffix in (".cmap", ".cset"):
                visit(resolve(charset + suffix))
    visit(resolve(str(effective["keyboard"]) + ".kbd"))
    if effective.get("use-other-dicts"):
        for key in ("personal-path", "repl-path"):
            path = Path(effective[key]).resolve()
            if path.is_file():
                visit(path)
    for name in effective.get("extra-dicts", []) + effective.get("wordlists", []):
        visit(resolve(name))
    locator_keys = {"actual-dict-dir", "dict-dir", "data-dir", "local-data-dir", "master", "master-path", "prefix",
                    "home-dir", "conf", "conf-dir", "conf-path", "per-conf", "per-conf-path", "personal", "personal-path",
                    "repl", "repl-path", "filter-path", "word-list-path", "extra-dicts", "wordlists", "dict-alias"}
    # Freeze every effective non-locator option; suggestion behavior must not
    # change silently through an unrecorded environment/configuration default.
    settings = {key: value for key, value in effective.items() if key not in locator_keys}
    identities = sorted(({"name": path.name, "sha256": content_hash(path.read_bytes())} for path in files), key=lambda item: item["name"])
    if not speller.check("English") or not speller.suggest("Englsh"):
        raise RuntimeError("Real English Aspell check/suggest preflight failed")
    return {"files": [str(path) for path in sorted(files)], "identities": identities,
            "settings": settings, "identity_hash": content_hash({"files": identities, "settings": settings}),
            "check_suggest_passed": True, "status": "verified", "simulation": False}


_WORKER = r'''
import sys,json,importlib.metadata,platform,hashlib,pathlib
config=json.loads(sys.stdin.readline())
sys.path.insert(0,config['source_dir'])
import numpy as np
import introduce_errors as official
import udpipe_tokenizer
import ufal.udpipe
udpipe_tokenizer.UDPipeTokenizer.MODELS['en']=config['udpipe_model']
model=ufal.udpipe.Model.load(config['udpipe_model'])
if model is None: raise RuntimeError('UDPipe binary is not loadable')
tokenizer=official.load_tokenizer('en')
if not tokenizer.tokenize('This is an English tokenizer check.'):
 raise RuntimeError('UDPipe tokenizer empty')
versions={n:importlib.metadata.version(n) for n in ['numpy','scipy','ufal.udpipe','aspell-python-py3','Unidecode']}
print(json.dumps({'ready':True,'versions':versions,'python':platform.python_version()}),flush=True)
aspects={}
for line in sys.stdin:
 try:
  req=json.loads(line); alpha=req['alpha']; np.random.seed(req['seed'])
  if alpha not in aspects:
   aspects[alpha]=official.get_aspects_generator(config['profile'],'en',float(alpha),0,False,0.3,alpha_std=0,alpha_uniformity_prob=0)()
  out=[]; changes=[]
  for segment in req['segments']:
   if segment['protected']:
    out.append(segment['text']); continue
   # Preserve newlines and whitespace at protection boundaries; internal
   # whitespace remains subject to the official whitespace aspect.
   import re
   for part in re.split('(\r\n|\r|\n)',segment['text']):
    if not part.strip() or part in ['\r','\n','\r\n']:
     out.append(part); continue
    match=re.fullmatch(r'(\s*)(.*?)(\s*)',part,re.DOTALL)
    prefix,body,suffix=match.groups()
    text,edits=official.introduce_errors_in_line(body,tokenizer,aspects[alpha],False,False,False,False,False,True,False,False,verbose=False,no_error_sentence_boost=0)
    out.append(prefix+text+suffix); changes.extend(edits)
  print(json.dumps({'reasoning':''.join(out),'changes':changes}),flush=True)
 except Exception as error:
  print(json.dumps({'error':type(error).__name__+': '+str(error)}),flush=True)
'''


class OfficialKaziText:
    """Persistent isolated official worker; no substitute noise implementation."""
    def __init__(self, source_dir, udpipe_model, *, aspell_dictionary_files: list,
                 expected_udpipe_sha256: str | None = None, timeout: float = 120,
                 python_executable: str = sys.executable):
        self.source_dir, self.model = Path(source_dir).resolve(), Path(udpipe_model).resolve()
        revision = subprocess.check_output(["git", "-C", str(self.source_dir), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(self.source_dir), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
        if revision != REVISION or dirty:
            raise ValueError("KaziText source must be the unmodified pinned revision")
        profile = self.source_dir / PROFILE_PATH
        if content_hash(profile.read_bytes()) != PROFILE_SHA256:
            raise ValueError("Official KaziText profile checksum mismatch")
        if self.model.name != UDPIPE_MODEL:
            raise ValueError("Exact English-EWT UD2.5 tokenizer model is required")
        model_bytes = self.model.read_bytes()
        if len(model_bytes) < 100_000 or model_bytes.lstrip().lower().startswith((b"<!doctype", b"<html")):
            raise ValueError("UDPipe model is not a plausible binary")
        model_hash = content_hash(model_bytes)
        if model_hash != UDPIPE_SHA256 or (expected_udpipe_sha256 and expected_udpipe_sha256 != model_hash):
            raise ValueError("UDPipe model checksum mismatch")
        if not aspell_dictionary_files:
            raise ValueError("Resolved English Aspell dictionary provenance is required")
        dictionary = discover_aspell_dictionary()
        if {str(Path(p).resolve()) for p in aspell_dictionary_files} != set(dictionary["files"]):
            raise ValueError("Declared Aspell files differ from the actual English speller dependencies")
        dictionaries = dictionary["identities"]
        self.asset_paths = {"source_dir": str(self.source_dir), "udpipe_model": str(self.model),
                            "aspell_dictionary_files": dictionary["files"]}
        self.provenance = {"source_revision": REVISION, "profile_path": PROFILE_PATH,
                           "profile_sha256": PROFILE_SHA256, "udpipe_model": UDPIPE_MODEL,
                           "udpipe_sha256": model_hash, "aspell_dictionaries": dictionaries,
                           "aspell_settings": dictionary["settings"], "aspell_identity_hash": dictionary["identity_hash"],
                           "settings": SETTINGS, "pythonhashseed": 0,
                           "protection_version": PROTECTION_VERSION,
                           "seed_convention": "sha256_42_kazitext_example_model_level_v1"}
        self.timeout, self.lock = timeout, threading.Lock()
        self.errors = tempfile.TemporaryFile(mode="w+t")
        env = dict(os.environ, PYTHONHASHSEED="0")
        self.process = subprocess.Popen([python_executable, "-u", "-c", _WORKER],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.errors, text=True, env=env)
        self.process.stdin.write(json.dumps({"source_dir": str(self.source_dir), "profile": str(profile), "udpipe_model": str(self.model)}) + "\n")
        self.process.stdin.flush()
        ready = self._read()
        if not ready.get("ready"):
            self.close()
            raise RuntimeError("KaziText worker preflight failed")
        self.provenance.update(runtime_versions=ready["versions"], python=ready["python"])
        atexit.register(self.close)

    def _read(self) -> dict:
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            if not selector.select(self.timeout):
                self.close()
                raise TimeoutError("Official KaziText operation timed out")
        line = self.process.stdout.readline()
        if not line:
            self.errors.seek(0)
            raise RuntimeError("Official KaziText worker terminated: " + self.errors.read()[-4000:])
        response = json.loads(line)
        if "error" in response:
            raise RuntimeError("Official KaziText failure: " + response["error"])
        return response

    def close(self):
        if getattr(self, "process", None) and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)

    def transform(self, reasoning: str, example_id: str, level: int, *, model_id: str) -> dict:
        if isinstance(level, bool) or level not in range(1, 11) or model_id not in {"7b", "14b"}:
            raise ValueError("Invalid KaziText model/level")
        segments = split_protected(reasoning)
        request_seed = derive_seed(42, "kazitext", example_id, model_id, level)
        request = {"segments": segments, "alpha": ALPHAS[level - 1], "seed": request_seed}
        with self.lock:
            if self.process.poll() is not None:
                raise RuntimeError("Official KaziText worker is unavailable")
            self.process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            self.process.stdin.flush()
            response = self._read()
        transformed = response["reasoning"]
        # Sequential exact span search plus numeric-list identity catches missing,
        # reordered or accidentally added numerical literals.
        pos = 0
        for segment in segments:
            if segment["protected"]:
                pos = transformed.find(segment["text"], pos)
                if pos < 0:
                    raise ValueError("KaziText protected-content integrity failure")
                pos += len(segment["text"])
        if _NUMBER.findall(reasoning) != _NUMBER.findall(transformed):
            raise ValueError("KaziText numerical-literal integrity failure")
        return {"reasoning": transformed, "method": "kazitext", "level": level,
                "alpha": ALPHAS[level - 1], "seed": 42, "request_seed": request_seed,
                "source_hash": content_hash(reasoning), "content_hash": content_hash(transformed),
                "changed_word_fraction": changed_word_fraction(reasoning, transformed),
                "protected_spans": [reasoning[a:b] for a, b in protected_spans(reasoning)],
                "protected_content_valid": True, "changes": response["changes"],
                "transformation": self.provenance, "algorithm_version": "official_kazitext_protected_v1",
                "status": "verified"}


def provision_kazitext_source(destination) -> dict:
    """Acquire public official source without altering pinned content."""
    path = Path(destination)
    if not path.exists():
        subprocess.run(["git", "clone", "https://github.com/ufal/kazitext.git", str(path)], check=True)
        subprocess.run(["git", "-C", str(path), "checkout", "--detach", REVISION], check=True)
    revision = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    profile_hash = content_hash((path / PROFILE_PATH).read_bytes())
    if revision != REVISION or profile_hash != PROFILE_SHA256:
        raise ValueError("Provisioned KaziText identity mismatch")
    return {"source_dir": str(path.resolve()), "revision": revision, "profile_sha256": profile_hash}


def provision_kazitext_assets(destination, *, source_dir=None, udpipe_path=None) -> dict:
    """Provision exact public auxiliary assets and resolve installed Aspell data.

    Native Aspell, its English dictionary and Python binding are prerequisites
    installed by the environment's pinned dependency workflow. This does not
    invoke package managers, change native dictionaries or accept a license.
    """
    root = Path(destination).resolve()
    source = provision_kazitext_source(source_dir or root / "kazitext")
    model = provision_udpipe_model(udpipe_path or root / "udpipe" / UDPIPE_MODEL)
    dictionary = discover_aspell_dictionary()
    result = {"source_dir": source["source_dir"], "udpipe_path": model["path"],
              "aspell_dictionary_files": dictionary["files"], "source": source,
              "udpipe": model, "aspell": dictionary, "status": "verified", "simulation": False}
    result["identity_hash"] = content_hash({"source": {"revision": source["revision"], "profile_sha256": source["profile_sha256"]},
              "udpipe_sha256": model["sha256"], "aspell_identity_hash": dictionary["identity_hash"]})
    path = root / "kazitext_auxiliary_manifest.json"
    atomic_write_json(path, result)
    notify_artifact(path, {"artifact_type": "auxiliary_provenance", "simulation": False})
    return result


def run_transformation_preflight(config: dict, output_dir) -> dict:
    """Actual auxiliary/tokenizer controls before clean-corpus generation.

    The control text is explicitly nonexperimental. Passing establishes runtime
    transformation readiness, never existence or eligibility of a clean corpus.
    """
    from transformers import AutoTokenizer
    from operations.assets import model_load_kwargs
    from data_generation_collection.sft_data.rot13.rot13_data_construction import transform_rot13
    from data_generation_collection.sft_data.clean_trace_collection.candidate_batch.clean_trace_batch import native_completion, total_sequence_tokens
    from training.sft.individual_fit.sft_fit import render_sft_completion
    from shared_procedures.configuration import configuration_hash
    cache = Path(config.get("runtime", {}).get("cache_root", "/workspace/model_cache"))
    options = config.get("operations", {})
    assets = provision_kazitext_assets(cache / "kazitext_auxiliary", source_dir=cache / "kazitext",
                                     udpipe_path=options.get("udpipe_path"))
    backend = OfficialKaziText(assets["source_dir"], assets["udpipe_path"], aspell_dictionary_files=assets["aspell_dictionary_files"],
                              expected_udpipe_sha256=config["interventions"]["kazitext"]["udpipe_sha256"])
    try:
        actual = {"files": backend.provenance["aspell_dictionaries"], "settings": backend.provenance["aspell_settings"]}
        if actual != config["interventions"]["kazitext"].get("aspell_dictionary_provenance"):
            raise ValueError("Actual Aspell dictionaries/settings differ from the scientific freeze")
        prompt = "Check the arithmetic for the runtime control."
        reasoning = ("The careful reader checks the original quantities before doing arithmetic. "
                     "We have 24 apples and x = 12 + 12. The expression $\\frac{24}{2}=12$ stays exact. "
                     "The comparison gives a consistent conclusion, with probability 0.75 and 1,234.50 units.\n"
                     "This separate English sentence checks casing and punctuation.")
        final = "\n12"
        checks = []
        tokenizer_identities = {}
        for model in ("7b", "14b"):
            spec = config["models"][model]
            tokenizer = AutoTokenizer.from_pretrained(spec["id"], revision=spec["tokenizer_revision"], trust_remote_code=False,
                 **model_load_kwargs(spec["id"], spec["tokenizer_revision"], cache, tokenizer_only=True))
            tokenizer_identities[model] = {"id": spec["id"], "revision": spec["tokenizer_revision"]}
            original = native_completion(reasoning, final)
            original_tokens = total_sequence_tokens(tokenizer, prompt, original)
            rendered, _ = render_sft_completion(prompt, original, tokenizer)
            if original not in rendered or original_tokens > 4096:
                raise ValueError("Native training renderer dropped or truncated the control reasoning")
            for family in ("rot13", "kazitext"):
                for level in range(1, 11):
                    def transform():
                        return (transform_rot13(reasoning, "transformation-runtime-control", level) if family == "rot13"
                            else backend.transform(reasoning, "transformation-runtime-control", level, model_id=model))
                    first, replay = transform(), transform()
                    if first != replay or first.get("protected_content_valid") is not True:
                        raise ValueError("Transformation control replay or protected-content check failed")
                    completion = native_completion(first["reasoning"], final)
                    text, _ = render_sft_completion(prompt, completion, tokenizer)
                    tokens = total_sequence_tokens(tokenizer, prompt, completion)
                    if completion not in text or tokens > 4096:
                        raise ValueError("Native transformed target failed intact 4096-token control")
                    checks.append({"model": model, "method": family, "level": level,
                                   "original_token_count": original_tokens, "transformed_token_count": tokens,
                                   "content_hash": first["content_hash"], "replay_passed": True,
                                   "protection_passed": True, "final_answer_unchanged": True})
        result = {"name": "transformation_runtime", "status": "passed",
                  "detail": "Both immutable native tokenizers and all forty intervention controls passed exact replay, protection and intact target checks.",
                  "evidence": {"live": True, "simulation": False, "control_preflight": True,
                               "scientific_results": False, "configuration_hash": configuration_hash(config),
                               "control_input_hash": content_hash({"prompt": prompt, "reasoning": reasoning, "final": final}),
                               "native_tokenizers": tokenizer_identities, "variants": checks,
                               "kazitext_provenance": backend.provenance}}
        path = Path(output_dir) / "transformation_runtime.json"
        atomic_write_json(path, result)
        notify_artifact(path, {"kind": "transformation_runtime_preflight", "simulation": False, "control_preflight": True})
        return result
    finally:
        backend.close()


def construct_kazitext_data(pool, tokenizers, output_dir, *, backend: OfficialKaziText, config_hash: str) -> dict:
    from data_generation_collection.sft_data.sft_data_preparation import build_sft_datasets
    return build_sft_datasets(pool, tokenizers, output_dir, config_hash=config_hash, kazitext=backend, families=("kazitext",))

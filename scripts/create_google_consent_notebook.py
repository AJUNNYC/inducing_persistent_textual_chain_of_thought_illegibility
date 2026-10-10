#!/usr/bin/env python3
"""Generate standalone restricted Google consent cells; never embed credentials."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import textwrap


HELPERS = r'''
import getpass
import hmac
import json
import logging
import os
from pathlib import Path
import tempfile
from urllib.parse import parse_qs, urlsplit, urlunsplit

SCOPE = "https://www.googleapis.com/auth/drive.file"
INTENDED_ACCOUNT = "antoniojun.nyc@gmail.com"
PRIVATE_ROOT = Path("/workspace/.secrets")
CLIENT_PATH = PRIVATE_ROOT / "google_client.json"
TOKEN_PATH = PRIVATE_ROOT / "google_token.json"
REDIRECT_URI = "http://localhost:8765/"
for logger_name in ("oauthlib", "requests_oauthlib", "google.auth", "google_auth_oauthlib", "urllib3"):
    logging.getLogger(logger_name).setLevel(logging.WARNING)

def checked_scopes(value):
    scopes = value.split() if isinstance(value, str) else value
    if not isinstance(scopes, (list, tuple)) or set(scopes) != {SCOPE}:
        raise ValueError("Restricted Google Drive consent was not established")

def checked_callback(callback, expected_state):
    if not isinstance(callback, str) or not isinstance(expected_state, str):
        raise ValueError("Invalid consent callback")
    parsed = urlsplit(callback.strip())
    if (parsed.scheme != "http" or parsed.hostname != "localhost" or parsed.port != 8765
        or parsed.username is not None or parsed.password is not None or parsed.path != "/"
        or parsed.fragment or parsed.netloc != "localhost:8765"):
        raise ValueError("Invalid consent callback")
    query = parse_qs(parsed.query, keep_blank_values=True)
    if (set(query) - {"state", "code", "scope", "authuser", "prompt", "hd", "iss"}
        or len(query.get("state", [])) != 1 or len(query.get("code", [])) != 1
        or not query["code"][0] or not hmac.compare_digest(query["state"][0], expected_state)):
        raise ValueError("Invalid consent callback state or authorization")
    # Google may identify the authorization server using RFC 9207's optional
    # issuer parameter. Accept only its exact issuer; never trust another server
    # or an ambiguous repeated/blank value, even with a matching request state.
    if "iss" in query and query["iss"] != ["https://accounts.google.com"]:
        raise ValueError("Invalid consent callback issuer")
    # OAuthlib requires a secure authorization-response URL for parsing. This
    # normalization matches run_local_server; the actual token POST stays TLS
    # to the validated Google token endpoint, with the original redirect URI.
    return urlunsplit(("https", parsed.netloc, parsed.path, parsed.query, ""))

def checked_client_path(path=CLIENT_PATH):
    if PRIVATE_ROOT.is_symlink() or path.is_symlink() or not path.is_file():
        raise ValueError("Protected desktop OAuth client file is unavailable")
    if path.parent != PRIVATE_ROOT:
        raise ValueError("Desktop OAuth client must use the protected configured path")
    PRIVATE_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(PRIVATE_ROOT, 0o700)
    os.chmod(path, 0o600)
    document = json.loads(path.read_text())
    try:
        installed = document.get("installed")
        if not isinstance(installed, dict) or "web" in document:
            raise ValueError("An existing desktop OAuth client is required")
        auth = urlsplit(installed.get("auth_uri", ""))
        if (auth.scheme != "https" or auth.netloc != "accounts.google.com"
            or auth.path not in {"/o/oauth2/auth", "/o/oauth2/v2/auth"} or auth.query or auth.fragment):
            raise ValueError("Desktop client Google authorization endpoint is invalid")
        if installed.get("token_uri") != "https://oauth2.googleapis.com/token":
            raise ValueError("Desktop client Google token endpoint is invalid")
        if not installed.get("client_id", "").endswith(".apps.googleusercontent.com") or not installed.get("client_secret"):
            raise ValueError("Existing desktop client identity is incomplete")
    finally:
        document.clear()

def private_write_json(path, value):
    if path.parent != PRIVATE_ROOT or PRIVATE_ROOT.is_symlink() or path.is_symlink():
        raise ValueError("Authorized credentials must remain at the protected configured path")
    PRIVATE_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(PRIVATE_ROOT, 0o700)
    fd, temporary = tempfile.mkstemp(dir=PRIVATE_ROOT, prefix=".google_authorized_")
    try:
        with os.fdopen(fd, "w") as stream:
            os.fchmod(stream.fileno(), 0o600)
            json.dump(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(PRIVATE_ROOT, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
'''


CONSENT = r'''
# Run this cell once, open the printed Google URL in your own browser, and
# sign in as antoniojun.nyc@gmail.com. Accept only the requested app-file scope.
# Google then redirects your browser to localhost:8765. "Cannot connect" is
# expected: copy the FULL URL from the browser address bar and paste it into
# the hidden input below. Do not paste the URL into a code cell or chat.
flow = credentials = callback_url = authorization_response = expected_state = None
consent_url = token_info = account = private_document = service = None
failure_message = "Consent setup failed. Confirm the Google Drive consent kernel and protected desktop client file, then rerun this cell."
try:
    import requests
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build

    checked_client_path()
    flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_PATH), scopes=[SCOPE],
                                                    autogenerate_code_verifier=True)
    flow.redirect_uri = REDIRECT_URI
    consent_url, expected_state = flow.authorization_url(access_type="offline", prompt="consent",
                                                        login_hint=INTENDED_ACCOUNT)
    print("Open this Google consent URL in your browser and sign in as " + INTENDED_ACCOUNT + ":")
    print(consent_url)
    print("After consent, copy the full localhost:8765 callback URL from the browser address bar. A localhost connection error is expected.")
    failure_message = "The callback was not accepted. Rerun this cell for a new consent URL and paste its complete matching localhost:8765 callback only into the hidden input."
    callback_url = getpass.getpass("Paste the full callback URL here (hidden input): ")
    authorization_response = checked_callback(callback_url, expected_state)
    # Both explicit comparison above and OAuthlib's stored session state are
    # enforced. Never send a callback from a previous consent attempt.
    if flow.oauth2session._state != expected_state:
        raise ValueError("OAuth session state does not match the consent request")
    failure_message = "Google token exchange failed. No token was saved. Rerun this cell for a new consent URL and matching callback; if it repeats, check network access to oauth2.googleapis.com."
    flow.fetch_token(authorization_response=authorization_response)
    credentials = flow.credentials
    failure_message = "Offline refresh authorization was not established. Rerun this cell and complete Google consent again."
    if not credentials.refresh_token:
        raise ValueError("Offline refresh authorization missing")
    checked_scopes(credentials.scopes)
    credentials.refresh(Request())
    if credentials.granted_scopes is not None:
        checked_scopes(credentials.granted_scopes)

    failure_message = "Actual token scope or account verification failed. No token was saved. Use the intended Google account and request only app-file access."
    token_info = requests.get("https://oauth2.googleapis.com/tokeninfo",
                              params={"access_token": credentials.token}, timeout=30,
                              allow_redirects=False)
    if not token_info.ok:
        raise ValueError("Actual token scope verification unavailable")
    checked_scopes(token_info.json().get("scope", ""))
    service = build("drive", "v3", credentials=credentials, cache_discovery=False)
    account = service.about().get(fields="user(emailAddress)").execute(num_retries=0)
    if account.get("user", {}).get("emailAddress") != INTENDED_ACCOUNT:
        raise ValueError("Authorized Google account differs from intended account")

    failure_message = "Verified authorization could not be saved to the protected token path. No private values are displayed; check workspace permissions and rerun consent."
    private_document = json.loads(credentials.to_json())
    checked_scopes(private_document.get("scopes"))
    private_write_json(TOKEN_PATH, private_document)
    print("Verified restricted Google Drive consent, refresh, and intended account. Credentials saved privately with mode 0600; no tokens or client secrets are displayed.")
except BaseException:
    # Never render exception text or a traceback: either can contain callback,
    # authorization code, access token, refresh token or client-secret values.
    print(failure_message)
finally:
    if isinstance(private_document, dict):
        private_document.clear()
    if flow is not None:
        try:
            flow.oauth2session.close()
        except Exception:
            pass
    flow = credentials = callback_url = authorization_response = expected_state = None
    consent_url = token_info = account = private_document = service = None
'''


def create_notebook(output: str | Path) -> Path:
    """Write only public workflow instructions; private files are read on the Pod."""
    def cell(kind: str, source: str) -> dict:
        result = {'cell_type': kind, 'id': hashlib.sha256(source.encode()).hexdigest()[:12],
                  'metadata': {}, 'source': textwrap.dedent(source).strip().splitlines(keepends=True)}
        if kind == 'code':
            result.update(execution_count=None, outputs=[])
        return result
    notebook = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {
        'kernelspec': {'display_name': 'Google Drive consent', 'language': 'python', 'name': 'illegibility_oauth'},
        'language_info': {'name': 'python', 'version': '3.11'}}, 'cells': [
        cell('markdown', '''
        # Private Google Drive consent

        Select **Google Drive consent** as the notebook kernel and run both code cells.
        The existing desktop client is already stored privately at `/workspace/.secrets/google_client.json`.
        This notebook requests only `drive.file` and verifies the intended account, `antoniojun.nyc@gmail.com`.

        Open the printed Google consent URL in your own browser. After consent, the browser redirects to
        `http://localhost:8765/` and may say that it cannot connect. Copy the **full browser address-bar URL**
        and paste it only into the notebook's **hidden input**. Never put that URL, token or client JSON into
        a code cell, chat, screenshot or notebook output. The authorization code works despite the local
        connection error. Rerun the consent cell if you used an old callback or chose the wrong account.

        Success saves refresh credentials at `/workspace/.secrets/google_token.json` with mode `0600`.
        Notebook outputs contain only the public consent URL and safe status messages.
        '''), cell('code', HELPERS), cell('code', CONSENT)]}
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook, indent=2) + '\n')
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='/workspace/google_drive_consent.ipynb')
    args = parser.parse_args()
    print(create_notebook(args.output))


if __name__ == '__main__':
    main()

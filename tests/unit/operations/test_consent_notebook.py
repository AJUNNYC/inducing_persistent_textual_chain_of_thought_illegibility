"""Standalone notebook checks use public synthetic fixtures, never credentials."""
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def generator():
    path = Path(__file__).resolve().parents[3] / 'scripts/create_google_consent_notebook.py'
    spec = importlib.util.spec_from_file_location('consent_notebook_generator', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def helpers(generator):
    namespace = {}
    exec(compile(generator.HELPERS, '<public_consent_helpers>', 'exec'), namespace)
    return namespace


def test_valid_matching_callback_normalizes_only_parser_transport(helpers):
    callback = 'http://localhost:8765/?state=public_fixture_nonce&code=public_fixture_code&scope=drive.file'
    assert helpers['checked_callback'](callback, 'public_fixture_nonce') == callback.replace('http:', 'https:', 1)
    assert helpers['REDIRECT_URI'] == 'http://localhost:8765/'


@pytest.mark.parametrize('callback', [
    'https://localhost:8765/?state=nonce&code=fixture',
    'http://evil.example:8765/?state=nonce&code=fixture',
    'http://localhost:8000/?state=nonce&code=fixture',
    'http://localhost:8765/elsewhere?state=nonce&code=fixture',
    'http://user@localhost:8765/?state=nonce&code=fixture',
    'http://localhost:8765/?state=nonce&code=fixture#fragment',
    'http://localhost:8765/?state=old_nonce&code=fixture',
    'http://localhost:8765/?state=nonce&state=nonce&code=fixture',
    'http://localhost:8765/?state=nonce&code=fixture&code=fixture',
    'http://localhost:8765/?state=nonce&error=access_denied',
    'http://localhost:8765/?state=nonce&code=',
    'http://localhost:8765/?state=nonce&code=fixture&unexpected=1',
])
def test_callback_mismatch_or_ambiguous_authorization_is_rejected(helpers, callback):
    with pytest.raises(ValueError):
        helpers['checked_callback'](callback, 'nonce')


def test_exact_scope_is_required_even_if_drive_file_is_present(helpers):
    scope = helpers['SCOPE']
    helpers['checked_scopes']([scope])
    helpers['checked_scopes'](scope)
    for broad in ([scope, 'https://www.googleapis.com/auth/drive'], [], None):
        with pytest.raises(ValueError):
            helpers['checked_scopes'](broad)


def test_private_client_and_atomic_token_paths_modes_and_symlink_rejection(helpers, tmp_path):
    private = tmp_path / '.secrets'
    private.mkdir()
    helpers['PRIVATE_ROOT'] = private
    client = private / 'google_client.json'
    client.write_text(json.dumps({'installed': {'client_id': 'fixture.apps.googleusercontent.com',
        'client_secret': 'explicit_public_test_fixture', 'auth_uri': 'https://accounts.google.com/o/oauth2/auth',
        'token_uri': 'https://oauth2.googleapis.com/token'}}))
    helpers['checked_client_path'](client)
    assert private.stat().st_mode & 0o777 == 0o700
    assert client.stat().st_mode & 0o777 == 0o600
    token = private / 'google_token.json'
    helpers['private_write_json'](token, {'refresh_token': 'explicit_public_test_fixture'})
    assert token.stat().st_mode & 0o777 == 0o600
    assert json.loads(token.read_text())['refresh_token'] == 'explicit_public_test_fixture'
    assert not list(private.glob('.google_authorized_*'))
    with pytest.raises(ValueError):
        helpers['private_write_json'](tmp_path / 'outside.json', {})
    token.unlink()
    token.symlink_to(tmp_path / 'outside.json')
    with pytest.raises(ValueError):
        helpers['private_write_json'](token, {})
    assert not (tmp_path / 'outside.json').exists()


def test_client_token_endpoint_must_be_google_https(helpers, tmp_path):
    private = tmp_path / '.secrets'
    private.mkdir()
    helpers['PRIVATE_ROOT'] = private
    client = private / 'google_client.json'
    client.write_text(json.dumps({'installed': {'client_id': 'fixture.apps.googleusercontent.com',
        'client_secret': 'explicit_public_test_fixture', 'auth_uri': 'https://accounts.google.com/o/oauth2/auth',
        'token_uri': 'https://evil.example/token'}}))
    with pytest.raises(ValueError):
        helpers['checked_client_path'](client)


def test_generated_notebook_is_standalone_empty_and_uses_dedicated_kernel(generator, tmp_path):
    path = generator.create_notebook(tmp_path / 'google_drive_consent.ipynb')
    notebook = json.loads(path.read_text())
    assert notebook['metadata']['kernelspec']['name'] == 'illegibility_oauth'
    assert len({cell['id'] for cell in notebook['cells']}) == len(notebook['cells'])
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            assert cell['outputs'] == [] and cell['execution_count'] is None
            compile(''.join(cell['source']), '<standalone_consent_cell>', 'exec')
    consent = generator.CONSENT
    assert 'getpass.getpass(' in consent and 'flow.oauth2session._state != expected_state' in consent
    assert 'allow_redirects=False' in consent and 'except BaseException:' in consent
    assert 'from operations' not in consent and 'from shared_procedures' not in consent

"""Standalone notebook checks use public synthetic fixtures, never credentials."""
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from urllib.parse import urlencode

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


def test_current_google_callback_with_exact_issuer_is_accepted(helpers):
    query = urlencode({'state': 'public_fixture_nonce', 'code': 'public_fixture_code',
                       'scope': helpers['SCOPE'], 'authuser': '0', 'prompt': 'consent',
                       'iss': 'https://accounts.google.com'})
    callback = 'http://localhost:8765/?' + query
    assert helpers['checked_callback'](callback, 'public_fixture_nonce') == callback.replace('http:', 'https:', 1)


@pytest.mark.parametrize('issuer_query', [
    'iss=',
    'iss=https%3A%2F%2Fevil.example',
    'iss=http%3A%2F%2Faccounts.google.com',
    'iss=https%3A%2F%2Faccounts.google.com%2F',
    'iss=https%3A%2F%2Faccounts.google.com&iss=https%3A%2F%2Faccounts.google.com',
    'iss=https%3A%2F%2Faccounts.google.com&iss=',
])
def test_untrusted_blank_or_duplicate_callback_issuer_is_rejected(helpers, issuer_query):
    callback = 'http://localhost:8765/?state=nonce&code=fixture&' + issuer_query
    with pytest.raises(ValueError):
        helpers['checked_callback'](callback, 'nonce')


def test_trusted_issuer_does_not_allow_mismatched_state(helpers):
    callback = 'http://localhost:8765/?state=old_nonce&code=fixture&iss=https%3A%2F%2Faccounts.google.com'
    with pytest.raises(ValueError):
        helpers['checked_callback'](callback, 'nonce')


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


@pytest.mark.parametrize('callback_state,expected_failure,exchange_attempted', [
    ('old_nonce', 'The callback was not accepted.', False),
    ('nonce', 'Google token exchange failed.', True),
])
def test_consent_distinguishes_callback_and_exchange_failures_without_private_output(
        generator, helpers, monkeypatch, capsys, callback_state, expected_failure, exchange_attempted):
    """Execute both failure boundaries with offline SDK doubles and public fixtures."""
    calls = []
    session = SimpleNamespace(_state='nonce', close=lambda: calls.append('close'))

    def fail_exchange(**kwargs):
        calls.append('exchange')
        raise RuntimeError('PUBLIC_FIXTURE_EXCEPTION_DETAIL_MUST_NOT_BE_DISPLAYED')

    flow = SimpleNamespace(oauth2session=session,
        authorization_url=lambda **kwargs: ('https://accounts.google.com/public_fixture', 'nonce'),
        fetch_token=fail_exchange)
    modules = {
        'requests': {},
        'google_auth_oauthlib.flow': {'InstalledAppFlow': SimpleNamespace(
            from_client_secrets_file=lambda *args, **kwargs: flow)},
        'google.auth.transport.requests': {'Request': object},
        'googleapiclient.discovery': {'build': lambda *args, **kwargs: pytest.fail('No service call expected')},
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    callback = 'http://localhost:8765/?' + urlencode({
        'state': callback_state, 'code': 'PUBLIC_FIXTURE_CODE_MUST_NOT_BE_DISPLAYED',
        'iss': 'https://accounts.google.com'})
    monkeypatch.setattr(helpers['getpass'], 'getpass', lambda prompt: callback)
    helpers['checked_client_path'] = lambda: None
    exec(compile(generator.CONSENT, '<offline_consent_fixture>', 'exec'), helpers)
    output = capsys.readouterr().out
    assert expected_failure in output
    assert ('exchange' in calls) is exchange_attempted
    assert 'close' in calls
    assert 'PUBLIC_FIXTURE_EXCEPTION_DETAIL_MUST_NOT_BE_DISPLAYED' not in output
    assert 'PUBLIC_FIXTURE_CODE_MUST_NOT_BE_DISPLAYED' not in output
    assert 'Traceback' not in output
    assert all(helpers[name] is None for name in ('flow', 'credentials', 'callback_url', 'authorization_response',
        'expected_state', 'consent_url', 'token_info', 'account', 'private_document', 'service'))

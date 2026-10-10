"""Provider state simulations test guards; no paid deployment is performed."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from operations.runpod import RunPodClient
from shared_procedures.storage import read_json


def authorization():
    return {'approved': True, 'authorization_id': 'fixture', 'pod_id': 'fixture_pod',
            'max_runtime_seconds': 30, 'max_total_usd': .1, 'authorized_hourly_usd': 2,
            'price_evidence': 'explicit_provider_fixture', 'billing_started_at_unix': 100.0}


def controlled_client():
    client = RunPodClient(api_key='explicit_fixture')
    elapsed = [100.0]
    stops, pauses = [], []
    client.stop_pod = lambda pod: stops.append(pod)
    def pause(seconds):
        pauses.append(seconds)
        elapsed[0] += seconds
    return client, elapsed, stops, pauses, pause


def test_stop_acknowledgment_waits_for_actual_termination(tmp_path):
    client, elapsed, stops, pauses, pause = controlled_client()
    observations = iter([
        {'desiredStatus': 'RUNNING', 'runtime': {'uptimeInSeconds': 40}, 'costPerHr': 2},
        {'desiredStatus': 'EXITED', 'runtime': {'uptimeInSeconds': 41}, 'costPerHr': 2},
        {'desiredStatus': 'EXITED', 'runtime': None, 'costPerHr': 2}])
    client.inspect_pod = lambda pod: next(observations)
    state = client.guarded_lifecycle('fixture_pod', authorization(), tmp_path / 'lifecycle.json',
                                    clock=lambda: elapsed[0], pause=pause)
    assert state['status'] == 'STOPPED' and stops == ['fixture_pod']
    assert pauses == [5] and state['stop_confirmation_observations'] == 2
    assert state['stop_evidence']['runtime_stopped'] is True
    assert state['stopped_at_unix'] == 105.0


def test_desired_exit_without_runtime_evidence_is_unconfirmed(tmp_path):
    client, elapsed, stops, pauses, pause = controlled_client()
    calls = []
    def inspect(pod):
        calls.append(pod)
        return {'desiredStatus': 'EXITED', 'costPerHr': 2}
    client.inspect_pod = inspect
    path = tmp_path / 'lifecycle.json'
    state = client.guarded_lifecycle('fixture_pod', authorization(), path,
                                    clock=lambda: elapsed[0], pause=pause)
    assert state['status'] == 'STOP_UNCONFIRMED' and stops == ['fixture_pod']
    assert len(calls) == 4  # One monitor observation, exactly three confirmation polls.
    assert pauses == [5, 5]
    assert 'stopped_at_unix' not in state and 'stop_evidence' not in state
    assert read_json(path)['status'] == 'STOP_UNCONFIRMED'


def test_timeout_of_stop_request_can_be_confirmed_by_provider(tmp_path):
    client, elapsed, stops, pauses, pause = controlled_client()
    stopped = [False]
    def stop(pod):
        stops.append(pod)
        stopped[0] = True
        raise TimeoutError('explicit ambiguous transport fixture')
    client.stop_pod = stop
    client.inspect_pod = lambda pod: ({'status': 'STOPPED', 'costPerHr': 2} if stopped[0]
                                      else {'desiredStatus': 'RUNNING', 'costPerHr': 2})
    state = client.guarded_lifecycle('fixture_pod', authorization(), tmp_path / 'lifecycle.json',
                                    clock=lambda: elapsed[0], pause=pause)
    assert state['status'] == 'STOPPED'
    assert state['stop_request_error'] == 'TimeoutError'
    assert state['stop_evidence']['actual_status'] == 'STOPPED'


def test_controller_exception_attempts_stop_but_preserves_failure(tmp_path):
    client, elapsed, stops, pauses, pause = controlled_client()
    calls = [0]
    def inspect(pod):
        calls[0] += 1
        if calls[0] == 1:
            raise ConnectionError('explicit provider fixture')
        return {'desiredStatus': 'RUNNING', 'runtime': {'uptimeInSeconds': 2}}
    client.inspect_pod = inspect
    path = tmp_path / 'lifecycle.json'
    with pytest.raises(ConnectionError):
        client.guarded_lifecycle('fixture_pod', authorization(), path,
                                clock=lambda: elapsed[0], pause=pause)
    state = read_json(path)
    assert state['status'] == 'STOP_UNCONFIRMED' and stops == ['fixture_pod']
    assert state['stop_reason'] == 'controller_failure'
    assert calls[0] == 4 and 'stopped_at_unix' not in state


def test_unconfirmed_restart_retains_original_billing_start(tmp_path):
    client, elapsed, stops, pauses, pause = controlled_client()
    path = tmp_path / 'lifecycle.json'
    client.inspect_pod = lambda pod: {'desiredStatus': 'RUNNING', 'costPerHr': 2}
    state = client.guarded_lifecycle('fixture_pod', authorization(), path,
                                    clock=lambda: elapsed[0], pause=pause)
    assert state['status'] == 'STOP_UNCONFIRMED'
    elapsed[0] = 200.0
    client.inspect_pod = lambda pod: {'desiredStatus': 'EXITED', 'runtime': None}
    repaired = client.guarded_lifecycle('fixture_pod', authorization(), path,
                                       clock=lambda: elapsed[0], pause=pause)
    assert repaired['status'] == 'STOPPED'
    assert repaired['started_at_unix'] == 100.0 and stops == ['fixture_pod', 'fixture_pod']


def test_saved_confirmed_stop_is_revalidated_after_external_restart(tmp_path):
    client, elapsed, stops, pauses, pause = controlled_client()
    path = tmp_path / 'lifecycle.json'
    client.inspect_pod = lambda pod: {'status': 'STOPPED'}
    assert client.guarded_lifecycle('fixture_pod', authorization(), path,
                                   clock=lambda: elapsed[0], pause=pause)['status'] == 'STOPPED'
    client.inspect_pod = lambda pod: {'desiredStatus': 'RUNNING', 'costPerHr': 2}
    state = client.guarded_lifecycle('fixture_pod', authorization(), path,
                                    clock=lambda: elapsed[0], pause=pause)
    assert state['status'] == 'STOP_UNCONFIRMED'
    assert state['stop_reason'] == 'provider_restarted_after_confirmed_stop'
    assert 'stopped_at_unix' not in state and 'stop_evidence' not in state


@pytest.mark.parametrize('key,value', [('max_total_usd', float('nan')), ('max_total_usd', float('inf')),
    ('max_runtime_seconds', float('inf')), ('max_runtime_seconds', True)])
def test_nonfinite_creation_authorization_never_sends_paid_post(key, value):
    calls = []
    session = SimpleNamespace(headers={}, request=lambda *a, **k: calls.append((a, k)))
    client = RunPodClient('explicit_fixture', session)
    payload = {'imageName': 'fixture@sha256:abc', 'name': 'fixture', 'gpuTypeIds': ['A100'],
               'gpuCount': 1, 'containerDiskInGb': 30, 'volumeMountPath': '/workspace'}
    permitted = {**authorization(), 'approved_payload': deepcopy(payload), key: value}
    with pytest.raises(RuntimeError):
        client.create_pod(payload, permitted)
    assert not calls

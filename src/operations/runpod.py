"""RunPod REST lifecycle guarded by concrete, limited spending authorization."""
from __future__ import annotations
import os
import math
import requests

API_ROOT = "https://rest.runpod.io/v1"


class RunPodClient:
    def __init__(self, api_key: str | None = None, session=None):
        key = api_key or os.environ.get("RUNPOD_API_KEY")
        if not key:
            raise RuntimeError("RUNPOD_API_KEY is required")
        self.session = session or requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {key}"})

    def _request(self, method, path, **kwargs):
        response = self.session.request(method, f"{API_ROOT}/{path}", timeout=60, **kwargs)
        if not response.ok:
            raise RuntimeError(f"RunPod {method} operation failed with HTTP {response.status_code}")
        return response.json() if response.content else {}

    def list_pods(self):
        return self._request("GET", "pods")

    def inspect_pod(self, pod_id: str):
        return self._request("GET", f"pods/{pod_id}")

    def create_pod(self, payload: dict, authorization: dict):
        required = {"imageName", "name", "gpuTypeIds", "gpuCount", "containerDiskInGb", "volumeMountPath"}
        if required - payload.keys():
            raise ValueError("Incomplete RunPod deployment payload")
        if authorization.get("approved") is not True or not authorization.get("authorization_id"):
            raise RuntimeError("Paid Pod creation requires an explicit authorization record")
        if payload != authorization.get("approved_payload"):
            raise RuntimeError("Deployment differs from the concrete approved payload")
        ceiling = authorization.get("max_total_usd", 0)
        if not isinstance(ceiling, (float, int)) or isinstance(ceiling, bool) or not math.isfinite(ceiling) or ceiling <= 0:
            raise RuntimeError("Paid deployment authorization needs a positive total ceiling")
        runtime=authorization.get("max_runtime_seconds",0)
        if type(runtime) not in (int,float) or not math.isfinite(runtime) or runtime<=0:
            raise RuntimeError("Paid deployment authorization needs a runtime limit")
        # Never retry creation blindly: a timeout may already have created a billable Pod.
        return self._request("POST", "pods", json=payload)

    def stop_pod(self, pod_id: str):
        return self._request("POST", f"pods/{pod_id}/stop")

    def resume_pod(self, pod_id: str, authorization: dict):
        if authorization.get("approved") is not True or authorization.get("pod_id") != pod_id:
            raise RuntimeError("Pod resume requires authorization for the selected billable Pod")
        return self._request("POST", f"pods/{pod_id}/start")

    @staticmethod
    def _stop_evidence(pod: dict) -> dict | None:
        """A requested desired status alone is not observed termination evidence."""
        actual = pod.get('actualStatus', pod.get('status'))
        desired = pod.get('desiredStatus')
        stopped = (actual in ('EXITED', 'STOPPED') and pod.get('runtime') is None) or (
            desired in ('EXITED', 'STOPPED') and 'runtime' in pod and pod['runtime'] is None)
        if not stopped:
            return None
        return {'actual_status': actual, 'desired_status': desired,
                'runtime_present': 'runtime' in pod, 'runtime_stopped': pod.get('runtime') is None}

    def guarded_lifecycle(self, pod_id: str, authorization: dict, state_path, *, poll_seconds: int = 30,
                          stop_confirmation_polls: int = 3, clock=None, pause=None, stop_requested=None):
        """Monitor an already approved Pod and stop it at conservative billing limits.

        Persist the first observation across controller restarts. Unavailable price
        evidence, status, or controller exceptions trigger a stop attempt. This
        controller must run outside the monitored Pod; it cannot guarantee stop if
        its own host disappears. An accepted stop request is confirmed by bounded
        provider observations; absent termination evidence returns STOP_UNCONFIRMED.
        Use the provider console as an independent fallback.
        """
        import time
        import math
        from pathlib import Path
        from shared_procedures.identity import content_hash
        from shared_procedures.storage import atomic_write_json, read_json
        clock=clock or time.time;pause=pause or time.sleep
        if authorization.get('approved') is not True or authorization.get('pod_id')!=pod_id or not authorization.get('authorization_id'):
            raise RuntimeError('Lifecycle monitoring requires authorization for this Pod')
        runtime=authorization.get('max_runtime_seconds');ceiling=authorization.get('max_total_usd')
        hourly=authorization.get('authorized_hourly_usd')
        if any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in (runtime,ceiling,hourly)) or not authorization.get('price_evidence'):
            raise RuntimeError('Lifecycle authorization requires finite runtime, total ceiling, hourly bound and actual price evidence')
        if type(poll_seconds) is not int or not 1<=poll_seconds<=60:
            raise ValueError('Lifecycle polling interval must be1..60seconds')
        if type(stop_confirmation_polls) is not int or not 1<=stop_confirmation_polls<=3:
            raise ValueError('Stop confirmation must use1..3provider observations')
        path=Path(state_path);identity=content_hash(authorization)
        if path.exists():
            state=read_json(path)
            if state.get('authorization_hash')!=identity or state.get('pod_id')!=pod_id:
                raise RuntimeError('Lifecycle state belongs to another Pod or authorization')
        else:
            # Include any time between provider creation and controller start.
            started=authorization.get('billing_started_at_unix')
            if type(started) not in (int,float) or not math.isfinite(started) or started>clock():
                raise RuntimeError('Actual billing start time is required')
            state={'pod_id':pod_id,'authorization_hash':identity,'started_at_unix':started,'status':'MONITORING'}
            atomic_write_json(path,state)
        def confirm_stop():
            state.pop('stopped_at_unix', None)
            state.pop('stop_evidence', None)
            for observation in range(1, stop_confirmation_polls+1):
                state['stop_confirmation_observations'] = observation
                try:
                    pod = self.inspect_pod(pod_id)
                    evidence = self._stop_evidence(pod)
                    state.pop('stop_confirmation_error', None)
                    if evidence is not None:
                        state.update(status='STOPPED', stopped_at_unix=clock(),
                                     stop_evidence={**evidence, 'observed_at_unix': clock()})
                        atomic_write_json(path, state)
                        return True
                except Exception as error:
                    # Persist only the type; provider exceptions may include credentials.
                    state['stop_confirmation_error'] = type(error).__name__
                atomic_write_json(path, state)
                if observation < stop_confirmation_polls:
                    pause(min(poll_seconds, 5))
            state.update(status='STOP_UNCONFIRMED', last_observed_at_unix=clock())
            atomic_write_json(path, state)
            return False

        def request_and_confirm_stop(reason):
            state.update(status='STOP_REQUESTED', stop_reason=reason)
            state.pop('stopped_at_unix', None)
            state.pop('stop_evidence', None)
            atomic_write_json(path, state)
            try:
                self.stop_pod(pod_id)
            except Exception as error:
                # A timed-out stop may already have been accepted. Observe the Pod
                # before deciding whether termination has actually been established.
                state['stop_request_error'] = type(error).__name__
                atomic_write_json(path, state)
            else:
                state.pop('stop_request_error', None)
            confirm_stop()
            return state

        try:
            if state['status'] in ('STOP_REQUESTED', 'STOP_UNCONFIRMED'):
                # Resume observation, rather than treating an accepted prior request
                # as completed or starting a fresh billing period.
                return request_and_confirm_stop(state.get('stop_reason', 'resume_unconfirmed_stop'))
            while True:
                elapsed=max(0,clock()-state['started_at_unix'])
                conservative_cost=hourly*(elapsed+poll_seconds)/3600
                pod=self.inspect_pod(pod_id)
                evidence=self._stop_evidence(pod)
                if evidence is not None:
                    # Revalidate even a previously confirmed saved STOPPED state:
                    # external actions may have restarted the Pod in the meantime.
                    if state['status'] == 'STOPPED':
                        return state
                    state.update(status='STOPPED', stop_reason='provider_already_stopped',
                                 stopped_at_unix=clock(), last_observed_at_unix=clock(),
                                 stop_evidence={**evidence, 'observed_at_unix': clock()})
                    atomic_write_json(path,state)
                    return state
                observed_rate=pod.get('costPerHr',pod.get('adjustedCostPerHr'))
                # Rates must be read from current provider evidence before another interval.
                if type(observed_rate) not in (int,float) or not math.isfinite(observed_rate) or observed_rate<0 or observed_rate>hourly:
                    reason='provider_price_unverified_or_above_authorized_bound'
                elif state['status'] == 'STOPPED':
                    reason='provider_restarted_after_confirmed_stop'
                elif pod.get('desiredStatus') in ('EXITED','STOPPED'):
                    reason='provider_stop_not_confirmed'
                elif pod.get('desiredStatus') != 'RUNNING' and pod.get('actualStatus',pod.get('status')) != 'RUNNING':
                    reason='provider_status_unverified'
                elif elapsed+poll_seconds>=runtime or conservative_cost>=ceiling:
                    reason='authorized_runtime_or_cost_limit'
                elif stop_requested and stop_requested(): reason='requested_safe_stop'
                else:
                    state.update(elapsed_seconds=elapsed,estimated_upper_cost_usd=conservative_cost,last_observed_at_unix=clock())
                    atomic_write_json(path,state);pause(poll_seconds);continue
                state.update(elapsed_seconds=elapsed,estimated_upper_cost_usd=conservative_cost)
                return request_and_confirm_stop(reason)
        except BaseException:
            try:
                request_and_confirm_stop('controller_failure')
            except Exception:
                state.update(status='STOP_UNCONFIRMED');atomic_write_json(path,state)
            raise

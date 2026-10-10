"""Disposable Docker executor for untrusted MBPP Python, never host execution.

The daemon and digest-pinned image are runtime prerequisites. No image is pulled
implicitly. Only an ephemeral read-only input directory is mounted; host secrets
and training artifacts are never mounted or passed in the container environment.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable
import json
import os
import subprocess
import tempfile
import uuid

from shared_procedures.identity import content_hash

SANDBOX_VERSION = "docker_mbpp_v1"
LIMITS = {"cpus": 1, "memory_bytes": 1073741824, "cpu_seconds": 5, "wall_seconds": 10}
# The trusted supervisor keeps original tests in its own root-only memory/files.
# Untrusted code runs as UID 65534 in a child with its stdout captured in a pipe.
# Child messages are primitive function return values, never correctness flags.
# Root capabilities are limited to changing child UID/GID and terminating it.
_WORKER = r'''import ast,base64,collections,collections.abc,contextlib,json,os,pickle,re,sys
source=open('/input/solution.py').read()
execute=exec
out=sys.stdout
inp=sys.stdin
namespace={'__name__':'__mbpp_solution__'}
objects={}
def encode_value(value,depth=0):
 if depth>64: raise ValueError('oversized nesting')
 kind=type(value)
 if kind in (type(None),bool,int,float,str,bytes,complex):
  representation=repr(value)
  ast.literal_eval(representation)
  return {'kind':'literal','value':representation}
 names={list:'list',tuple:'tuple',set:'set',frozenset:'frozenset',dict:'dict',collections.Counter:'counter',collections.OrderedDict:'ordered_dict',collections.deque:'deque',collections.defaultdict:'defaultdict'}
 if kind not in names: raise ValueError('unsupported value')
 packet={'kind':names[kind]}
 if isinstance(value,dict):
  packet['items']=[[encode_value(k,depth+1),encode_value(v,depth+1)] for k,v in value.items()]
  if kind is collections.defaultdict:
   factories={list:'list',set:'set',dict:'dict',tuple:'tuple',int:'int',float:'float',str:'str',bool:'bool',None:None}
   if value.default_factory not in factories: raise ValueError('unsupported factory')
   packet['factory']=factories[value.default_factory]
 else:
  packet['items']=[encode_value(v,depth+1) for v in value]
  if kind is collections.deque: packet['maxlen']=value.maxlen
 return packet
def encode(value):
 if type(value) is re.Match:
  handle=str(len(objects));objects[handle]=value
  return {'regex_match':handle}
 try:
  packet=encode_value(value)
  if len(json.dumps(packet))>1048576: raise ValueError('oversized return')
  return {'value':packet}
 except (ValueError,SyntaxError,RecursionError):
  if isinstance(value,collections.abc.Iterator):
   handle=str(len(objects));objects[handle]=value
   return {'handle':handle,'iterator':True}
  return {'unsupported':True}

try:
 with open(os.devnull,'w') as sink,contextlib.redirect_stdout(sink),contextlib.redirect_stderr(sink):
  execute(compile(source,'<generated_solution>','exec'),namespace)
 out.write(json.dumps({'ready':True})+'\n');out.flush()
except BaseException:
 out.write(json.dumps({'ready':False})+'\n');out.flush();sys.exit(1)
for line in inp:
 try:
  packet=json.loads(line)
  with open(os.devnull,'w') as sink,contextlib.redirect_stdout(sink),contextlib.redirect_stderr(sink):
   if packet.get('operation')=='match':
    if packet['method'] not in ('group','groups','groupdict','start','end','span','expand','__getitem__'): raise ValueError('unsupported match operation')
    args,kwargs=pickle.loads(base64.b64decode(packet['arguments']))
    result=getattr(objects[packet['handle']],packet['method'])(*args,**kwargs)
    reply=encode(result)
   elif packet.get('operation')=='next':
    result=next(objects[packet['handle']])
    reply=encode(result)
   elif packet.get('operation')=='iter':
    result=iter(objects[packet['handle']])
    reply=encode(result)
   else:
    args,kwargs=pickle.loads(base64.b64decode(packet['arguments']))
    result=namespace[packet['function']](*args,**kwargs)
    reply=encode(result)
    reply['arguments_after']=encode_value((args,kwargs))
 except StopIteration:
  reply={'stop':True}
 except BaseException:
  reply={'error':True}
 out.write(json.dumps(reply)+'\n');out.flush()
'''
_HARNESS = r'''import ast,base64,collections,json,os,pickle,signal,subprocess,sys,threading,time
if os.getpid()!=1 or os.geteuid()!=0: raise RuntimeError('trusted supervisor requires disposable namespace PID1')
inputs=json.load(sys.stdin)
original_tests=inputs['tests']
function_names=set(inputs['entrypoints'])
compiled_tests=[compile(t,'<original_mbpp_test>','exec') for t in original_tests]
if not function_names: raise RuntimeError('frozen tests lack approved callable entrypoints')
def cpu_usage():
 try:
  values=dict(line.split() for line in open('/sys/fs/cgroup/cpu.stat'))
  return int(values['usage_usec'])/1000000
 except (OSError,KeyError,ValueError):
  return None
origin=cpu_usage()
if origin is None: raise RuntimeError('cgroup CPU accounting unavailable')
def least_privilege():
 os.setgroups([])
 os.setgid(65534)
 os.setuid(65534)
started=time.monotonic()
process=subprocess.Popen([sys.executable,'-I','-B','/input/worker.py'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,preexec_fn=least_privilege,start_new_session=True,bufsize=1)
limit=[None]
stop=threading.Event()
def monitor():
 while not stop.wait(0.01):
  usage=cpu_usage()
  if usage is None or usage-origin>=5 or time.monotonic()-started>=10:
   limit[0]='wall_time_limit' if time.monotonic()-started>=10 else 'cpu_time_limit'
   try: os.kill(-1,signal.SIGKILL)
   except ProcessLookupError: pass
   return
thread=threading.Thread(target=monitor,daemon=True);thread.start()
class UnsupportedTransport(Exception): pass
def read_reply():
 line=process.stdout.readline(1048577)
 if not line or len(line)>1048576: raise ValueError('missing or oversized worker reply')
 reply=json.loads(line)
 if not isinstance(reply,dict): raise ValueError('invalid reply')
 return reply
def request(packet):
 process.stdin.write(json.dumps(packet)+'\n');process.stdin.flush()
 return read_reply()
class RemoteIterable:
 def __init__(self,handle,iterator): self.handle=handle;self.iterator=iterator
 def __iter__(self):
  return self if self.iterator else decode(request({'operation':'iter','handle':self.handle}))
 def __next__(self):
  return decode(request({'operation':'next','handle':self.handle}))
class RemoteRegexMatch:
 def __init__(self,handle): self.handle=handle
 def __bool__(self): return True
 def _invoke(self,name,args,kwargs):
  serialized=base64.b64encode(pickle.dumps((args,kwargs))).decode('ascii')
  return decode(request({'operation':'match','handle':self.handle,'method':name,'arguments':serialized}))
 def group(self,*args,**kwargs): return self._invoke('group',args,kwargs)
 def groups(self,*args,**kwargs): return self._invoke('groups',args,kwargs)
 def groupdict(self,*args,**kwargs): return self._invoke('groupdict',args,kwargs)
 def start(self,*args,**kwargs): return self._invoke('start',args,kwargs)
 def end(self,*args,**kwargs): return self._invoke('end',args,kwargs)
 def span(self,*args,**kwargs): return self._invoke('span',args,kwargs)
 def expand(self,*args,**kwargs): return self._invoke('expand',args,kwargs)
 def __getitem__(self,item): return self._invoke('__getitem__',(item,),{})
def decode_value(packet,depth=0):
 if depth>64 or not isinstance(packet,dict): raise UnsupportedTransport('invalid value packet')
 kind=packet.get('kind')
 if kind=='literal' and set(packet)=={'kind','value'}:
  try: return ast.literal_eval(packet['value'])
  except (ValueError,SyntaxError,TypeError): raise UnsupportedTransport('unsupported literal')
 fields={'kind','items'}
 if kind=='deque': fields.add('maxlen')
 if kind=='defaultdict': fields.add('factory')
 if set(packet)!=fields or not isinstance(packet.get('items'),list): raise UnsupportedTransport('unsupported value packet')
 if kind in ('dict','counter','ordered_dict','defaultdict'):
  pairs=[]
  for pair in packet['items']:
   if not isinstance(pair,list) or len(pair)!=2: raise UnsupportedTransport('invalid mapping')
   pairs.append((decode_value(pair[0],depth+1),decode_value(pair[1],depth+1)))
  if kind=='dict': return dict(pairs)
  if kind=='counter': return collections.Counter(dict(pairs))
  if kind=='ordered_dict': return collections.OrderedDict(pairs)
  factories={'list':list,'set':set,'dict':dict,'tuple':tuple,'int':int,'float':float,'str':str,'bool':bool,None:None}
  if packet['factory'] not in factories: raise UnsupportedTransport('unsupported mapping factory')
  return collections.defaultdict(factories[packet['factory']],pairs)
 values=[decode_value(value,depth+1) for value in packet['items']]
 constructors={'list':list,'tuple':tuple,'set':set,'frozenset':frozenset}
 if kind in constructors: return constructors[kind](values)
 if kind=='deque' and (packet['maxlen'] is None or type(packet['maxlen']) is int): return collections.deque(values,maxlen=packet['maxlen'])
 raise UnsupportedTransport('unsupported value kind')
def decode(reply):
 if set(reply)=={'value'}: return decode_value(reply['value'])
 if set(reply)=={'regex_match'} and isinstance(reply['regex_match'],str): return RemoteRegexMatch(reply['regex_match'])
 if set(reply)=={'handle','iterator'} and isinstance(reply['handle'],str) and type(reply['iterator']) is bool:
  return RemoteIterable(reply['handle'],reply['iterator'])
 if reply=={'stop':True}: raise StopIteration
 if reply=={'unsupported':True}: raise UnsupportedTransport('unsupported return value')
 raise ValueError('task call failed')
def synchronize(original,updated):
 if isinstance(original,list) and isinstance(updated,list):
  desired=[synchronize(original[index],value) if index<len(original) else value for index,value in enumerate(updated)]
  original[:]=desired
 elif isinstance(original,dict) and isinstance(updated,dict):
  desired={key:synchronize(original[key],value) if key in original else value for key,value in updated.items()}
  original.clear();original.update(desired)
 elif isinstance(original,set) and isinstance(updated,set):
  original.clear();original.update(updated)
 elif isinstance(original,tuple) and isinstance(updated,tuple):
  return tuple(synchronize(before,after) for before,after in zip(original,updated))
 else: return updated
 return original
def proxy(name):
 def invoke(*args,**kwargs):
  try: serialized=base64.b64encode(pickle.dumps((args,kwargs))).decode('ascii')
  except (TypeError,pickle.PicklingError): raise UnsupportedTransport('unsupported task argument')
  reply=request({'function':name,'arguments':serialized})
  after=reply.pop('arguments_after',None)
  if after is not None:
   try: updated_args,updated_kwargs=decode_value(after)
   except (ValueError,SyntaxError): raise UnsupportedTransport('unsupported mutation transport')
   synchronize(args,updated_args);synchronize(kwargs,updated_kwargs)
  return decode(reply)
 return invoke
result={'status':'outcome','correct':False,'reason':'task_failure'}
try:
 if read_reply()!={'ready':True}: raise ValueError('candidate load failed')
 namespace={'sys':sys,**{name:proxy(name) for name in function_names}}
 for test in compiled_tests: exec(test,namespace)
 result={'status':'outcome','correct':True,'reason':'passed'}
except UnsupportedTransport:
 result={'status':'infrastructure_failure','correct':None,'reason':'unsupported_value_transport'}
except BaseException:
 if limit[0]: result['reason']=limit[0]
finally:
 stop.set()
 try: os.kill(-1,signal.SIGKILL)
 except ProcessLookupError: pass
 process.wait()
result['candidate_wall_seconds']=time.monotonic()-started
print(json.dumps(result),flush=True)
'''


class DockerMBPPSandbox:
    def __init__(self, image: str, *, runner: Callable = subprocess.run, scratch_root: str | Path | None = None):
        if not isinstance(image, str) or "@sha256:" not in image or len(image.rsplit("@sha256:", 1)[1]) != 64:
            raise ValueError("MBPP image must be pinned by a verified sha256 digest")
        compile(_HARNESS, "<trusted_mbpp_supervisor>", "exec")
        compile(_WORKER, "<trusted_mbpp_worker>", "exec")
        self.image, self.runner = image, runner
        self.simulation = runner is not subprocess.run
        self.scratch_root = str(scratch_root) if scratch_root else None
        self._availability_proven = False

    @property
    def identity(self) -> dict:
        return {"version": SANDBOX_VERSION, "image": self.image, "limits": LIMITS,
                "harness_hash": content_hash(_HARNESS), "worker_hash": content_hash(_WORKER), "isolation": "docker_private_namespace"}

    def command(self, inputs: Path, name: str) -> list[str]:
        return ["docker", "run", "--interactive", "--rm", "--pull=never", "--name", name,
            "--network=none", "--user=0:0", "--read-only", "--cap-drop=ALL",
            "--cap-add=SETUID", "--cap-add=SETGID", "--cap-add=KILL", "--cgroupns=private",
            "--security-opt=no-new-privileges:true", "--pids-limit=64", "--cpus=1",
            "--memory=1g", "--memory-swap=1g", "--ulimit=cpu=5:5", "--ulimit=nofile=64:64",
            "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=64m", "--log-driver=none",
            "--mount", f"type=bind,source={inputs},target=/input,readonly",
            self.image, "python", "-I", "-B", "/input/harness.py"]

    def available(self) -> dict:
        try:
            daemon = self.runner(["docker", "info", "--format", "{{.ServerVersion}}"],
                                  capture_output=True, text=True, timeout=5, check=False)
            if daemon.returncode != 0:
                return {"status": "blocked", "reason": "docker_daemon_unavailable"}
            inspected = self.runner(["docker", "image", "inspect", self.image], capture_output=True,
                                     text=True, timeout=5, check=False)
            if inspected.returncode != 0:
                return {"status": "blocked", "reason": "digest_pinned_image_unavailable"}
            return {"status": "passed", "identity": self.identity}
        except (OSError, subprocess.SubprocessError):
            return {"status": "blocked", "reason": "docker_runtime_unavailable"}

    def execute(self, code: str, tests: list[str], *, entrypoints: list[str] | None = None) -> dict:
        if not isinstance(code, str) or not isinstance(tests, list) or not tests or any(not isinstance(t, str) for t in tests):
            raise ValueError("Sandbox requires exact extracted code and nonempty original tests")
        # Broken source tests are a frozen input failure rather than model error.
        for test in tests:
            compile(test, "<original_mbpp_test>", "exec")
        if entrypoints is None:
            import ast
            try:
                defined = {node.name for node in ast.parse(code).body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
            except SyntaxError:
                defined = set()
            called = {node.func.id for test in tests for node in ast.walk(ast.parse(test))
                      if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
            entrypoints = sorted(defined & called)
            # Malformed generated code still receives the frozen task's names
            # when provided by capability scoring. Direct callers with invalid
            # source are ordinary candidate failures, never repaired.
            if not entrypoints:
                return {"status": "outcome", "correct": False, "reason": "no_callable_entrypoint",
                        "sandbox": self.identity, "code_hash": content_hash(code),
                        "tests_hash": content_hash(tests), "simulation": self.simulation}
        if not isinstance(entrypoints, list) or not entrypoints or any(not isinstance(name,str) or not name.isidentifier() for name in entrypoints):
            raise ValueError("Sandbox entrypoints must be the approved public function names")
        dependencies = {"sandbox": self.identity, "code_hash": content_hash(code), "tests_hash": content_hash(tests), "simulation": self.simulation}
        availability = {"status": "passed"} if self._availability_proven else self.available()
        if availability["status"] != "passed":
            return {"status": "infrastructure_failure", "correct": None, "reason": availability["reason"], **dependencies}
        self._availability_proven = True
        with tempfile.TemporaryDirectory(prefix="mbpp_", dir=self.scratch_root) as directory:
            inputs = Path(directory)
            os.chmod(inputs, 0o755)
            for filename, contents in [("solution.py", code), ("harness.py", _HARNESS), ("worker.py", _WORKER)]:
                path = inputs / filename
                path.write_text(contents)
                os.chmod(path, 0o444)
            name = "mbpp_" + uuid.uuid4().hex
            try:
                process = self.runner(self.command(inputs, name), capture_output=True, text=True, input=json.dumps({"tests":tests,"entrypoints":entrypoints}),
                                      timeout=45, check=False)
            except subprocess.TimeoutExpired:
                self._cleanup(name)
                return {"status": "infrastructure_failure", "correct": None, "reason": "executor_watchdog_timeout", **dependencies}
            except (OSError, subprocess.SubprocessError):
                self._cleanup(name)
                return {"status": "infrastructure_failure", "correct": None, "reason": "executor_failure", **dependencies}
            if process.returncode in {125, 126, 127}:
                return {"status": "infrastructure_failure", "correct": None, "reason": "container_launch_failure", **dependencies}
            if process.returncode in {137, 152, -9, -24}:
                return {"status": "outcome", "correct": False, "reason": "task_resource_limit", **dependencies}
            if process.returncode != 0:
                return {"status": "infrastructure_failure", "correct": None, "reason": "trusted_executor_failure", **dependencies}
            try:
                result = json.loads(process.stdout)
            except (TypeError, ValueError):
                return {"status": "outcome", "correct": False, "reason": "task_terminated_without_result", **dependencies}
            if (set(result) not in ({"status", "correct", "reason"},{"status","correct","reason","candidate_wall_seconds"})
                or result["status"] not in {"outcome", "infrastructure_failure"}
                or (result["status"] == "outcome" and type(result["correct"]) is not bool)
                or (result["status"] == "infrastructure_failure" and result["correct"] is not None)):
                return {"status": "outcome", "correct": False, "reason": "invalid_task_completion", **dependencies}
            if "candidate_wall_seconds" in result:
                import math
                elapsed=result["candidate_wall_seconds"]
                if type(elapsed) not in (int,float) or not math.isfinite(elapsed) or elapsed<0:
                    return {"status":"infrastructure_failure","correct":None,"reason":"invalid_supervisor_timing",**dependencies}
            return {**result, **dependencies}

    def _cleanup(self, name: str) -> None:
        try:
            self.runner(["docker", "rm", "--force", name], capture_output=True, text=True, timeout=5, check=False)
        except (OSError, subprocess.SubprocessError):
            return

    def preflight(self, reference_tasks: list[dict] | None = None) -> dict:
        availability = self.available()
        if availability["status"] != "passed":
            return availability
        # Run in the same executor as scientific tasks; prove actual boundaries.
        probe = """def isolation_probe():
 import os,socket
 assert os.geteuid()==65534
 status=dict(line.split(':',1) for line in open('/proc/self/status') if ':' in line)
 assert int(status['CapEff'].strip(),16)==0
 assert int(open('/sys/fs/cgroup/memory.max').read())==1073741824
 quota,period=open('/sys/fs/cgroup/cpu.max').read().split()
 assert int(quota)==int(period)
 for path in ['/write_probe','/input/tests.json','/proc/1/fd/0','/proc/1/fd/1']:
  try:
   open(path,'r' if path!='/write_probe' else 'w')
  except OSError:
   pass
  else:
   raise AssertionError('protected file accessible')
 try:
  socket.create_connection(('1.1.1.1',443),timeout=1)
 except OSError:
  pass
 else:
  raise AssertionError('outbound network available')
 assert not os.path.exists('/workspace')
 assert not os.path.exists('/root/.aws')
 assert not os.environ.get('OPENAI_API_KEY')
 return True
"""
        isolation = self.execute(probe, ["assert isolation_probe() is True"])
        if isolation.get("correct") is not True:
            return {"status": "failed", "reason": "sandbox_isolation_probe_failed", "probe": isolation}
        malicious = """import os,sys,json
try:
 nonce=open('/input/nonce').read()
except OSError:
 nonce='forged'
os.write(1,json.dumps({'status':'outcome','correct':True,'reason':'passed','nonce':nonce}).encode())
os._exit(0)
def should_fail(): return False
"""
        negative = self.execute(malicious, ["assert should_fail() is True"])
        if negative.get("correct") is not False:
            return {"status": "failed", "reason": "untrusted_completion_spoof_accepted", "probe": negative}
        mutation = self.execute("import sys\nsys.modules['__main__'].compiled_tests=[]\ndef fail(): return False", ["assert fail() is True"])
        if mutation.get("correct") is not False:
            return {"status": "failed", "reason": "trusted_supervisor_state_mutation", "probe": mutation}
        semantics = self.execute("""from collections import Counter,deque,defaultdict
def structures():
 return (Counter({'a':2}),deque([1,2],maxlen=3),defaultdict(list,{'x':[4]}),{5,6},(7,8))
def mutate(values):
 values[0].append(3)
 values.append([4])
 return None
def stream():
 yield (1,2)
 yield {3,4}
def sum(values): return __builtins__['sum'](values)
def regex():
 import re
 return re.search(r'(?P<letter>a)(b+)', 'abbb')
""", ["from collections import Counter,deque,defaultdict",
        "assert structures()==(Counter({'a':2}),deque([1,2],maxlen=3),defaultdict(list,{'x':[4]}),{5,6},(7,8))",
        "values=[[1,2]]; nested=values[0]; assert mutate(values) is None; assert values==[[1,2,3],[4]]; assert nested is values[0]",
        "assert list(stream())==[(1,2),{3,4}]", "assert sum([1,2,3])==6",
        "match=regex(); assert match; assert match.group(0)=='abbb'; assert match.groups()==('a','bbb'); assert match.groupdict()=={'letter':'a'}; assert match.span()==(0,4); assert match[1]=='a'"])
        if semantics.get("correct") is not True:
            return {"status":"failed","reason":"supported_python_semantics_probe_failed","probe":semantics}
        resource_probes = {}
        for name, code in [
            ("cpu", "def exhaust():\n while True: pass"),
            ("wall", "def exhaust():\n import time\n time.sleep(20)\n return True"),
            ("descendant_wall", "def exhaust():\n import os,time\n if os.fork()==0:\n  os.setsid()\n  time.sleep(20)\n  os._exit(0)\n time.sleep(20)\n return True"),
            ("memory", "def exhaust():\n x=[bytearray(1024*1024) for _ in range(2048)]\n return True")]:
            probe_result = self.execute(code, ["assert exhaust() is True"])
            resource_probes[name] = probe_result
            if name=="descendant_wall" and (probe_result.get("reason")!="wall_time_limit" or not 0<=probe_result.get("candidate_wall_seconds",float("inf"))<=11):
                return {"status":"failed","reason":"descendant_wall_limit_escape","probe":probe_result}
            if probe_result.get("status") != "outcome" or probe_result.get("correct") is not False:
                return {"status": "failed", "reason": "resource_limit_probe_failed", "resource": name, "probe": probe_result}
        for task in reference_tasks or []:
            tests = ([task["test_setup_code"]] if task.get("test_setup_code") else []) + task["scoring_tests"]
            score = self.execute(task["reference_code"], tests)
            if score.get("correct") is not True:
                return {"status": "failed", "reason": "reference_tests_failed", "question_id": task.get("question_id"), "record": score}
        return {"status": "passed", "simulation": self.simulation, "live": not self.simulation, "identity": self.identity, "reference_tasks": len(reference_tasks or []),
                "isolation_probe": isolation, "early_exit_probe": negative,
                "supervisor_mutation_probe": mutation, "value_semantics_probe":semantics, "resource_probes": resource_probes}

DEFAULT_MBPP_IMAGE = "python@sha256:e88e9763f943ec1834f992a4b51e0f24500486803e8bc534e5767af9ea65f6ce"


def _selected_reference_binding(panel: dict) -> list[dict]:
    rows = [item for item in panel.get("items", []) if item.get("domain") == "mbpp"]
    if len(rows) != 203 or len({row["question_id"] for row in rows}) != 203:
        raise ValueError("Reference certification requires the exact fixed 203 MBPP items")
    return [{"question_id":row["question_id"], "source_content_hash":row["source_content_hash"],
             "source_revision":row["source_revision"], "public_interface_hash":row["public_interface_hash"],
             "code_hash":content_hash(row["reference_code"]),
             "tests_hash":content_hash(([row["test_setup_code"]] if row.get("test_setup_code") else []) + row["scoring_tests"])}
            for row in rows]


def verified_reference_callback(report_path: str | Path, *, panel: dict | str | Path,
                                image: str = DEFAULT_MBPP_IMAGE):
    """Certify only the immutable selected 203 from actual original suites.

    The exhaustive 427 diagnostic remains intact. An unselected reference that
    exceeds a frozen resource cap does not change the selected question IDs.
    """
    from shared_procedures.storage import read_json
    report = read_json(report_path)
    selected = _selected_reference_binding(read_json(panel) if isinstance(panel,(str,Path)) else panel)
    saved = report.get("report_hash")
    if saved != content_hash({k:v for k,v in report.items() if k != "report_hash"}):
        raise ValueError("MBPP reference report integrity failure")
    sandbox = DockerMBPPSandbox(image)
    if (report.get("simulation") is not False or report.get("live") is not True
        or report.get("source_count") != 427 or report.get("sandbox_identity") != sandbox.identity
        or len(report.get("records", [])) != 427 or len({row["task_id"] for row in report.get("records", [])}) != 427
        or report.get("selected_reference_status") != "passed" or report.get("selected_count") != 203
        or report.get("selected_passed_count") != 203 or report.get("selected_binding_hash") != content_hash(selected)):
        raise ValueError("The exact selected 203 original suites have not passed this real sandbox")
    scores = {(row["score"]["code_hash"], row["score"]["tests_hash"]): row["score"] for row in report["records"]}
    if any(scores.get((row["code_hash"],row["tests_hash"]),{}).get("correct") is not True for row in selected):
        raise ValueError("Selected reference success claim conflicts with retained outcomes")
    permitted={(row["code_hash"],row["tests_hash"]) for row in selected}
    def verified(code, tests):
        key=(content_hash(code),content_hash(tests))
        score = scores.get(key) if key in permitted else None
        return bool(score and score.get("status") == "outcome" and score.get("correct") is True)
    return verified


def run_mbpp_preflight(config: dict, output_dir: str | Path) -> dict:
    """Canonical actual CheckResult with isolation and fixed selected references.

    This performs free local validation only. It does not create infrastructure,
    pull an image, invoke model APIs, or generate scientific outcomes.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from shared_procedures.configuration import configuration_hash
    from shared_procedures.identity import file_hash
    from shared_procedures.storage import read_json, atomic_write_json, notify_artifact
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    image = config.get("evaluation", {}).get("mbpp", {}).get("image") or config.get("operations",{}).get("mbpp_image") or DEFAULT_MBPP_IMAGE
    sandbox = DockerMBPPSandbox(image)
    binding = {"live": False, "simulation": False, "configuration_hash": configuration_hash(config),
               "implementation_hash": file_hash(Path(__file__)), "sandbox_identity": sandbox.identity}
    availability = sandbox.available()
    if availability["status"] != "passed":
        return {"name": "mbpp_sandbox", "status": "blocked", "detail": availability["reason"], "evidence": binding}
    source = Path(config.get("runtime", {}).get("data_root", "data")) / "evaluation" / "official_pools.json"
    if not source.is_file():
        return {"name": "mbpp_sandbox", "status": "blocked", "detail": "Pinned sanitized MBPP source pool is missing", "evidence": binding}
    source_document = read_json(source)
    rows = source_document.get("pools",source_document).get("mbpp", [])
    if len(rows) != 427 or len({row["task_id"] for row in rows}) != 427:
        return {"name": "mbpp_sandbox", "status": "failed", "detail": "Invalid frozen sanitized MBPP source count/IDs", "evidence": binding}
    source_hash = file_hash(source)
    journal_identity = content_hash({"sandbox":sandbox.identity,"source_file_sha256":source_hash})
    journal = output / "reference_journal" / journal_identity
    fallback_journal = Path(tempfile.gettempdir()) / "illegibility_mbpp_reference_journal" / journal_identity
    journal.mkdir(parents=True,exist_ok=True)
    fallback_journal.mkdir(parents=True,exist_ok=True)
    reference_path = output / "reference_tests.json"
    report = read_json(reference_path) if reference_path.is_file() else None
    valid_cached = bool(report and report.get("source_file_sha256") == source_hash
        and report.get("source_count") == 427 and report.get("sandbox_identity") == sandbox.identity
        and report.get("simulation") is False and report.get("live") is True
        and report.get("report_hash") == content_hash({k:v for k,v in report.items() if k != "report_hash"}))
    if not valid_cached:
        def score(row):
            row_hash = content_hash(row)
            name = row_hash + ".json"
            for candidate in (journal/name,fallback_journal/name):
                if candidate.is_file():
                    cached = read_json(candidate)
                    if (cached.get("record_hash") != content_hash({key:value for key,value in cached.items() if key!="record_hash"})
                        or cached.get("task_id") != row["task_id"] or cached.get("source_hash") != row_hash
                        or cached.get("score",{}).get("sandbox") != sandbox.identity
                        or cached.get("score",{}).get("simulation") is not False):
                        raise ValueError("MBPP reference journal integrity/identity failure")
                    # Ordinary task outcomes are retained, including negatives.
                    # An unavailable executor can be retried after recovery.
                    if cached["score"].get("status") == "outcome":
                        return cached
            tests = (["\n".join(row["test_imports"])] if row.get("test_imports") else []) + list(row["test_list"])
            record = {"task_id": row["task_id"], "source_hash":row_hash,"score":sandbox.execute(row["code"],tests)}
            record["record_hash"] = content_hash(record)
            # The secondary journal preserves completed local checks if the
            # workspace runs out of space during an unrelated asset download.
            atomic_write_json(fallback_journal/name,record)
            try:
                atomic_write_json(journal/name,record)
            except OSError as error:
                import errno
                if error.errno != errno.ENOSPC:
                    raise
            return record
        with ThreadPoolExecutor(max_workers=8) as executor:
            records = [future.result() for future in as_completed([executor.submit(score,row) for row in rows])]
        records.sort(key=lambda row: row["task_id"])
        passed = sum(row["score"].get("correct") is True for row in records)
        report = {"status": "passed" if passed == 427 else "failed", "simulation": False, "live": True,
                  "source_file_sha256": source_hash, "source_count": 427, "passed_count": passed,
                  "sandbox_identity": sandbox.identity, "records": records}
        report["report_hash"] = content_hash(report)
        atomic_write_json(fallback_journal/"reference_tests.json",report)
        atomic_write_json(reference_path, report)
        notify_artifact(reference_path, {"kind": "mbpp_reference_preflight", "report_hash": report["report_hash"], "simulation": False})
    data_dir = source.parent
    requested_panel = config.get("evaluation",{}).get("mbpp",{}).get("panel_path")
    candidates = [Path(requested_panel)] if requested_panel else [data_dir/name for name in
                  ("capability_manifest.json","capability_panel.json","capability_panel_prepared.json")]
    panel_path = next((path for path in candidates if path.is_file()),None)
    if panel_path is None:
        return {"name":"mbpp_sandbox","status":"blocked","detail":"Frozen selected panel is missing; exhaustive reference diagnostics retained", "evidence":binding}
    selected = _selected_reference_binding(read_json(panel_path))
    scores = {(row["score"]["code_hash"],row["score"]["tests_hash"]):row["score"] for row in report["records"]}
    selected_failures = [row for row in selected if scores.get((row["code_hash"],row["tests_hash"]),{}).get("correct") is not True]
    report.update(selected_reference_status="passed" if not selected_failures else "failed", selected_count=203,
                  selected_passed_count=203-len(selected_failures), selected_binding_hash=content_hash(selected),
                  selected_binding=selected, selected_failures=selected_failures)
    report["report_hash"] = content_hash({key:value for key,value in report.items() if key!="report_hash"})
    atomic_write_json(reference_path,report)
    notify_artifact(reference_path,{"kind":"mbpp_reference_preflight","report_hash":report["report_hash"],"simulation":False})
    isolation = sandbox.preflight()
    atomic_write_json(output / "isolation_checks.json", isolation)
    binding.update(live=True, reference_report_hash=report["report_hash"], reference_report=str(reference_path),
                   source_file_sha256=source_hash, source_count=427, passed_reference_count=report["passed_count"],
                   exhaustive_reference_status=report["status"], selected_reference_status=report["selected_reference_status"],
                   selected_count=203, selected_passed_count=report["selected_passed_count"], selected_binding_hash=content_hash(selected),
                   isolation=isolation, failures=[row for row in report["records"] if row["score"].get("correct") is not True],
                   selected_failures=selected_failures)
    status = "passed" if not selected_failures and isolation["status"] == "passed" else "failed"
    return {"name": "mbpp_sandbox", "status": status,
            "detail": "Actual restricted executor isolation and exact selected 203 original reference suites " + status,
            "evidence": binding}

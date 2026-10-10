import subprocess,sys
import pytest
from shared_procedures.identity import content_hash,derive_seed,canonical_json

def test_hash_is_order_independent_and_bytes_exact():
    assert content_hash({'b':2,'a':1})==content_hash({'a':1,'b':2})
    assert content_hash('a')==content_hash(b'a')
    assert content_hash('a')!=content_hash('a\n')

def test_seed_stable_across_process():
    expression="from shared_procedures.identity import derive_seed;print(derive_seed(42,'id',1))"
    output=subprocess.check_output([sys.executable,'-c',expression],text=True)
    assert int(output)==derive_seed(42,'id',1)
    assert derive_seed(42,'id',1)!=derive_seed(42,'id',2)

def test_nonfinite_not_hashable():
    with pytest.raises(ValueError): canonical_json({'x':float('nan')})

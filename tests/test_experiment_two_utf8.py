import os
import subprocess
import sys
from pathlib import Path


def test_utf8_entry_overrides_gbk_for_parent_and_child_without_running_experiment(tmp_path):
    root=Path(__file__).resolve().parents[1]
    entry=root/'tools/run_experiment_two_utf8.py'
    assert entry.exists(), 'UTF-8 entry missing'
    code='''import runpy,sys,types,subprocess
module=types.ModuleType('run_experiment_two_sharded')
def fake_main():
    assert sys.argv[1:]==['--batch-id','dummy','--run','--resume']
    print('parent: \\U0001f937',flush=True)
    child=subprocess.run([sys.executable,'-c',"print('child: \\U0001f937')"],capture_output=True)
    assert child.returncode==0,child.stderr
    assert child.stdout.decode('utf-8').strip()=='child: \\U0001f937'
    print('child OK',flush=True)
    return 0
module.main=fake_main
sys.modules['run_experiment_two_sharded']=module
sys.argv=[sys.argv[1],'--batch-id','dummy','--run','--resume']
runpy.run_path(sys.argv[0],run_name='__main__')
'''
    result=subprocess.run([sys.executable,'-c',code,str(entry)],cwd=root,
        env={**os.environ,'PYTHONIOENCODING':'gbk'},capture_output=True)
    assert result.returncode==0,result.stderr.decode('utf-8',errors='replace')
    assert 'parent: 🤷' in result.stdout.decode('utf-8')
    assert 'child OK' in result.stdout.decode('utf-8')

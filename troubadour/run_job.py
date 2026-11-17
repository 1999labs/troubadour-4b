"""Run one local job with logs, AC/battery checks, and persistent status."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import re
import signal
import subprocess
import time


def power():
    text=subprocess.run(['pmset','-g','batt'],capture_output=True,text=True).stdout
    m=re.search(r'(\d+)%;',text)
    return {'percent':int(m[1]) if m else None,'status':text.split(';')[1].strip() if ';' in text else 'unknown','ac':'AC Power' in text}


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    cmd=a.command[1:] if a.command and a.command[0]=='--' else a.command
    if not cmd:raise SystemExit('Missing command')
    Path('runs').mkdir(exist_ok=True);status=Path('runs',a.name+'.status.json')
    start=time.time();payload={'name':a.name,'command':cmd,'started':datetime.now(timezone.utc).isoformat(),'status':'running'}
    with Path('runs',a.name+'.log').open('a') as log:
        child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
        awake=subprocess.Popen(['caffeinate','-i','-w',str(child.pid)])
        try:
            while child.poll() is None:
                batt=power();payload.update({'power':batt,'elapsed_seconds':time.time()-start,'pid':child.pid,'updated':datetime.now(timezone.utc).isoformat()})
                temporary=status.with_suffix('.tmp');temporary.write_text(json.dumps(payload,indent=2));temporary.replace(status)
                if batt['percent'] is not None and batt['percent']<=18 and batt['status'] not in ('charging','charged'):
                    child.send_signal(signal.SIGINT)
                    try:child.wait(timeout=30)
                    except subprocess.TimeoutExpired:child.terminate();child.wait(timeout=10)
                    payload['status']='paused_low_battery';break
                time.sleep(10)
            else:payload['status']='complete' if child.returncode==0 else 'failed'
            payload.update({'exit_code':child.poll(),'elapsed_seconds':time.time()-start})
            temporary=status.with_suffix('.tmp');temporary.write_text(json.dumps(payload,indent=2));temporary.replace(status)
        finally:
            if child.poll() is None:child.terminate()
            if awake.poll() is None:awake.terminate()
    print(json.dumps(payload,indent=2))
    if payload['status']!='complete':raise SystemExit(1)

if __name__=='__main__':main()

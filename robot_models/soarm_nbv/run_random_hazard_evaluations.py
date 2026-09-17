"""Sequential GUI evaluation with process/GPU recovery gates; no policy tuning."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time

from soarm_nbv.audit_random_hazard_eval import audit

FRAMEWORK = Path('/home/iy/Isaac/Go2_Intelligence_Framework')
POLICY = '/home/iy/Isaac/Robotics/data/smolvla_runs/binary_tree_hazard_102_v2/checkpoints/020000/pretrained_model'
PROCESS_PATTERN = r'go2_soarm.py|run_active_slam_.*ros2.sh|run_binary_tree_.*demo.sh|smolvla_hazard_policy_runner.py|binary_tree_hazard_supervisor|rtabmap|rviz2|leader_bridge_7dof.py'
child = None


def emit(event, **data):
    print(json.dumps(dict(event=event, **data), ensure_ascii=False), flush=True)


def command(args):
    return subprocess.check_output(args, text=True, timeout=15).strip()


def gpu_memory():
    return int(command(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits']).splitlines()[0])


def lingering():
    result = subprocess.run(['pgrep', '-af', PROCESS_PATTERN], capture_output=True, text=True)
    if result.returncode not in (0, 1):
        raise RuntimeError('Process inspection failed')
    return result.stdout.strip()


def recover(baseline):
    consecutive = 0
    for _ in range(36):
        processes = lingering()
        memory = gpu_memory()
        consecutive = consecutive + 1 if not processes and memory <= baseline + 128 else 0
        if consecutive >= 3:
            emit('RECOVERED', gpu_mib=memory, consecutive_checks=consecutive)
            return
        time.sleep(5)
    raise RuntimeError('GPU/process recovery failed; no next trial will launch')


def stop(signum, frame):
    if child is not None and child.poll() is None:
        child.send_signal(signal.SIGTERM)
    raise KeyboardInterrupt


def main():
    global child
    parser = argparse.ArgumentParser()
    parser.add_argument('plan', type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    root = args.plan.parent
    if lingering():
        raise RuntimeError('Existing simulator/robotics process; refusing batch')
    baseline = gpu_memory()
    emit('BASELINE', gpu_mib=baseline, memory=command(['free', '-h']))
    start_trial = int(plan.get('start_trial', 1))
    if start_trial < 1:
        raise ValueError('start_trial must be positive')
    for index, pattern in enumerate(plan['patterns'], start_trial):
        recover(baseline)
        compute = command(['nvidia-smi', '--query-compute-apps=pid,used_memory,name', '--format=csv,noheader'])
        if compute:
            raise RuntimeError('Another GPU compute process is present: ' + compute)
        trial = root / f'trial{index:02d}_{pattern}'
        trial.mkdir(exist_ok=False)
        env = os.environ.copy()
        env.update(HEADLESS='0', DISPLAY=':0', XAUTHORITY='/run/user/1000/gdm/Xauthority',
                   BINARY_TREE_VLA_ARM_POLICY='1', BINARY_TREE_MANUAL_ARM_TELEOP='0',
                   BINARY_TREE_COLLECT_NBV_TEACHER='0', BINARY_TREE_COLLECT_HUMAN='0',
                   BINARY_TREE_CUBE_PATTERN=pattern, BINARY_TREE_OPEN_RETURN_CONNECTORS='1',
                   BINARY_TREE_ROUTE_SEED='47', BINARY_TREE_SEED='47',
                   BINARY_TREE_RUN_ROOT=str(trial), BINARY_TREE_USD=str(trial/'map.usda'),
                   BINARY_TREE_LAYOUT=str(trial/'layout.json'), BINARY_TREE_SMOLVLA_POLICY_PATH=POLICY,
                   BINARY_TREE_NBV_COLLECTION_LAPS='1', BINARY_TREE_NBV_COLLECTION_START_LAP='1')
        emit('START', trial=index, pattern=pattern, gpu_mib=gpu_memory(), memory=command(['free', '-h']))
        started = time.time()
        with (trial/'launcher.log').open('x') as log:
            child = subprocess.Popen(['bash', './scripts/run_binary_tree_vision_demo.sh'],
                                     cwd=FRAMEWORK, env=env, stdout=log, stderr=subprocess.STDOUT)
            while child.poll() is None:
                time.sleep(5)
        rc = child.returncode
        child = None
        policy_log = Path('/tmp/soarm_smolvla_hazard_policy.log')
        if policy_log.exists() and policy_log.stat().st_mtime >= started:
            shutil.copy2(policy_log, trial/'policy_runner.log')
        result = audit(trial)
        result.update(exit_code=rc, wall_seconds=time.time()-started)
        (trial/'audit.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
        emit('FINISHED', trial=index, pattern=pattern, exit_code=rc,
             passed=result['signal_and_route_audit_pass'], state=result['last_state'], holds=result['holds'])
        recover(baseline)
        if rc not in (0, 1) or (rc == 1 and not result['holds']):
            raise RuntimeError('Unexpected process failure, batch stopped for inspection')
        # Deliberately leave a cooldown/recording interval, even after recovery.
        if index < start_trial + len(plan['patterns']) - 1:
            time.sleep(30)
    emit('BATCH_DONE', count=len(plan['patterns']))


if __name__ == '__main__':
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        main()
    finally:
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)
            child.wait(timeout=60)

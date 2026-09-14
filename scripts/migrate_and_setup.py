import os
import shutil
import glob

RT = '/home/sengar/workspace_unified_build/repo_runtime'
BENCH = '/home/sengar/workspace_unified_build/repo_benchmark'
TARGET = '/home/sengar/edge-quality-intelligence'

print('Step 3: Populating from repo_runtime...')

runtime_copies = [
    ('src/policy.py', 'src/runtime/policy.py'),
    ('src/sensor_simulator.py', 'src/runtime/sensor_simulator.py'),
    ('src/inference_service.py', 'src/runtime/inference_service.py'),
    ('src/spooler.py', 'src/runtime/spooler.py'),
    ('src/mqtt_publisher.py', 'src/runtime/mqtt_publisher.py'),
    ('src/mqtt_subscriber.py', 'src/runtime/mqtt_subscriber.py'),
    ('src/audit_log.py', 'src/runtime/audit_log.py'),
    ('src/evidence_manager.py', 'src/runtime/evidence_manager.py'),
    ('src/stream_models.py', 'src/runtime/stream_models.py'),
    ('src/trace_replay.py', 'src/runtime/trace_replay.py'),
    ('src/fault_injector.py', 'src/runtime/fault_injector.py'),
    ('src/config.py', 'src/runtime/config.py'),
    ('src/config.py', 'src/config.py'),
    ('src/metrics/queue_model.py', 'src/metrics/queue_model.py'),
    ('src/metrics/evaluator.py', 'src/metrics/evaluator.py'),
    ('src/metrics/significance.py', 'src/metrics/significance.py'),
]

for src_rel, dst_rel in runtime_copies:
    s = os.path.join(RT, src_rel)
    d = os.path.join(TARGET, dst_rel)
    os.makedirs(os.path.dirname(d), exist_ok=True)
    shutil.copy2(s, d)
    print(f'Copied {src_rel} -> {dst_rel}')

# Copy runtime configs and scenarios
if os.path.isdir(os.path.join(RT, 'configs/scenarios')):
    shutil.copytree(os.path.join(RT, 'configs/scenarios'), os.path.join(TARGET, 'configs/scenarios'), dirs_exist_ok=True)
    print('Copied configs/scenarios')

for cfg_file in glob.glob(os.path.join(RT, 'configs/*.*')):
    shutil.copy2(cfg_file, os.path.join(TARGET, 'configs', os.path.basename(cfg_file)))
    print(f'Copied config: {os.path.basename(cfg_file)}')

# Copy tests from repo_runtime
existing_tests = set(os.listdir(os.path.join(TARGET, 'tests')))
for t_file in glob.glob(os.path.join(RT, 'tests/test_*.py')):
    base = os.path.basename(t_file)
    if base in existing_tests:
        dst_name = f'test_rt_{base[5:]}'
        print(f'Collision resolved: {base} -> {dst_name}')
    else:
        dst_name = base
    shutil.copy2(t_file, os.path.join(TARGET, 'tests', dst_name))
    print(f'Copied test: {base} -> {dst_name}')

print('Step 3 completed successfully.')

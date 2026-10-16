#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import requests

PROJECT = Path('/home/aimldl/LH-VLN')
NAV_GEN = PROJECT / 'nav_gen'
RUN_ROOT = NAV_GEN / 'runs/audit_regenerate_cs_aligned_v5_levels2_5_downstream_success2500_20260704'
LOGS_DIR = RUN_ROOT / 'logs'
MANIFEST_JSONL = LOGS_DIR / 'downstream_manifest.jsonl'
MANIFEST_CSV = LOGS_DIR / 'downstream_manifest.csv'
SUMMARY_PATH = LOGS_DIR / 'latest_summary.json'
SPLIT_DIR = RUN_ROOT / 'split'
SPLIT_BATCH_DIR = RUN_ROOT / 'split_batches'
STAGING_DIR = RUN_ROOT / 'staging'
STEP_TASK_PATH = RUN_ROOT / 'step_task'
TRAIL_LIST = SPLIT_DIR / 'trail_list.txt'
RAM_MODEL = PROJECT / 'data/models/ram_plus_swin_large_14m.pth'
PROMPT_PATH = NAV_GEN / 'prompt'
RAM_LOGS = LOGS_DIR / 'step_task_logs.txt'
REQUEST_LOG = LOGS_DIR / 'step_qwen_requests.jsonl'
PROGRESS_LOG = LOGS_DIR / 'step_qwen_progress.jsonl'
API_BASE = os.environ.get('OLLAMA_API_BASE', 'http://localhost:11434/v1')
MODEL = os.environ.get('OLLAMA_MODEL', 'qwen2.5:7b-instruct')
API_KEY = os.environ.get('OLLAMA_API_KEY', 'dummy')
STEP_TEMPERATURE = float(os.environ.get('STEP_TEMPERATURE', '0.2'))
STEP_MAX_TOKENS = int(os.environ.get('STEP_MAX_TOKENS', '256'))
STEP_TIMEOUT = int(os.environ.get('STEP_TIMEOUT', '180'))
MAPPING_POLICY = 'range_based_target_and_start_pose'
EXPECTED_TASKS = 1294
SOURCES = [
    ('v2', NAV_GEN / 'runs/audit_regenerate_cs_aligned_v5_levels2_5_habitat_validate_2300_qwen7b_20260701_v2/combined_task'),
    ('topup', NAV_GEN / 'runs/audit_regenerate_cs_aligned_v5_levels2_5_habitat_validate_topup2500_qwen7b_20260703/combined_task'),
]


def now() -> str:
    return time.strftime('%Y-%m-%d %H:%M:%S')


def ensure_dirs() -> None:
    for p in [RUN_ROOT, LOGS_DIR, SPLIT_DIR, SPLIT_BATCH_DIR, STAGING_DIR, STEP_TASK_PATH]:
        p.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)


def safe_name(text: str, max_len: int = 90) -> str:
    s = re.sub(r'[^A-Za-z0-9._-]+', '_', text).strip('._-')
    return (s[:max_len].strip('._-') or 'task')


def task_status(task_dir: Path) -> str:
    if list(task_dir.glob('success/trial_*/task.json')):
        return 'success'
    if list(task_dir.glob('fail/trial_*')):
        return 'fail'
    if (task_dir / 'temp').exists():
        return 'temp'
    return 'pending'


def discover_success() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_name, combined in SOURCES:
        for task_json in sorted(combined.glob('*/*/success/trial_*/task.json')):
            task_dir = task_json.parents[2]
            level = task_dir.parent.name
            task_name = task_dir.name
            rows.append({
                'id': f'{source_name}_{len(rows):05d}_{safe_name(task_name, 48)}',
                'source_run': source_name,
                'level': level,
                'task_name': task_name,
                'source_path': str(task_dir),
                'source_task_json': str(task_json),
                'split_status': 'pending',
                'split_batch': '',
                'split_trail_count': 0,
                'step_status': 'pending',
            })
    return rows


def read_manifest() -> list[dict[str, Any]]:
    if not MANIFEST_JSONL.exists():
        return []
    rows = []
    with MANIFEST_JSONL.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_manifest(rows: list[dict[str, Any]]) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST_JSONL.with_suffix('.jsonl.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    tmp.replace(MANIFEST_JSONL)
    fields = ['id', 'source_run', 'level', 'task_name', 'source_path', 'split_status', 'split_batch', 'split_trail_count', 'step_status']
    with MANIFEST_CSV.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def trail_count(path: Path = TRAIL_LIST) -> int:
    if not path.exists():
        return 0
    with path.open('r', encoding='utf-8') as f:
        return sum(1 for line in f if line.strip())


def step_json_count() -> int:
    if not STEP_TASK_PATH.exists():
        return 0
    return sum(1 for _ in STEP_TASK_PATH.glob('step_*.json'))


def batch_files() -> list[Path]:
    return sorted(SPLIT_BATCH_DIR.glob('batch_*.trail_list.txt'))


def merge_trails() -> int:
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = TRAIL_LIST.with_suffix('.txt.tmp')
    total = 0
    with tmp.open('w', encoding='utf-8') as out:
        for bf in batch_files():
            with bf.open('r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        out.write(line.rstrip('\n') + '\n')
                        total += 1
    tmp.replace(TRAIL_LIST)
    return total


def summary(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    rows = read_manifest()
    total = len(rows)
    split_done = sum(1 for r in rows if r.get('split_status') == 'done')
    split_failed = sum(1 for r in rows if r.get('split_status') == 'failed')
    data = {
        'run_root': str(RUN_ROOT),
        'expected_new_success_tasks': EXPECTED_TASKS,
        'manifest_count': total,
        'split_done': split_done,
        'split_pending': total - split_done - split_failed,
        'split_failed': split_failed,
        'trail_list': str(TRAIL_LIST),
        'trail_list_count': trail_count(),
        'step_task_path': str(STEP_TASK_PATH),
        'step_task_json_count': step_json_count(),
        'status': 'completed' if total == EXPECTED_TASKS and split_done == total and trail_count() > 0 and step_json_count() >= trail_count() else 'running',
        'updated_at': now(),
    }
    if extra:
        data.update(extra)
    write_json(SUMMARY_PATH, data)
    return data


def prepare() -> int:
    ensure_dirs()
    rows = read_manifest()
    if not rows:
        rows = discover_success()
        if len(rows) != EXPECTED_TASKS:
            write_manifest(rows)
            summary({'stage': 'prepare', 'preflight_error': f'expected {EXPECTED_TASKS} success tasks, found {len(rows)}'})
            print(f'preflight_failed: expected {EXPECTED_TASKS}, found {len(rows)}')
            return 2
        write_manifest(rows)
    if not RAM_MODEL.exists():
        summary({'stage': 'prepare', 'preflight_error': f'missing RAM model: {RAM_MODEL}'})
        return 3
    if not PROMPT_PATH.exists():
        summary({'stage': 'prepare', 'preflight_error': f'missing prompt path: {PROMPT_PATH}'})
        return 4
    print(json.dumps(summary({'stage': 'prepare', 'preflight': 'ok'}), indent=2, ensure_ascii=False))
    return 0


def make_stage(batch_id: str, rows: list[dict[str, Any]]) -> Path:
    stage_root = STAGING_DIR / batch_id / 'task'
    if stage_root.exists():
        shutil.rmtree(stage_root)
    stage_root.mkdir(parents=True, exist_ok=True)
    for row in rows:
        level_dir = stage_root / row['level']
        level_dir.mkdir(parents=True, exist_ok=True)
        link = level_dir / row['id']
        os.symlink(row['source_path'], link, target_is_directory=True)
    return stage_root


def run_split_batch(batch_size: int) -> int:
    rc = prepare()
    if rc != 0:
        return rc
    rows = read_manifest()
    pending = [r for r in rows if r.get('split_status') != 'done']
    if not pending:
        merge_trails()
        print(json.dumps(summary({'stage': 'split', 'message': 'no split pending'}), indent=2, ensure_ascii=False))
        return 0
    selected = pending[:batch_size]
    batch_index = len(batch_files())
    batch_id = f'batch_{batch_index:04d}'
    stage_task = make_stage(batch_id, selected)
    out_path = SPLIT_BATCH_DIR / f'{batch_id}.trail_list.txt'
    tmp_out = out_path.with_suffix('.txt.tmp')
    sys.path.insert(0, str(NAV_GEN))
    os.chdir(NAV_GEN)
    import split_task
    args = SimpleNamespace(
        API_KEY='',
        task_path=str(stage_task) + '/',
        step_task_path=str(STEP_TASK_PATH) + '/',
        split_save_path=str(tmp_out),
        ram_model=str(RAM_MODEL),
        ram_logs=str(RAM_LOGS),
        prompt_path=str(PROMPT_PATH) + '/',
    )
    append_jsonl(PROGRESS_LOG, {'phase': 'split_batch_start', 'batch_id': batch_id, 'task_count': len(selected), 'time': now()})
    try:
        split_task.split_traj(args)
        tmp_out.replace(out_path)
    except Exception as exc:
        for row in rows:
            if row['id'] in {r['id'] for r in selected}:
                row['split_status'] = 'failed'
                row['split_error'] = repr(exc)
        write_manifest(rows)
        summary({'stage': 'split', 'last_batch': batch_id, 'last_error': repr(exc)})
        append_jsonl(PROGRESS_LOG, {'phase': 'split_batch_failed', 'batch_id': batch_id, 'error': repr(exc), 'time': now()})
        raise
    count = trail_count(out_path)
    selected_ids = {r['id'] for r in selected}
    for row in rows:
        if row['id'] in selected_ids:
            row['split_status'] = 'done'
            row['split_batch'] = batch_id
            row['split_trail_count'] = count
    write_manifest(rows)
    total_trails = merge_trails()
    data = summary({'stage': 'split', 'last_batch': batch_id, 'last_batch_task_count': len(selected), 'last_batch_trail_count': count, 'trail_list_count': total_trails})
    append_jsonl(PROGRESS_LOG, {'phase': 'split_batch_done', 'batch_id': batch_id, 'task_count': len(selected), 'trail_count': count, 'total_trails': total_trails, 'time': now()})
    print(json.dumps(data, indent=2, ensure_ascii=False))
    return 0


def load_trails() -> list[list[dict[str, Any]]]:
    trails: list[list[dict[str, Any]]] = []
    with TRAIL_LIST.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                trails.append(ast.literal_eval(line.strip()))
    return trails


def config_object_label(value: Any) -> str:
    if isinstance(value, list) and value:
        return str(value[0])
    if isinstance(value, dict):
        return str(value.get('name') or value.get('object') or value)
    return str(value)


def object_labels(config: dict[str, Any]) -> list[str]:
    return [config_object_label(x) for x in config['Object']]


def trial_ranges(config: dict[str, Any]) -> list[dict[str, Any]]:
    ranges: list[dict[str, Any]] = []
    labels = object_labels(config)
    step_former = -1
    for index, obj in enumerate(labels):
        trial_name = 'trial_' + str(index)
        pos_len = len(config['trial'][trial_name]['pos'])
        start = step_former + 1
        end = step_former + pos_len
        ranges.append({'index': index, 'object': obj, 'region': config['Region'][index], 'trial_name': trial_name, 'pos_len': pos_len, 'global_start': start, 'global_end': end, 'step_former': step_former})
        step_former += pos_len - 1
    return ranges


def resolve_trials(config: dict[str, Any], trail: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    global_start = trail[0]['start']
    global_end = trail[-1]['end']
    target = str(trail[0]['target'])
    ranges = trial_ranges(config)
    same_name = [item for item in ranges if item['object'] == target]
    if not same_name:
        raise ValueError({'reason': 'target_trial_resolution_failed_no_object', 'target': target, 'candidate_count': 0, 'objects': object_labels(config)})

    resolution = 'unique_object'
    if len(same_name) == 1:
        target_trial = same_name[0]
    else:
        end_matches = [item for item in same_name if item['global_start'] <= global_end <= item['global_end']]
        if len(end_matches) == 1:
            target_trial = end_matches[0]
            resolution = 'end_containment'
        else:
            def overlap(item: dict[str, Any]) -> int:
                return max(0, min(global_end, item['global_end']) - max(global_start, item['global_start']) + 1)
            max_overlap = max(overlap(item) for item in same_name)
            overlap_matches = [item for item in same_name if overlap(item) == max_overlap and max_overlap > 0]
            if len(overlap_matches) == 1:
                target_trial = overlap_matches[0]
                resolution = 'max_overlap'
            else:
                pool = overlap_matches if overlap_matches else same_name
                target_trial = min(pool, key=lambda item: min(abs(global_end - item['global_start']), abs(global_end - item['global_end'])))
                resolution = 'nearest_end'

    start_candidates = [item for item in ranges if item['global_start'] <= global_start <= item['global_end']]
    if len(start_candidates) == 1:
        start_pose_trial = start_candidates[0]
    elif target_trial in start_candidates:
        start_pose_trial = target_trial
    elif start_candidates:
        start_pose_trial = start_candidates[-1]
    else:
        raise ValueError({'reason': 'start_pose_trial_resolution_failed', 'target': target, 'global_start': global_start})
    local_start = global_start - start_pose_trial['step_former'] - 1
    if not (0 <= local_start < start_pose_trial['pos_len']):
        raise ValueError({'reason': 'local_start_out_of_range', 'local_start': local_start, 'start_pose_trial': start_pose_trial})
    resolution_info = {
        'target_duplicate_count': len(same_name),
        'duplicate_target_resolution': resolution if len(same_name) > 1 else None,
        'segment_global_start': global_start,
        'segment_global_end': global_end,
    }
    return target_trial, start_pose_trial, resolution_info


def clean_instruction(text: str) -> str:
    text = text.strip().strip('`').strip()
    if text.lower().startswith('instruction:'):
        text = text.split(':', 1)[1].strip()
    return ' '.join(text.split()).replace('/', ' ').replace('\\', ' ')


def safe_slug(text: str, max_len: int = 80) -> str:
    slug = re.sub(r'[^A-Za-z0-9._ -]+', '_', text)
    slug = re.sub(r'\s+', ' ', slug).strip(' ._')
    return (slug[:max_len].strip(' ._') or 'instruction')


def output_path_for(index: int, instruction: str) -> Path:
    return STEP_TASK_PATH / f'step_{index:05d}_{safe_slug(instruction)}.json'


def existing_output_for(index: int) -> Path | None:
    matches = sorted(STEP_TASK_PATH.glob(f'step_{index:05d}_*.json'))
    return matches[0] if matches else None


def check_ollama() -> None:
    resp = requests.get(f'{API_BASE.rstrip("/")}/models', timeout=10)
    resp.raise_for_status()
    models = [m.get('id') for m in resp.json().get('data', [])]
    if MODEL not in models:
        raise RuntimeError(f'Model {MODEL} not found; available={models}')


def ollama_generate(prompt_path: Path, ex_prompt: str, segment_index: int) -> str:
    prompt_text = prompt_path.read_text(encoding='utf-8')
    payload = {
        'model': MODEL,
        'messages': [
            {'role': 'system', 'content': 'You generate concise navigation instructions. Return only the instruction text.'},
            {'role': 'user', 'content': prompt_text + '\n\nINPUT:\n' + ex_prompt},
        ],
        'temperature': STEP_TEMPERATURE,
        'max_tokens': STEP_MAX_TOKENS,
    }
    started = time.time()
    last_exc: Exception | None = None
    for attempt in range(1, 4):
        try:
            response = requests.post(
                f'{API_BASE.rstrip("/")}/chat/completions',
                headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {API_KEY}'},
                json=payload,
                timeout=STEP_TIMEOUT,
            )
            response.raise_for_status()
            text = clean_instruction(response.json()['choices'][0]['message']['content'])
            append_jsonl(REQUEST_LOG, {'segment_index': segment_index, 'attempt': attempt, 'status': 'ok', 'elapsed_sec': round(time.time() - started, 3), 'instruction_chars': len(text), 'time': now()})
            return text
        except Exception as exc:
            last_exc = exc
            append_jsonl(REQUEST_LOG, {'segment_index': segment_index, 'attempt': attempt, 'status': 'error', 'error': repr(exc), 'elapsed_sec': round(time.time() - started, 3), 'time': now()})
            time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f'Ollama generation failed for segment {segment_index}: {last_exc!r}')


def build_step_task(index: int, trail: list[dict[str, Any]]) -> dict[str, Any]:
    tags: dict[str, Any] = {'target': trail[0]['target']}
    for i, segment in enumerate(trail):
        tags[f'step_{i}'] = {'action': segment['label'], 'tags': segment['scene tags']}
    config_path = Path(trail[0]['trajectory']) / 'success/trial_1/task.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    target_trial, start_pose_trial, resolution_info = resolve_trials(config, trail)
    local_start = trail[0]['start'] - start_pose_trial['step_former'] - 1
    labels = object_labels(config)
    task: dict[str, Any] = {}
    task['trajectory path'] = '/'.join(str(trail[0]['trajectory']).split('/')[-3:]) + '/success/trial_1'
    task['start'] = trail[0]['start']
    task['end'] = trail[-1]['end']
    task['Robot'] = config['Robot']
    task['Scene'] = config['Scene']
    task['target'] = [labels[target_trial['index']]]
    task['Region'] = [config['Region'][target_trial['index']]]
    task['start_pos'] = config['trial'][start_pose_trial['trial_name']]['pos'][local_start]
    task['start_yaw'] = config['trial'][start_pose_trial['trial_name']]['yaw'][local_start]
    task['mapping'] = {'policy': MAPPING_POLICY, 'target_trial': target_trial['trial_name'], 'target_trial_index': target_trial['index'], 'start_pose_trial': start_pose_trial['trial_name'], 'start_pose_trial_index': start_pose_trial['index'], 'local_start_index': local_start, **resolution_info}
    task['Task instruction'] = ollama_generate(PROMPT_PATH / 'gen_task.txt', json.dumps(tags), index)
    return task


def run_step_batch(max_segments: int) -> int:
    rc = prepare()
    if rc != 0:
        return rc
    merge_trails()
    if trail_count() <= 0:
        summary({'stage': 'step', 'step_status': 'blocked_missing_trail_list'})
        return 5
    rows = read_manifest()
    if any(r.get('split_status') != 'done' for r in rows):
        print(json.dumps(summary({'stage': 'step', 'step_status': 'blocked_split_pending'}), indent=2, ensure_ascii=False))
        return 0
    check_ollama()
    trails = load_trails()
    completed_before = step_json_count()
    processed = 0
    append_jsonl(PROGRESS_LOG, {'phase': 'step_batch_start', 'completed': completed_before, 'total': len(trails), 'max_segments': max_segments, 'time': now()})
    for index, trail in enumerate(trails):
        if existing_output_for(index) is not None:
            continue
        task = build_step_task(index, trail)
        out = output_path_for(index, task['Task instruction'])
        tmp = out.with_suffix(out.suffix + '.tmp')
        tmp.write_text(json.dumps(task, indent=4, ensure_ascii=False), encoding='utf-8')
        tmp.replace(out)
        processed += 1
        if processed >= max_segments:
            break
    completed = step_json_count()
    status = 'completed' if completed >= len(trails) else 'partial'
    data = summary({'stage': 'step', 'step_status': status, 'step_batch_processed': processed, 'total_segments': len(trails), 'completed_segments': completed})
    append_jsonl(PROGRESS_LOG, {'phase': 'step_batch_done', 'processed': processed, 'completed': completed, 'total': len(trails), 'status': status, 'time': now()})
    print(json.dumps(data, indent=2, ensure_ascii=False))
    return 0


def run_next(batch_size: int, step_segments: int) -> int:
    rc = prepare()
    if rc != 0:
        return rc
    rows = read_manifest()
    if any(r.get('split_status') != 'done' for r in rows):
        return run_split_batch(batch_size)
    merge_trails()
    if step_json_count() < trail_count():
        return run_step_batch(step_segments)
    print(json.dumps(summary({'stage': 'complete'}), indent=2, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd', required=True)
    sub.add_parser('prepare')
    p_split = sub.add_parser('split-batch'); p_split.add_argument('--batch-size', type=int, default=50)
    p_step = sub.add_parser('step-batch'); p_step.add_argument('--max-segments', type=int, default=50)
    p_next = sub.add_parser('run-next'); p_next.add_argument('--batch-size', type=int, default=50); p_next.add_argument('--step-segments', type=int, default=50)
    sub.add_parser('summary')
    args = parser.parse_args()
    if args.cmd == 'prepare': return prepare()
    if args.cmd == 'split-batch': return run_split_batch(args.batch_size)
    if args.cmd == 'step-batch': return run_step_batch(args.max_segments)
    if args.cmd == 'run-next': return run_next(args.batch_size, args.step_segments)
    print(json.dumps(summary({'stage': 'summary'}), indent=2, ensure_ascii=False)); return 0


if __name__ == '__main__':
    raise SystemExit(main())
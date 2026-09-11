#!/usr/bin/env python3
"""Build the curated, data-free GitHub release directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(root: Path, output: Path, source: str, target: str) -> Path:
    destination = output / target
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / source, destination)
    return destination


def toolbox_ignore(directory: str, names: list[str]) -> set[str]:
    path = Path(directory)
    ignored = {name for name in names if name in {'__pycache__', '.git', '.pytest_cache'} or name.endswith('.pyc')}
    if path.name == 'rPPG-Toolbox':
        ignored.update(name for name in names if name in {'model_outputs', 'tools', 'wip'})
    if path.name == 'face_detector':
        ignored.update(name for name in names if name == 'ckpts')
    return ignored


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('github_release/lightweight-rppg-reproduction'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = (root / args.output).resolve() if not args.output.is_absolute() else args.output.resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite existing release: {output}')
    output.mkdir(parents=True)

    for directory in ('src', 'scripts', 'configs', 'docs', 'tests', 'patches'):
        shutil.copytree(root / directory, output / directory, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.pytest_cache'))
    shutil.copytree(root / 'third_party' / 'rPPG-Toolbox', output / 'third_party' / 'rPPG-Toolbox', ignore=toolbox_ignore)
    for filename in ('pyproject.toml', 'requirements.txt', '.gitignore'):
        shutil.copy2(root / filename, output / filename)
    shutil.copy2(root / 'docs' / 'github_package_readme.md', output / 'README.md')

    replacements = {
        str(root / 'outputs/toolbox/ubfc/TSCAN/UBFC-rPPG_TSCAN_within_622/PreTrainedModels/ubfc_ubfc_ubfc_TSCAN_Epoch16.pth'): 'weights/tscan/ubfc_epoch16.pth',
        str(root): '.',
    }
    for path in (output / 'configs').rglob('*'):
        if path.is_file() and path.suffix in {'.yaml', '.yml', '.json'}:
            content = path.read_text(encoding='utf-8')
            for old, new in replacements.items():
                content = content.replace(old, new)
            path.write_text(content, encoding='utf-8')

    cached_tscan = output / 'configs' / 'toolbox_baselines' / 'ubfc' / 'tscan_cpu_cached.yaml'
    cached_tscan.write_text(
        cached_tscan.read_text(encoding='utf-8').replace(
            '  RESUME: ./outputs/toolbox/ubfc/TSCAN/UBFC-rPPG_TSCAN_within_622/PreTrainedModels/ubfc_ubfc_ubfc_TSCAN_Epoch19.pth',
            "  RESUME: ''",
        ),
        encoding='utf-8',
    )

    release_tscan = output / 'configs' / 'toolbox_baselines' / 'ubfc' / 'tscan_release_test.yaml'
    release_tscan.write_text(
        'BASE:\n- tscan_cpu_cached.yaml\nTOOLBOX_MODE: only_test\nINFERENCE:\n'
        '  MODEL_PATH: weights/tscan/ubfc_epoch16.pth\n',
        encoding='utf-8',
    )

    specs = [
        {
            'name': 'Light-rPPGViT UBFC best',
            'source': 'outputs/ubfc/light_rppgvit/trials/physical_batch4/best.pt',
            'path': 'weights/light_rppgvit/ubfc_best.pt',
            'framework': 'lightrppg',
            'config': 'configs/ubfc_light_rppgvit_physical_batch4.yaml',
            'metrics_source': 'outputs/ubfc/light_rppgvit/trials/physical_batch4/metrics.json',
        },
        {
            'name': 'rPPGViT UBFC best',
            'source': 'outputs/ubfc/rppgvit/trials/window64_cosine_physical_batch4/best.pt',
            'path': 'weights/rppgvit/ubfc_best.pt',
            'framework': 'lightrppg',
            'config': 'configs/ubfc_rppgvit_window64_cosine_physical_batch4.yaml',
            'metrics_source': 'outputs/ubfc/rppgvit/trials/window64_cosine_physical_batch4/metrics.json',
        },
        {
            'name': 'TSCAN UBFC epoch 16',
            'source': 'outputs/toolbox/ubfc/TSCAN/UBFC-rPPG_TSCAN_within_622/PreTrainedModels/ubfc_ubfc_ubfc_TSCAN_Epoch16.pth',
            'path': 'weights/tscan/ubfc_epoch16.pth',
            'framework': 'rppg-toolbox',
            'config': 'configs/toolbox_baselines/ubfc/tscan_release_test.yaml',
            'metrics_source': 'outputs/toolbox-results/ubfc/tscan/metrics.json',
        },
    ]
    manifest_entries = []
    summary = {'scope': 'approximate reproduction', 'dataset': 'UBFC-rPPG', 'models': {}}
    for spec in specs:
        destination = copy_file(root, output, spec['source'], spec['path'])
        metrics = json.loads((root / spec['metrics_source']).read_text(encoding='utf-8'))
        manifest_entries.append({
            'name': spec['name'],
            'path': spec['path'],
            'bytes': destination.stat().st_size,
            'sha256': file_hash(destination),
            'framework': spec['framework'],
            'config': spec['config'],
            'metrics': metrics,
        })
        summary['models'][spec['name']] = metrics

    (output / 'weights' / 'manifest.json').write_text(
        json.dumps({'format_version': 1, 'weights': manifest_entries}, indent=2) + '\n', encoding='utf-8'
    )
    (output / 'results').mkdir(parents=True, exist_ok=True)
    (output / 'results' / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    copy_file(root, output, 'outputs/comparison/paper_vs_local.csv', 'results/paper_vs_local.csv')
    copy_file(root, output, 'outputs/benchmarks/apple_m2_cpu.json', 'results/apple_m2_cpu.json')
    copy_file(root, output, 'outputs/optimization/light_rppgvit.json', 'results/light_rppgvit_optimization.json')
    for model, source in (
        ('light_rppgvit', 'outputs/ubfc/light_rppgvit/trials/physical_batch4'),
        ('rppgvit', 'outputs/ubfc/rppgvit/trials/window64_cosine_physical_batch4'),
    ):
        for filename in ('metrics.json', 'per_video_metrics.csv', 'split_subjects.json', 'resolved_config.json', 'iteration_status.json'):
            copy_file(root, output, f'{source}/{filename}', f'results/{model}/{filename}')
    copy_file(root, output, 'outputs/toolbox-results/ubfc/tscan/metrics.json', 'results/tscan/metrics.json')

    print(json.dumps({'output': str(output), 'weights': len(manifest_entries)}, indent=2))


if __name__ == '__main__':
    main()

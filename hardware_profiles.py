"""GPU rules are independent of observed inventory and purchase promises."""
import json
from pathlib import Path
import re

MAX_GPUS = 64
SOURCE = 'https://developer.nvidia.com/cuda/gpus'
# Capacity and board IDs vary by SKU; only B300 retains its documented variant
# check in authenticity.py. Never infer a purchase BOM from discovered devices.
PROFILES = {
    'auto': dict(title='自动识别 / 未指定采购型号', capability=None),
    'generic': dict(title='其他 NVIDIA / 仅一致性核验', capability=None),
    'b300': dict(title='B300', capability=[10, 3]),
    'b200': dict(title='B200', capability=[10, 0]),
    'h200': dict(title='H200', capability=[9, 0]),
    'h100': dict(title='H100', capability=[9, 0]),
    'a100': dict(title='A100', capability=[8, 0]),
    'a30': dict(title='A30', capability=[8, 0]),
    'a40': dict(title='A40', capability=[8, 6]),
    'a10': dict(title='A10', capability=[8, 6]),
    'l40s': dict(title='L40S', capability=[8, 9]),
    'l40': dict(title='L40', capability=[8, 9]),
    'l4': dict(title='L4', capability=[8, 9]),
    'rtx4090': dict(title='RTX 4090', capability=[8, 9]),
    'rtx3090': dict(title='RTX 3090', capability=[8, 6]),
    'rtx5090': dict(title='RTX 5090', capability=[12, 0]),
    't4': dict(title='T4', capability=[7, 5]),
}


def matches(profile, name):
    if profile not in PROFILES:
        raise ValueError('Unknown hardware profile')
    token = PROFILES[profile]['title']
    return profile in ('auto', 'generic') or bool(re.search(r'(?<![A-Za-z0-9])' + re.escape(token) + r'(?![A-Za-z0-9])', str(name), re.I))


def identify(name):
    return next((key for key in PROFILES if key not in ('auto', 'generic') and matches(key, name)), 'generic')


def policy(out=None, selected=None):
    if selected is None:
        try:
            selected = json.loads((Path(out) / 'plan.json').read_text(encoding='utf-8')).get('hardware_profile', 'b300')
        except (OSError, ValueError, TypeError):
            selected = 'b300'  # Legacy evidence keeps the original strict rule.
    if selected not in PROFILES:
        raise ValueError('Unknown hardware profile')
    return selected


def capability_for(name, selected='auto'):
    return PROFILES[identify(name) if selected in ('auto', 'generic') else selected]['capability']


def valid_capability(value):
    return isinstance(value, list) and len(value) == 2 and all(type(x) is int for x in value) and 1 <= value[0] <= 99 and 0 <= value[1] <= 9


def public_compatible(gpus):
    return len(gpus) == 8 and all(matches('b300', gpu.get('name', '')) for gpu in gpus)


def options():
    return [dict(id=key, **value) for key, value in PROFILES.items()]

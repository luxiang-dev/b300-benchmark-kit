"""Validate a complete physical-GPU sample before a watchdog accepts it."""
import math


def validate_sample(samples, expected_count, expected_uuids=None):
    if len(samples) != expected_count:
        raise ValueError('Telemetry did not cover the expected GPU count')
    uuids = [sample['uuid'].strip() for sample in samples]
    if any(not value or not value.startswith('GPU-') for value in uuids) or len(set(uuids)) != expected_count:
        raise ValueError('Missing or duplicate GPU UUID in telemetry')
    if expected_uuids is not None and set(uuids) != set(expected_uuids):
        raise ValueError('Telemetry GPU identities differ from the preflight inventory')
    if any('index' in sample for sample in samples):
        indices = [sample.get('index') for sample in samples]
        if any(type(index) is not int for index in indices) or set(indices) != set(range(expected_count)):
            raise ValueError('Telemetry GPU indices are not a complete bijection')
    for sample in samples:
        for field, lower, upper in [('temperature_c', 0, 120), ('power_w', 0, None),
                                    ('memory_used_mib', 0, None), ('gpu_util_percent', 0, 100)]:
            if field not in sample and field not in ('temperature_c', 'power_w'):
                continue
            value = sample[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError('Non-finite or invalid telemetry: ' + field)
            if value < lower or (field == 'temperature_c' and value == 0) or (upper is not None and value > upper) or (field == 'temperature_c' and value == 120):
                raise ValueError('Out-of-range telemetry: ' + field)


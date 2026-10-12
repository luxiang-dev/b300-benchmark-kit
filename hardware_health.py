"""Conservative ECC, row-remapper, fabric and NVLink history checks."""
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET


def leaves(node, prefix=''):
    values = {}
    if node is None:
        return values
    for child in node:
        key = prefix + child.tag
        if len(child):
            values.update(leaves(child, key + '/'))
        else:
            text = (child.text or '').strip()
            if re.fullmatch(r'\d+', text):
                values[key] = int(text)
    return values


def nvlink_counters(path):
    values, gpu, link = {}, None, None
    raw = Path(path).read_text(encoding='utf-8', errors='replace')
    for line in raw.splitlines():
        header = re.search(r'GPU\s+\d+.*UUID:\s*(GPU-[0-9a-fA-F-]+)', line)
        if header:
            gpu, link = header[1], None
            continue
        match = re.match(r'\s*Link\s+(\d+)\s*:(.*)', line)
        if match:
            link, line = int(match[1]), match[2]
        counter = re.match(r'\s*([A-Za-z][A-Za-z0-9 /()_-]*):\s*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)\s*(?:Errors?)?\s*$', line)
        if counter and gpu is not None and link is not None:
            label = counter[1].strip().lower()
            if re.search('error|crc|replay|recovery|malformed|buffer|integrity|discard|fec|ber|symbol', label):
                value = float(counter[2])
                if not math.isfinite(value):
                    raise ValueError('NVLink counter is non-finite')
                key = (gpu, link, label)
                if key in values:
                    raise ValueError('Duplicate NVLink counter')
                values[key] = value
    return values


def evaluate(out, strict=False, generic=False):
    out = Path(out)
    failures, review = [], []
    try:
        before_root, after_root = [ET.parse(out / name).getroot() for name in ('before.xml', 'after.xml')]
        before = {gpu.findtext('uuid'): gpu for gpu in before_root.findall('gpu')}
        after = {gpu.findtext('uuid'): gpu for gpu in after_root.findall('gpu')}
        if len(before) != len(before_root.findall('gpu')) or len(after) != len(after_root.findall('gpu')):
            failures.append('硬件快照存在重复 GPU UUID。')
        for gpu_uuid, current in after.items():
            baseline = before.get(gpu_uuid)
            if baseline is None:
                continue
            ecc_required = strict and (not generic or current.findtext("ecc_mode/current_ecc") == "Enabled")
            for scope in ('volatile', 'aggregate'):
                old = leaves(baseline.find('ecc_errors/' + scope))
                new = leaves(current.find('ecc_errors/' + scope))
                if ecc_required and not new:
                    review.append(f'{gpu_uuid}: {scope} ECC 计数缺失或不可读。')
                if ecc_required and 'dram_uncorrectable' not in new and not any('dram' in key and ('uncorrectable' in key or 'double_bit' in key) for key in new):
                    review.append(f'{gpu_uuid}: {scope} DRAM 不可纠正 ECC（dram_uncorrectable）计数不可读。')
                threshold = current.findtext('ecc_errors/' + scope + '/sram_threshold_exceeded')
                if threshold in ('Yes', 'True', '1'):
                    failures.append(f'{gpu_uuid}: {scope} SRAM ECC threshold exceeded。')
                for field, count in new.items():
                    if 'correctable' not in field and 'double_bit' not in field and 'single_bit' not in field:
                        continue
                    prior = old.get(field)
                    text = f'{gpu_uuid}: {scope}/{field}'
                    uncorrected = 'uncorrectable' in field or 'double_bit' in field
                    if prior is None:
                        review.append(text + ': 缺少前置基线。')
                    elif count > prior:
                        (failures if uncorrected else review).append(text + f' 增加 {count-prior}；需复核 ECC/RAS。')
                    elif count < prior:
                        review.append(text + ': 计数下降，不能用重置掩盖历史。')
                    elif count > 0:
                        review.append(text + f': 存在历史 ECC 计数 {count}，采购验货需厂商解释。')
            remap = current.find('remapped_rows')
            previous = baseline.find('remapped_rows')
            if remap is None:
                if strict and not generic:
                    review.append(f'{gpu_uuid}: 显存行重映射信息不可读。')
            else:
                pending = remap.findtext('pending')
                if pending in ('Yes', 'Pending'):
                    failures.append(f'{gpu_uuid}: 显存行重映射 pending，不能视为健康通过。')
                elif strict and not generic and pending != 'No':
                    review.append(f'{gpu_uuid}: 行重映射 pending 状态不可确认。')
                for field in ('correctable_error', 'uncorrectable_error'):
                    now = leaves(remap).get(field)
                    old = leaves(previous).get(field)
                    if now is None or old is None:
                        if strict and not generic:
                            review.append(f'{gpu_uuid}: 行重映射 {field} 前后计数不可读。')
                    elif now > old:
                        (failures if field == 'uncorrectable_error' else review).append(f'{gpu_uuid}: 行重映射 {field} 增加 {now-old}。')
                    elif now < old:
                        review.append(f'{gpu_uuid}: 行重映射 {field} 计数下降。')
                    elif now > 0:
                        review.append(f'{gpu_uuid}: 存在历史行重映射 {field}={now}，需 OEM 解释。')
                for field, count in leaves(remap).items():
                    if 'histogram' in field and field.rsplit('/', 1)[-1] in ('none', 'low') and count:
                        review.append(f'{gpu_uuid}: 显存重映射备用行 {field}={count}，需厂商复核。')
            fabric = current.find('fabric')
            if fabric is None:
                if strict and not generic:
                    review.append(f'{gpu_uuid}: NVSwitch fabric 状态不可读，需 OEM 复核。')
            elif fabric.findtext("state") not in (("Completed", "N/A", "Not Supported") if generic else ("Completed",)):
                review.append(f'{gpu_uuid}: fabric state={fabric.findtext("state")} 未确认完成。')
            elif fabric.findtext("state") == "Completed" and fabric.findtext("status") not in ("Success", "NVML_SUCCESS"):
                failures.append(f'{gpu_uuid}: fabric status={fabric.findtext("status")} 非 Success。')
            if fabric is not None:
                health = fabric.find('health')
                if health is not None:
                    for field in health.iter():
                        value = (field.text or '').strip()
                        if field.tag in ('route_unhealthy', 'route_recovery_in_progress', 'access_timeout_recovery') and value.lower() in ('true', 'yes', '1'):
                            failures.append(f'{gpu_uuid}: fabric {field.tag}={value}。')
                        elif field.tag in ('bandwidth', 'link') and value.lower() in ('degraded', 'unhealthy', 'failed'):
                            failures.append(f'{gpu_uuid}: fabric {field.tag}={value}。')
            recovery = current.findtext('gpu_recovery_action')
            if recovery not in (None, 'None', 'N/A', 'No Action'):
                failures.append(f'{gpu_uuid}: GPU Recovery Action={recovery}，不能继续按健康通过。')
        require_nvlink = strict and not generic
        link_raw = (out / "nvlink.txt").read_text(encoding="utf-8", errors="replace") if (out / "nvlink.txt").is_file() else ""
        has_links = bool(re.search(r"Link\s+\d+", link_raw, re.I))
        if require_nvlink:
            for name in ('topology.txt', 'nvlink.txt'):
                if not (out / name).is_file() or not (out / name).read_text(encoding='utf-8', errors='replace').strip():
                    review.append(name + ': 互联拓扑/链路证据缺失。')
        if (require_nvlink or has_links) and (out / 'nvlink.txt').is_file() and re.search(r'\bInactive\b|\bSleep\b|Not Supported|Error', (out / 'nvlink.txt').read_text(encoding='utf-8', errors='replace'), re.IGNORECASE):
            review.append('NVLink 链路状态包含 Inactive/Sleep/不可读；需在通信负载时复测并由 OEM 核对预期拓扑。')
        paths = [out / f'nvlink-errors-{phase}.txt' for phase in ('before', 'after')]
        if require_nvlink or has_links or (not generic and any(path.exists() for path in paths)):
            old, new = [nvlink_counters(path) for path in paths]
            if not old or not new or set(old) != set(new):
                review.append('NVLink 错误计数缺失、不可解析或前后字段不同，不能视为零错误。')
            if require_nvlink and {key[0] for key in new} != set(after):
                review.append('NVLink 错误计数未覆盖全部物理 GPU。')
            for key in old.keys() & new.keys():
                gpu, link, label = key
                if new[key] != old[key]:
                    delta = new[key]-old[key]
                    corrected = any(word in label for word in ('fec', 'corrected', 'ber', 'symbol'))
                    (review if delta < 0 or corrected else failures).append(f'{gpu}: NVLink {link}/{label} 变化 {delta:g}（{old[key]:g}→{new[key]:g}）。')
                elif new[key] > 0:
                    review.append(f'{gpu}: NVLink {link}/{label} 有历史计数 {new[key]:g}。')
    except Exception as error:
        if strict or any((out / f'nvlink-errors-{phase}.txt').exists() for phase in ('before', 'after')):
            review.append('扩展 GPU 健康证据不可复核: ' + str(error))
    return failures, review

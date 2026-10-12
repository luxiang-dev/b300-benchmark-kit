"""Keep technical test outcomes distinct from a buyer's final acceptance."""

PENDING = [
    ('purchase_bom', '采购清单逐项核对：整机型号、CPU、DIMM、SSD、NIC、PSU 和 8 张 GPU 的具体 SKU/序列号'),
    ('vendor_confirmation', '实物板标签、厂商序列号/保修资格、供货来源及整机部件身份确认'),
    ('extended_cpu_ram', '按该机型验收规范完成 CPU 全核与主机内存完整诊断，复核 ECC/RAS'),
    ('storage', '全部合同存储设备的健康、寿命、容量和介质诊断；临时文件抽查不能替代'),
    ('network', '全部约定 NIC/端口及外部链路的速率、正确性、错误计数和实际吞吐验收'),
    ('oem_diagnostics', '对应 OEM 的整机诊断及 BMC 电源、风扇、温度、固件/FRU 复核'),
    ('power_boot', '厂商指导下的冷启动、重启及约定供电冗余验证，保留前后日志'),
    ('stability', '按双方确认的持续时长/负载/温度条件进行稳定性与同配置重复测量'),
    ('human_signoff', '买方与厂商/验收人员复核全部原始证据及尚未解决的事项'),
]


def evaluate(summary):
    technical = summary.get('overall', 'INCOMPLETE')
    blockers = list(summary.get('failures', [])) + list(summary.get('incomplete', [])) + list(summary.get('review', []))
    return dict(schema=1, status='BLOCKED' if technical != 'PASS' or blockers else 'NOT_ACCEPTED',
                automatic_checks=technical, blockers=blockers,
                pending_checks=[dict(id=key, title=title, status='PENDING') for key, title in PENDING],
                statement='技术检查通过不等于采购验收完成。此程序不自动签发整机收货合格或正品认证。',
                scope='当前自动测试没有证明合同配置完全相符或全部部件已完成厂商验收。',
                final_acceptance='NOT_PERFORMED')

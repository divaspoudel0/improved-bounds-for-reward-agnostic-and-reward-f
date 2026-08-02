"""Run all claim audits for the reproduction of Ridel & Cohen (2602.16363).

Theory paper -> numerical audits (no GPU required). Prints a structured JSON
report to stdout that is the evidence channel for the logbook.
"""
import json
import time
import numpy as np
from . import claim1, claim3, claim4, claim25


def main():
    t0 = time.time()
    report = {
        'paper': 'Improved Bounds for Reward-Agnostic and Reward-Free Exploration',
        'arxiv': '2602.16363',
        'audit_type': 'numerical audit (theory paper, no GPU needed)',
        'claims': {},
        'environment': {},
    }
    import sys
    report['environment']['python'] = sys.version.split()[0]
    report['environment']['numpy'] = np.__version__

    report['claims']['claim4_logbarrier'] = claim4.audit_claim4()
    report['claims']['claim3_reward_sequence'] = claim3.audit_claim3()
    report['claims']['claim1_sample_complexity'] = claim1.audit_claim1()
    report['claims']['claims25_lower_bound'] = claim25.audit_claims25()

    report['wall_clock_s'] = round(time.time() - t0, 2)
    print(json.dumps(report, indent=2, default=str))


if __name__ == '__main__':
    main()

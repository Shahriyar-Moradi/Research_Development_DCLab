"""Run only the frozen, offline TP-06 DEV comparison."""
import argparse
import json
from . import run


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=('dev',), default='dev')
    args = parser.parse_args(argv)
    path = run(split=args.split)
    doc = json.loads(path.read_text())
    print(doc['verdict'])
    print(path)
    return 1 if doc['verdict'] == 'FAIL' else 0


if __name__ == '__main__':
    raise SystemExit(main())

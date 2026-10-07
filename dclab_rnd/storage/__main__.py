"""python -m dclab_rnd.storage upgrade | status | migrate SOURCE --to TARGET | backup --to DIR | restore BACKUP --home DIR: the product's database schema, moving a file workspace into it, and backups."""

import argparse
from pathlib import Path
import sys

from . import db
from .models import TABLES


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.storage", description=__doc__.strip().splitlines()[0])
    parser.add_argument("command", choices=["upgrade", "status", "migrate", "backup", "restore"])
    parser.add_argument("source", nargs="?", type=Path, help="migrate: the file workspace folder to copy (only read, never changed); "
                                                              "restore: the backup folder")
    parser.add_argument("--to", dest="target", type=Path, help="migrate: the folder of the database workspace it becomes; backup: where backups go")
    parser.add_argument("--home", type=Path, default=None, help="backup: the workspace folder (default DCLAB_AGENT_HOME); "
                                                                 "restore: the new, empty folder it becomes")
    args = parser.parse_args(argv)
    if args.command == "migrate" and (args.source is None or args.target is None):
        parser.error("migrate needs SOURCE and --to TARGET")
    if args.command == "restore" and (args.source is None or args.home is None):
        parser.error("restore needs BACKUP and --home DIR (a new folder; DCLAB_DATABASE_URL names a new, empty database)")
    try:
        db.database_url()
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 2
    problem = db.reachable()
    if problem:
        print(problem + ". Is PostgreSQL running, and does the database exist (createdb dclab_dev)?", file=sys.stderr)
        return 2
    if args.command == "upgrade":
        db.upgrade()
    if args.command in ("backup", "restore"):
        from . import backup

        try:
            if args.command == "backup":
                from ..settings import Settings

                made = backup.backup(db.database_url(), args.home or Path(Settings.load().agent_home), args.target or Path("backups"))
                print(f"backup {made['folder']}: schema {made['schema']}, {sum(made['rows'].values())} rows, {made['files']} files ({made['bytes']} bytes)")
                return 0
            report = backup.restore(args.source, db.database_url(), args.home)
        except backup.BackupError as error:
            print(f"Nothing was {'backed up' if args.command == 'backup' else 'restored'}: {error}", file=sys.stderr)
            return 1
        for line in report["differences"]:
            print("differs: " + line, file=sys.stderr)
        print(f"restored schema {report['schema']}: {sum(report['rows'].values())} rows, {report['files']} files"
              + ("" if not report["differences"] else "; the counts above differ from the backup's"))
        return 0 if not report["differences"] else 1
    if args.command == "migrate":
        from .migrate import migrate, print_report

        db.upgrade()  # the schema first: a migration into an old schema would fail half way
        try:
            report = migrate(args.source, args.target)
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001 — the type only: a database error's text can carry rows and connection details
            print(f"The migration stopped: {type(exc).__name__}. Nothing was copied over what the target holds; run it again after fixing the cause.",
                  file=sys.stderr)
            return 1
        print_report(report)
        return 0 if report["ok"] else 1
    print(f"schema revision: {db.current() or 'none (run: python -m dclab_rnd.storage upgrade)'}; tables: {', '.join(TABLES)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

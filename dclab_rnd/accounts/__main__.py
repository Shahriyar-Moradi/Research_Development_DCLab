"""Accounts from the command line (package 10.2): the first owner, before anyone can sign in to add members.

    python -m dclab_rnd.accounts add --email you@example.com --role owner [--name "Your Name"] [--workspace ID]
    python -m dclab_rnd.accounts list [--workspace ID]
    python -m dclab_rnd.accounts workspaces

The password is read from DCLAB_NEW_PASSWORD, or asked for (never an argument: it would stay in the shell history).
The workspace defaults to the server's own (DCLAB_AGENT_HOME). Needs DCLAB_DATABASE_URL.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.accounts", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    add = sub.add_parser("add", help="add a member (creates the user when the email is new)")
    add.add_argument("--email", required=True)
    add.add_argument("--role", required=True, choices=("owner", "data_scientist", "reviewer", "viewer"))
    add.add_argument("--name", default="")
    add.add_argument("--workspace", default=None)
    add.add_argument("--no-password", action="store_true", help="a user who signs in through the identity provider only")
    show = sub.add_parser("list", help="the members of a workspace")
    show.add_argument("--workspace", default=None)
    sub.add_parser("workspaces", help="every workspace")
    args = parser.parse_args(argv)

    from dotenv import load_dotenv

    from ..settings import ROOT, Settings

    load_dotenv(ROOT / ".env", override=False)
    from ..storage import db

    if not db.database_url(required=False):
        print("Accounts need PostgreSQL: set DCLAB_DATABASE_URL.", file=sys.stderr)
        return 2
    from .store import AccountError, Accounts

    accounts = Accounts()
    workspace = getattr(args, "workspace", None) or db.workspace_for(Path(Settings.load().agent_home))
    if args.command == "workspaces":
        import sqlalchemy as sa

        from ..storage.models import workspace as t

        with accounts.engine.connect() as c:
            for row in c.execute(sa.select(t.c.id, t.c.name).order_by(t.c.name)):
                print(f"{row.id}  {row.name}")
        return 0
    if args.command == "list":
        for m in accounts.members(workspace):
            print(f"{m['email']:<40} {m['role']:<15} {'password' if m['password'] else ''}{' sso' if m['single_sign_on'] else ''}")
        return 0
    password = None
    found = accounts.by_email(args.email)
    if found is None and not args.no_password:
        password = os.environ.get("DCLAB_NEW_PASSWORD") or getpass.getpass(f"Password for {args.email} (10 characters or more): ")
    try:
        user = found or accounts.create_user(args.email, args.name, password=password)
        accounts.set_member(workspace, user["id"], args.role)
    except (AccountError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"{user['email']} is {args.role} of workspace {workspace}{'' if found is None else ' (an existing user)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

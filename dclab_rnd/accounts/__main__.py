"""Accounts from the command line (package 10.2): the first owner, before anyone can sign in to add members.

    python -m dclab_rnd.accounts add --username you [--email you@example.com] --role owner [--name "Your Name"] [--workspace ID]
    python -m dclab_rnd.accounts passwd --username you     a new password for a person (asked for twice, never an argument)
    python -m dclab_rnd.accounts list [--workspace ID]
    python -m dclab_rnd.accounts workspaces
    python -m dclab_rnd.accounts admin-set --username you   the Admin area's own sign-in on a machine without accounts (no PostgreSQL)
    python -m dclab_rnd.accounts admin-remove               remove it: the Admin area is open again on this machine

The password is read from DCLAB_NEW_PASSWORD, or asked for (never an argument: it would stay in the shell history).
The workspace defaults to the server's own (DCLAB_AGENT_HOME). Needs DCLAB_DATABASE_URL.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from pathlib import Path


def _admin(args: argparse.Namespace) -> int:
    """A1 (2026-11-03): the Admin area's own sign-in, in the workspace folder; it needs no database."""
    from dotenv import load_dotenv

    from ..settings import ROOT, Settings
    from . import admin_login

    load_dotenv(ROOT / ".env", override=False)
    home = Path(Settings.load().agent_home)
    if args.command == "admin-remove":
        print("removed: the Admin area is open again on this machine" if admin_login.remove(home) else "no admin login was set")
        return 0
    new = os.environ.get("DCLAB_NEW_PASSWORD") or getpass.getpass(f"Admin password for {args.username} (10 characters or more): ")
    if not os.environ.get("DCLAB_NEW_PASSWORD") and new != getpass.getpass("Again: "):
        print("The two passwords differ: nothing changed.", file=sys.stderr)
        return 2
    try:
        admin_login.set_login(home, args.username, new)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"The Admin area now asks for {args.username} and the password (workspace {home}); every admin session ended.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.accounts", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    add = sub.add_parser("add", help="add a member (creates the user when the username or email is new)")
    add.add_argument("--username", default=None, help="the handle that signs the person in (3 to 32 letters, digits, dot, dash, underscore)")
    add.add_argument("--email", default=None, help="optional when there is a username")
    add.add_argument("--role", required=True, choices=("owner", "data_scientist", "reviewer", "viewer"))
    add.add_argument("--name", default="")
    add.add_argument("--workspace", default=None)
    add.add_argument("--no-password", action="store_true", help="a user who signs in through the identity provider only")
    pw = sub.add_parser("passwd", help="set a new password for a person (asked for twice; DCLAB_NEW_PASSWORD skips the questions)")
    pw.add_argument("--username", default=None)
    pw.add_argument("--email", default=None)
    show = sub.add_parser("list", help="the members of a workspace")
    show.add_argument("--workspace", default=None)
    sub.add_parser("workspaces", help="every workspace")
    adm = sub.add_parser("admin-set", help="the Admin area's username and password on this machine (asked for twice; DCLAB_NEW_PASSWORD skips the questions)")
    adm.add_argument("--username", required=True)
    sub.add_parser("admin-remove", help="remove the Admin area's sign-in on this machine")
    args = parser.parse_args(argv)
    if args.command in ("admin-set", "admin-remove"):
        return _admin(args)

    from dotenv import load_dotenv

    from ..settings import ROOT, Settings

    load_dotenv(ROOT / ".env", override=False)
    from .. import secret_files

    secret_files.load(strict=False)  # a NAME_FILE written in .env (the import-time load ran before .env was read)
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
            print(f"{(m['username'] or m['email'] or ''):<40} {m['role']:<15} {'password' if m['password'] else ''}{' sso' if m['single_sign_on'] else ''}")
        return 0
    if args.command == "passwd":
        who = (args.username or args.email or "").strip()
        found = accounts.by_username(args.username) if args.username else accounts.by_email(args.email) if args.email else None
        if found is None:
            print("No such person: name them with --username or --email.", file=sys.stderr)
            return 2
        new = os.environ.get("DCLAB_NEW_PASSWORD") or getpass.getpass(f"New password for {who} (10 characters or more): ")
        if not os.environ.get("DCLAB_NEW_PASSWORD") and new != getpass.getpass("Again: "):
            print("The two passwords differ: nothing changed.", file=sys.stderr)
            return 2
        try:
            accounts.update_user(found["id"], password=new)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(f"{who}: the password is changed. Sessions already signed in stay until they end.")
        return 0
    password = None
    who = (args.username or args.email or "").strip()
    if not who:
        print("A username or an email is needed (--username, --email).", file=sys.stderr)
        return 2
    found = accounts.by_username(args.username) if args.username else accounts.by_email(args.email)
    if found is None and not args.no_password:
        password = os.environ.get("DCLAB_NEW_PASSWORD") or getpass.getpass(f"Password for {who} (10 characters or more): ")
    try:
        user = found or accounts.create_user(args.email, args.name, password=password, username=args.username)
        accounts.set_member(workspace, user["id"], args.role)
    except (AccountError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"{user.get('username') or user['email']} is {args.role} of workspace {workspace}{'' if found is None else ' (an existing user)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

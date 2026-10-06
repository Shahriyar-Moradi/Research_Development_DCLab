"""Accounts, workspaces and roles (package 10.2).

    roles.py      the four roles, their shown names, and what each may do
    principal.py  who is asking, for one request or one job (a context variable)
    passwords.py  argon2 password hashes
    store.py      users, members, sessions and API tokens in PostgreSQL
    guard.py      the one role check every route passes, and how a request is identified
    api.py        sign-in, sign-out, who am I, switch workspace, members, API tokens

Sign-in is on only when configured (``DCLAB_AUTH=password`` or ``oidc``; the owner's choice, 2026-10-07). Without it
the app is today's single owner on this machine, and accounts need PostgreSQL.
"""

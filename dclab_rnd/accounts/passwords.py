"""Password hashes (package 10.2): argon2id, with the library's current parameters. Password sign-in is for local
development and small teams; a company signs in through its identity provider (OIDC)."""

from __future__ import annotations

MIN_LENGTH = 10
_DUMMY = None  # a real hash to verify against when the user is unknown, so both cases take the same time


def _hasher():
    from argon2 import PasswordHasher

    return PasswordHasher()


def hash_password(password: str) -> str:
    if len(password or "") < MIN_LENGTH:
        raise ValueError(f"a password has at least {MIN_LENGTH} characters")
    return _hasher().hash(password)


def verify(stored: str | None, password: str) -> bool:
    from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

    global _DUMMY
    if stored is None:
        if _DUMMY is None:
            _DUMMY = _hasher().hash("not a real password, only for timing")
        stored, password = _DUMMY, password + "\x00"  # never matches
    try:
        return _hasher().verify(stored, password or "")
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False

"""Read-only SQL through SQLAlchemy, on connections the server defines.

A connection is an environment variable on the server, ``DCLAB_DB_<NAME>=<SQLAlchemy URL>`` (for example
``DCLAB_DB_WAREHOUSE=postgresql+psycopg2://reader:...@host/db``); the user only ever sees ``<name>``.
The URL, and so the password, never leaves this module: not in a source record, not in an error.

What runs is either ``SELECT * FROM <table>`` or one ``SELECT``/``WITH`` query, always wrapped as
``SELECT * FROM (<query>) AS dclab_q LIMIT :n``. On PostgreSQL the transaction is also declared
``READ ONLY``, and nothing is ever committed.

Limits of the query check: it is a small tokenizer (comments, quoted strings and identifiers, PostgreSQL
dollar quotes) that rejects write keywords and extra statements. It cannot see what a function does
(``SELECT some_function_with_side_effects()`` passes), so the real guarantee is the database user:
configure connections with a user that can only read.
"""

from __future__ import annotations

import os
import re
from decimal import Decimal
from pathlib import Path
from typing import Any

from . import BadInput, ConnectorError, sha256_file, target_path, temp_path

PREFIX = "DCLAB_DB_"
NAME = re.compile(r"^[A-Za-z0-9_]{1,60}$")
TABLE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?$")
FORBIDDEN = {"INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "GRANT", "REVOKE", "TRUNCATE", "MERGE", "COPY",
             "CALL", "EXEC", "EXECUTE", "INTO", "ATTACH", "DETACH", "PRAGMA", "VACUUM", "LOCK", "RENAME", "UPSERT"}
MAX_LIMIT = 1_000_000
MAX_QUERY = 20_000
DRIVERS = {"psycopg2": "psycopg2-binary", "psycopg": "psycopg[binary]", "pymysql": "pymysql", "MySQLdb": "mysqlclient",
           "pyodbc": "pyodbc", "oracledb": "oracledb", "cx_Oracle": "oracledb", "pymssql": "pymssql", "duckdb": "duckdb-engine",
           "snowflake": "snowflake-sqlalchemy", "asyncpg": "psycopg2-binary (async drivers are not supported)", "aiomysql": "pymysql"}


def connections() -> list[str]:
    """The names of the connections configured on this server (lowercase)."""
    names = {key[len(PREFIX):].lower() for key, value in os.environ.items()
             if key.upper().startswith(PREFIX) and value.strip() and NAME.match(key[len(PREFIX):])}
    return sorted(names)


def _url(name: str) -> str:
    wanted = PREFIX + name.upper()
    for key, value in os.environ.items():
        if key.upper() == wanted and value.strip():
            return value.strip()
    raise ConnectorError(f"No database connection named {name} is configured on this server. "
                         f"The administrator defines it as the environment variable {wanted}.")


# ---------------------------------------------------------------------------- the query check
def _scan(sql: str, mysql: bool) -> tuple[list[str], list[int], int]:
    """Words outside comments, strings and quoted identifiers; the positions of ``;`` there; the last code position.

    Two lexical rule sets, because the same text can mean different things: ``mysql=False`` follows
    PostgreSQL/SQLite/standard SQL (``--`` comments, ``$tag$`` strings, no backslash escapes) and
    ``mysql=True`` follows MySQL (``-- `` and ``#`` comments, backslash escapes, no dollar quoting,
    ``/*! */`` refused because MySQL runs it). ``check_query`` requires both readings to pass.
    Positions are indexes in ``sql``; "code" is everything except whitespace and comments.
    """
    words: list[str] = []
    semicolons: list[int] = []
    last = -1
    i, n = 0, len(sql)
    while i < n:
        c = sql[i]
        if sql.startswith("--", i) and (not mysql or i + 2 >= n or sql[i + 2].isspace()) or (mysql and c == "#"):
            end = sql.find("\n", i)
            i = n if end < 0 else end
        elif sql.startswith("/*", i):
            if mysql and sql.startswith("/*!", i):
                raise BadInput("MySQL executable comments (/*! ... */) are not allowed.")
            end = sql.find("*/", i + 2)
            if end < 0:
                raise BadInput("The query has a comment that is never closed (/* without */).")
            i = end + 2
        elif c in ("'", '"', "`"):
            j = i + 1
            while True:
                if j >= n:
                    raise BadInput("The query has a quote that is never closed.")
                if mysql and c != "`" and sql[j] == "\\":
                    j += 2  # a backslash escapes the next character in MySQL strings
                    continue
                if sql[j] == c:
                    if j + 1 < n and sql[j + 1] == c:  # a doubled quote is an escaped quote
                        j += 2
                        continue
                    break
                j += 1
            last, i = j, j + 1
        elif c == "$" and not mysql and _DOLLAR.match(sql, i):
            tag = _DOLLAR.match(sql, i).group(0)  # PostgreSQL $$...$$ or $tag$...$tag$
            end = sql.find(tag, i + len(tag))
            if end < 0:
                raise BadInput("The query has a dollar-quoted string that is never closed.")
            last, i = end + len(tag) - 1, end + len(tag)
        elif c.isalpha() or c == "_":
            j = i
            while j < n and (sql[j].isalnum() or sql[j] in "_$"):
                j += 1
            words.append(sql[i:j].upper())
            last, i = j - 1, j
        else:
            if c == ";":
                semicolons.append(i)
            if not c.isspace():
                last = i
            i += 1
    return words, semicolons, last


_DOLLAR = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)?\$")


def check_query(query: str) -> str:
    """The query, without a trailing ``;``, if it is one read-only SELECT/WITH statement; else BadInput.

    The text must pass under both lexical readings of ``_scan`` (standard SQL and MySQL), so a string or
    a comment cannot hide a statement from one database while looking harmless to the other.
    """
    query = str(query or "")
    if len(query) > MAX_QUERY:
        raise BadInput(f"The query is longer than {MAX_QUERY:,} characters.")
    cut = None
    for mysql in (False, True):
        words, semicolons, last = _scan(query, mysql)
        if not words:
            raise BadInput("Write a SELECT query first.")
        if words[0] not in ("SELECT", "WITH"):
            raise BadInput("Only read queries are allowed: start with SELECT or WITH.")
        blocked = sorted({w for w in words if w in FORBIDDEN})
        if blocked:
            raise BadInput(f"Only read queries are allowed; remove {', '.join(blocked)}.")
        if semicolons:
            if len(semicolons) > 1 or semicolons[0] != last or cut not in (None, semicolons[0]):
                raise BadInput("Send one statement at a time (a single SELECT, no ';' in the middle).")
            cut = semicolons[0]
        elif cut is not None:
            raise BadInput("Send one statement at a time (a single SELECT, no ';' in the middle).")
    return (query[:cut] if cut is not None else query).strip()


# ---------------------------------------------------------------------------- fetch
def fetch(connection: str, directory: Path, table: str | None = None, query: str | None = None, limit: int = 200_000) -> dict[str, Any]:
    """Run the read on ``connection`` and save the rows as parquet in ``directory``; returns {path, filename, source}."""
    name = str(connection or "").strip().lower()
    if not NAME.match(name):
        raise BadInput("Choose one of the database connections configured on the server.")
    table = (table or "").strip() or None
    query = (query or "").strip() or None
    if (table is None) == (query is None):
        raise BadInput("Give either a table name or a query, not both.")
    try:
        limit = max(1, min(int(limit), MAX_LIMIT))
    except (TypeError, ValueError):
        raise BadInput("The row limit must be a number.") from None
    if table is not None and not TABLE.match(table):
        raise BadInput("A table is written name or schema.name (letters, digits and underscores).")
    inner = f"SELECT * FROM {table}" if table else check_query(query)
    url = _url(name)

    import pandas as pd
    import sqlalchemy as sa
    from sqlalchemy import exc as sa_exc

    try:
        engine = sa.create_engine(url, poolclass=sa.pool.NullPool)
    except ModuleNotFoundError as exc:
        raise ConnectorError(_driver_message(name, exc)) from None
    except sa_exc.NoSuchModuleError:
        raise ConnectorError(f"The database type of connection {name} is not supported by this server "
                             "(its SQLAlchemy dialect is not installed).") from None
    except (sa_exc.ArgumentError, ValueError):
        raise ConnectorError(f"The server's setting for connection {name} ({PREFIX}{name.upper()}) is not a valid "
                             "SQLAlchemy URL.") from None
    dialect = engine.dialect.name
    statement = sa.text(_wrap(inner, dialect))
    try:
        try:
            raw = engine.connect()
        except ModuleNotFoundError as exc:
            raise ConnectorError(_driver_message(name, exc)) from None
        except sa_exc.SQLAlchemyError as exc:
            raise ConnectorError(f"Could not connect to the database {name} ({_kind(exc)}). Check that it is reachable "
                                 f"from the server and that the server's {PREFIX}{name.upper()} setting is right.") from None
        with raw as conn:
            conn = conn.execution_options(stream_results=True)
            try:
                if dialect == "postgresql":
                    conn.exec_driver_sql("SET TRANSACTION READ ONLY")
                frame = pd.read_sql(statement, conn, params={"n": limit})
            except (sa_exc.SQLAlchemyError, pd.errors.DatabaseError) as exc:
                raise ConnectorError(f"The database {name} rejected the query ({_kind(exc)}). Check the table and column "
                                     "names, and that the server's database user may read them.") from None
            finally:
                try:
                    conn.rollback()  # read only: nothing is ever committed
                except sa_exc.SQLAlchemyError:
                    pass
    finally:
        engine.dispose()
    if frame.empty:
        raise ConnectorError(f"The {'table' if table else 'query'} returned no rows.")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    final = target_path(directory, f"{name}_{table.replace('.', '_') if table else 'query'}.parquet")
    scratch = temp_path(directory)
    try:
        _to_parquet(frame, scratch)
        scratch.replace(final)
    finally:
        scratch.unlink(missing_ok=True)
    source: dict[str, Any] = {"kind": "database", "connection": name, "dialect": dialect}
    source["table" if table else "query"] = table or inner
    source.update(limit=limit, rows=int(len(frame)), columns=int(frame.shape[1]), truncated=bool(len(frame) >= limit),
                  sha256=sha256_file(final))
    return {"path": str(final), "filename": final.name, "source": source}


def _wrap(inner: str, dialect: str) -> str:
    """The limit around the read. The inner query sits on its own lines, so a trailing ``--`` comment cannot
    swallow the wrapper; its ``:words`` are escaped, so only ``:n`` is a bind parameter."""
    inner = re.sub(r"(?<![:\w\\]):(\w+)(?!:)", r"\\:\1", inner)
    if dialect == "mssql":
        return f"SELECT TOP (:n) * FROM (\n{inner}\n) AS dclab_q"
    if dialect == "oracle":
        return f"SELECT * FROM (\n{inner}\n) dclab_q FETCH FIRST :n ROWS ONLY"
    return f"SELECT * FROM (\n{inner}\n) AS dclab_q LIMIT :n"


def _kind(exc: BaseException) -> str:
    """The error's class name only (the driver's message can contain the host, the user or the URL)."""
    orig = getattr(exc, "orig", None)
    return type(orig).__name__ if orig is not None else type(exc).__name__


def _driver_message(name: str, exc: ModuleNotFoundError) -> str:
    module = (getattr(exc, "name", None) or "").split(".")[0]
    package = DRIVERS.get(module, module or "the database driver")
    return (f"The database driver for connection {name} is not installed: install `{package}` in the server "
            "environment and restart the server.")


def _to_parquet(frame, path: Path) -> None:
    """Parquet needs one type per column: decimals become floats, bytes become hex, other mixed objects text."""
    import pandas as pd

    frame = frame.copy()
    frame.columns = [str(c) for c in frame.columns]
    for col in frame.columns:
        if frame[col].dtype != object:
            continue
        values = frame[col].dropna()
        if len(values) and all(isinstance(v, Decimal) for v in values):
            frame[col] = pd.to_numeric(frame[col].map(lambda v: None if v is None else float(v)), errors="coerce")
        elif not all(isinstance(v, str) for v in values):
            frame[col] = frame[col].map(_as_text)
    frame.to_parquet(path, index=False)


def _as_text(value: Any) -> str | None:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, float) and value != value:
        return None
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).hex()
    return str(value)

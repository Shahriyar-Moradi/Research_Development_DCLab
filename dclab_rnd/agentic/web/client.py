"""The product frontend's API client, generated from the server's OpenAPI schema (package 13.1).

    python -m dclab_rnd.agentic.web.build      # writes src/api.js with the rest of the frontend

``src/api.js`` defines ``DC.client``: one function per route under /api, named after the route's Python function in
camelCase (``get_project`` → ``DC.client.getProject``), with the path's parameters as arguments in order and an
options object last (``{query, body, headers}``). Each function says in JSDoc what it takes and returns, and checks at
run time that every path parameter is given and that no query parameter is unknown, so a renamed route or a typo
fails at once instead of sending a request to a path that does not exist. ``DC.client.path.<name>(…)`` gives the
same path as a string, for an EventSource; ``DC.client.href.<name>(…)`` the full URL of a GET, for a download link. The file is generated, never edited: the build's --check fails when it
differs from what the schema gives.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any
from unittest import mock

HEADER = ("/* GENERATED from the server's OpenAPI schema by `make web` (dclab_rnd/agentic/web/client.py, package 13.1).\n"
          "   Do not edit: change the route, then rebuild. One function per route under /api: DC.client.<name>(path params…, {query, body, headers}). */\n")


def _app():
    """A server made on a scratch folder, on files, without accounts, before .env or a secret file can say otherwise:
    nothing real is opened (no workspace folder, no database), so the build works with the database stopped."""
    home = tempfile.mkdtemp(prefix="dclab-schema-")
    clean = {k: "" for k in ("DCLAB_DATABASE_URL", "DCLAB_AUTH", "DCLAB_FILES_URL", "DCLAB_METRICS_PORT", "DCLAB_ALLOWED_HOSTS",
                             "DCLAB_STUDIO_HOME", "DCLAB_DRAFT_HOME", "DCLAB_INTERN_HOME")}
    with mock.patch.dict(os.environ):
        for key in [k for k in os.environ if k.startswith("DCLAB_") and k.endswith("_FILE")]:
            del os.environ[key]  # a secret file would fill the database URL again
        os.environ.update(clean, DCLAB_AGENT_HOME=home)
        from ..server import create_app

        return create_app(Path(home))


def schema() -> dict[str, Any]:
    return _app().openapi()


def route_name(op_id: str, method: str, path: str) -> str:
    """The route's Python function name: FastAPI's operationId is that name, the path with every non-word character
    as "_", and the method (fastapi.utils.generate_unique_id)."""
    suffix = re.sub(r"\W", "_", path) + "_" + method.lower()
    if not op_id.endswith(suffix):
        raise ValueError(f"unexpected operationId {op_id} for {method.upper()} {path}")
    return op_id[: -len(suffix)]


def camel(name: str) -> str:
    head, *rest = name.split("_")
    return head + "".join(w[:1].upper() + w[1:] for w in rest)


def _type(s: dict[str, Any] | None) -> str:
    if not s:
        return "*"
    if "$ref" in s:
        return s["$ref"].rsplit("/", 1)[-1]
    for key in ("anyOf", "oneOf"):
        if key in s:
            kinds = []
            for part in s[key]:
                t = _type(part)
                if t not in kinds:
                    kinds.append(t)
            return "(" + "|".join(kinds) + ")" if len(kinds) > 1 else kinds[0]
    kind = s.get("type")
    if kind == "array":
        return f"Array<{_type(s.get('items'))}>"
    if kind == "object":
        extra = s.get("additionalProperties")
        return f"Object<string, {_type(extra)}>" if isinstance(extra, dict) and extra else "Object"
    return {"string": "string", "integer": "number", "number": "number", "boolean": "boolean", "null": "null"}.get(kind, "*")


def _response(op: dict[str, Any]) -> str:
    for code in ("200", "201", "202"):
        content = (op.get("responses", {}).get(code) or {}).get("content") or {}
        for media in ("application/json",):
            if media in content:
                return _type(content[media].get("schema"))
    return "null" if "204" in op.get("responses", {}) else "*"


def _body(op: dict[str, Any]) -> str | None:
    content = (op.get("requestBody") or {}).get("content") or {}
    if "application/json" in content:
        return _type(content["application/json"].get("schema"))
    if content:
        return "Blob"
    return None


def generate(spec: dict[str, Any] | None = None) -> str:
    spec = spec or schema()
    lines = [HEADER, "(function () {", "  'use strict';", ""]
    # typedefs: the models the routes take and give, as editors and readers see them
    for name, model in sorted((spec.get("components") or {}).get("schemas", {}).items()):
        lines.append(f"  /** @typedef {{Object}} {name}")
        required = set(model.get("required") or [])
        for prop, ps in (model.get("properties") or {}).items():
            optional = "" if prop in required else "="
            key = prop if re.fullmatch(r"[A-Za-z_$][\w$]*", prop) else json.dumps(prop)
            lines.append(f"   *  @property {{{_type(ps)}{optional}}} {key}")
        lines.append("   */")
    lines.append("")
    routes = []
    for path, item in sorted(spec["paths"].items()):
        if not path.startswith("/api/"):
            continue
        for method, op in sorted(item.items()):
            name = route_name(op["operationId"], method, path)
            params = re.findall(r"\{(\w+)\}", path)
            query = [p["name"] for p in op.get("parameters", []) if p.get("in") == "query"]
            routes.append((camel(name), method.upper(), path[len("/api"):], params, query, op))
    seen: set[str] = set()
    lines += ["  const ROUTES = {"]
    for name, method, path, params, query, op in routes:
        if name in seen:
            raise ValueError(f"two routes are named {name}")
        seen.add(name)
        lines.append(f"    {name}: [{json.dumps(method)}, {json.dumps(path)}, {json.dumps(params)}, {json.dumps(query)}],")
    lines += ["  };", ""]
    lines += ["""  function build(name, args) {
    const [, template, params, query] = ROUTES[name];
    let i = 0;
    const path = template.replace(/\\{(\\w+)\\}/g, (_, key) => {
      const value = args[i++];
      if (value === undefined || value === null || value === '') throw new Error(`DC.client.${name} needs ${key}`);
      return encodeURIComponent(value);
    });
    const opts = args[params.length] || {};
    const pairs = [];
    Object.entries(opts.query || {}).forEach(([key, value]) => {
      if (!query.includes(key)) throw new Error(`DC.client.${name} has no query parameter ${key}`);
      if (value !== undefined && value !== null && value !== '') pairs.push(encodeURIComponent(key) + '=' + encodeURIComponent(value));
    });
    return { path: pairs.length ? path + '?' + pairs.join('&') : path, opts };
  }
  function call(name, args) {
    const { path, opts } = build(name, args);
    const { query, ...rest } = opts;
    return window.DC.api(path, Object.assign({}, rest, { method: ROUTES[name][0] }));
  }
""", "  const client = { path: {}, href: {} };"]
    for name, method, path, params, query, op in routes:
        doc = (op.get("summary") or name).strip().replace("*/", "* /")
        body = _body(op)
        options = ["query?: {" + ", ".join(f"{q}?: *" for q in query) + "}"] if query else []
        if body:
            options.append(f"body{'' if (op.get('requestBody') or {}).get('required') else '?'}: {body}")
        options.append("headers?: Object")
        lines.append(f"  /** {method} /api{path}: {doc}")
        for p in params:
            lines.append(f"   *  @param {{string}} {p}")
        lines.append(f"   *  @param {{{{{', '.join(options)}}}}} [opts]")
        lines.append(f"   *  @returns {{Promise<{_response(op)}>}} */")
        lines.append(f"  client.{name} = (...args) => call('{name}', args);")
        lines.append(f"  client.path.{name} = (...args) => build('{name}', args).path;")
        if method == "GET":
            lines.append(f"  client.href.{name} = (...args) => '/api' + build('{name}', args).path;")
    lines += ["", "  window.DC.client = client;", "})();", ""]
    return "\n".join(lines)

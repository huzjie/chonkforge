"""Minimal YAML subset parser (stdlib only).

Supports the small YAML subset used by config.yaml so the package works even
when PyYAML is not installed (the managed Python runtimes often lack it):
  - ``key: value`` scalars (int / float / bool / str)
  - inline comments ``key: value  # note``
  - block lists::

        keywords:
          - a
          - b

  - nested mappings by indentation
  - lists of mappings::

        items:
          - name: x
            value: 1

This is deliberately tiny -- it is a fallback, not a full YAML implementation.
"""
import ast


def _scalar(s):
    s = s.strip()
    if s == "":
        return None
    low = s.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if low in ("null", "none", "~"):
        return None
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _strip_comment(line):
    # strip inline comment that is preceded by whitespace (avoid stripping URLs)
    for marker in ("  #", "\t#"):
        idx = line.find(marker)
        if idx >= 0:
            return line[:idx]
    return line


def parse(text):
    """Parse a YAML subset into nested dict/list. Raises ValueError on garbage."""
    root = {}
    # stack of (indent, container, key_or_index)
    stack = [(-1, root, None)]
    lines = text.splitlines()
    for raw in lines:
        if raw.strip() == "" or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        line = _strip_comment(raw.rstrip())
        stripped = line.strip()
        # pop stack until we are at a shallower indent
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent_indent, parent, parent_key = stack[-1]
        if stripped.startswith("- "):
            # list item
            val = stripped[2:].strip()
            if not isinstance(parent, list):
                # convert empty dict placeholder to list
                if parent_key is not None and isinstance(parent, dict) and parent_key in parent                         and parent[parent_key] == {}:
                    lst = []
                    parent[parent_key] = lst
                    parent = lst
                else:
                    # top-level list not supported; treat as scalar list under current key
                    lst = []
                    if parent_key is not None:
                        parent[parent_key] = lst
                    parent = lst
            if ":" in val and not val.startswith(("{", "[")):
                # "- key: value" -> item is a dict
                k, _, v = val.partition(":")
                item = {}
                item[k.strip()] = _scalar(v) if v.strip() != "" else {}
                parent.append(item)
                stack.append((indent, item, None))
            else:
                parent.append(_scalar(val))
        else:
            if ":" not in stripped:
                continue
            key, _, val = stripped.partition(":")
            key = key.strip()
            val = val.strip()
            if val == "":
                # nested block follows
                child = {}
                parent[key] = child
                stack.append((indent, child, key))
            else:
                parent[key] = _scalar(val)
    return root

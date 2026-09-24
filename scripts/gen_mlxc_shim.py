#!/usr/bin/env python3
"""Generate the MLX C API runtime dispatch shim.

sinter/mlx does not link libmlxc at build time (no #cgo LDFLAGS -lmlx/-lmlxc):
entry points are resolved at runtime via dlopen (mlx/mlx_shim.c). This script
emits the two generated files:

  mlx/mlx_shim_dispatch_gen.h  - mlx_shim_dispatch_t: one typed function
                                 pointer per entry point
  mlx/mlx_shim_dispatch_gen.c  - one C wrapper per entry point (real name,
                                 forwards through the dispatch table) plus
                                 mlx_shim_resolve_dispatch()

The entry-point set is the intersection of (a) C.mlx_* references in the
package's .go files and (b) mlx_* call sites in the package's .c shims, and
(c) prototypes present in the vendored mlx-c headers (mlx/mlxc_headers/).

Re-run after adding a new C.mlx_* call or bumping the vendored headers:
  python3 scripts/gen_mlxc_shim.py
"""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PKG = ROOT / "mlx"
HDR_DIR = PKG / "mlxc_headers" / "mlx" / "c"

NAME_RE = re.compile(r"\bmlx_[a-z0-9_]+\s*\(")
OWN_PREFIXES = ("mlx_go_", "mlx_shim_")


def strip_comments(src: str) -> str:
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    return re.sub(r"//[^\n]*", "", src)


def collect_prototypes() -> dict:
    """name -> (ret, args) for every mlx_* function prototype in the headers."""
    protos = {}
    for hdr in sorted(HDR_DIR.glob("*.h")):
        text = strip_comments(hdr.read_text())
        lines = [l for l in text.splitlines() if not l.lstrip().startswith("#")]
        text = re.sub(r'extern\s+"C"\s*\{?', "", "\n".join(lines))
        for m in NAME_RE.finditer(text):
            name = m.group(0)[:-1].strip()
            open_paren = text.index("(", m.start())
            depth, k = 0, open_paren
            while k < len(text):
                if text[k] == "(":
                    depth += 1
                elif text[k] == ")":
                    depth -= 1
                    if depth == 0:
                        break
                k += 1
            if depth != 0:
                continue
            args = " ".join(text[open_paren + 1 : k].split())
            # Declaration boundary: the nearest preceding ';', '{', or '}'.
            end = max(text.rfind(";", 0, m.start()),
                      text.rfind("{", 0, m.start()),
                      text.rfind("}", 0, m.start()))
            ret = " ".join(text[end + 1 : m.start()].split())
            if not re.fullmatch(r"(const\s+)?[A-Za-z_]\w*(\s*\*)*", ret):
                continue
            if name in protos and protos[name] != (ret, args):
                sys.exit(f"conflicting prototypes for {name}: {protos[name]} vs ({ret}, {args})")
            protos[name] = (ret, args)
    return protos


def collect_go_names() -> set:
    names = set()
    for f in PKG.glob("*.go"):
        for m in re.finditer(r"C\.mlx_[a-z_0-9_]+", f.read_text()):
            names.add(m.group(0)[2:])
    return names


def collect_c_names() -> set:
    names = set()
    files = list(PKG.glob("*.c")) + list(PKG.glob("*.go"))
    for f in files:
        if "dispatch_gen" in f.name or "resolve_gen" in f.name:
            continue
        for m in re.finditer(r"\bmlx_[a-z_0-9_]+\s*\(", f.read_text()):
            n = m.group(0)[:-1].strip()
            if not any(n.startswith(p) for p in OWN_PREFIXES):
                names.add(n)
    return names


def param_name(param: str) -> str:
    p = param.strip()
    m = re.match(r"^\w[\w\s\*]*?\(\s*\*\s*(\w+)", p)
    if m:
        return m.group(1)
    toks = p.split()
    return toks[-1] if toks else "arg"


def collect_struct_types(text: str) -> set:
    """typedef'd struct names (handle types) — their zero form is (T){0}."""
    names = set()
    for m in re.finditer(r"typedef\s+struct\s+\w+_?\s*\{[^}]*\}\s*(\w+)\s*;", text):
        names.add(m.group(1))
    return names


def dummy_for(ret: str, struct_types: set):
    r = ret.strip()
    if r == "int":
        return "1"
    if r == "void":
        return None
    if r == "bool":
        return "false"
    if r == "size_t":
        return "0"
    if r in ("double", "float"):
        return "0.0"
    if re.fullmatch(r"(const\s+)?[A-Za-z_]\w*\s*\*", r):
        return "NULL"
    if re.fullmatch(r"[A-Za-z_]\w*", r):
        if r in struct_types:
            return f"({r}){{0}}"
        return "0"
    sys.exit(f"unhandled return type: {r!r}")


def split_params(args: str):
    if not args or args == "void":
        return []
    out, depth, buf = [], 0, []
    for ch in args:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    out.append("".join(buf))
    return [p for p in (x.strip() for x in out) if p]


def main() -> None:
    protos = collect_prototypes()
    struct_types = set()
    for hdr in HDR_DIR.glob("*.h"):
        struct_types |= collect_struct_types(strip_comments(hdr.read_text()))
    used = collect_go_names() | collect_c_names()
    required = sorted(n for n in used if n in protos)
    if not required:
        sys.exit("no entry points resolved — check mlx/mlxc_headers and C.mlx_* usage")

    own_or_types = sorted(
        n for n in used
        if n not in protos and not any(n.startswith(p) for p in OWN_PREFIXES)
    )
    print(f"entry points: {len(required)}")
    print(f"used but not function prototypes (types/enums, expected): {own_or_types}")

    header = [
        "// GENERATED by scripts/gen_mlxc_shim.py — do not edit.",
        "// Entry points: " + ", ".join(required),
        "",
        '#ifndef MLX_SHIM_DISPATCH_GEN_H',
        '#define MLX_SHIM_DISPATCH_GEN_H',
        "",
        "#include <mlx/c/mlx.h>",
        "",
        "typedef struct {",
    ]
    wrappers = [
        "// GENERATED by scripts/gen_mlxc_shim.py — do not edit.",
        "#include \"mlx_shim.h\"",
        '#include "mlx_shim_dispatch_gen.h"',
        "",
    ]
    resolve = [
        "// GENERATED by scripts/gen_mlxc_shim.py — do not edit.",
        "#include <dlfcn.h>",
        '',
        '#include "mlx_shim_dispatch_gen.h"',
        "",
        "int mlx_shim_resolve_dispatch(void* handle, mlx_shim_dispatch_t* d) {",
        "  void* p;",
    ]

    for name in required:
        ret, args = protos[name]
        params = split_params(args)
        arg_types = args if args else "void"
        header.append(f"  {ret} (*{name})({arg_types});")

        names = [param_name(p) for p in params]
        call = ", ".join(names)
        dummy = dummy_for(ret, struct_types)
        if ret.strip() == "void":
            body = (
                f"  if (mlx_shim_ready()) mlx_dispatch.{name}({call});"
            )
        else:
            body = (
                f"  if (!mlx_shim_ready()) {{ mlx_shim_report_notloaded(); return {dummy}; }}\n"
                f"  return mlx_dispatch.{name}({call});"
            )
        wrappers.append(f"{ret} {name}({args}) {{\n{body}\n}}")

        resolve.append(f'  p = dlsym(handle, "{name}");')
        resolve.append(f"  if (!p) return 0; d->{name} = ({ret} (*)({arg_types}))p;")

    header += ["} mlx_shim_dispatch_t;", "", "#endif", ""]
    resolve += ["  return 1;", "}", ""]
    constraint = "//go:build darwin && arm64 && cgo\n\n"
    (PKG / "mlx_shim_dispatch_gen.h").write_text(constraint + "\n".join(header))
    (PKG / "mlx_shim_dispatch_gen.c").write_text(constraint + "\n".join(wrappers))
    (PKG / "mlx_shim_resolve_gen.c").write_text(constraint + "\n".join(resolve))
    print("wrote mlx_shim_dispatch_gen.h, mlx_shim_dispatch_gen.c, mlx_shim_resolve_gen.c")


if __name__ == "__main__":
    main()

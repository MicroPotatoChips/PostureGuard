"""Check local XML/resource wiring without compiling or executing the Android app.

Optional Kotlin syntax checking: install binary wheels for tree-sitter and
tree-sitter-kotlin in your environment, or pass their directory with --parser-path.
This does not replace Android Lint, Kotlin type checking, or device testing.
"""
from pathlib import Path
import argparse
import collections
import re
import sys
import xml.etree.ElementTree as ET

PROJECT = Path(__file__).resolve().parents[1]
RES = PROJECT / "app/src/main/res"
ANDROID = "{http://schemas.android.com/apk/res/android}"
errors = []
resources = collections.defaultdict(set)
xml_files = sorted(RES.rglob("*.xml"))
trees = {}
for path in xml_files + [PROJECT / "app/src/main/AndroidManifest.xml"]:
    try:
        trees[path] = ET.parse(path).getroot()
    except ET.ParseError as error:
        errors.append(f"{path.relative_to(PROJECT)}: {error}")

for path in RES.rglob("*"):
    if not path.is_file():
        continue
    kind = path.parent.name.split("-")[0]
    if kind != "values":
        resources[kind].add(path.stem)
for path, root in trees.items():
    if path.parent.name.startswith("values"):
        names = set()
        for element in root:
            if "name" not in element.attrib:
                continue
            kind = element.attrib.get("type", element.tag)
            key = (kind, element.attrib["name"])
            if key in names:
                errors.append(f"{path.name}: duplicate {kind}/{key[1]}")
            names.add(key)
            resources[kind].add(key[1])
    for element in root.iter():
        ident = element.attrib.get(ANDROID + "id", "")
        if ident.startswith("@+id/"):
            resources["id"].add(ident.split("/", 1)[1])

external_styles = {
    "Widget.Material3.Button.OutlinedButton",
    "Widget.Material3.Button.TonalButton",
    "Widget.Material3.Button.TextButton",
    "Theme.Material3.Light.NoActionBar",
}
for path, root in trees.items():
    for element in root.iter():
        for value in element.attrib.values():
            match = re.fullmatch(r"@\+?(\w+)/([\w.]+)", value)
            if match:
                kind, name = match.groups()
                if name not in resources[kind] and not (kind == "style" and name in external_styles):
                    errors.append(f"{path.relative_to(PROJECT)}: missing {kind}/{name}")
        if element.tag == "style":
            parent = element.attrib.get("parent")
            if parent and parent not in external_styles and parent not in resources["style"] and not parent.startswith("android:"):
                errors.append(f"{path.name}: missing parent style {parent}")

layout_path = RES / "layout/activity_main.xml"
layout = trees[layout_path]
layout_ids = [e.attrib[ANDROID+"id"].split("/", 1)[1]
              for e in layout.iter() if ANDROID+"id" in e.attrib]
duplicates = [name for name, count in collections.Counter(layout_ids).items() if count > 1]
if duplicates:
    errors.append(f"activity_main.xml: duplicate view IDs {duplicates}")
custom_view = PROJECT / "app/src/main/java/com/postureguard/ScoreRingView.kt"
if not custom_view.exists():
    errors.append("Missing ScoreRingView class")
if len(layout) != 1:
    errors.append("NestedScrollView must contain exactly one child")

kotlin_files = sorted((PROJECT / "app/src").rglob("*.kt"))
for path in kotlin_files:
    code = path.read_text(encoding="utf-8")
    for kind, name in re.findall(r"(?<![\w.])R\.(\w+)\.(\w+)", code):
        if name not in resources[kind]:
            errors.append(f"{path.relative_to(PROJECT)}: missing R.{kind}.{name}")
        if kind == "id" and name not in layout_ids:
            errors.append(f"{path.relative_to(PROJECT)}: view {name} absent from activity_main")
    for model in re.findall(r'setModelAssetPath\("([^"]+)"\)', code):
        if not (PROJECT/"app/src/main/assets"/model).is_file():
            errors.append(f"Missing model asset: {model}")
    for ref in re.findall(r"PostureState\.(\w+)", code):
        if ref not in {"SEARCHING", "CHECKING", "GOOD", "ADJUST", "CALIBRATING", "ERROR"}:
            errors.append(f"Unknown state {ref}")

parser = argparse.ArgumentParser()
parser.add_argument("--parser-path", type=Path)
args = parser.parse_args()
if args.parser_path:
    sys.path.insert(0, str(args.parser_path.resolve()))
try:
    from tree_sitter import Language, Parser
    import tree_sitter_kotlin
    kotlin_parser = Parser(Language(tree_sitter_kotlin.language()))
    for path in kotlin_files:
        tree = kotlin_parser.parse(path.read_bytes())
        def visit(node):
            if node.type == "ERROR" or node.is_missing:
                errors.append(f"{path.relative_to(PROJECT)}: syntax error at line {node.start_point.row + 1}")
            for child in node.children:
                visit(child)
        visit(tree.root_node)
    print(f"Kotlin syntax: checked {len(kotlin_files)} files")
except ImportError:
    print("Kotlin syntax: skipped (optional parser unavailable)")

print(f"XML: parsed {len(trees)} files")
print(f"Resources: checked {sum(map(len, resources.values()))} names, {len(layout_ids)} view IDs")
print("No compilation, type checking, app execution, or device tests performed.")
if errors:
    print("\n".join(errors))
    raise SystemExit(1)
print("Static source checks passed.")

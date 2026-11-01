"""Check source/artifact inventories for unexpected files; no external tools needed."""

import argparse
from pathlib import Path, PurePosixPath
import tarfile
import zipfile


ROOT_FILES = {
    "README.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "CONTRIBUTING.md",
    "CHANGELOG.md", "pyproject.toml", "MANIFEST.in", ".gitignore",
}
IGNORED_DIRS = {
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".ruff_cache",
    "build", "dist",
}


def allowed(name, artifact=False):
    path = PurePosixPath(name)
    if path.name in ROOT_FILES and len(path.parts) == 1:
        return True
    if name == "docs/images/gui-empty.png" or name == ".github/workflows/ci.yml":
        return True
    if path.suffix == ".py" and path.parts[0] in {"src", "tests", "tools"}:
        return True
    if path.suffix == ".md" and path.parts[0] == "docs":
        return True
    if artifact:
        # Setuptools synthesizes this configuration in source archives.
        if name == "setup.cfg":
            return True
        if path.parts[0] == "eis_fitter" and path.suffix == ".py":
            return True
        if name == "PKG-INFO":
            return True
        if any(part.endswith((".dist-info", ".egg-info")) for part in path.parts):
            return path.name in {
                "METADATA", "WHEEL", "RECORD", "entry_points.txt", "top_level.txt",
                "PKG-INFO", "SOURCES.txt", "dependency_links.txt", "requires.txt",
                "LICENSE", "THIRD_PARTY_NOTICES.md",
            }
    return False


def inspect_source(root):
    names = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in IGNORED_DIRS or part.endswith(".egg-info")
               for part in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError(f"Symlinks are not release inputs: {relative}")
        if path.is_file():
            if not allowed(relative.as_posix()):
                raise ValueError(f"Unexpected release file: {relative}")
            names.append(relative.as_posix())
    required = ROOT_FILES | {"docs/images/gui-empty.png", ".github/workflows/ci.yml"}
    if missing := required - set(names):
        raise ValueError(f"Missing release files: {sorted(missing)}")
    return names


def inspect_artifacts(folder):
    paths = sorted(folder.glob("*.whl")) + sorted(folder.glob("*.tar.gz"))
    if not any(path.suffix == ".whl" for path in paths) or not any(
            path.name.endswith(".tar.gz") for path in paths):
        raise ValueError("Both a wheel and a source archive are required.")
    for archive in paths:
        if archive.suffix == ".whl":
            with zipfile.ZipFile(archive) as handle:
                names = [name for name in handle.namelist() if not name.endswith("/")]
        else:
            with tarfile.open(archive) as handle:
                if any(member.issym() or member.islnk() for member in handle.getmembers()):
                    raise ValueError(f"Archive contains links: {archive.name}")
                names = [name.split("/", 1)[1] for name in handle.getnames()
                         if "/" in name and handle.getmember(name).isfile()]
        for name in names:
            if ".." in PurePosixPath(name).parts or name.startswith("/"):
                raise ValueError(f"Unsafe archive path: {name}")
            if not allowed(name, artifact=True):
                raise ValueError(f"Unexpected archive member: {archive.name}: {name}")
        print(f"{archive.name}: {len(names)} permitted files")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    names = inspect_source(root)
    print(f"Source inventory: {len(names)} permitted files")
    if args.artifacts:
        inspect_artifacts(args.artifacts.resolve())


if __name__ == "__main__":
    main()

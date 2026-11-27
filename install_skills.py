#!/usr/bin/env python3
"""Sync gs-modelflows skills across repo agent directories or install globally.

Usage:
    # Keep .agent/skills, .agents/skills, and .claude/skills in sync with skills/
    python install_skills.py --sync-repo

    # Install globally for Antigravity, Claude Code, and/or Codex
    python install_skills.py --target all
    python install_skills.py --target claude
    python install_skills.py --target codex
    python install_skills.py --target antigravity
"""
import argparse
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
SKILLS_SRC = REPO_ROOT / "skills"
SKILL_NAMES = ("gs-modeling", "gs-to-sprite", "gs-to-mesh", "gs-stage")

REPO_AGENT_DIRS = [
    REPO_ROOT / ".agent" / "skills",
    REPO_ROOT / ".agents" / "skills",
    REPO_ROOT / ".claude" / "skills",
]


def copy_skill_tree(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        src,
        dst,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
    )


def sync_repo() -> None:
    for agent_dir in REPO_AGENT_DIRS:
        for name in SKILL_NAMES:
            src = SKILLS_SRC / name
            if not src.exists():
                raise FileNotFoundError(f"Missing canonical skill: {src}")
            dst = agent_dir / name
            copy_skill_tree(src, dst)
        print(f"[sync-repo] Updated {agent_dir.relative_to(REPO_ROOT)}")


def global_target_dirs(target: str) -> list[tuple[str, Path]]:
    home = Path.home()
    mapping = {
        "antigravity": home / ".gemini" / "config" / "skills",
        "claude": home / ".claude" / "skills",
        "codex": home / ".codex" / "skills",
    }
    if target == "all":
        return list(mapping.items())
    return [(target, mapping[target])]


def install_global(target: str, force: bool = False) -> None:
    spark_dir = (REPO_ROOT / "renderer" / "spark_demo").resolve()
    for label, root_dir in global_target_dirs(target):
        installed = 0
        for name in SKILL_NAMES:
            src = SKILLS_SRC / name
            dst = root_dir / name
            if (dst.exists() or dst.is_symlink()) and not force:
                print(f"[install] Skipped existing {dst} (pass --force to overwrite)")
                continue
            copy_skill_tree(src, dst)
            if (src / "scripts" / "render_views.py").exists() and (spark_dir / "capture.mjs").exists():
                # global skills live outside the repo: ship the renderer next to the scripts
                shutil.copytree(spark_dir, dst / "renderer" / "spark_demo", dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns("node_modules", "vendor", "out*"))
            installed += 1
        print(f"[install] Installed {installed}/{len(SKILL_NAMES)} skills for {label} -> {root_dir}")
    print(f"[install] Renderer copied into the installed skills; GS_SPARK_DIR={spark_dir} overrides it.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync or install gs-modelflows skills.")
    parser.add_argument(
        "--sync-repo",
        action="store_true",
        help="Synchronize .agent/skills, .agents/skills, and .claude/skills from skills/.",
    )
    parser.add_argument(
        "--target",
        choices=("all", "antigravity", "claude", "codex"),
        help="Install skills into global agent skill directory.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing skills in global target directories.",
    )
    args = parser.parse_args()
    if not args.sync_repo and not args.target:
        parser.print_help()
        return
    if args.sync_repo:
        sync_repo()
    if args.target:
        install_global(args.target, force=args.force)


if __name__ == "__main__":
    main()

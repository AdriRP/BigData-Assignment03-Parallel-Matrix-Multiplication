"""Compile the Java sources into build/java."""
import subprocess

from common import JAVA_CLASSES, ROOT


def main():
    JAVA_CLASSES.mkdir(parents=True, exist_ok=True)
    sources = sorted(str(p) for p in (ROOT / "src" / "java").glob("*.java"))
    cmd = ["javac", "--release", "21", "-d", str(JAVA_CLASSES), *sources]
    print(">", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("build OK")


if __name__ == "__main__":
    main()

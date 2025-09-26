import sys, pathlib, pytest

def main() -> int:
    root = pathlib.Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(root / "src"))  # ensure "src" is importable
    reports = root / "reports"
    reports.mkdir(exist_ok=True)

    targets = [p for p in ("src/data/db/tests", "tests") if (root / p).exists()]

    args = [
        *targets,
        "-q", "-rA", "--maxfail=0", "--disable-warnings", "--durations=10",
        f"--junitxml={reports.as_posix()}/junit.xml",
    ]
    rc = pytest.main(args)
    print("\n✅ All tests passed." if rc == 0 else f"\n❌ Some tests failed. Exit code: {rc}\nSee reports/junit.xml")
    return rc

if __name__ == "__main__":
    raise SystemExit(main())

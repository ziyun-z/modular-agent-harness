# paste into a Python REPL or save as scripts/smoke_sandbox.py
import sys
sys.path.insert(0, ".")
from src.sandbox import DockerSandbox

class FakeTask:
    instance_id = "smoke-test"
    repo = "pallets/flask"
    base_commit = "2.3.3"   # a real tag
    test_patch = None

sandbox = DockerSandbox()
try:
    sandbox.setup(FakeTask())

    # 1. exec
    r = sandbox.exec("python --version")
    assert r.ok, f"exec failed: {r.stdout}"
    print("exec OK:", r.stdout.strip())

    # 2. write + read
    sandbox.write_file("hello.txt", "hello world\n")
    content = sandbox.read_file("hello.txt")
    assert "hello world" in content
    print("write/read OK")

    # 3. edit
    sandbox.edit_file("hello.txt", "hello world", "hello sandbox")
    content = sandbox.read_file("hello.txt")
    assert "hello sandbox" in content
    print("edit OK")

    # 4. search_code
    results = sandbox.search_code("Flask", file_glob="*.py")
    print("search_code OK, hits:", len(results.splitlines()))

    # 5. get_diff (should be empty — no tracked files changed)
    diff = sandbox.get_diff()
    print("get_diff OK, diff length:", len(diff))

    print("\nAll checks passed.")
finally:
    sandbox.teardown()
    print("Sandbox torn down.")

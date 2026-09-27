"""Run the suite with one `unittest` process per test class, N at a time.

Usage: python run_parallel.py [-j N]   (from this directory)
"""
from __future__ import annotations

import argparse
import os
import queue
import re
import subprocess
import sys
import threading
import time
import unittest
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))


def test_classes() -> Counter:
    loader = unittest.TestLoader()
    suite = loader.discover(HERE)
    if loader.errors:
        sys.exit("\n".join(loader.errors))
    classes: Counter = Counter()

    def walk(node):
        for item in node:
            if isinstance(item, unittest.TestSuite):
                walk(item)
            else:
                classes[item.id().rsplit(".", 1)[0]] += 1

    walk(suite)
    return classes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--jobs", type=int, default=8)
    jobs = parser.parse_args().jobs

    classes = test_classes()
    # Biggest classes first, so a long one does not start last and set the wall time.
    work: queue.Queue = queue.Queue()
    for name in sorted(classes, key=classes.get, reverse=True):
        work.put(name)
    ran, failed = 0, []
    lock = threading.Lock()

    def worker():
        nonlocal ran
        while True:
            try:
                name = work.get_nowait()
            except queue.Empty:
                return
            result = subprocess.run(
                [sys.executable, "-m", "unittest", name], cwd=HERE,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            count = re.search(r"^Ran (\d+) test", result.stderr, re.M)
            with lock:
                ran += int(count.group(1)) if count else 0
                if result.returncode != 0:
                    failed.append(name)
                    sys.stderr.write(result.stderr)

    started = time.perf_counter()
    threads = [threading.Thread(target=worker) for _ in range(jobs)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    elapsed = time.perf_counter() - started
    print(f"Ran {ran} tests in {len(classes)} classes in {elapsed:.0f}s with {jobs} processes")
    if failed:
        print("FAILED: " + ", ".join(sorted(failed)))
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

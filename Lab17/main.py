import re
import sys
from pathlib import Path


ANSI_CODES = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")

class Tee:
    """Sends output to the terminal and to a log file."""

    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file

    def write(self, text):
        self.stream.write(text)
        self.log_file.write(ANSI_CODES.sub("", text))
        self.log_file.flush()
        return len(text)

    def flush(self):
        self.stream.flush()
        self.log_file.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)

# Runs before crewai is imported, because its loggers keep a reference to the
# streams that exist at import time
log_file = open(Path(__file__).with_name("run.txt"), "w", encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8") # type: ignore
sys.stderr.reconfigure(encoding="utf-8") # type: ignore
sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = Tee(sys.stderr, log_file)

from crew import crew, memory, memory_crew

def main():

    # Start empty so the memory demo only recalls what this run saved
    memory.reset_all()

    result = crew.kickoff()

    print("\n")
    print("=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    print(result)
    print("\n")
    print("=" * 60)
    print("DEMO 2: CREWAI MEMORY")
    print("=" * 60)

    memory_result = memory_crew.kickoff()

    print("\n")
    print("MEMORY RESULT")
    print("=" * 60)

    print(memory_result)

if __name__ == "__main__":
    main()

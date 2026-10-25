from rich.console import Console
from rich.markdown import Markdown
from manager import run_manager
from state import create_state

console = Console()

def main():

     print("=" * 50)
     print(" INTELLIGENT MULTI-AGENT SYSTEM")
     print("=" * 50)

     task = input("\nEnter your task: ")

     state = run_manager(task)

     print("\n" + "=" * 50)
     print(" RESEARCH")
     print("=" * 50)

     console.print(Markdown(state["research"] or ""))

     print("\n" + "=" * 50)
     print(" CODE")
     print("=" * 50)

     console.print(Markdown(state["code"] or ""))

     print("\n" + "=" * 50)
     print(" REVIEW")
     print("=" * 50)
     
     review = state["review"] or {}

     console.print(Markdown(
          f"**Status:** {review.get('status', '')}\n\n"
          f"**Feedback:** {review.get('feedback', '')}"
     ))

if __name__ == "__main__":
    main()


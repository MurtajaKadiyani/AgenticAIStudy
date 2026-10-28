from rich.console import Console
from rich.markdown import Markdown
from agent.manager import run_manager
from agent.state import create_state
from permission import demonstraton_permission

console = Console()

def main():

     print("=" * 50)
     print(" INTELLIGENT MULTI-AGENT SYSTEM")
     print("=" * 50)

     task = input("\nEnter your task: ")

     try:

       state = run_manager(task)

     except ValueError as e:
         print(f"\nValidation error: {e}")
         return

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

     demonstraton_permission()

if __name__ == "__main__":
    main()


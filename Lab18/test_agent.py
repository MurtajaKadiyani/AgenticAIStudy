from agent import (
    research_agent,
    generate_report
)

from tools import (
    get_verified_research_notes
)


request = (
    "Research the role of AI agents "
    "in software development. "
    "Identify major use cases, benefits, "
    "and limitations."
)


research_agent(
    request,
    max_steps=12
)


verified_notes = (
    get_verified_research_notes()
)


if not verified_notes:

    print(
        "\nNo verified research findings "
        "were collected."
    )

    raise SystemExit



print(
    "\nVerified findings:",
    len(verified_notes)
)


report = generate_report(
    user_request=request,
    verified_notes=verified_notes
)


print("\n")
print("=" * 60)
print("FINAL RESEARCH REPORT")
print("=" * 60)
print("\n")

print(report)
import os
from crewai import Crew,Memory,Process
from crewai.rag.embeddings.factory import build_embedder
from dotenv import load_dotenv
from agents import (researcher,coder,reviewer)
from tasks import (research_task,coding_task,review_task,memory_task)
from llm import manager_llm,memory_llm

load_dotenv()

embedder = {
    "provider": os.getenv("EMBED_PROVIDER"),
    "config": {
        "model_name": os.getenv("EMBED_MODEL"),
        "url": os.getenv("EMBED_URL")
    }
}

# Both crews share one memory so memory_crew can recall what crew saved
memory = Memory(
    llm=memory_llm,
    embedder=build_embedder(embedder), # type: ignore
    root_scope="/crew/lab17"
)

crew = Crew(
    agents=[researcher,coder,reviewer],
    tasks=[research_task,coding_task,review_task],
    process=Process.hierarchical,
    manager_llm=manager_llm,
    memory=memory,
    verbose=True
)

memory_crew = Crew(
    agents=[researcher],
    tasks=[memory_task],
    memory=memory,
    verbose=True
)
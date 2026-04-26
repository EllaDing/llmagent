# Draft Notes

## F24 Lecture 3 — LLM Agent Infrastructure

<!-- Jot notes here while watching. Tell Claude "save notes to F24 Lecture 3" when done. -->

### Core Topics
1. What will future AI applications look like?
2. How to empower every developer to build them?

---

### What Will Future AI Applications Look Like?

**[The Shift from Models to Compound AI Systems, Zaharia et al., BAIR Blog 2024](https://bair.berkeley.edu/blog/2024/02/18/compound-ai-systems/)**

**The trend:** best AI results are increasingly obtained by *compound systems* — multiple interacting components — not monolithic models alone.

**Compound AI System:** a system that tackles AI tasks using multiple interacting components, including multiple calls to models, retrievers, or external tools.

**The trend from generative content → agent:**
- Early AI apps: single model call → generates content (text, image, code)
- Next: compound systems — model + retriever + tools + multiple LLM calls + filtering
- Example: AlphaCode 2 — generates up to 1 million candidate solutions, then filters down; SOTA on programming benchmarks
- Agents are the natural endpoint: compound systems that reason, act, and adapt dynamically rather than executing a fixed pipeline

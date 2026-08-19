# F24 Lecture 3 — Agentic AI Frameworks & Multimodal Knowledge Assistants
**Sept 23, 2024** | [Video](https://www.youtube.com/live/OOdtmCMSOo4)
**Part 1:** Chi Wang (AutoGen-AI) — Agentic AI Frameworks & AutoGen · [slides](https://rdi.berkeley.edu/llm-agents-mooc/slides/autogen.pdf)
**Part 2:** Jerry Liu (LlamaIndex) — Building a Multimodal Knowledge Assistant · [slides](https://rdi.berkeley.edu/llm-agents-mooc/slides/MKA.pdf)

---

## Framing: Two Questions

1. What will future AI applications look like?
2. How do we empower every developer to build them?

**The bet (made early 2023):** future AI applications are *agentic* — agents as a new way for humans to interact with the digital world. Many doubted agentic AI was viable at the time; Chi Wang cites [the Berkeley compound AI systems article](https://bair.berkeley.edu/blog/2024/02/18/compound-ai-systems/).

**What's newly possible:** old agentic applications (assistants, bots, game agents) are far easier and more capable to build; genuinely new ones appear — science discovery agents, web automation, software agents building from scratch. Demo: a multi-agent system builds a working website, then self-heals after a critical line is deleted.

**Three benefits of agentic AI:** natural-language interface · complex tasks with minimal supervision · **a software architecture** — the under-discussed one.

---

## Part 1 — AutoGen (Chi Wang)

**Core abstraction:** the *conversable agent*. An agent's backend can be an LLM, a tool, or a human — one concept covers all three. Build in two steps: **define the agents, then get them to talk.**

**Conversation patterns:**
| Pattern | Use |
|---|---|
| Two-agent | Reflection — writer + critic iterating |
| Nested chat | An agent whose "single" reply is really an inner multi-agent conversation |
| Group chat | A manager selects the next speaker; optional state-machine constraints |

Nesting is the key idea: agents contain agents, composed recursively.

**Tool grounding — conversational chess.** Two LLM agents playing directly make illegal moves. Add a board agent backed by a Python chess library, and legality is enforced outside the model. *The tool owns the truth.*

**Multi-agent vs. single-agent:** splitting writer/safeguard beat a single agent by ~20% on GPT-4, more on GPT-3.5. Chi Wang's own caveat: the benefit **scales with task complexity and inversely with model capacity.**

**Design axes any framework must handle:** static vs. dynamic workflow · flexibility vs. controllability · shared vs. isolated context · cooperation vs. competition · centralized vs. decentralized · automation vs. intervention.

**AutoBuild:** given a task, auto-generate the agent team; adaptive variant builds per-step teams and grows a reusable agent library.

---

## Part 2 — Multimodal Knowledge Assistant (Jerry Liu)

**Basic RAG's four limits:** primitive parsing/chunking · LLM used only for synthesis, never reasoning or planning · one-shot · stateless.

**Four ingredients of a better assistant:** high-quality multimodal retrieval · richer outputs (reports, actions) · agentic reasoning over inputs · deployment.

**Garbage in, garbage out.** Parsing quality is the ceiling on everything downstream — a bad PDF parser hallucinates before the LLM ever sees the text.

**Indexing trick:** don't embed the source element directly. Extract *representations* that point to it — summaries → table/image, small chunk → parent chunk — then dereference on retrieval and feed text + images to a multimodal model.

**Agentic RAG:** treat retrieval as a *tool*, put an agent reasoning layer on top. Not a fixed pre-step.

**Constrained vs. unconstrained flows:**
- *Constrained* — you write the if/else and loops; router → tool → reflection → response. Reliable, cheaper, less expressive.
- *Unconstrained* — hand the agent tools and let it plan (ReAct, LLM Compiler). Expressive, but can loop forever, call the wrong things, costs more.
- Most production systems lean constrained, for trust.

**Agent (rough definition):** a program with a nonzero number of LLM calls.

**Workflows:** event-driven step orchestration — steps listen for and emit messages. Deploy each as a microservice behind a central message queue, with human-in-the-loop as a server-side pause (the Devin pattern).

---

## ⚠️ Dated as of Aug 2026

This lecture is about *infrastructure*, the fastest-moving layer. The framing aged well; the tooling did not.

**Still holds:** unified agent abstraction · recursive nesting (= today's subagents, now used for **context isolation**) · the design axes · tool-as-source-of-truth · parsing quality as the ceiling · retrieval-as-tool · constrained vs. unconstrained · event-driven workflows · human-in-the-loop pauses.

**Out of date:**
| In the lecture | Now |
|---|---|
| AutoGen as a framework to adopt | Wang + Wu left Microsoft Nov 2024 (~2 mo. after this talk) → forked to **AG2**. Microsoft: AutoGen to maintenance mode Oct 2025, merged with Semantic Kernel into **Microsoft Agent Framework** (1.0 GA Apr 2026). Don't learn the API. |
| Conversation as the universal primitive | Lost — the tool-calling loop with typed events won. AutoGen itself rewrote to event-driven actors in v0.4. |
| Multi-agent gives +20% | Directionally reversed on frontier models. Split for context isolation, not quality. |
| AutoBuild (auto-generated agent teams) | Superseded by human-authored **Agent Skills** + subagent definitions. |
| "Keep tools under ~10" | Obsolete — the fix for tool sprawl is **progressive disclosure**, not a cap. |
| Hand-built multimodal RAG; CLIP image embeddings | Long context + native document/vision understanding thins the pipeline; ColPali-style visual retrieval replaced CLIP. |
| Agents as microservices behind a custom message layer | Right instinct, hand-rolled. **MCP** (Nov 2024) standardized the tool half; **A2A** (Apr 2025) the agent-to-agent half. |

**The real gap — Agent Skills.** Neither speaker has a concept of portable, model-readable capability packaging; both assume capability = Python wired up inside their framework. Skills invert it: a folder of Markdown + scripts loaded on demand. That kills AutoBuild and the tool ceiling — but *vindicates* the modularity thesis. The module boundary just moved from "a Python class in your framework" to "a folder on disk / an MCP server," which survives model swaps and requires no one to adopt your framework.

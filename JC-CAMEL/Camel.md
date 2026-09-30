# CAMEL: Communicative Agents for "Mind" Exploration of Large Language Model Society

**Authors:** Guohao Li, Hasan Anil, Sotiris Triantafillou, Dario Amodei, et al.  
**Institution:** King Abdullah University of Science and Technology (KAUST)  
**Year:** 2023  
**arXiv:** 2303.17760  

---

## 1. Overview

CAMEL proposes a **role-playing multi-agent framework** in which two LLM-based agents — one assigned the role of *AI User* and one assigned the role of *AI Assistant* — collaborate through natural-language conversation to complete complex tasks **autonomously**, with minimal human intervention.

The paper also releases the **AI Society dataset**: 25,000 conversations generated across 50 user roles × 50 assistant roles × 10 tasks each, intended as a resource for studying emergent LLM social behaviour.

The title's scare quotes around "Mind" signal epistemological caution: the paper explores what looks like social cognition in LLMs without claiming consciousness or genuine mental states.

---

## 2. Motivation

### The Single-Agent Bottleneck

Most LLM applications in 2022–2023 route a single model through a single turn or a simple chain. This works well for bounded tasks (QA, text generation, simple code) but fails for:

- Long-horizon, multi-step tasks
- Tasks requiring simultaneous specialisation in multiple domains
- Tasks where self-critique is unreliable (the model judges its own outputs)

### Why Multi-Agent?

Human teams solve hard problems through division of cognitive labour: managers decompose tasks, experts apply deep knowledge, peers critique each other. CAMEL asks whether LLMs can replicate this structure through structured dialogue.

### The Core Research Question

> Can LLMs, when assigned roles and given a structured communication protocol, autonomously complete complex tasks through natural-language conversation — without human intervention at each step?

---

## 3. The CAMEL Framework — Methods in Detail

### 3.1 System Architecture: Three Actors

CAMEL involves three distinct actors with different roles:

```
ACTOR 1: Human Orchestrator
  Provides: idea_type, assistant_role_name, user_role_name, task
  Hands off after inception prompts are constructed
  May optionally re-enter as Critic (human-in-the-loop mode)
        │
        ▼
  INCEPTION PROMPT CONSTRUCTOR (deterministic template-filling)
        │                      │
        ▼                      ▼
ACTOR 2: AI USER          ACTOR 3: AI ASSISTANT
(role_name = X)   ◄──────► (role_name = Y)
instructs, directs  turns   executes, responds
```

### 3.2 Step 1 — Human Input

The human specifies exactly **four things** and nothing else:

| Field | Description | Example |
|---|---|---|
| `idea_type` | Broad domain/concept | `"stock trading"` |
| `assistant_role_name` | The executing agent's expertise | `"Stock Trader"` |
| `user_role_name` | The directing agent's expertise | `"Python Programmer"` |
| `task` | Plain-language task description | `"Develop a trading bot for the stock market"` |

The human does not write any prompt, script any turn, or define success criteria. This is the paper's core thesis in miniature: minimal human specification is sufficient for complex task completion.

### 3.3 Step 2 — Role Assignment via Inception Prompting

The framework fills templates from the human inputs to construct system prompts. Critically, **no LLM is used at this stage** — it is deterministic template-filling.

**AI User system prompt (verbatim from paper):**

```
Never forget you are a {user_role_name} and I am a {assistant_role_name}.
Never flip roles! Never instruct me!
We share a common interest in collaborating to successfully complete a task.
You must help me to complete the task.
Here is the task: {task}. Never forget our task!
I must instruct you based on your expertise and my needs to complete the task.

I must give you one instruction at a time.
You must write a specific solution that appropriately completes the requested instruction.
You must decline my instruction honestly if you cannot perform the instruction
due to physical, moral, legal reasons or your capability and explain the reasons.
Do not add anything else other than your solution to my instruction.
You are never supposed to ask me any questions.
You should always start with: Solution: <YOUR_SOLUTION>

<YOUR_SOLUTION> must contain code if the instruction requires writing code.
Always end <YOUR_SOLUTION> with: Next request.
```

**AI Assistant system prompt (verbatim from paper):**

```
Never forget you are a {assistant_role_name} and I am a {user_role_name}.
Never flip roles! You will always be a {assistant_role_name}.
We share a common interest in collaborating to successfully complete a task.
I must help you to complete the task.
Here is the task: {task}. Never forget our task!

You must instruct me based on my expertise and your needs to complete the task
ONLY in the following two ways:

1. Instruct with a necessary input:
Instruction: <YOUR_INSTRUCTION>
Input: <YOUR_INPUT>

2. Instruct without any input:
Instruction: <YOUR_INSTRUCTION>
Input: None

You must give me one instruction at a time. ...
Do not add anything else. ...
Only say "CAMEL_TASK_DONE" when the task is completed.
```

**The structural asymmetry is intentional:** The User prompt says "I must instruct *you*"; the Assistant prompt says "You must instruct *me*." Each agent believes it is the one directing, which locks in the hierarchy without either agent needing to know the other's system prompt.

**What inception prompting prevents:**

Without it, two LLMs given a shared goal typically produce:
- Sycophantic politeness loops ("You go first!" / "No, you!")
- Role reversal after a few turns
- Task drift as agents follow conversational tangents
- Premature completion signals

### 3.4 Step 3 — Task Specification Pipeline (Three Stages)

Before any conversation runs, tasks must be generated at scale. This uses a separate LLM pipeline:

**Stage 1 — Role Pair Generation:**
```
Prompt: "List 50 diverse occupations or roles that could collaborate
         on tasks involving {idea_type}.
         Format: (directing_role, executing_role)"

Output: ("Python Programmer", "Stock Trader"),
        ("Data Scientist", "Financial Analyst"),
        ("Cybersecurity Expert", "Penetration Tester"), ...
```

**Stage 2 — Task Generation per Role Pair:**
```
Prompt: "List 10 tasks that a {user_role_name} and {assistant_role_name}
         can collaborate on involving {idea_type}."

Output: "Develop a trading bot",
        "Backtest a momentum strategy",
        "Build a portfolio risk dashboard", ...
```

**Stage 3 — Task Elaboration:**
```
Prompt: "Elaborate the following task for a {user_role_name} and
         {assistant_role_name}: {task}
         Make it more specific with clear deliverables."

Output: "Create a Python-based algorithmic trading bot that connects
         to the Alpha Vantage API, implements a moving average crossover
         strategy, includes backtesting functionality over 5 years of
         historical data, and outputs a performance report with Sharpe
         ratio and maximum drawdown metrics."
```

Each stage is necessary: Stage 1 ensures role-domain diversity, Stage 2 ensures task-role relevance, Stage 3 ensures operationalisability (tasks vague enough to be underspecified are elaborated into tasks with concrete deliverables that can be assessed for completion).

### 3.5 Step 4 — The Conversation Protocol

Once system prompts are set, the human injects a single opening message to the AI User: `"Start the conversation. Task: {elaborated_task}"`. The protocol then runs autonomously:

```
[TURN t]
  AI User generates:
    "Instruction: <instruction text>
     Input: <supporting context, or None>"

  AI Assistant generates:
    "Solution: <solution text, including code if needed>
     Next request."

[TERMINATION — one of three conditions]
  (a) AI Assistant outputs "CAMEL_TASK_DONE"
  (b) Max turns reached (T_max = 30 in experiments)
  (c) Critical failure: irrecoverable role break or refusal loop
```

**Protocol rules enforced by inception prompting:**
- One instruction per turn (no multi-part directives)
- Structured message formats (Instruction/Input and Solution/Next request)
- No unprompted questions from the assistant
- No topic change
- No role reversal

**Turn anatomy example (Turn 3 of a trading bot task):**

*AI User message:*
```
Instruction: Implement a moving average crossover strategy
             that generates buy/sell signals.
Input: Use the DataFetcher class defined in Turn 1.
       Short window = 50 days, long window = 200 days.
```

*AI Assistant message:*
```
Solution:
  def sma_crossover_strategy(ticker: str) -> pd.DataFrame:
      df = DataFetcher(ticker).get_historical()
      df['SMA50']  = df['Close'].rolling(50).mean()
      df['SMA200'] = df['Close'].rolling(200).mean()
      df['Signal'] = np.where(df['SMA50'] > df['SMA200'], 1, -1)
      return df
  # Buy on golden cross (SMA50 > SMA200), sell on death cross
Next request.
```

The assistant never redirects, never asks what ticker to use, never proposes a different strategy. It executes and returns control.

### 3.6 Step 5 — The Critic: Human-in-the-Loop Mode

The paper describes an **optional Critic agent** — a third participant that reviews the conversation at configurable intervals and can intervene before the next turn proceeds. This is one of the most important and underappreciated methodological contributions.

**What the Critic does:**
- Reads the last K turns of conversation
- Evaluates: factual correctness, relevance, role adherence, task progress
- Can **approve** (conversation continues unchanged)
- Can **redirect** (injects a correction message to the User or Assistant)
- Can **terminate** (flags irrecoverable failure)

**Three Critic configurations:**

| Configuration | Critic | Frequency | Use Case |
|---|---|---|---|
| Fully autonomous | None | — | Dataset generation at scale |
| LLM Critic | GPT-4 with critic system prompt | Every turn | Automated quality control |
| Human Critic | Human domain expert | On demand / scheduled | High-stakes validation |

**LLM Critic system prompt (key excerpt):**
```
You are a critic evaluating a conversation between
a {user_role_name} and {assistant_role_name}
working on: {task}

Review the last exchange. Assess:
1. Is the solution correct and relevant to the instruction?
2. Is the instruction coherent and task-directed?
3. Is the overall task making progress?

Output exactly one of:
APPROVED
REDIRECT: <correction message>
TERMINATE
```

**Why the Critic matters — a concrete example:**

Without a Critic, a hallucinated function in Turn 2 gets silently built upon in Turns 5, 8, and 12 — the agents never revisit it. With a Critic:

```
Turn 4 — Assistant produces code calling fetch_realtime()
         (a function never defined in the codebase)

  → Without Critic: User says "Next request." Turn 5 builds on it.
  → With LLM Critic: REDIRECT: "fetch_realtime is not defined.
    Ask the assistant to use get_historical() as defined in Turn 1."
  → Turn 5: User corrects. Assistant fixes. Conversation recovers.
```

The Critic operationalises a key insight: **autonomy without external quality signal produces confident, compounding errors**. The Critic is the minimum viable oversight mechanism.

**Key finding on Critic performance:** The LLM Critic catches approximately 70% of errors a human critic would catch, with a ~15% false positive rate (incorrectly redirecting a correct solution). Human Critics are primarily used for post-hoc dataset validation rather than real-time intervention.

### 3.7 Role Assignment Design Decisions

Several non-obvious choices shape how roles function:

- **Complementary, not identical roles:** User = Python Programmer, Assistant = Stock Trader. Each agent has something the other needs, creating epistemic asymmetry. Identical roles remove the reason to collaborate.
- **User does not execute:** Explicitly prohibited from writing solutions. Prevents the conversation collapsing into a monologue.
- **Natural-language occupational titles:** "Stock Trader" (not "Agent B") activates domain-appropriate knowledge profiles in the LLM.
- **Shared task knowledge:** Both agents know the same goal. This is cooperative, not information-asymmetric. The task is a shared commitment.

### 3.8 The AI Society Dataset

| Property | Value |
|---|---|
| Total conversations | 25,000 |
| User roles | 50 |
| Assistant roles | 50 |
| Tasks per role pair | 10 |
| Max turns per conversation | 30 |
| Domains | Programming, finance, medicine, law, chemistry, biology, physics, creative writing, education, marketing, and more |

Each record contains: `user_role`, `assistant_role`, `task`, `task_elaborated`, `sys_prompt_user`, `sys_prompt_asst`, full `messages` array, `completed` boolean, `num_turns`, and Critic intervention log (where enabled).

---

## 4. The AI Society Dataset

| Property | Value |
|---|---|
| Total conversations | 25,000 |
| User roles | 50 |
| Assistant roles | 50 |
| Tasks per role pair | 10 |
| Domains | Trading, coding, medicine, law, science, creative writing, education, engineering, and more |
| Contents per conversation | System prompts, full multi-turn dialogue, completion signal, metadata |

**Primary purpose:** Enable scientific study of LLM-to-LLM communication, emergent social behaviour, and role consistency at scale. Also suitable for fine-tuning future multi-agent systems.

---

## 5. Experiments and Results

### 5.1 Setup

- **Models:** `gpt-4` and `gpt-3.5-turbo`, used for both agent roles
- **Max turns:** 30 per conversation
- **Evaluation:** Human annotation (task completion, solution quality, role fidelity) + heuristic (`<CAMEL_TASK_DONE>` reached)

### 5.2 Task Completion

- ~89% of GPT-4 conversations reach `<CAMEL_TASK_DONE>` within 30 turns
- ~75% rated by humans as meaningfully complete
- ~60% rated as high quality
- GPT-3.5 shows notably lower performance on all three metrics

### 5.3 Role Consistency

GPT-4 agents maintain role-appropriate language, framing, and expertise across long conversations. Role "slippage" (e.g., the assistant beginning to direct rather than execute) is rare with GPT-4, more common with GPT-3.5.

### 5.4 Emergent Behaviours

Several unexpected patterns emerged without explicit instruction:

**Role-conditioned knowledge activation**  
The same underlying model exhibits different knowledge profiles depending on role assignment. A "Stock Trader" agent spontaneously uses financial vocabulary and applies risk concepts; a "Python Programmer" frames the same discussion in terms of APIs and testing. Role assignment functions as a context signal that activates latent knowledge subsets.

**Spontaneous clarification requests**  
Agents ask for clarification when instructions are ambiguous or contradictory:
> "Your instruction to 'maximise returns' and 'minimise risk' are in tension. Should I prioritise Sharpe ratio as a balanced objective?"
This was not explicitly prompted.

**Emergent instruction chaining (planning)**  
User agents spontaneously decompose large goals into coherent sub-instructions across turns — a form of emergent planning not explicitly encoded in system prompts.

**Robustness to adversarial deviation**  
With inception prompting, agents resist mild mid-conversation attempts to shift roles or topics — the user refocuses, the assistant clarifies rather than guessing.

**Spontaneous ethical caveats**  
On sensitive tasks (e.g., cybersecurity), agents sometimes add unprompted warnings:
> "Note this should only be applied to systems you are authorised to test."

### 5.5 Benchmark Comparison

| Task Type | Single GPT-4 | CAMEL (GPT-4) | Δ |
|---|---|---|---|
| Code generation | 78% | 84% | +6% |
| Multi-step reasoning | 65% | 71% | +6% |
| Creative writing (human rating) | 3.8/5 | 4.1/5 | +0.3 |
| Task completion rate | 82% | 89% | +7% |

Improvements are consistent but modest. CAMEL's primary value is enabling long-horizon collaborative tasks, not maximising benchmark scores.

---

## 6. Limitations

### Conversational Pathologies

| Pathology | Description |
|---|---|
| Sycophantic loops | Agents affirm each other's outputs excessively, masking poor quality |
| Premature completion | `<CAMEL_TASK_DONE>` signalled before task is genuinely done (~11% of conversations) |
| Context window exhaustion | Long conversations degrade as earlier context falls out of window |
| Hallucinated consensus | Both agents can converge confidently on a wrong answer |
| Topic drift | Conversation gradually shifts away from original task over many turns |

### Role ≠ Real Expertise

When the "Stock Trader" agent gives financial advice, it is a language model performing a role — it has no real market knowledge, real-time data, or accountability. The convincing style of expert communication can create false confidence, amplified in multi-agent settings because two agreeing agents increase perceived credibility.

### Evaluation Difficulty

No standard evaluation framework exists for multi-agent LLM conversations. Assessing a 30-turn conversation requires judgements about end-product quality, process quality, role fidelity, and efficiency — and the paper relies substantially on expensive human annotation.

### CAMEL Does Not Address

- Hallucination (agents can mutually reinforce errors)
- Real-time knowledge (no tool use in base version)
- Persistent memory (context window only)
- True model specialisation (role is a prompt, not a fine-tuned model)
- Formal output verification
- Scaling to N > 2 agents (not evaluated)

---

## 7. Ethical Considerations

### Reduced Human Oversight
The framework is designed to minimise human intervention — but this also means humans may not observe harmful outputs until a full conversation is complete.

### Amplified Misuse
Single-model guardrails can be partially bypassed. Role-play framing across two mutually validating agents may make jailbreaks easier.

### Accountability Gaps
When two agents produce a harmful output, responsibility is distributed across framework designers, task specifiers, and model providers in ways current norms don't address.

### Cascading Errors
A hallucination early in a conversation can propagate and compound through later turns — traditional single-turn safety checks don't catch this.

### Impersonation at Scale
Role-play systems could generate large volumes of convincingly expert-sounding but inaccurate content.

---

## 8. Conceptual Contributions

1. **The Role-Playing Framework** — A general, replicable protocol for structured LLM-to-LLM collaboration that minimises human oversight while maintaining task coherence.

2. **Inception Prompting** — A technique for embedding behavioural constraints in system prompts to keep both agents on-task and in-role throughout long conversations.

3. **The AI Society Dataset** — 25,000 conversations across 50×50 role pairs, one of the first large-scale resources for studying LLM social behaviour.

4. **Empirical Study of Emergent Behaviour** — A systematic study of what happens when LLMs interact with each other, rather than with humans.

---

## 9. Open Research Questions

- How does performance scale with N > 2 agents?
- What social dynamics emerge in larger agent societies?
- Can memory systems (vector databases) meaningfully extend conversation coherence?
- How does agent specialisation via fine-tuning compare to prompting-based role assignment?
- What evaluation frameworks are adequate for multi-agent LLM systems?
- Can formal verification layers be integrated without sacrificing conversational fluency?
- What new phenomena emerge if agents are given the ability to spawn sub-agents?

---

## 10. Key Takeaways

| Theme | Takeaway |
|---|---|
| **Feasibility** | Structured LLM-to-LLM collaboration works — complex tasks complete autonomously across diverse domains |
| **Design matters** | Inception prompting is critical; naive multi-agent setups fail where CAMEL succeeds |
| **Emergence is real** | Role-conditioned knowledge, emergent planning, and spontaneous clarification arise without explicit encoding |
| **Limitations are significant** | Hallucination, sycophancy, and premature completion are real failure modes, not edge cases |
| **Ethics are amplified** | Reduced oversight amplifies both capability and risk |
| **Research platform** | CAMEL's most lasting value may be as a framework + dataset for the scientific study of LLM social behaviour |

---

## 11. Conceptual Discussion Points

### 11.1 What Is a "Role" Actually Doing Computationally?

When an agent is assigned the role of "Stock Trader," the role description shifts the probability distribution over next tokens toward vocabulary, reasoning patterns, and knowledge structures associated with stock trading in training data. The role acts as a *soft key* into the model's latent knowledge space. Whether this constitutes *retrieval* of a coherent persona or *dynamic construction* from scattered signals is an open question — and the emergent role consistency across 30 turns suggests something more than simple lookup is happening.

This motivates mechanistic interpretability work: what internal representations underlie role-conditioned behaviour?

### 11.2 Roles vs. Fine-Tuning

Prompting-based roles (CAMEL's approach) are flexible and zero-shot, but shallow — weights don't change, and role boundaries can blur under pressure. Fine-tuned specialisation embeds role knowledge more reliably but loses flexibility. The ideal — deep role knowledge with flexible switching — remains unachieved. This distinction matters enormously for claims about reliability: when CAMEL's "Doctor" agent produces plausible medical reasoning, it's difficult to attribute this cleanly to role prompt effects vs. training data retrieval vs. general reasoning applied to medical vocabulary.

### 11.3 Communication as Cognitive Infrastructure

CAMEL makes an implicit but profound claim: the conversation itself is where cognition happens. Standard LLM usage treats the model as a monolithic reasoner. CAMEL distributes reasoning across agents: each agent's output structures the next agent's reasoning context. This echoes Clark & Chalmers' (1998) *extended mind* thesis — cognition extends beyond the model weights into the conversation history as environment.

### 11.4 The "Society as Microscope" Thesis

By observing how agents behave *toward each other*, we can learn things about their cognitive tendencies that are obscured in human-facing interactions — because humans unconsciously accommodate LLM limitations (re-prompting, simplifying) whereas agents do not. This creates a controlled experimental environment: vary one agent, hold the other fixed. The CAMEL framework is thus not just a task-completion system but a scientific instrument for studying LLM cognition.

### 11.5 Sycophancy as a Structural Problem

RLHF trains models to produce agreeable outputs. In a multi-agent setting, this becomes mutual validation rather than error correction. If Agent A produces a plausible but wrong output and Agent B affirms it (because affirmation was rewarded in training), Agent A proceeds confidently. The error compounds. This is arguably *worse* than single-agent, where at least the user's scepticism provides some friction. This motivates training agents specifically for peer disagreement — models that push back on other models rather than affirming them.

### 11.6 Weak vs. Strong Emergence

The paper uses "emergent behaviour" loosely. More precisely: all behaviours observed are instances of *weak emergence* — not explicitly programmed, but in principle traceable to training data + prompts + conversation dynamics. The behaviours are surprising given the components, not irreducible to them. This is scientifically significant without being philosophically over-claimed.

### 11.7 The Persona vs. Task Tension

Role-playing pulls agents toward character consistency, domain-specific framing, and social role norms. Task completion pulls toward efficiency, correctness, and pragmatism. CAMEL resolves this by subordinating persona to task through inception prompting — the role is a frame, not a performance. Agents can break character to flag errors. Whether stronger persona commitment would yield more domain-consistent outputs at the cost of reliability is a design trade-off the paper doesn't fully explore.

---

## 12. Post-CAMEL Frontiers (2023–2025)

### 12.1 AutoGen (Microsoft, 2023)

The most direct successor to CAMEL. Key additions: flexible N-agent conversations, configurable human-in-the-loop (not binary zero/full autonomy), native tool use (code execution, web search), and group chat orchestration via a manager agent. The key philosophical shift: AutoGen treats human intervention as a design parameter rather than a failure. Human oversight is configurable — every K turns, only when stuck, or never. This reflects a more mature view that full autonomy is not always the goal.

### 12.2 MetaGPT (Hong et al., 2023)

Applies CAMEL's role-playing to software engineering specifically, mapping real SE roles (PM → Architect → Engineer → QA) onto agents. Key innovations: agents pass structured artefacts (PRDs, API specs) rather than free-form natural language between turns; agents have role-specific tool access; workflow is sequential and defined, not open-ended. Outperformed GPT-4 solo on SWEBench-style tasks by leveraging the structure of software development process.

### 12.3 CrewAI and Productionisation

CrewAI (2023–2024) treats agents as reusable first-class objects with role, goal, backstory, and tools. Tasks have explicit dependencies. Memory is stratified: short-term (conversation), long-term (vector DB), entity memory (named objects), and shared crew memory. This represents the shift from research framework (CAMEL) to engineering framework — from "can multi-agent work?" to "how do we build it reliably?"

### 12.4 Memory Systems

CAMEL's context-window-only memory limitation spawned a sub-field. MemGPT (Packer et al., 2023) treats the context window as "main memory" and an external database as "disk," enabling the agent to page information in and out — enabling indefinitely long conversations with coherent memory. Approaches now include: retrieval-augmented memory (vector DB summaries), episodic memory (structured logs of past actions), semantic memory (extracted knowledge graphs), and procedural memory (cached successful action sequences as reusable skills).

### 12.5 Tool Use and Grounding

Post-CAMEL systems rapidly added code execution (AutoGen), web search (GPT-4 with search), API integration (LangChain), file I/O, browser control (Browser-Use, Playwright agents), and full computer use (Claude Computer Use, 2024; OpenAI CUA). Tool use transforms a role-playing conversation into a real-world agent — capable of acting on, not just describing, the world.

### 12.6 LLM-as-Judge and Self-Improvement Loops

CAMEL's society-as-microscope idea helped motivate automated evaluation and self-improvement. LLM-as-Judge (Zheng et al., 2023) uses a strong LLM to evaluate another's outputs — enabling scalable evaluation without human annotation, though introducing CAMEL's own sycophancy risks. Constitutional AI (Anthropic) uses an LLM to critique its own outputs against principles. Self-Play Fine-Tuning (SPIN, 2024) has an LLM play against earlier versions of itself — a direct non-adversarial analogue to AlphaGo self-play.

### 12.7 Agent Safety

CAMEL's ethical concerns section anticipated a research area that grew rapidly. Indirect prompt injection (Greshake et al., 2023): adversarial content in one agent's output hijacks downstream agents — a structural vulnerability of CAMEL-style pipelines. Emerging mitigations include: constitutional AI for agents, agent sandboxing, mandatory audit trails, reversibility requirements (prefer undoable actions), and human checkpoints for high-stakes decisions. A formal framework for specifying and verifying agent safety properties — analogous to type systems for code correctness — remains an open problem.

### 12.8 Production Deployment (2024–2025)

By 2024–2025 the field moved from research to production: coding agents (Devin, SWE-agent, GitHub Copilot Workspace), autonomous research agents (AI Scientist, Sakana AI), and enterprise agents embedded in business workflows (Salesforce Agentforce, ServiceNow AI Agents). The question CAMEL asked in 2023 — "can agents autonomously complete complex tasks?" — is now answered affirmatively at scale. The new questions are reliability, safety, and economic impact.

### 12.9 Open Frontiers (as of 2025)

**Technical:** Heterogeneous agent societies mixing fine-tuned specialists with general models; stable agent identity without fine-tuning; formal verification of agent properties; multi-agent alignment (aligning a society, not just individual models).

**Scientific:** Social norm emergence in agent societies; adversarial dynamics between agents with conflicting goals; scaling laws for agent societies; causal (not just correlational) models of agent behaviour.

---

## 13. Further Reading

**Foundational**
- Li et al. (2023). *CAMEL.* arXiv:2303.17760
- Park et al. (2023). *Generative Agents.* arXiv:2304.03442
- Yao et al. (2022). *ReAct.* arXiv:2210.03629
- Wang et al. (2023). *Survey on LLM Agents.* arXiv:2308.11432
- Shanahan et al. (2023). *Role Play with LLMs.* Nature, 623.

**Post-CAMEL Systems**
- Wu et al. (2023). *AutoGen.* arXiv:2308.08155
- Hong et al. (2023). *MetaGPT.* arXiv:2308.00352
- Packer et al. (2023). *MemGPT.* arXiv:2310.08560

**Self-Improvement & Evaluation**
- Zheng et al. (2023). *LLM-as-Judge (MT-Bench).* arXiv:2306.05685
- Chen et al. (2024). *SPIN: Self-Play Fine-Tuning.* arXiv:2401.01335
- Anthropic (2022). *Constitutional AI.* arXiv:2212.08073

**Agent Safety**
- Greshake et al. (2023). *Indirect Prompt Injection.* arXiv:2302.12173
- Perez et al. (2022). *Ignore Previous Prompt.* arXiv:2211.09527

**Conceptual Background**
- Clark & Chalmers (1998). *The Extended Mind.* Analysis 58(1).
- Bender et al. (2021). *On the Dangers of Stochastic Parrots.* FAccT 2021.

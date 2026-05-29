# Reinforcement Learning — Temporal-Difference (TD) Learning

> **Context:** Personal learning notes on Chapter 6 of Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed., 2018), pp. 116–137. Section numbers (§6.1–6.8) refer to that textbook. Companion to the [Dynamic Programming notes](rl_dynamic_programming.md). These consolidate TD prediction, control (Sarsa, Q-learning, Expected Sarsa), and the bias/variance refinements.

---

## 0. The Big Picture

**Temporal-difference (TD) learning is a combination of Monte Carlo (MC) ideas and dynamic programming (DP) ideas.**

- Like **MC**: learns directly from raw experience, **no model** of the environment needed.
- Like **DP**: updates estimates based in part on other learned estimates — it **bootstraps**, without waiting for a final outcome.

TD sits at the intersection of two axes:

| Property | DP | Monte Carlo | TD |
|---|---|---|---|
| Needs a model? | **Yes** | No | No |
| Bootstraps? | **Yes** | No | **Yes** |
| Update type | expected (all successors) | sample (full return) | sample (one step) |
| When it learns | anytime (needs model) | end of episode | **every step** |
| Continuing tasks? | yes | no | yes |

The whole chapter is organized around two problems:

- **Prediction** (§6.1–6.3): evaluate a *fixed* policy $\pi$ → estimate $v_\pi$ or $q_\pi$.
- **Control** (§6.4–6.7): find an *optimal* policy → estimate $q_\*$ and $\pi_\*$, via **Generalized Policy Iteration (GPI)**.

---

## 1. TD Prediction (§6.1)

Goal: estimate $V \approx v_\pi$ for a fixed policy $\pi$.

**TD(0) / one-step TD update** (Eq. 6.2):

$$V(S_t) \leftarrow V(S_t) + \alpha\bigl[\underbrace{R_{t+1} + \gamma V(S_{t+1})}_{\text{TD target}} - V(S_t)\bigr]$$

Compare with **constant- $\alpha$ MC** (Eq. 6.1), which waits for the full return $G_t$:

$$V(S_t) \leftarrow V(S_t) + \alpha\bigl[G_t - V(S_t)\bigr]$$

- **MC target** $G_t$ — estimate of $\mathbb{E}[G_t \mid S_t]$ (Eq. 6.3); samples, no bootstrap.
- **DP target** $\mathbb{E}[R_{t+1} + \gamma v_\pi(S_{t+1})]$ (Eq. 6.4) — bootstraps, no sampling.
- **TD target** $R_{t+1} + \gamma V(S_{t+1})$ — does **both**: samples the expectation *and* bootstraps off the current estimate.

### TD error (Eq. 6.5)

$$\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$$

- It's the error in the estimate $V(S_t)$, but only available at time $t+1$.
- If $V$ is fixed during an episode, the **MC error equals the sum of discounted TD errors** (Eq. 6.6):

$$G_t - V(S_t) = \sum_{k=t}^{T-1} \gamma^{\,k-t}\,\delta_k$$

  (Only approximate when $V$ changes mid-episode, as in real TD(0).)

### Tabular TD(0) algorithm
```
Initialize V(s) arbitrarily, V(terminal)=0
Loop per episode:
  Initialize S
  Loop per step:
    A ← action from π for S
    Take A, observe R, S'
    V(S) ← V(S) + α[R + γV(S') − V(S)]
    S ← S'
  until S terminal
```

---

## 2. Advantages of TD (§6.2)

**Over DP:** model-free — no reward/transition distributions required.

**Over MC:** online, fully incremental — update every step, not just at episode end. Critical when:
- episodes are very long,
- tasks are *continuing* (no episodes at all),
- MC would have to discard/discount episodes containing exploratory actions.

**Soundness:** For a fixed $\pi$, TD(0) converges to $v_\pi$ — in the mean for small constant $\alpha$, and w.p. 1 under the usual stochastic-approximation step-size conditions (Eq. 2.7).

**Speed:** No proof that TD beats MC in general, but empirically TD usually converges faster than constant- $\alpha$ MC on stochastic tasks (Random Walk, Example 6.2).

---

## 3. Optimality of TD(0) (§6.3)

**Batch updating:** with finite data, repeatedly present all experience until convergence; apply the summed increments once per pass.

- **Batch MC** → estimates that **minimize MSE on the training set** (sample averages of observed returns).
- **Batch TD(0)** → the **certainty-equivalence estimate**: the values that are exactly correct for the **maximum-likelihood Markov model** of the data (empirical transition probabilities & rewards).

**"You are the Predictor" (Example 6.4):** 8 episodes; `A,0,B,0` once + mostly `B,1`.
- $V(B) = 3/4$ (everyone agrees).
- $V(A)$: MC says **0** (the one return seen from A → zero training error); TD says **3/4** (A always goes to B, and B's value is 3/4).
- If the process is Markov, TD's answer generalizes better to future data.

**Why TD is faster:** it implicitly exploits the Markov structure (converges toward certainty-equivalence), while MC ignores it. The full certainty-equivalence estimate is optimal but expensive ($\sim n^2$ memory, $\sim n^3$ compute); TD approximates it with $O(n)$ memory — often the only feasible route for large state spaces.

---

## 4. Prediction → Control: the bridge

Two shifts when moving from prediction to control:

1. **State values → action values.** Model-free, $V(s)$ alone can't tell you which action to take (you'd need to look ahead through the dynamics). So learn **$Q(s,a)$** directly.
2. **Exploration/exploitation tradeoff appears.** Must visit all $(s,a)$ pairs → can't act purely greedily. Splits into **on-policy** vs **off-policy**.

All control methods follow **GPI** — the only thing that changes between DP/MC/TD control is *how the evaluation step is done*:

```
   evaluate          improve
π ────────► q_π ────────► π' (greedier)  ──► ... ──► π*
```

TD control plugs the TD update into this loop and interleaves improvement at every step.

---

## The Bellman View — One Equation per Method

The three TD control methods are sample-based versions of three closely related Bellman equations. Knowing which equation each one samples explains the on-policy / off-policy split and ties everything back to DP (Ch. 4).

### Two Bellman equations for action values

**Bellman expectation equation** (for a fixed policy $\pi$):

$$q_\pi(s,a) = \mathbb{E}_\pi\bigl[R_{t+1} + \gamma\, q_\pi(S_{t+1}, A_{t+1}) \mid S_t=s,\, A_t=a\bigr]$$

where $A_{t+1} \sim \pi(\cdot \mid S_{t+1})$. This characterizes the value function *for whatever policy you are following*. Its DP solution method is **iterative policy evaluation** — repeatedly apply the RHS as an update.

**Bellman optimality equation** (for $q_\*$):

$$q_\*(s,a) = \mathbb{E}\bigl[R_{t+1} + \gamma\, \max_{a'} q_\*(S_{t+1}, a') \mid S_t=s,\, A_t=a\bigr]$$

The $\max$ replaces the policy — there is no $\pi$ here. Its DP solution method is **value iteration**.

### How TD methods sample these equations

Both Bellman equations have the form "$q(s,a)$ = expectation of one-step return + next-state value". TD methods drop the model-based expectation over $(S_{t+1}, R_{t+1})$ and replace it with a **single sampled transition** $(S_t, A_t, R_{t+1}, S_{t+1})$. What differs is how they handle the next-state value term:

| Method | Underlying Bellman equation | Next-state value term |
|---|---|---|
| **Sarsa** | Bellman expectation (for current $\pi$) | $Q(S_{t+1}, A_{t+1})$ — one *sampled* next action from $\pi$ |
| **Expected Sarsa** | Bellman expectation (for current $\pi$) | $\sum_a \pi(a\|S_{t+1}) Q(S_{t+1}, a)$ — *exact* over actions |
| **Q-learning** | Bellman optimality | $\max_a Q(S_{t+1}, a)$ — the optimal continuation |

### Why this matches the on-policy / off-policy split

- **Sarsa and Expected Sarsa sample the Bellman *expectation* equation**, so the value they learn is tied to whichever $\pi$ generates $A_{t+1}$. That is why they are **on-policy**: as $\pi$ changes ($\epsilon$-greedy on the latest $Q$), the equation being solved also changes. Each Sarsa update is one step of **sample-based policy evaluation** under the current policy; the surrounding GPI loop handles improvement.

- **Q-learning samples the Bellman *optimality* equation.** The $\max$ is independent of the behavior policy — whichever actions the agent actually takes ($\epsilon$-greedy, fully random, replay from a log), the target $R_{t+1} + \gamma \max_a Q(S_{t+1}, a)$ is always a sample of $q_\*$'s defining equation. That is why Q-learning is **off-policy** and converges to $q_\*$ directly — it is **sample-based value iteration**.

### Mapping back to DP (Ch. 4)

| DP method (model-based) | TD analogue (model-free, sampled) |
|---|---|
| Iterative policy evaluation | Sarsa / Expected Sarsa (under current $\pi$) |
| Value iteration | Q-learning |
| Policy iteration (eval + greedy improve) | Sarsa + $\epsilon$-greedy GPI |

The substantive change going from DP to TD is not the Bellman equation — it is how you take the expectation. DP averages over *all* successors using the model; TD averages over *sampled* successors over time. Bootstrapping is unchanged.

---

## 5. Sarsa — On-Policy TD Control (§6.4)

Apply TD(0) to **action values**. Uses the quintuple $(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$ → "Sarsa."

**Update (Eq. 6.7):**

$$Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha\bigl[R_{t+1} + \gamma Q(S_{t+1},A_{t+1}) - Q(S_t,A_t)\bigr]$$

- $A_{t+1}$ = the action **actually taken next** by the current (e.g. $\epsilon$-greedy) policy.
- **On-policy**: evaluates & improves the *same* policy it uses to act.
- Converges to optimal w.p. 1 if all $(s,a)$ visited infinitely often and the policy → greedy in the limit (e.g. $\epsilon = 1/t$).

### Algorithm
```
Initialize Q(s,a) arbitrarily, Q(terminal,·)=0
Loop per episode:
  Initialize S; choose A from S (ε-greedy on Q)
  Loop per step:
    Take A, observe R, S'
    Choose A' from S' (ε-greedy on Q)
    Q(S,A) ← Q(S,A) + α[R + γQ(S',A') − Q(S,A)]
    S ← S'; A ← A'
  until S terminal
```

### Windy Gridworld (Example 6.5)
- Crosswind pushes the agent upward; undiscounted, reward $-1$/step until goal.
- Params: $\epsilon=0.1$, $\alpha=0.5$, $Q_0=0$. Greedy policy becomes optimal well before 8000 steps.
- Residual $\epsilon$ keeps average episode length ~17 vs. the 15-step minimum.
- **MC can't easily be used here** — some policies never terminate; Sarsa fixes bad policies *mid-episode*.

#### $\alpha$ (step-size) vs. exploration — key clarification
- **$\epsilon$ controls exploration** (random vs. greedy actions) — *not* $\alpha$.
- **$\alpha$ controls learning rate** — how fast/stably $Q$ updates toward its target.
  - Higher $\alpha$ → faster learning; deterministic dynamics (like Windy GW) tolerate large $\alpha$.
  - In a *stochastic* environment (Exercise 6.10), large $\alpha$ makes $Q$ thrash; small $\alpha$ needed for good asymptotic performance.
  - $\alpha$ can *indirectly* affect the trajectory (which path looks best as $Q$ evolves), but it never changes the exploration *rule*.

---

## 6. Q-learning — Off-Policy TD Control (§6.5)

**Update (Eq. 6.8):**

$$Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha\bigl[R_{t+1} + \gamma \max_a Q(S_{t+1},a) - Q(S_t,A_t)\bigr]$$

- Target uses the **greedy** action ($\max$), regardless of what's actually taken.
- **Off-policy**: $Q$ directly approximates $q_\*$ independent of the behavior policy.
- Sample-based analogue of the DP value-iteration / Bellman-optimality backup.
- Converges w.p. 1 to $q_\*$ as long as all pairs keep being updated (+ usual step-size conditions).

### Sarsa vs. Q-learning — Cliff Walking (Example 6.6)
Reward $-1$/step, stepping into the cliff $= -100$ + reset. $\epsilon=0.1$.

| | Sarsa (on-policy) | Q-learning (off-policy) |
|---|---|---|
| Target action | $A'$ actually taken | greedy ($\max$) |
| Converges to | values of the followed policy (optimal as $\epsilon\to0$) | $q_\*$ directly |
| Accounts for exploration cost? | **Yes** | No |
| On this task | learns **safer** detour (higher online reward) | learns optimal cliff-edge path but **keeps falling off** |

**Insight:** Q-learning learns the better *policy*; Sarsa gets the better *results while learning*. If $\epsilon\to0$, both converge to the same optimum.
**If acting greedily ($\epsilon=0$), they're identical** (Exercise 6.12).

**When to use which:**
- **Sarsa** — online performance matters during learning; exploration is costly/dangerous (robots, safety).
- **Q-learning** — you just want the optimal policy, exploration mistakes are cheap (sim), or learning off-policy from logged/other-policy data.

---

## 7. Expected Sarsa (§6.6)

Like Q-learning, but use the **expectation** over next actions under $\pi$ instead of the max (Eq. 6.9):

$$Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha\bigl[R_{t+1} + \gamma \underbrace{\textstyle\sum_a \pi(a\mid S_{t+1})\, Q(S_{t+1},a)}_{\mathbb{E}_\pi[Q(S_{t+1},A_{t+1})]} - Q(S_t,A_t)\bigr]$$

Three targets side by side:

| Method | Next-action term |
|---|---|
| Sarsa | $Q(S',A')$ — one sampled action |
| Expected Sarsa | $\sum_a \pi(a\|S')Q(S',a)$ — policy-weighted average |
| Q-learning | $\max_a Q(S',a)$ — greedy action |

- **Benefit:** eliminates the **variance from randomly sampling $A'$** → moves deterministically in Sarsa's expected direction. Generally better given the same experience.
- **Cost:** more compute per step (sum over actions).
- **Cliff Walking (Fig. 6.3):** keeps Sarsa's edge over Q-learning AND beats Sarsa across a wide $\alpha$ range. With deterministic transitions it can set **$\alpha = 1$** with no degradation (Sarsa needs small $\alpha$).
- **Generalizes Q-learning:** with a greedy target policy, the expectation collapses to the max → Expected Sarsa *is* Q-learning. With behavior $\neq$ target it's off-policy. "May completely dominate both."

---

## 8. Maximization Bias & Double Learning (§6.7)

### The problem: maximization bias
All these methods build their target with a **$\max$** over estimates. Using $\max Q$ as an estimate of the *true* max is **positively biased**:

> If all true values $q(s,a)=0$ but estimates $Q(s,a)$ are noisy (some $+$, some $-$), the true max is 0 but $\max Q$ is **positive**.

Root cause: the **same samples** are used both to *pick* the maximizing action and to *evaluate* it.

**Example 6.7 (Fig. 6.5):** Start A → `left` (→B, then reward $\sim \mathcal{N}(-0.1,1)$) or `right` (→terminal, 0). `left` has expected return $-0.1$, always a mistake — yet Q-learning strongly favors `left` early and still picks it ~5% too often at asymptote.

### The fix: double learning
Keep **two independent estimates** $Q_1, Q_2$:
- Use $Q_1$ to *select*: $A^\* = \arg\max_a Q_1(a)$
- Use $Q_2$ to *evaluate*: $Q_2(A^\*)$ → unbiased, since $\mathbb{E}[Q_2(A^\*)] = q(A^\*)$
- Swap roles symmetrically.

**Double Q-learning (Eq. 6.10):** flip a coin each step. With probability 0.5:

$$Q_1(S,A) \leftarrow Q_1(S,A) + \alpha\bigl[R + \gamma\, Q_2\bigl(S', \arg\max_a Q_1(S',a)\bigr) - Q_1(S,A)\bigr]$$

otherwise the same update with $Q_1$ and $Q_2$ swapped. Behavior is $\epsilon$-greedy on $Q_1 + Q_2$.

- Only **one** estimate updated per step → **doubles memory, same compute/step.**
- Essentially eliminates maximization bias in Fig. 6.5.
- Double versions of Sarsa and Expected Sarsa exist too.

---

## 9. The TD Control Family — Map

```
              on-policy          off-policy
            ┌───────────────┬──────────────────┐
  sampled   │    Sarsa      │   Q-learning      │  ← max → maximization bias
  target    │  (uses A′)    │   (uses max)      │
            ├───────────────┴──────────────────┤
  expected  │       Expected Sarsa             │  ← lower variance; subsumes Q-learning
  target    │  (uses Σ π(a)Q)                  │
            ├──────────────────────────────────┤
  bias-fix  │  Double Q-learning / Double *    │  ← two estimates, unbiased max
            └──────────────────────────────────┘
```

- **Variance axis:** Expected Sarsa removes variance from sampling $A'$.
- **Bias axis:** Double learning removes overestimation from the max.
- All remain within the same **GPI + sample + bootstrap** TD framework.

---

## 10. Afterstates (§6.8)

A specialized case: value the position **after** the agent's move (e.g., the tic-tac-toe agent from Ch. 1) — an **afterstate value function**.
- Useful when you know the **immediate effect** of your action but not the full dynamics (e.g., your chess move's result, not the opponent's reply).
- **Efficiency:** different position–move pairs that lead to the **same resulting position** share one value — learning transfers automatically, unlike a conventional $Q(s,a)$.
- Still describable via GPI; still faces the on-policy vs off-policy choice.

---

## Quick Reference — All Update Rules

| Method | Target |
|---|---|
| Constant- $\alpha$ MC | $G_t$ |
| TD(0) prediction | $R_{t+1} + \gamma V(S_{t+1})$ |
| Sarsa | $R_{t+1} + \gamma Q(S_{t+1}, A_{t+1})$ |
| Q-learning | $R_{t+1} + \gamma \max_a Q(S_{t+1}, a)$ |
| Expected Sarsa | $R_{t+1} + \gamma \sum_a \pi(a\|S_{t+1}) Q(S_{t+1}, a)$ |
| Double Q-learning | $R_{t+1} + \gamma Q_2(S_{t+1}, \arg\max_a Q_1(S_{t+1},a))$ |

All updates have the form $Q(S,A) \leftarrow Q(S,A) + \alpha\bigl[\text{target} - Q(S,A)\bigr]$.

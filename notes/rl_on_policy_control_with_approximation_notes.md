# Reinforcement Learning — On-Policy Control with Approximation

> **Context:** Personal learning notes on Chapter 10 of Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed., 2018), pp. 243–256. Section numbers (§10.1–10.6) refer to that textbook. Companion to the [Dynamic Programming](rl_dynamic_programming.md), [TD learning](rl_td_learning.md), and [On-policy prediction with approximation](rl_on_policy_prediction_with_approximation.md) notes.
>
> **Theme:** the control counterpart of Ch. 9. Learn $\hat{q}(s, a, w) \approx q_\*(s, a)$ with a weight vector, still on-policy. Off-policy is deferred to Ch. 11.

---

## 0. The Big Picture — Two Halves

1. **Episodic case (§10.1–10.2):** extending Ch. 9 prediction to **action values + control** is *immediate* — swap $\hat{v}(s, w)$ for $\hat{q}(s, a, w)$ and wrap in $\epsilon$-greedy GPI → **semi-gradient Sarsa**.
2. **Continuing case (§10.3–10.5):** something breaks. Under FA, **discounting stops being meaningful** for control, and we switch to a new **average-reward** formulation with **differential** value functions.

Carry-over from Ch. 9: semi-gradient (biased) updates, the n-step spectrum, "intermediate $n$ best," on-policy stability.

---

## 1. Episodic Semi-Gradient Control (§10.1)

Extend the prediction update from state values to **state–action** examples $(S_t, A_t) \mapsto U_t$:

$$w_{t+1} = w_t + \alpha\bigl[U_t - \hat{q}(S_t, A_t, w_t)\bigr]\, \nabla \hat{q}(S_t, A_t, w_t) \qquad \text{(10.1)}$$

**One-step Sarsa** target $U_t = R_{t+1} + \gamma\, \hat{q}(S_{t+1}, A_{t+1}, w_t)$:

$$w_{t+1} = w_t + \alpha\bigl[R_{t+1} + \gamma\, \hat{q}(S_{t+1}, A_{t+1}, w_t) - \hat{q}(S_t, A_t, w_t)\bigr]\, \nabla \hat{q}(S_t, A_t, w_t) \qquad \text{(10.2)}$$

- For a **fixed policy**, converges like TD(0) with the same $\tfrac{1}{1-\gamma}$-type bound (9.14).
- **Control via GPI:** for discrete (not-too-large) action sets, compute $\hat{q}(S_t, a, w)$ for all $a$, take greedy $A^\* = \arg\max_a \hat{q}$, act $\epsilon$-greedy. (Continuous / huge action sets remain open research.)

### Episodic Semi-Gradient Sarsa (pseudocode)
```
Loop per episode:
  S, A ← initial state, action (ε-greedy)
  Loop per step:
    take A, observe R, S′
    if S′ terminal: w ← w + α[R − q̂(S,A,w)] ∇q̂(S,A,w); next episode
    A′ ← ε-greedy(q̂(S′,·,w))
    w ← w + α[R + γ q̂(S′,A′,w) − q̂(S,A,w)] ∇q̂(S,A,w)
    S, A ← S′, A′
```

### Example 10.1 — Mountain Car
Underpowered car must drive *away* from the goal first (things get worse before better). 2 continuous dims (position, velocity) + 3 actions → **tile coding** (8 tilings, asymmetric offsets) + linear $\hat{q} = w^\top x(s, a)$. **Optimistic init** ($w = 0$, all true values negative) drives systematic exploration even with $\epsilon = 0$ — the agent is repelled from visited states because reality is worse than the optimistic estimate.

---

## 2. Semi-Gradient n-step Sarsa (§10.2)

Plug the n-step return into the Sarsa update:

$$G_{t:t+n} = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n\, \hat{q}(S_{t+n}, A_{t+n}, w_{t+n-1}) \qquad \text{(10.4)}$$

$$w_{t+n} = w_{t+n-1} + \alpha\bigl[G_{t:t+n} - \hat{q}(S_t, A_t, w_{t+n-1})\bigr]\, \nabla \hat{q}(S_t, A_t, w_{t+n-1}) \qquad \text{(10.5)}$$

**Intermediate $n$ is best** (Fig. 10.3–10.4): on Mountain Car, $n = 8$ learns faster + better asymptote than $n = 1$; the sweep finds $n = 4$ optimal. Same recurring Ch. 6/7/9 lesson — some bootstrapping, but not one-step.

---

## 3. Average Reward: A New Problem Setting (§10.3)

A **third** formulation (alongside episodic and discounted), for **continuing** tasks with **no discounting** — the agent values delayed reward as much as immediate.

**Quality of a policy** = average rate of reward:

$$r(\pi) = \lim_{h\to\infty} \frac{1}{h} \sum_{t=1}^{h} \mathbb{E}[R_t \mid S_0, A_{0:t-1} \sim \pi] = \sum_s \mu_\pi(s) \sum_a \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a)\, r \qquad \text{(10.7)}$$

where $\mu_\pi$ is the steady-state distribution under $\pi$ (Eq. 10.8). Requires **ergodicity** (start state / early actions only matter transiently). Optimal = any policy maximizing $r(\pi)$.

### Differential returns & value functions
Returns are defined relative to the average reward — the **differential return**:

$$G_t = (R_{t+1} - r(\pi)) + (R_{t+2} - r(\pi)) + (R_{t+3} - r(\pi)) + \cdots \qquad \text{(10.9)}$$

Differential value functions $v_\pi, q_\pi, v_\*, q_\*$ defined on this $G_t$. Their Bellman equations are the old ones with **$\gamma$ removed** and **every reward replaced by $r - r(\pi)$**, e.g.:

$$v_\pi(s) = \sum_a \pi(a \mid s) \sum_{r, s'} p(s', r \mid s, a)\bigl[r - r(\pi) + v_\pi(s')\bigr]$$

### Differential TD errors

$$\delta_t = R_{t+1} - \bar{R}_t + \hat{v}(S_{t+1}, w_t) - \hat{v}(S_t, w_t) \qquad \text{(10.10)}$$

$$\delta_t = R_{t+1} - \bar{R}_t + \hat{q}(S_{t+1}, A_{t+1}, w_t) - \hat{q}(S_t, A_t, w_t) \qquad \text{(10.11)}$$

where $\bar{R}_t$ = running estimate of $r(\pi)$. With these, most algorithms carry over unchanged.

### Differential Semi-Gradient Sarsa

$$w_{t+1} = w_t + \alpha\, \delta_t\, \nabla \hat{q}(S_t, A_t, w_t) \qquad \text{(10.12)}$$

Two step sizes ($\alpha$ for $w$, $\beta$ for $\bar{R}$):
```
Init w, R̄, S, A
Loop per step:
  take A, observe R, S′;  A′ ← ε-greedy(q̂(S′,·,w))
  δ ← R − R̄ + q̂(S′,A′,w) − q̂(S,A,w)
  R̄ ← R̄ + β δ          # update via full TD error δ (not R − R̄), Ex 10.8
  w ← w + α δ ∇q̂(S,A,w)
  S, A ← S′, A′
```

- On-policy analog of **R-learning** (Schwartz 1993); "R" ≈ **relative/differential** values.
- **Example 10.2 — Access-Control Queuing:** 10 servers, 4 priority levels; accept/reject for maximum long-term undiscounted reward. Differential Sarsa learns policy + value; learned $\bar{R} \approx 2.31$.

---

## 4. Deprecating the Discounted Setting (§10.4)

**The surprising core result:** under FA, **discounting is futile for continuing control**.

**Why:** with no episode starts/ends and states represented only by features, performance can only be assessed by averaging over time. The average of discounted returns over the on-policy distribution is:

$$\text{average discounted return} = \frac{r(\pi)}{1 - \gamma}$$

i.e. average reward scaled by a constant — so **$\gamma$ does not change the policy ordering at all** (it could even be 0).

- **Intuition:** every reward appears once in each return position; its total weight is $1 + \gamma + \gamma^2 + \cdots = \tfrac{1}{1-\gamma}$. Since all states are weighted equally, average of returns $= \tfrac{r(\pi)}{1-\gamma}$.
- Consequence: $\gamma$ demotes from a **problem parameter** to a **solution-method parameter**.

### The deeper reason — we lost the policy improvement theorem

Under FA, the **policy improvement theorem (§4.2) no longer holds**. Improving one state's value doesn't guarantee a better overall policy. Affects **all** settings (episodic, average-reward, discounted) — no local improvement guarantee for action-value methods. $\epsilon$-greedification can **chatter** among good policies instead of converging (Gordon 1996). **Ch. 13 (policy-gradient methods) restores a guarantee** via the policy-gradient theorem.

---

## 5. Differential Semi-Gradient n-step Sarsa (§10.5)

Combine average-reward with n-step bootstrapping. Differential n-step return:

$$G_{t:t+n} = \sum_{i=1}^{n} (R_{t+i} - \bar{R}_{t+n-1}) + \hat{q}(S_{t+n}, A_{t+n}, w_{t+n-1}) \qquad \text{(10.14)}$$

$$\delta_t = G_{t:t+n} - \hat{q}(S_t, A_t, w) \qquad \text{(10.15)}$$

then the usual update (10.12). Pseudocode mirrors §10.2 with $(R_i - \bar{R})$ in the return and $\bar{R} \leftarrow \bar{R} + \beta \delta$.

> **Ex 10.9:** $\beta$ must be small for $\bar{R}$ to be a good long-term estimate, but small $\beta$ leaves $\bar{R}$ init-biased for many steps. Use the **unbiased constant-step-size trick** (Ex 2.7) for $\bar{R}$.

---

## 6. Summary (§10.6)

- **Episodic control:** immediate extension of Ch. 9 → **semi-gradient (n-step) Sarsa** on action values with $\epsilon$-greedy GPI.
- **Continuing control:** adopt the **average-reward** setting maximizing $r(\pi)$. Discounting cannot be carried over to control under FA — most policies can't be exactly represented, and the rest are ranked by the scalar $r(\pi)$.
- The average-reward machinery (**differential** returns, value functions, Bellman eqns, TD errors) **parallels the discounted machinery** with small changes: $\gamma$ removed, rewards offset by $r(\pi)$.
- **Theoretical gap:** no policy improvement theorem under FA (addressed by policy gradients, Ch. 13).

---

## Quick Reference

| Concept | Key formula |
|---|---|
| Action-value SGD update | $w \leftarrow w + \alpha[U_t - \hat{q}(S_t, A_t, w)]\, \nabla \hat{q}(S_t, A_t, w)$ |
| One-step Sarsa target | $U_t = R_{t+1} + \gamma\, \hat{q}(S_{t+1}, A_{t+1}, w)$ |
| n-step Sarsa return | $G_{t:t+n} = \sum_i \gamma^{i-1} R_{t+i} + \gamma^n\, \hat{q}(S_{t+n}, A_{t+n}, w)$ |
| Average reward | $`r(\pi) = \sum_s \mu_\pi(s) \sum_a \pi(a \mid s) \sum_{s',r} p(s', r \mid s, a)\, r`$ |
| Differential return | $G_t = \sum_k (R_{t+k} - r(\pi))$ |
| Differential TD error | $\delta_t = R_{t+1} - \bar{R}_t + \hat{q}(S_{t+1}, A_{t+1}, w) - \hat{q}(S_t, A_t, w)$ |
| Differential Sarsa | $\bar{R} \leftarrow \bar{R} + \beta \delta$, $w \leftarrow w + \alpha \delta\, \nabla \hat{q}$ |
| Discounting futility | average discounted return $= \tfrac{r(\pi)}{1-\gamma}$ → $\gamma$ doesn't change policy ordering |

---

## Connections to Earlier Chapters

- **Direct lift from Ch. 9:** semi-gradient updates, n-step bootstrapping, intermediate-$n$ optimum, on-policy stability. See [On-policy prediction notes](rl_on_policy_prediction_with_approximation.md).
- **Sarsa = on-policy control** (uses $A'$ actually taken); FA version of the Ch. 6 algorithm. Off-policy (Q-learning-style) deferred to Ch. 11 (deadly triad). See [TD learning notes](rl_td_learning.md).
- **Prediction → control via GPI:** $\epsilon$-greedy improvement wrapped around semi-gradient evaluation — same skeleton as tabular control.
- **Lost guarantees:** the policy improvement theorem (§4.2, [Dynamic programming notes](rl_dynamic_programming.md)) fails under FA → motivates policy-gradient methods (Ch. 13).

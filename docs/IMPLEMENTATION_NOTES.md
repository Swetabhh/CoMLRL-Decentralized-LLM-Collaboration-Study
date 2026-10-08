# CoMLRL: Implementation Notes

This is my implementation study of **Learning Decentralized LLM Collaboration with Multi-Agent Actor Critic** by Shuo Liu, Tianle Chen, Ryan Amiri, and Christopher Amato (ICML 2026). It's based on the authors' official CoMLRL code, specifically the paper-reference release, `v1.3.2`.

The research, the code, the datasets, the experimental setup, and the reported results all belong to the authors. I'm not claiming the algorithms or the codebase. What I set out to do was understand the implementation properly, trace the training pipeline, check the assumptions the code makes, and work out what's realistic on a consumer GPU.

**Paper:** https://proceedings.mlr.press/v306/liu26k.html (PMLR volume 306, pp. 75603 to 75623)
**Official repo:** https://github.com/OpenMLRL/CoMLRL

---

## The problem

The paper asks how several LLM agents can learn to collaborate through multi-agent reinforcement learning instead of following a fixed protocol. The motivation is that standard Monte Carlo policy-gradient methods tend to have high variance and need a lot of samples, especially over long horizons. The authors test whether actor-critic learning gives better value estimates and a lower-variance learning signal.

They study two versions, and the difference is what the critic gets to see during training:

- **CoLLM-CC** uses a centralized critic.
- **CoLLM-DC** uses decentralized critics, one per agent.

## The pipeline at a glance

A dataset item gets formatted into an agent-specific prompt, tokenized, and fed to the LLM to generate a rollout. Rewards are computed, a value estimate is made, and the return or TD target is built. From that comes the advantage, then the policy and critic are optimized, and the run is evaluated. In multi-turn settings an external transition mechanism updates the interaction state between turns.

## The three files that matter

**`comlrl/models/actor_critic.py`** holds `ActorCriticOutput`, `ValueHead`, and `CausalLMWithValueHead`. The value head turns transformer hidden states into a scalar value estimate. The wrapper puts a value head on a causal LM's backbone, so you can skip loading a separate critic model.

**`comlrl/trainers/iac.py`** implements Independent Actor-Critic (IAC). It covers tokenizer and model setup, prompt formatting, rollout generation, reward handling, policy and critic evaluation, sequence statistics, advantage preparation, the optimization step, and evaluation.

**`comlrl/trainers/maac.py`** implements Multi-Agent Actor-Critic (MAAC) with a shared critic. Actors generate independently, while the critic is conditioned on joint agent information during training. This is the code that corresponds to the centralized-critic approach.

## IAC

IAC keeps an actor and a critic for every agent, and the code supports two ways of setting up the critic.

With a **separate critic**, a second causal LM is loaded just for value estimation. That cleanly separates actor from critic, but the extra models have to stay in memory during training. With a **shared model**, a value head sits directly on the actor's backbone, so the two share transformer representations. This adds much less memory than a full separate critic.

## MAAC

MAAC runs multiple actors with a single shared critic. Actor inputs stay agent-specific, but the critic gets joint information, built from the participating agents' prompts. The critic can estimate either `V(s)` from joint state and history, or `Q(s,a)` from joint state and action. What I like about this is that the centralized-critic assumption is explicit in the code, not just an idea in the paper.

## Rollout generation

This is one of the heavier steps. The code tokenizes the prompt, calls the actor's generate method, separates prompt tokens from response tokens, measures the response length, decodes the completion, and computes the matching value estimate. Sampling is controlled through `max_new_tokens`, `temperature`, `top_p`, `top_k`, and `num_return_sequences`. If a generation comes back empty, the code raises an error.

## Rewards

The reward function is configurable. The code accepts several reward-output shapes and maps them into the agent and return structure the training loop expects. In cooperative settings that flexibility matters, because the trainer may get a shared reward, one reward per agent, or rewards lined up with multiple generated returns. The reward processor can also transform values before they reach optimization.

## Advantages and values

Each rollout sample stores the old policy information and the old value estimate. The advantage comes from the value target and the stored old value:

```text
advantage = value_target - old_value
```

Advantages can be normalized across the rollout batch. The policy loss is proportional to `-log π(a|s) × advantage`, and the value function is trained toward its return or value target. Before continuing, the code checks for non-finite log probabilities, advantages, returns, actor loss, and value loss.

## Multi-turn training

When `num_turns > 1`, an `external_transition` function is required. The trainer uses it to build later interaction states and prompts. Multi-turn configs are also limited to `num_return_sequences == 1`, which is enforced when the configuration is validated.

---

## What I checked

I'm not trying to reproduce the ICML experiments here. The question was whether the main assumptions and control paths in the code hang together and are explicitly guarded. By reading the released code and its checks, here is what I found:

| Area | Finding | Status |
|---|---|---|
| IAC configuration | Bad batch size, agent count, turn count, and multi-turn sampling settings are rejected | Pass |
| MAAC configuration | A missing critic model and an invalid critic type are rejected | Pass |
| Rollout generation | An empty completion raises a runtime error | Pass |
| Reward handling | Reward shape is validated before mapping to agents and returns | Pass |
| Advantages | Value targets and old values are used explicitly to form them | Pass |
| Numerical stability | Non-finite log probs, advantages, returns, and losses are checked | Pass |
| Multi-turn setup | A missing `external_transition` is rejected | Pass |
| Centralized critic | MAAC builds joint critic inputs | Pass |
| Source integrity | I left the original source unchanged | Pass |

These are findings from tracing the code. They aren't benchmark results.

## Failure guards in detail

- **Batch size.** IAC and MAAC both require `per_device_train_batch_size == 1`, and anything else fails configuration.
- **Multi-turn sampling.** Multi-turn training needs `num_return_sequences == 1`, otherwise initialization fails.
- **Transition function.** Multi-turn IAC and MAAC need an `external_transition`. Without one, training is rejected before any rollouts are collected.
- **MAAC critic.** MAAC can't be initialized without a critic model identifier. That reflects the architecture, not an optional optimization.
- **Critic type.** MAAC accepts `v` or `q` and rejects everything else.
- **Empty completions.** If generation returns nothing beyond the prompt, rollout generation errors out instead of quietly continuing with bad training data.
- **Non-finite values.** The update path checks policy statistics, advantages, returns, and losses. That matters for actor-critic training in particular, since an unstable value estimate can otherwise corrupt every policy update after it.

---

## Hardware

```text
OS    Windows 11
CPU   Intel Core i7
GPU   NVIDIA GeForce RTX 3050 6GB
RAM   16GB
```

My machine is far smaller than what the paper used. The paper reports much larger GPU memory needs for its actor-critic runs. For CoopHE, training overhead is roughly 126.3 GB of VRAM for CoLLM-DC and 107.4 GB for CoLLM-CC. Those figures come from the authors' infrastructure, not from my machine.

Multi-agent actor-critic training needs several LLMs, so trying to run the paper's configuration on a 6 GB card wouldn't be a fair reproduction. The most useful practical result of this study is therefore a feasibility boundary, not a made-up benchmark score:

- **Full paper-scale actor-critic training: not feasible on 6 GB.** It can need multiple actor models, MAAC adds a critic on top, the paper's runs use far more memory, and training would blow past my VRAM by a wide margin.
- **A lighter implementation study: feasible.** That means source tracing, config validation, inspecting the model wrapper and the reward and rollout paths, analyzing implementation constraints, running unit tests where dependencies allow, and small synthetic or tiny-model sanity checks.

Understanding the implementation is possible even when paper-scale training isn't.

---

## Results reported in the paper

The paper evaluates on TLDR summarization, ArXiv introduction expansion, CoopHumanEval, StrBuild, and HouseBuild. These numbers are **author-reported**. I haven't reproduced them.

| Method | TLDR | ArXiv | CoopHE Pass@1 | StrBuild IoU | HouseBuild IoU |
|---|---:|---:|---:|---:|---:|
| MAGRPO | 93.5 | 93.1 | 74.3 | 50.6 | 50.9 |
| CoLLM-DC | 95.4 | 94.1 | 59.1 | 44.6 | 46.8 |
| CoLLM-CC | 95.2 | 95.0 | 75.2 | 68.5 | 52.7 |

The main takeaway is that CoLLM-DC can keep up with CoLLM-CC on short-horizon, dense-reward tasks, but falls behind on longer-horizon or sparse-reward ones. The paper points to convergence trouble for CoLLM-DC in those harder settings. I'm including the numbers only for context on the code I studied.

## The cost side

The paper also reports training overhead on CoopHE:

| Method | Training time | VRAM |
|---|---:|---:|
| MAGRPO | 4.5 H200 hours | 93.8 GB |
| CoLLM-DC | 13.4 H200 hours | 126.3 GB |
| CoLLM-CC | 11.1 H200 hours | 107.4 GB |

So the trade-off is a better value-based learning signal in exchange for extra model and critic computation, which means more training memory and compute. I treated cost as part of understanding the method, not something separate from the benchmark scores.

---

## What I did and didn't do

What I did:

- worked through the original implementation by hand,
- traced the IAC and MAAC execution paths,
- studied how the value head integrates with the model,
- followed rollout generation,
- looked at reward and return handling,
- traced how advantages are built,
- compared centralized and decentralized critics,
- catalogued the explicit failure guards,
- assessed feasibility under a 6 GB GPU.

I didn't change the original actor-critic source files to alter the algorithm.

What I didn't do, and am not claiming:

- a full reproduction of the ICML 2026 experiments,
- the five-run benchmark reproduction,
- the paper-scale training curves,
- the reported training times or VRAM usage,
- the full set of benchmark scores,
- a hardware-matched replication of the authors' setup.

## Questions this raised

These came from the code and the paper's findings. They aren't validated results.

1. How much of the centralized critic's advantage comes from seeing joint history versus simply estimating values better?
2. How does critic instability build up as the number of turns grows?
3. How sensitive is MAAC to the size and quality of the critic model?
4. Could decentralized critics get more stable with stronger value regularization or shared representations?
5. What's the smallest critic that keeps most of the benefit of a full LLM critic?
6. Can actor-critic training become practical on much smaller hardware with parameter-efficient critics or quantized models?

---

## Attribution

This builds on the official CoMLRL implementation of **Learning Decentralized LLM Collaboration with Multi-Agent Actor Critic** by Shuo Liu, Tianle Chen, Ryan Amiri, and Christopher Amato (ICML 2026).

- Official repository: https://github.com/OpenMLRL/CoMLRL
- Paper: https://proceedings.mlr.press/v306/liu26k.html

The implementation, the research, the datasets, and the reported results belong to the original authors.

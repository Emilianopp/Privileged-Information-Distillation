# Twitter Thread Draft

Paper: [Privileged Information Distillation for Language Models](https://arxiv.org/abs/2602.04942)

## Tweet 1 - Hook

We wrote a blog post on self-distillation and privileged information distillation for language models with interactive visualizations.

How do you train a model when it can't sample successful outcomes and you don't have access to a larger model?

Blog: [LINK]

[attach: start.png]

---

## Tweet 2 - Failures of RL

RL relies on the policy being able to sample high-reward outputs. When the policy cannot produce any successful trajectories, there is no positive signal to reinforce.

The model only ever learns what *not* to do, never what *to* do.

[attach: rl-zero.png]

---

## Tweet 3 - Shaping a LM by Conditioning

Unlike other ML systems, LMs can be freely conditioned on additional information. Even a small amount of *privileged information* can enable models to succeed at tasks they previously could not.

We call this the "teacher." The catch? The privileged information won't be available at test time.

[attach: start-w-teacher.png]

---

## Tweet 4 - SFT (Forward-KL)

By far the simplest way to transfer this knowledge is through SFT. Generate samples with the teacher, train the student on them.

But forward-KL is *mode covering*. The student expands its support to cover all of the teacher rather than accurately approximating it.

[attach: sft.mov]

---

## Tweet 5 - Self-Distillation (Reverse-KL)

A natural fix is to optimize reverse-KL, which is *mode seeking*. The student seeks easy-to-fit modes of the teacher.

This is the most widely used objective in recent self-distillation works. But the student can fit a lower-reward mode of the teacher.

[attach: rKL.mov]

---

## Tweet 6 - Reward-Tilted Self-Distillation

The main problem with pure self-distillation is its bias towards easier-to-fit modes. A simple fix is to define the target as a reward-tilted variant of the teacher.

This explicitly rewards high-reward outputs while still matching the teacher's structure. Same form as RL-as-inference, just with the teacher as the prior.

[attach: rklr.mov]

---

## Tweet 7 - What if the Teacher is Bad?

Self-distillation assumes the teacher has support over high-reward regions. But what if even conditioning on privileged information doesn't give sufficient coverage?

Most of the teacher's mass can still lie in the low-reward region. We are limited by the abilities of the teacher.

[attach: bad-teacher.png]

---

## Tweet 8 - Variational EM

To fix this, we can first train the teacher to approximate the target via RL, then distill to the student via SFT.

This is variational EM. It can improve over the base policy, but maintaining two separate training phases with dual parameter sets is inefficient.

[attach: em3.mov]

---

## Tweet 9 - pi-Distill

In our work, we show variational EM is inefficient and less effective. A simple solution is to allow the teacher and student to share parameters and jointly train them, greatly simplifying the training process.

Paper: https://arxiv.org/abs/2602.04942

---

## Tweet 10 - Visualizing pi-Distill

A key component of pi-Distill is modulating between teacher and student objectives using alpha:

- alpha=1: only teacher trains, student improves through shared params
- alpha=0: only student trains from teacher traces
- alpha=0.5: both optimized jointly, most stable across settings

[attach: joint.mov]

---

## Tweet 11 - Closing

Full blog post with derivations, interactive visualizations, supplemental reading, as well as further discussion on future opportunities are discussed in the full post: [[LINK](https://emilianopp.github.io/Privileged-Information-Distillation-and-Self-Distillation/)]


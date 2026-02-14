---
title: "Privileged Information Distillation"
date: 2026-02-04
author: "Your Name"
---

The purpose of this post is to be a quick introduction to some recent works on self-distillation and privileged information distillation, including [Shenfeld et al. (2026)](#ref-shenfeld2026), [Zhao et al. (2026)](#ref-zhao2026), [Hübotter et al. (2026)](#ref-hubotter2026), and [Penaloza et al. (2026)](#ref-penaloza2026). We hope to give some intuition on how these algorithms work, where they come from, as well as comment on when their assumptions are valid or invalid.

<details markdown="1">
<summary><strong>Background Reading</strong> (click to expand)</summary>

The intended audience of this blog is for people that have a decent understanding of variational inference and reinforcement learning, as well as their relationships. For those unfamiliar, we recommend the following:
- [Control as Inference](https://jasonppy.github.io/deeprl/deeprl-12-control-as-inference/) for an introduction to the RL-as-inference framework (or see [Levine's tutorial paper](https://arxiv.org/abs/1805.00909))
- [Variational Inference: A Review for Statisticians](https://arxiv.org/abs/1601.00670) for a comprehensive overview of variational inference
- [The RLHF Book](https://rlhfbook.com/) for RL + LLMs

</details>

<details markdown="1">
<summary><strong>Notation</strong> (click to expand)</summary>

We work in a single-turn setting where $x$ denotes the prompt, $z$ the chain-of-thought, and $y$ the final output response. We write policies as $\pi(y \mid x)$ or $\pi(y \mid x, z)$ when conditioning on chain-of-thought.

</details>

The goal of RL is to maximize reward

$$
\max_\pi \mathbb{E}_{y \sim \pi(\cdot \mid x)}\left[R(y, x)\right]
$$

with the goal of obtaining a target policy $\pi^*$. A slightly different perspective treats RL as an inference problem, wherein we define the target policy as a distribution over outputs:

$$
\pi^*(y \mid x) = \frac{1}{Z(x)}\,\pi_{\mathrm{ref}}(y \mid x)\exp\left(\frac{R(y, x)}{\tau}\right)
$$

Here, the partition function $Z$ makes direct inference intractable, making us rely on approximations. To do this, we can parameterize a policy $\piSphi$ and approximate the target policy using the reverse-KL:

$$
D_{\text{kl}}(\piSphi \;\|\; \pi^*) = \mathbb{E}_{y\sim\piSphi}\left[\log\frac{\piSphi(y \mid x)}{\pi^*(y \mid x)}\right]
$$

If we write the target policy in the usual RL-as-inference form

$$
\pi^*(y \mid x) = \frac{1}{Z(x)}\,\pi_{\mathrm{ref}}(y \mid x)\exp\left(\frac{R(y, x)}{\tau}\right),
$$

then the reverse-KL decomposes as

$$
\begin{aligned}
D_{\text{kl}}(\piSphi \;\|\; \pi^*) &= \mathbb{E}_{y\sim\piSphi}\big[\log\piSphi(y \mid x)-\log\pi^*(y \mid x)\big] \\
&= \mathrm{KL}(\piSphi\;\|\;\pi_{\mathrm{ref}}) - \frac{1}{\tau}\,\mathbb{E}_{y\sim\piSphi}[R(y, x)] + \log Z(x).
\end{aligned}
$$

Thus minimizing the reverse-KL is equivalent (up to constants) to maximizing expected reward while penalizing deviation from a reference policy.

This is equivalent to the reward-maximization term with an added KL penalty. But this objective also has a failure mode: it requires us to sample successful trajectories in order to reinforce them. For instance, consider a hard task where the model always yields zero success, as shown in the figure below:

![Student-only policy failing example](/figures/start.png)
*Figure 1: Student-only agent fails to sample correct trajectories (illustrative).*

In this case the model cannot sample any correct trajectories and cannot bootstrap itself onto the optimal policy $\pi^*$. While theoretically an LM could eventually sample a correct trajectory, since it has full support over the sampling distribution (i.e., non-zero probability over every token at each step), this is infeasible in practice.


One exciting property of LMs, unlike most other ML systems, is that we can freely contextualize them on additional information. This property has many benefits — notably, even a small amount of *Privileged Information* (PI) can enable models to sample tasks they could not previously. See the figure below, where $\piTthetafull$ denotes the teacher contextualized on privileged information $\mathbf{I}$.

![With teacher example](/figures/start-w-teacher.png)
*Figure 2: Teacher policy contextualized on privileged information can now sample successful trajectories.*

After contextualizing on $\mathbf{I}$, we see that the model can now sample successful traces. The only problem now is that we won't have access to $\mathbf{I}$ at test time, since it is typically *task-specific*. So we need to find a way to transfer the information embedded within it to $\piS$, as this is the only policy we have access to at test time.

The remainder of this blog post focuses on different ways of doing this.

1. SFT
2. Self-Distillation
3. Variational EM
4. Privileged Information Distillation
5. Cool opportunities


### SFT

By far the simplest and most popular way of transferring this knowledge is through Supervised Fine-Tuning (SFT). This is equivalent to fitting forward-KL between the teacher policy $\piT$ and the student $\piS$.



*Figure: Supervised Fine-Tuning (SFT) demonstration.*

$$D_{\text{kl}}(\piT \;\|\; \piS) = \mathbb{E}_{y\sim\piT}\left[\log\frac{\piT(y \mid x)}{\piS(y \mid x)}\right]$$


While this approach is powerful, it is inherently limited by the properties of forward-KL, namely that it is *mode covering*. To see this behavior, see the image below:
![SFT illustration](/figures/sft-gif.gif)

While this can be desirable, it can lead to the model outputting samples that are not quite like the teacher model's outputs, where $\piS$ focuses on expanding its support to match that of $\piT$, rather than accurately approximating it.

### Self-Distillation via Reverse KL

An easy fix to the mode covering behavior of forward-KL is to instead optimize reverse-KL, which is *mode seeking*. When using $\piTthetafull$ as the teacher, this is referred to as self-distillation, which optimizes the following objective:

$$
D_{\text{kl}}(\piStheta \;\|\; \piTtheta) = \mathbb{E}_{y\sim\piStheta}\left[\log\frac{\piSthetafull}{\piTthetafull}\right].
$$

Minimizing this encourages the student to place mass where it already samples, but guided by the teacher's density.

![Self-distillation illustration](/figures/reverse-kl.gif)

This idea comes from [Agarwal et al. (2023)](#ref-onpoldistill), which shows that distilling on-policy can lead to significantly better performance and generalization on a variety of tasks when compared to SFT.

But as seen in the figure above, this can lead to some suboptimal behavior, where the policy fits a suboptimal mode of the teacher.


### Reward-Tilted Self-Distillation

The main problem with pure self-distillation is its bias towards easier-to-fit modes, which may inherently be suboptimal. This comes from $\piTthetafull$ being the target distribution. A simple way to alleviate this is to define our target distribution $\pi^*$ as a reward-tilted variant of $\piTthetafull$:

$$
\pi^*(y \mid x, \mathbf{I}) \propto \piTthetafull\exp\left(\frac{R(y, x)}{\tau}\right).
$$

Assuming $\theta$ is fixed with respect to $\piT$, we can optimize the reverse-KL between the student and this tilted target. This yields the following objective:

$$
\max_\theta \; \mathbb{E}_{y\sim\piSthetafull}\left[R(y, x)\right] - \tau \, D_{\text{kl}}\big(\piSthetafull \;\|\; \text{sg}[\piTthetafull]\big)
$$

where $\text{sg}[\cdot]$ denotes stop-gradient. This objective explicitly rewards high-reward outputs while still matching the teacher's structure.

<details>
<summary><strong>Derivation</strong> (click to expand)</summary>

$$
\begin{aligned}
D_{\text{kl}}(\piS \;\|\; \pi^*) &= \mathbb{E}_{y\sim\piS}\left[\log\frac{\piS(y \mid x)}{\pi^*(y \mid x, \mathbf{I})}\right] \\
&= \mathbb{E}_{y\sim\piS}\left[\log\frac{\piS(y \mid x)}{\frac{1}{\tilde{Z}(x)}\piT(y \mid x, \mathbf{I})\exp\left(\frac{R(y, x)}{\tau}\right)}\right] \\
&= \mathbb{E}_{y\sim\piS}\left[\log\piS(y \mid x) - \log\piT(y \mid x, \mathbf{I}) - \frac{R(y, x)}{\tau} + \log \tilde{Z}(x)\right] \\
&= \mathbb{E}_{y\sim\piS}\left[\log\frac{\piS(y \mid x)}{\piT(y \mid x, \mathbf{I})} - \frac{R(y, x)}{\tau}\right] + \log \tilde{Z}(x) \\
&= D_{\text{kl}}\big(\piS(y \mid x) \;\|\; \piT(y \mid x, \mathbf{I})\big) - \frac{1}{\tau}\mathbb{E}_{y\sim\piS}\left[R(y, x)\right] + \log \tilde{Z}(x).
\end{aligned}
$$

Since $\log \tilde{Z}(x)$ is constant with respect to $\piS$, minimizing this KL is equivalent to maximizing:

$$
\max_\theta \; \mathbb{E}_{y\sim\piSthetafull}\left[R(y, x)\right] - \tau \, D_{\text{kl}}\big(\piSthetafull \;\|\; \piTthetafull\big).
$$

</details>

![Reward-tilted self-distillation illustration](/figures/RKLR.gif)

Introducing this reward bias should enable us to effectively fit higher reward modes.




### Variational EM
One assumption that both self-distillation variants rely on is that the teacher model $\piT$ has support over high-reward regions. While this assumption is likely valid in many settings, it may be severely limited in cases where even when conditioned on $\mathbf{I}$, the model still does not have coverage over high-reward regions.



<!-- figure -->

In this case, regardless of which algorithm we use, we are limited by the abilities of the teacher.


An easy solution is that since $\piT$ does have some coverage over successful trajectories, we can leverage it to approximate the target policy. In this case though, rather than directly trying to approximate $\piT$, we first make it approximate $\pi^*$ itself. Once $\piT$ resembles $\pi^*$, we can then use it as a target for $\piS$. Plainly put: we train the teacher to approximate the target, and once the teacher looks like the target, we fit the student onto the teacher. The figure below visualizes this procedure:




While this approach can improve over the base policy, in [Penaloza et al. (2026)](#ref-penaloza2026) we show that it is inefficient and in many cases simply training the teacher suffices. Thus, one can greatly simplify the process by simultaneously training the teacher and student.

## $\pi$-Distill

While variational EM fits the teacher and student sequentially with different parameters for $\piS$ and $\piT$, in [Penaloza et al. (2026)](#ref-penaloza2026) we show this is inefficient and less effective. Rather, a simple solution is to allow $\piS$ and $\piT$ to share parameters and jointly train the model, greatly simplifying the training process. Moreover, rather than training the student using naive SFT, we can directly use the same loss used for the teacher but via importance sampling:

$$
J_{\text{Student}}(\theta) = \mathbb{E}_{y \sim \piTthetafull}\left[\frac{\piSthetafull}{\text{sg}[\piTthetafull]} R(y, x)\right] - \beta \, D_{\text{kl}}\big(\text{sg}[\piTthetafull] \;\|\; \piSthetafull\big)
$$

We can then jointly optimize the objectives for the teacher and the student:

$$
J_{\pi\text{-Distill}}(\theta) = \alpha \, J_{\text{Teacher}}(\theta) + (1 - \alpha) \, J_{\text{Student}}(\theta)
$$

A key component of $\pi$-Distill is its ability to modulate between the student and teacher objectives using $\alpha$. This allows for varying training configurations depending on the properties of the teacher:

- When $\alpha = 1$, optimization focuses entirely on the teacher, although the student may still improve through shared parameters.
- When $\alpha = 0$, training focuses on the student learning from the teacher's current behavior. Interestingly, we observe that under certain conditions, parameter sharing can still lead to improvements in the teacher without explicit teacher updates.
- When $\alpha = 0.5$, both are optimized jointly. Shared parameters allow representations learned for using $\mathbf{I}$ to transfer to the student, while student updates keep those representations effective without $\mathbf{I}$.

### Teacher Training $\alpha = 1$
When $\alpha = 1$, only the teacher is being trained. This is similar to other work that explores conditional training with language models.

For visualizing $\pi$-Distill, we add an axis, where now the x-axis represents the KL between the displayed models and the base policy $\pi_{\text{base}}$ while the y-axis represents the reward for the trajectories. This allows us to visualize how things change as we vary the portions we optimize.

<!-- insert teacher gif -->

For instance, as we optimize $\piS$ we see the model is incentivized to drift away from the base model, where adding the KL term prevents it from drifting too far. By sharing parameters though, knowledge can transfer between $\piStheta$ and $\piTtheta$.



### Student Training $\alpha = 0$

This case is the opposite of the previous one, in which we only train the student via off-policy RL with traces coming from the teacher $\piT$. The figure below visualizes what happens.



Training the student directly attracts it towards the teacher model, where again by sharing parameters the teacher can also improve. Directly training on the teacher's distribution though is a significantly stronger attractor for the student compared to hoping knowledge transfers when just training the teacher.

We note, as per our results in Section 7 in [Penaloza et al. (2026)](#ref-penaloza2026), fitting teacher distributions can be hard, specifically in higher KL settings, where this illustration depicts an ideal case.



### Joint Training $0 < \alpha < 1$

In this case, the teacher is explicitly tasked with improving, while the student is also explicitly tasked with approximating the teacher. This creates a self-regularizing behavior where the student actively seeks the teacher, not allowing the teacher to drift too far.




Notice how the teacher can drift away but not as far as before, and with explicit student training, we can approximate the teacher directly. This leads to both policies being closer to each other than in teacher training and does not allow the teacher to drift as far as in teacher training.



## Discussion

We hope this blog serves as a good starting point to get some intuition into the different variants that exist when training with PI. While this blog greatly simplifies the underlying mechanics, we hope it gives good intuition.

There are many different directions that are still unexplored. For instance, we discussed $\pi$-Distill which fits the student using off-policy traces from the teacher; in [Penaloza et al. (2026)](#ref-penaloza2026) we show the success of this depends heavily on the properties of the PI, e.g., the utility or KL between student and teacher model. Another approach is to do teacher training followed by self-distillation; this would be fully on-policy as teacher training is on-policy and student training is also on-policy. This could potentially lead to superior performance as reverse-KL is generally easier to fit, at the cost of having to do a two-step procedure. Alternatively, one could jointly train them like in $\pi$-Distill, but this would require sampling from the teacher and student at the same time (since both are trained on-policy).

Moreover, more effective types of PI mining are a good direction. [Liao et al. (2026)](#ref-liao2026) are a good instantiation of this, where by conditioning on self-reflection the agents are able to improve. Our work in [Penaloza et al. (2026)](#ref-penaloza2026) suggests that for self-distillation, what matters most is the informativeness of the PI; if the teacher is too close to the student in terms of KL, they can collapse onto each other, making optimization harder. Thus, mining helpful PI with sufficient but not excessive KL is a promising avenue.

Finally, recent work suggests that fitting policies to data that is more off-policy induces more forgetting. One can think of ways to slowly but surely generate PI for questions that help the student improve without incentivizing the policy to substantially drift from itself. The work by [Shenfeld et al. (2026)](#ref-shenfeld2026) is an instantiation of this. Extending to different settings, such as harder agentic tasks where context can grow exponentially, or personalization, could yield interesting results.



---

## References

<a id="ref-shenfeld2026"></a>
**[Shenfeld et al., 2026]** Shenfeld, I., Damani, M., Hübotter, J., & Agrawal, P. (2026). *Self-Distillation Enables Continual Learning*. [arXiv:2601.19897](https://arxiv.org/abs/2601.19897)

<a id="ref-zhao2026"></a>
**[Zhao et al., 2026]** Zhao, S., Xie, Z., Liu, M., Huang, J., Pang, G., Chen, F., & Grover, A. (2026). *Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models*. [arXiv:2601.18734](https://arxiv.org/abs/2601.18734)

<a id="ref-hubotter2026"></a>
**[Hübotter et al., 2026]** Hübotter, J., Lübeck, F., Behric, L., Baumann, A., Bagatella, M., Marta, D., Hakimi, I., Shenfeld, I., Kleine Buening, T., Guestrin, C., & Krause, A. (2026). *Reinforcement Learning via Self-Distillation*. [arXiv:2601.20802](https://arxiv.org/abs/2601.20802)

<a id="ref-penaloza2026"></a>
**[Penaloza et al., 2026]** Penaloza, E., Vattikonda, D., Gontier, N., Lacoste, A., Charlin, L., & Caccia, M. (2026). *Privileged Information Distillation for Language Models*. [arXiv:2602.04942](https://arxiv.org/abs/2602.04942)

<a id="ref-onpoldistill"></a>
**[Agarwal et al., 2023]** Agarwal, R., Vieillard, N., Stanczyk, P., Ramos, S., Geist, M., & Bachem, O. (2023). *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes*. [arXiv:2306.13649](https://arxiv.org/abs/2306.13649)

<a id="ref-hinton2015"></a>
**[Hinton et al., 2015]** Hinton, G., Vinyals, O., & Dean, J. (2015). *Distilling the Knowledge in a Neural Network*. [arXiv:1503.02531](https://arxiv.org/abs/1503.02531)

<a id="ref-vapnik2009"></a>
**[Vapnik & Vashist, 2009]** Vapnik, V., & Vashist, A. (2009). *A New Learning Paradigm: Learning Using Privileged Information*. Neural Networks, 22(5-6), 544-557.

<a id="ref-levine2018"></a>
**[Levine, 2018]** Levine, S. (2018). *Reinforcement Learning and Control as Probabilistic Inference: Tutorial and Review*. [arXiv:1805.00909](https://arxiv.org/abs/1805.00909)

<a id="ref-chen2026"></a>
**[Chen et al., 2026]** Chen, J. C.-Y., Peng, B. X., Choubey, P. K., Huang, K.-H., Zhang, J., Bansal, M., & Wu, C.-S. (2026). *Nudging the Boundaries of LLM Reasoning*. [arXiv:2509.25666](https://arxiv.org/abs/2509.25666)

<a id="ref-hatamizadeh2026"></a>
**[Hatamizadeh et al., 2026]** Hatamizadeh, A., Prabhumoye, S., Gitman, I., Lu, X., Han, S., Ping, W., Choi, Y., & Kautz, J. (2026). *iGRPO: Self-Feedback-Driven LLM Reasoning*. [arXiv:2602.09000](https://arxiv.org/abs/2602.09000)

<a id="ref-liao2026"></a>
**[Liao et al., 2026]** Liao, B., Dong, H., Xu, X., Monz, C., & Bian, J. (2026). *Self-Hinting Language Models Enhance Reinforcement Learning*. [arXiv:2602.03143](https://arxiv.org/abs/2602.03143)

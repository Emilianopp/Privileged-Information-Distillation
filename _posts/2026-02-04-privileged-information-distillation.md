---
title: "Privileged Information Distillation"
date: 2026-02-04
author: "Your Name"
---

This post provides an introductory overview of recent works on self-distillation and privileged information distillation, including [Shenfeld et al. (2026)](#ref-shenfeld2026), [Zhao et al. (2026)](#ref-zhao2026), [Hübotter et al. (2026)](#ref-hubotter2026), and [Penaloza et al. (2026)](#ref-penaloza2026). We aim to build intuition for how these algorithms work, where they come from, and when their assumptions hold. Throughout, we favour gaining intuition over providing a complete nuanced view, using simplified 2D/3D visualizations of these algorithms to describe the core ideas.


Our work spans the intersection of variational inference and reinforcement learning applied to LMs. We hope this blog is approachable to a broad audience assuming only a small background in reinforcement learning. Regardless, we provide some supplemental reading for those interested.
<details markdown="1">
<summary><strong>Supplemental Reading</strong> (click to expand)</summary>

- [Control as Inference](https://jasonppy.github.io/deeprl/deeprl-12-control-as-inference/) for an introduction to the RL-as-inference framework (or see [Levine's tutorial paper](https://arxiv.org/abs/1805.00909))
- [Variational Inference: A Review for Statisticians](https://arxiv.org/abs/1601.00670) for a comprehensive overview of variational inference
- [The RLHF Book](https://rlhfbook.com/) for RL + LLMs

</details>

<details markdown="1">
<summary><strong>Notation</strong> (click to expand)</summary>

We work in a single-turn setting where $x$ denotes the prompt, $z$ the chain-of-thought, and $y$ the final output response. We write policies as $\pi(y \mid x)$ or $\pi(y \mid x, z)$ when conditioning on chain-of-thought.

</details>

## Reinforcement Learning as Inference

The goal of RL is to maximize reward

$$
\piStar = \arg\max_\pi \mathbb{E}_{y \sim \pi(\cdot \mid x)}\left[R(y, x)\right]
$$

to obtain a target policy $\piStar$. A slightly different perspective treats RL as an inference problem, wherein we define the target policy as a distribution over outputs:

$$
\piStarfull = \frac{1}{Z(x)}\,\pi_{\mathrm{ref}}(y \mid x)\exp\left(\frac{R(y, x)}{\tau}\right)
$$

Since the partition function $Z$ makes direct inference intractable, we instead parameterize an approximate policy $\piSphi$ and minimize the reverse-KL divergence to $\piStar$. Where we must use reverse-KL as we can only sample from $\piSphi$, not from $\piStar$:

$$
\underbrace{D_{\text{kl}}(\piSphi \;\|\; \piStar)}_{\text{Reverse-KL}} = \mathbb{E}_{y\sim\piSphi}\left[\log\frac{\piSphi(y \mid x)}{\piStarfull}\right]
$$


<details markdown="1">
<summary><strong>Full Derivation</strong> (click to expand)</summary>

Starting from the reverse-KL and expanding the definition of $\piStar$:

$$
\begin{aligned}
D_{\text{kl}}(\piSphi \;\|\; \piStar)
&= \mathbb{E}_{y\sim\piSphi}\left[\log\frac{\piSphi(y \mid x)}{\piStarfull}\right] \\
&= \mathbb{E}_{y\sim\piSphi}\left[\log\frac{\piSphi(y \mid x)}{\frac{1}{Z(x)}\,\pi_{\mathrm{ref}}(y \mid x)\exp\!\left(\frac{R(y,x)}{\beta}\right)}\right] \\
&= \mathbb{E}_{y\sim\piSphi}\left[\log\piSphi(y \mid x) - \log\pi_{\mathrm{ref}}(y \mid x) - \frac{R(y,x)}{\beta} + \log Z(x)\right] \\
&= D_{\text{kl}}(\piSphi\;\|\;\pi_{\mathrm{ref}}) - \frac{1}{\beta}\,\mathbb{E}_{y\sim\piSphi}\left[R(y, x)\right] + \log Z(x)
\end{aligned}
$$

Since $\log Z(x)$ is constant w.r.t. $\phi$, minimizing the reverse-KL is equivalent to:

$$
\max_\phi\; \mathbb{E}_{y\sim\piSphi}\left[R(y, x)\right] - \beta \, D_{\text{kl}}(\piSphi\;\|\;\pi_{\mathrm{ref}})
$$
$$
\max_\phi\; \mathbb{E}_{y\sim\piSphi}\left[R(y, x)\right] - \beta \, D_{\text{kl}}(\piSphi\;\|\;\pi_{\mathrm{ref}})
$$



</details>
Note that when $\beta = 0$ we recover the original reward-maximization objective. Otherwise, the KL term acts as a regularizer, encouraging the policy to stay close to the reference.


To illustrate this, consider the figure below, where $\beta =0$. Here the model can quickly fit $\piStar$, but has its entropy (bell curveness) collapsed onto a high-reward mode. 
<video src="/figures/rl-collapse.mov" autoplay loop muted playsinline></video>



When $\beta>0$, we can expect the policy do behave similarly, but instead drifts less from the reference policy $\pi_{\text{ref}}$ while also preserving more entropy (bell curve shape). 

<video src="/figures/rl-kl.mov" autoplay loop muted playsinline></video>

Both of these properties can be useful in different contexts, for instance, when the entropy of the model has collapsed it can reliably sample from a high reward mode, but in the other case, it is able to preserve some reward over even higher-reward regions which can be useful property we'll discuss later. Moreover, the tradeoff becomes more nuanced when distributions are multi-model (multiple peaks) which is the setting we discuss in the remainder of the blog. 

Regardless, the choice of $\beta$ is largely task-specific and often decided via hyperparameter tuning ([Shah et al., 2026](#ref-shah2026)). 


## Failures of RL

While this objective has proven widely successful, it relies on the policy $\piSphi$ being able to sample high-reward outputs. When the policy cannot produce any successful trajectories, there is no positive signal to reinforce.

For instance, consider the following setting:

![Student-only policy failing example](/figures/rl-zero.png)
*Figure 1: Failure case of RL, here we visualize $\piSphi$ and $\piStar$ where $\piSphi$ has no support over successful trajectories.*

Notice that $\piSphi$ does not have support over correct trajectories, so applying RL in this setting would lead to the model only ever learning what *not* to do, never what *to* do. Without any high-reward samples to reinforce, it cannot bootstrap itself toward $\piStar$. While an LM technically has full support over the token space, making it theoretically possible to eventually sample a correct trajectory, this is infeasible in practice.


### Shaping a LM distribution by Conditioning

Although typical RL fails in these settings, LMs provide a useful property that can alleviate this problem. Unlike other ML systems, LMs can be freely conditioned on additional information. Notably, even a small amount of *Privileged Information* (PI) can enable models to sample tasks they previously could not. The figure below shows $\piTthetafull$, the same model now conditioned on privileged information $\mathbf{I}$, which we refer to as the *teacher*.

![With teacher example](/figures/teacher-self.png)
*Figure 2: Teacher policy contextualized on privileged information can now sample successful trajectories.*

After contextualizing on $\mathbf{I}$, we see that the model can now sample successful traces. The only problem now is that we won't have access to $\mathbf{I}$ at test time, since it is typically *task-specific*. So we need to find a way to transfer the information embedded within it to $\piS$, as this is the only policy we have access to at test time.

The remainder of this post explores different approaches to achieving this:

1. [SFT](#sft) — supervised fine-tuning on teacher trajectories
2. [Self-Distillation](#self-distillation-via-reverse-kl) — matching the student to the teacher via reverse KL
3. [Reward-Tilted Self-Distillation](#reward-tilted-self-distillation) — incorporating reward into the distillation target
4. [Variational EM](#variational-em) — alternating between fitting the teacher and distilling to the student
5. [$\pi$-Distill](#pi-distill) — joint teacher-student training with shared parameters


## Supervised Fine-Tuning 

By far the simplest and most popular way of transferring this knowledge is through Supervised Fine-Tuning (SFT). This is equivalent to fitting forward-KL between the teacher policy $\piT$ and the student $\piS$. 




$$D_{\text{kl}}(\piT \;\|\; \piS) = \mathbb{E}_{y\sim\piT}\left[\log\frac{\piT(y \mid x)}{\piS(y \mid x)}\right]$$

Since $\piT$ is fixed, minimizing the forward-KL reduces to maximizing the expected log-likelihood under the student:

$$
\max_\phi\; \mathbb{E}_{y\sim\piT}\left[\log \piSphi(y \mid x)\right]
$$

Typically $\piT$ is a larger model, but in this specific case it can be the same starting model with added context i.e. $\piTthetafull$. 

While this approach is powerful, it is inherently limited by the properties of forward-KL, namely that it is *mode covering*. To see this behavior, see the image below:

<video src="/figures/sft.mov" autoplay loop muted playsinline></video>


This can often lead to the model outputting samples that are not likely under the teacher model. This can happen as forward-KL makes $\piS$ focus on expanding its support to match that of $\piT$, rather than accurately approximating it.

## Self-Distillation via Reverse KL

An easy fix to the mode covering behavior of forward-KL is to instead optimize reverse-KL, which is *mode seeking*. When using $\piTthetafull$ as the teacher, this is referred to as self-distillation, which optimizes the following objective:

$$
D_{\text{kl}}(\piSphi \;\|\; \piTtheta) = \mathbb{E}_{y\sim\piSphi}\left[\log\frac{\piSphifull}{\piTthetafull}\right].
$$

Minimizing this encourages the student to place mass where it already samples, but guided by the teacher's density. We note that typically most works allow $\phi$ to equal $\theta$ as in [Penaloza et al. (2026)](#ref-penaloza2026), be an exponential moving average as in [Hübotter et al. (2026)](#ref-hubotter2026), [Shenfeld et al. (2026)](#ref-shenfeld2026), and [Zhao et al. (2026)](#ref-zhao2026), or simply be the base model. 

<video src="/figures/rKL.mov" autoplay loop muted playsinline></video>

This idea comes from [Agarwal et al. (2023)](#ref-onpoldistill), which shows that distilling on-policy can lead to significantly better performance and generalization on a variety of tasks when compared to SFT.

But as seen in the figure above, this can lead to some suboptimal behavior, where the policy fits a suboptimal mode of the teacher.


## Reward-Tilted Self-Distillation

The main problem with pure self-distillation is its bias towards easier-to-fit modes, which may inherently be suboptimal. This comes from $\piTthetafull$ being the target distribution. A simple way to alleviate this is to define our target distribution $\piStar$ as a reward-tilted variant of $\piTthetafull$:

$$
\piStarfulli \propto \piTthetafull\exp\left(\frac{R(y, x)}{\tau}\right).
$$

Assuming $\theta$ is fixed with respect to $\piT$, we can optimize the reverse-KL between the student and this tilted target. This yields the following objective:

$$
\max_\phi \; \mathbb{E}_{y\sim\piSphifull}\left[R(y, x)\right] - \tau \, D_{\text{kl}}\big(\piSphifull \;\|\; \piTthetafull\big)
$$

 This objective explicitly rewards high-reward outputs while still matching the teacher's structure.

<details>
<summary><strong>Derivation</strong> (click to expand)</summary>

$$
\begin{aligned}
D_{\text{kl}}(\piSphi \;\|\; \piStar) &= \mathbb{E}_{y\sim\piSphi}\left[\log\frac{\piSphifull}{\piStarfulli}\right] \\
&= \mathbb{E}_{y\sim\piSphi}\left[\log\frac{\piSphifull}{\frac{1}{\tilde{Z}(x)}\piTthetafull\exp\left(\frac{R(y, x)}{\tau}\right)}\right] \\
&= \mathbb{E}_{y\sim\piSphi}\left[\log\piSphifull - \log\piTthetafull - \frac{R(y, x)}{\tau} + \log \tilde{Z}(x)\right] \\
&= \mathbb{E}_{y\sim\piSphi}\left[\log\frac{\piSphifull}{\piTthetafull} - \frac{R(y, x)}{\tau}\right] + \log \tilde{Z}(x) \\
&= D_{\text{kl}}\big(\piSphi \;\|\; \piTthetafull\big) - \frac{1}{\tau}\mathbb{E}_{y\sim\piSphi}\left[R(y, x)\right] + \log \tilde{Z}(x).
\end{aligned}
$$

Since $\log \tilde{Z}(x)$ is constant with respect to $\piSphi$, minimizing this KL is equivalent to maximizing:

$$
\max_\phi \; \mathbb{E}_{y\sim\piSphifull}\left[R(y, x)\right] - \tau \, D_{\text{kl}}\big(\piSphifull \;\|\; \piTthetafull\big).
$$

</details>

<video src="/figures/rklr.mov" autoplay loop muted playsinline></video>


Introducing this reward bias should enable us to effectively fit higher reward modes. Also, notice how this objective is the same as the one outlined above to RL-as-inference, simply with a different prior distribution. 

While this objective is powerful, it relies on the teacher already being a good approximation of $\piStar$, which can be unrealistic in many settings. 




## Variational EM
One assumption that self-distillation relies on is that  the teacher model $\piT$ has support over high-reward regions,  i.e., $\piTtheta \approx \piStar$. While this assumption is likely valid in many settings, it may be severely limited in cases where even when conditioned on $\mathbf{I}$, the model still does not have coverage over high-reward regions.



![Teacher with limited coverage](/figures/bad-teacher.png)

In this case, regardless of which algorithm we use, we are limited by the abilities of the teacher.
<video src="/figures/em3.mov" autoplay loop muted playsinline></video>

An easy solution is that since $\piT$ does have some coverage over successful trajectories, we can leverage it to approximate the target policy. In this case though, rather than directly trying to approximate $\piT$, we first make it approximate $\piStar$ itself. Once $\piT$ resembles $\piStar$, we can then use it as a target for $\piS$. Plainly put: we train the teacher to approximate the target, and once the teacher looks like the target, we fit the student onto the teacher. The figure below visualizes this procedure:




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

<a id="ref-shah2026"></a>
**[Shah et al., 2026]** Shah, V., Obando-Ceron, J., Jain, V., Bartoldson, B., Kailkhura, B., Mittal, S., Berseth, G., Castro, P. S., Bengio, Y., Malkin, N., Jain, M., Venkatraman, S., & Courville, A. (2026). *A Comedy of Estimators: On KL Regularization in RL Training of LLMs*. [arXiv:2512.21852](https://arxiv.org/abs/2512.21852)

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

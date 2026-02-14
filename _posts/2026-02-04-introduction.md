---
title: "Privileged Information Distillation"
date: 2026-02-04
author: "Your Name"
---

LLMs have the unique ability to converse, which provides a superior user interface and more straightforward interactions than other machine learning systems. Yet this ability is rarely leveraged during RL-post training. Leveraging this capability by contextualizing models at train-time with privileged information (PI) could have many benefits.

For instance take a hard task where the model always yields zero success as can be seen in the figure below:

<!-- TODO: Add figure here -->

In this case the model cannot sample any correct trajectories and cannot bootstrap itself onto the optimal policy $\pi^*$. But maybe, we could leverage the ability of LM's to be contextualized on additional information, we denote this as $\pi(\cdot|s,\mathbf{I})$, where $\mathbf{I}$ stands for privileged information.


<!-- TODO: Add figure here -->

After this contextualization the model is able to sample successful trajectories, possibly being able to bootstrap itself onto approximating $\pi^*$.


But there is a problem, doing this trains the contextualized policy $\piT$ to expect that PI will be available/relevant at test time. This is unrealistic as PI is often task specific and not-relevant for other tasks. But something we can do, is that once we have approximated $\pi^*$ we can treat the contextualized policy as a teacher $\piT$ to train the unconditioned student $\piS$

<!-- Continue writing your blog post here... -->








### Problems

This framework is equivalent to Variational Expectation Maximization (EM), yet when we implmment this procedure we find it generally underperforms compared to baselines. Yet something we find is that when we simply train the teacher, the student learns when both $\piS$ and $\piT$ share parameters. With this in mind, we decided to allow teacher and student to share parameters,  optimizing them jointly. This setup leads to the following two lossess which can be derived from a variational prespective:




<!--- teacher pi distill loss-->

<!--- Comments on student loss-->

<!--- Comments on  -->


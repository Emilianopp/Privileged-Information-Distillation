# Privileged Information Distillation


LLMs have the unique ability to converse, which provides a superior user interface and more
straightforward interactions than other machine learning systems. Yet this ability is rarely leveraged during RL-post training. Leveraging this capability by contextualizing models at train-time with privielged information (PI) could have many benefits. 


For instance take a hard task where the model always yields zero success as can be seen in the figure below:






In this case the model cannot sample any correct trajectories and cannot bootstrap itself onto the optimal policy $\pi^*$. But may be, we could leverage the ability of LM's to be contextualized on additional information, we denot this as $\pi(.|s,\mathbf{I})$, where $\mathbf{I}$ stands for privileged pnformation. 








After this contextualization the model is able to sample successful trajectories, possibly being able to bootstrap itself onto approximating $\pi^*$. 



But there is a problem, doing this trains the contextualized $$ that PI will be available/relevant at test time
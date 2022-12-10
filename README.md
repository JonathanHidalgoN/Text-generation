About this repo.
This repo contains different algorithms to generate tweets like a human would do. Every function is documented, so you can quickly understand what it does.

Contents
LSMT_text_generation.py is a recurrent neural network with a little cell that works as “memory.
fine_tuning_gpt2.py fine-tune medium gpt2, this model is a transformer.
markov_generator.py is a probabilistic generator, you can choose between character-level generation or word-level generation.
transformers_text_generation.py is a transformed trained from 0, with a custom 256 embedding.
Some examples
(The promt is bolded)
Working on github right at 10 am when facebook decides to change cnns around again i do expect jax to benefit even less than on the r gd which could make some sense now we already have the next model fit into jax in tf keras
(Transformer GPT2 Francois Collet)
I love 2007 in agreed cool order a tesla powerwall battery for blackout protection adri note we are implementing additional safeguards to prevent impersonation as well as collective use of verified accounts by a single individual or organization widespread verification will democratize journalism.
(Markov generator on Tweets)
Politics is one of the few places which rewards winners losers who play by the rules by you guessed it cheating so you need winning incentives that are distributed by reward it also applies to competition as a whole but it changes game completely
(Transformer GPT2 Francois Collet)Educations is one of the most important things in the word because they open more sides of human existence you could argue this is our birthright the human mind has many other kinds in life now not the most powerful yet but to live for ourselves all you has to be in control of your body is an internet connection
(Transformer GPT2 Francois Collet)
Money is the law we all understand irony haha the rules make no sense lol with the exception ftw everyone benefits significantly at current tax there no limit on one person s company in 7 12 not a scam the most people know that rules exist but others fail to even after decades of study
(Transformer GPT2 Elon Musk)
no i didnt know he would have a great job in the up during your record like sinking more of the harper always people have to fight and a member and comes to ask
(Custom transformer on tweets)
Where did you get the data?
In https://github.com/JonathanHidalgoN/TwitterInferenceProject, there is a pipeline where you can store tweets in a MySQL database.

Can I use the models?
I am working on it, right now you can use fine_tuning_gpt2.py
at https://www.kaggle.com/code/jonathanaxel/fine-tuing-gpt-2
I will try to provide pre-trained models and tweets so everyone can check the model.

References
[1] Chollet, F. (2021). Deep learning with Python. Simon and Schuster.
[2] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., … & Polosukhin, I. (2017). Attention is all you need. Advances in neural information processing systems, 30.
[3] Shivamb. (2019, January 5). Beginners Guide to text generation using lstms. Kaggle. Retrieved December 6, 2022, from https://www.kaggle.com/code/shivamb/beginners-guide-to-text-generation-using-lstms

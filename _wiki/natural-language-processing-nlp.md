---
title: A Beginner's Guide to Natural Language Processing (NLP)
short_title: Natural Language Processing (NLP)
description: An introduction to extracting information from natural language and raw text.
---

Natural language processing applies computers to understanding human language, to the words we use. You speak human, and your computer speaks machine. That sounds like the first sentence of a post on couples counseling... 

*Natural language* refers to language that is spoken and written by people, and natural language processing (NLP) attempts to extract information from the spoken and written word using algorithms. In this post, we're going to focus on the written word in order to avoid the additional complexity of transcribing speech to text or generating natural human voices. 

NLP encompasses active and a passive modes: *natural language generation* (NLG), or the ability to formulate phrases that humans might emit, and *natural language understanding* (NLU), or the ability to build a comprehension of a phrase, what the words in the phrase refer to, and its intent. In a conversational system, NLU and NLG alternate, as algorithms parse and comprehend a natural-language statement, and formulate a satisfactory response to it. 

But let's start with something simpler than a chatbot. 

## The problem 

Most of the work of computer science is devoted to translating human ideas into a form that machines can understand. Code can be high-level like Python or Java or Ruby, which makes it easier for humans to read and write. But underneath those languages, the way thoughts are expressed must get closer and closer to the bits themselves through assembly language and object code, the 1s and 0s. 

You could say that the history of programming has been a steady march away from the machine and toward the human, moving more and more of the work of translation into compute (which has become cheaper) and relieving the human experts (who are always too rare). Natural language processing, like the graphical user interfaces (GUIs) we came to know through personal computers, is another big step in that direction. The only problem is, there are real limits to what NLP can do. 

Natural language processing tries to do two things: understand and generate human language. You might call these the passive and active sides of NLP. Natural language understanding can come in many forms. On the simplest level, you could classify a text: for example, you might have a bunch of emails and you want to know whether they are angry or happy, because you work in customer service. NLP can do that, and it's called sentiment analysis. Or maybe you're an HR department and you want to categorize resumes coming in for job descriptions; i.e. is the person applying for the role of UX designer someone who has UX experience, or someone who is parachuting into the profession from a previous career as a trapeze artist? NLP can do that, too. 

But that's not the level of understanding we need to relate to natural language in deeper and more interesting ways. 

Before we get to those deeper understandings, let's talk for a moment about what it means for a computer to store written language, like the sentence you are reading now. While these words echo in your mind, and carry with them energy and meaning, to the computer they are simply patterns of pixels printed on a screen. These arrays of characters that you call words are known as "strings" in programming. 

And most of the computer processing applied to human language is just a shuffling of strings, skating lightly over symbols that are just the petrified artifact of a live intelligence. The computer does not know what they signify. It has no visceral intuition of the objects to which they refer. Feeding a computer a string about a "little house in the big woods near the bright creek where the trout used to jump" will evoke no image or nostalgia, at least not on its own. For most of the history of computers, we have stored text in machines in order to relay the words later to other humans, who were called upon to supply the meaning. 

You could say that NLP tries to change that. Or at least make the question of whether machines understand what we say irrelevant. 

## Further Reading on Natural Language Processing

- [Stanford course: Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/) by Dan Jurafsky and James Martin
- [NLP-progress: Tracking progress in Natural Language Processing, including datasets & current state-of-the-art for common NLP tasks](http://nlpprogress.com/)
- [BERT](https://arxiv.org/abs/1810.04805), and [TensorFlow code and pre-trained models for BERT](https://github.com/google-research/bert) 
- [Deep Chit Chat: Deep Learning for Chatbots](http://ruiyan.me/pubs/tutorial-emnlp18.pdf)
- [top-k sampling using PyTorch](https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317)

[The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751)

Despite considerable advancements with deep neural language models, the enigma of neural text degeneration persists when these models are tested as text generators. The counter-intuitive empirical observation is that even though the use of likelihood as training objective leads to high quality models for a broad range of language understanding tasks, using likelihood as a decoding objective leads to text that is bland and strangely repetitive. In this paper, we reveal surprising distributional differences between human text and machine text. In addition, we find that decoding strategies alone can dramatically effect the quality of machine text, even when generated from exactly the same neural language model. Our findings motivate Nucleus Sampling, a simple but effective method to draw the best out of neural generation. By sampling text from the dynamic nucleus of the probability distribution, which allows for diversity while effectively truncating the less reliable tail of the distribution, the resulting text better demonstrates the quality of human text, yielding enhanced diversity without sacrificing fluency and coherence.

[Hierarchical Neural Story Generation](https://arxiv.org/abs/1805.04833)

We explore story generation: creative systems that can build coherent and fluent passages of text about a topic. We collect a large dataset of 300K human-written stories paired with writing prompts from an online forum. Our dataset enables hierarchical story generation, where the model first generates a premise, and then transforms it into a passage of text. We gain further improvements with a novel form of model fusion that improves the relevance of the story to the prompt, and adding a new gated multi-scale self-attention mechanism to model long-range context. Experiments show large improvements over strong baselines on both automated and human evaluations. Human judges prefer stories generated by our approach to those from a strong non-hierarchical model by a factor of two to one.


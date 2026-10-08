# Methods

I conducted a 3x3x3 grid search to determine optimal parameters to improve the CLEF RAG system. 
The following parameters were used for the 3x3x3 grid search:


| Weight Pairs (Lexical, Semantic) | Candidate Retrieval Amounts | Top K |
| -------------------------------- | --------------------------- | ----- |
| 0.2, 0.8                         | 5                           | 3     |
| 0.5, 0.5                         | 20                          | 6     |
| 0.8, 0.2                         | 25                          | 9     |

The default parameters (the control) was weight pair = 0.5, 0.5, candidate retrieval amount = 20, and top K = 6. 
Thus, the grid search would evaluate how increasing or decreasing the parameters would affect the CLEF RAG system.

The 3x3x3 grid search was chosen to keep the evaluation within a reasonable runtime as the total runtime for the experiment was already greater than 3.5 hours.

The weight pairs were chosen to evaluate how a heavier weighting towards a lexical ranker vs semantic ranker would affect the question & answer performance. The candidate retrieval amounts were chosen to determine how a small retrieval amount and a marginally higher retrieval amount (since 20 candidates is already a relatively high amount) would affect the question & answer performance. The top Ks were chosen to determine how a relatively lower and a relatively higher amount would affect the question & answer performance.

The following questions and answers were used to evaluate the CLEF RAG system. The number of questions and answers were kept minimal but variety to keep the runtime reasonable.

| Question | Answer |
| --- | --- |
| What is disagreement-aware ensemble? | four detectors with fundamentally different assumptions are fused by a meta-learner that receives not just their individual scores but their variance and range as explicit features, encoding the degree of inter-detector inconsistency |
| How does DeBERTa improves upon BERT-style encoders? | Through disentangled attention and enhanced decoding pre-training. |
| What does the result show about JobBERT in paper484? | The results show that JobBERT is already a strong dense baseline. |
| A teacher ensemble of the triplet members labelled what in paper333? | Soundscape, training, and prior-year windows. |
| What's the bibliography size for paper501? | 17 references cited. |
| According to paper160's official EXIST 2026 leaderboard results, what rank and ICM-Hard-Norm score did the NeverChorizoInMyPaella team achieve on Task 3.1? | Rank 38th, with an ICM-Hard Norm of 0.5662. |
| Which team ranked higher in the official EXIST 2026 leaderboard for Task 3.1 hard-hard, the team for paper160 or the team for paper163? | Team ELiRF-UPV of paper163 ranked higher at 1st place than Team NeverChorizoInMyPaella of paper160 at 38th place. |
| Which team ranked lower in the FinMMEval 2026 Task 2 leaderboard, the team for paper187 or the team for paper190? | HU_LLM_Fin in paper187 ranked lower. |
| Did the team in paper172 rank higher than the team in paper173 in the official EXIST 2026 leaderboard for Task 2.1 ALL? Compare the F1-YES score difference. | No, the team in paper172 ranked lower than the team in paper173. The F1-YES score difference was 0.0191. |
| Are the authors in paper241 and paper242 written by the same author? | Yes, they are both written by Anubhav. |
| What is the home address of paper6's lead author? | Insufficient evidence. |
| What is a agreement-aware ensemble? | No such term in the paper; the paper describes disagreement-aware ensemble. |
| How does BERT improves upon DeBERTa-style encoders? | It does not; DeBERTa improves upon BERT-style encoders. |
| What ensemble of the tuplet members labelled the soundscape, training, and prior-year windows in paper333. | Cannot answer; no tuplet members mentioned. |

# Results
I experimented different retriever weights, retrieval depth, and retrieval candidates compared to the default values of (0.5, 0.5), 6, and 20.

The CLEF RAG system performed well on most paper type questions. However, the CLEF RAG system performance did vary with the following questions:
- What's the bibliography size for paper501?
    - This question was a catalog-style question for the paper category questions, and more of a limitation of RAG in general.
- According to paper160's official EXIST 2026 leaderboard results, what rank and ICM-Hard-Norm score did the NeverChorizoInMyPaella team achieve on Task 3.1?
    - The CLEF RAG system either could not answer the question or stated the ICM-Hard score instead of the ICM-Hard-Norm score.
- A teacher ensemble of the triplet members labelled what in paper333?
    - The CLEF RAG system could not answer the questions under certain parameters.

The CLEF RAG system also did well on most sythesis type questions. Some observations:
- Are the authors in paper241 and paper242 written by the same author?
    - At certain parameters, the CLEF RAG system started also including "both papers list the email address", and "both papers list the ORCID ID...".
    - The CLEF RAG system never explicitly states yes they are written by the same author.
- Did the team in paper172 rank higher than the team in paper173 in the official EXIST 2026 leaderboard for Task 2.1 ALL? Compare the F1-YES score difference.
    - The CLEF RAG system was not able to answer for some parameters (I don't have enough evidence in the available corpus to answer this question.).
    - The calculation of the difference varied.
- Which team ranked higher in the official EXIST 2026 leaderboard for Task 3.1 hard-hard, the team for paper160 or the team for paper163?
    - The CLEF RAG system was not able to answer for some parameters (The answer hit its output limit. Try a narrower question.
    ).
    - The CLEF RAG system did not explicitly say which team ranked higher.
    - The CLEF RAG system also got the ranking wrong (Team ELiRF-UPV ranked 1st, not 3rd).
- Which team ranked lower in the FinMMEval 2026 Task 2 leaderboard, the team for paper187 or the team for paper190?
    - The CLEF RAG system did not explicitly say which team ranked lower.

Most notably, the CLEF RAG system performed poorly on all unsupported-type questions. Below summarizes what the CLEF RAG system answered across all parameters:
- What ensemble of the tuplet members labelled the soundscape, training, and prior-year windows in paper333.
    - The CLEF RAG system failed to differentiate "tuplet" vs "triplet", and answered using the definition for "triplet" members. Tuplet was never mentioned in the corpus.
- What is a agreement-aware ensemble?
    - The CLEF RAG system failed to differentiate "agreement-aware" vs "disagreement-aware", and answered using the definition for "disagreement-aware". "Agreement-aware" was never mentioned in the corpus.
- What is the home address of paper6's lead author?
    - The CLEF RAG system identified the author's university affiliation's address as the author's home address. 
- How does BERT improves upon DeBERTa-style encoders?
    - The CLEF RAG system did not explicitly say that BERT does not improve upon DeBERTa-style encoders. The CLEF RAG system did state that DeBERTa improves upon BERT-style encoders, which was what was mentioned in the corpus.

# Discussion
> 🚧 **Work in Progress:** This section is currently being updated.

# Appendix: Few-shot experiment in Ollama with Gemma3:12B to improve answering "What is a agreement-aware ensemble?"

```
Experiment 1:
>>> Answer my question using only information from the evidence.
... Evidence: [evidences obtained from retrieval], 
... QUESTION: What is an agreement-aware ensemble?

According to the evidence, a disagreement-aware ensemble is a system that fuses four detectors with fundamentally different assumptions. It uses a
meta-learner that receives not just the individual scores of the detectors, but also their variance and range as explicit features, which encodes the
degree of inter-detector inconsistency.

>>> The evidence does not include any information about agreement-aware ensemble, just disagreement-aware ensemble. It is not the same thing. Do not hallucinate
...  the answer.

You are absolutely correct. My apologies for the error.

According to the evidence, a disagreement-aware ensemble is a system that fuses four detectors with fundamentally different assumptions. It uses a
meta-learner that receives not just the individual scores of the detectors, but also their variance and range as explicit features, encoding the degree of
inter-detector inconsistency.

Thank you for catching my mistake. I am still under development and learning to adhere strictly to the provided evidence.

>>> The evidence does not include any information about agreement-aware ensemble, just disagreement-aware ensemble. It is not the same thing. Do not hallucinate
...  the answer. If there is not enough information about agreement-aware emsemble, say "Not enough evidence".
You are absolutely correct to call me out again.

Not enough evidence.
```

```
>>> Answer my question using the only information provided from the evidence. Do not hallucinate the answer, especially if the evidence contains information sim
... ilar to the question. If there is something that is similar to the question in the evidence, do not explicitly answer. Instead, say "I did not find exactly
... what you are looking for, but this may help" before answering. If there is not enough information in the evidence, say "Not enough evidence".
...
... QUESTION: What is an agreement-aware ensemble?
...
... EVIDENCES:
... [evidences obtained from retrieval]
...
I did not find exactly what you are looking for, but this may help: A disagreement-aware ensemble is four detectors with fundamentally different
assumptions that are fused by a meta-learner that receives not just their individual scores but their variance and range as explicit features, encoding
the degree of inter-detector inconsistency.
```

```
>>> Answer my question using the only information provided from the evidence. Do not hallucinate the answer, especially if the evidence contains information sim
... ilar to the question. If there is something that is similar to the question in the evidence, do not explicitly answer. Instead, say "I did not find exactly
... what you are looking for, but this may help" before answering. Include the exact term in the question and the term you found that may be relevant before ans
... wering. If there is not enough information in the evidence, say "Not enough evidence".
...
... QUESTION: What is an agreement-aware ensemble?
...
... EVIDENCES:
... [evidences obtained from retrieval]
...
The question asks for a definition of **agreement-aware ensemble**.

I did not find exactly what you are looking for, but this may help. According to the evidence, an **agreement-aware ensemble** is "four detectors with
fundamentally different assumptions are fused by a meta-learner that receives not just their individual scores but their variance and range as explicit
features, encoding the degree of inter-detector inconsistency."
```

```
(base) PS C:\Users\david> ollama run gemma3:12b
>>> Answer my question using the only information provided from the evidence. Do not hallucinate the answer, especially if the evidence contains information sim
... ilar to the question. If there is something that is similar to the question in the evidence, do not explicitly answer. Instead, say "I did not find exactly
... what you are looking for, but this may help" before answering. Include the exact term in the question and the term you found that may be relevant before ans
... wering. Do not replace the exact term with the term found in the evidence. If there is not enough information in the evidence, say "Not enough evidence".
...
... QUESTION: What is an agreement-aware ensemble?
...
... EVIDENCES:
... [evidences obtained from retrieval]
...
I did not find exactly what you are looking for, but this may help.

**What is an agreement-aware ensemble?** The evidence describes an **agreement-aware ensemble** as "four detectors with fundamentally different
assumptions are fused by a meta-learner that receives not just their individual scores but their variance and range as explicit features, encoding the
degree of inter-detector inconsistency."

>>> You replaced the term
You are absolutely right to call me out. I apologize for that error. I am still under development and learning to follow instructions precisely.

Here's the response again, adhering strictly to the guidelines:

QUESTION: What is an agreement-aware ensemble?

EVIDENCES:
[evidences obtained from retrieval]
```

```
>>> Answer my question using the only information provided from the evidence. Do not hallucinate the answer, especially if the evidence contains information sim
... ilar to the question. If there is something that is similar to the question in the evidence, do not explicitly answer. Instead, say "I did not find exactly
... what you are looking for, but this may help" before answering. Include the exact term in the question and the term you found that may be relevant before ans
... wering. Do not replace the exact term with the term found in the evidence when you answer. If there is not enough information in the evidence, say "Not enou
... gh evidence".
...
... QUESTION: What is an agreement-aware ensemble?
...
... EVIDENCES:
... [evidences obtained from retrieval]
...
You asked about what an **agreement-aware ensemble** is. I did not find exactly what you are looking for, but this may help: An **agreement-aware
ensemble** is four detectors with fundamentally different assumptions that are fused by a meta-learner that receives not just their individual scores but
their variance and range as explicit features, encoding the degree of inter-detector inconsistency.

>>> You just hallucinated with disagreement-aware ensemble
You are absolutely right. My apologies. I am still under development and learning to adhere strictly to the provided evidence.

Not enough evidence.
```

```
>>> Answer my question using the only information provided from the evidence. Do not hallucinate the answer, especially if the evidence contains information sim
... ilar to the question. If there is something that is similar to the question in the evidence, do not explicitly answer. Instead, say "I did not find exactly
... what you are looking for, but this may help" before answering. When answering, do not hallucinate. If there is not enough information in the evidence, say "
... Not enough evidence".
...
... QUESTION: What is an agreement-aware ensemble?
...
... EVIDENCES:
... [evidences obtained from retrieval]
...
I did not find exactly what you are looking for, but this may help: A disagreement-aware ensemble is four detectors with fundamentally different
assumptions that are fused by a meta-learner that receives not just their individual scores but their variance and range as explicit features, encoding
the degree of inter-detector inconsistency.
```

```
>>> Answer the question using ONLY the provided evidence. Follow these rules strictly:
...
... 1. EXACT MATCH: If the evidence directly defines or answers the exact concept in the QUESTION, provide a direct answer.
...
... 2. PARTIAL / SIMILAR MATCH: If the evidence does not contain the exact concept, but contains a highly similar or related term:
...    - State EXACTLY: "I did not find [Target Concept], but I found information on [Found Concept]: [Brief Explanation from Evidence]"
...    - Do NOT redefine or change the found term to match the question.
...    - Do NOT invent or extrapolate any details.
...
... 3. NO MATCH: If there is no relevant or similar information in the evidence, respond ONLY with: "Not enough evidence."
...
... QUESTION: What is an agreement-aware ensemble?
...
... EVIDENCES:
... [evidences obtained from retrieval]

An agreement-aware ensemble is four detectors with fundamentally different assumptions are fused by a meta-learner that receives not just their individual
scores but their variance and range as explicit features, encoding the degree of inter-detector inconsistency.
```

```
>>> You are a strict factual extraction system. You must answer questions based ONLY on the provided EVIDENCES.
...
... ### RULES:
... 1. Verify if the EXACT term in the question exists in the evidence.
... 2. If the EXACT term is NOT in the evidence, but an opposite/similar term IS in the evidence, you MUST NOT substitute the question's term into the definitio
... n.
... 3. Follow the output format strictly as shown in the examples.
...
... ### EXAMPLES:
...
... QUESTION: What is a supervised neural network?
... EVIDENCES: "This paper introduces an unsupervised neural network that clusters unlabeled data."
... ANSWER: I did not find information on supervised neural network, but I found information on unsupervised neural network: A neural network that clusters unla
... beled data.
...
... QUESTION: What is a linear regressor?
... EVIDENCES: "We evaluate decision trees on tabular datasets."
... ANSWER: Not enough evidence.
...
... QUESTION: What is a transformer model?
... EVIDENCES: "A transformer model is an architecture relying on self-attention mechanisms."
... ANSWER: A transformer model is an architecture relying on self-attention mechanisms.
...
... ---
...
... ### TASK:
...
... QUESTION: What is an agreement-aware ensemble?
...
... EVIDENCES:
... [evidences obtained from retrieval]
...
... ANSWER:
...
A disagreement-aware ensemble: four detectors with fundamentally different assumptions are fused by a meta-learner that receives not just their individual
scores but their varianceandrangeas explicit features, encoding the degree of inter-detector inconsistency.
```

```
>>> You are a strict factual extraction system. You must process the input in two steps:
... 1. FIRST, analyze whether the requested term exists in the text.
... 2. SECOND, generate the final output following the exact format.
...
... ### EXAMPLES:
...
... QUESTION: What is a supervised neural network?
... EVIDENCES: "This paper introduces an unsupervised neural network that clusters unlabeled data."
... ANALYSIS: Target term "supervised neural network" is NOT in evidence. Similar term "unsupervised neural network" IS in evidence.
... ANSWER: I did not find information on supervised neural network, but I found information on unsupervised neural network: A neural network that clusters unla
... beled data.
...
... QUESTION: What is a linear regressor?
... EVIDENCES: "We evaluate decision trees on tabular datasets."
... ANALYSIS: Target term "linear regressor" is NOT in evidence. No similar terms found.
... ANSWER: Not enough evidence.
...
... ---
...
... ### TASK:
...
... QUESTION: What is an agreement-aware ensemble?
...
... EVIDENCES:
... [evidences obtained from retrieval]
...
... ANALYSIS:
...
Target term "agreement-aware ensemble" IS in evidence.
ANSWER: An agreement-aware ensemble: four detectors with fundamentally different assumptions are fused by a meta-learner that receives not just their
individual scores but their variance and range as explicit features, encoding the degree of inter-detector inconsistency.
```

```
>>> Answer the following question based on the evidence provided. If the question cannot be answered using the evidence only, respond with "Not enough evidence.".
...
... QUESTION: What is an agreement-aware ensemble?
...
... EVIDENCE:
...
... ". When a test text has been paraphrased, style-shifted, or rewritten by a second model, surface signatures vanish and most existing detectors fail catastrophically. This paper is built around one observation:obfuscation breaks different detector families in different ways.A
... paraphrase that erases the vocabulary fingerprint of a TF-IDF classifier rarely also removes the structural regularity that stylometric features measure. A style-transfer attack that defeats sentence-level statistics rarely also cancels the cross-model perplexity anomaly meas
... ured by Binoculars. When an AI text is successfully obfuscated against one detector, the disagreement among detectors increases— and that disagreement itself is a detection signal. We operationalise this as adisagreement-aware ensemble: four detectors with fundamentally diffe
... rent assumptions are fused by a meta-learner that receives not just their individual scores but their varianceandrangeas explicit features, encoding the degree of inter-detector inconsistency. The PAN@CLEF 2026 Voight-Kampff task [9, 10] frames AI detection as soft classifica
... tion: given a single text, output a score𝑠∈[0,1] where 0.0 signals human authorship, 1.0 signals AI generation, and 0.5 is an explicit abstention for uncertain cases"
According to the evidence, a disagreement-aware ensemble is "four detectors with fundamentally different assumptions are fused by a meta-learner that receives not just their individual scores but their variance and range as explicit features, encoding the degree of
inter-detector inconsistency."

>>> The evidence did not mention agreement-aware ensemble.
You are absolutely correct. My apologies. I misread the text and responded based on a similar phrase.

The correct response is: **Not enough evidence.**
```

```
>>> Answer the following question based on the evidence provided. If the question cannot be answered using the evidence only, respond with "Not enough evidence.".
...
... QUESTION: What is an agreement-aware ensemble?
...
... EVIDENCE:
...
... "We operationalise this as a disagreement-aware ensemble: four detectors with fundamentally different assumptions are fused by a meta-learner that receives not just their individual scores but their variance and range as explicit features, encoding the degree of inter-detecto
... r inconsistency."
An agreement-aware ensemble is a system where four detectors with different assumptions are combined using a "meta-learner." This meta-learner utilizes not only the detectors' scores but also their variance and range to understand the level of inconsistency between them.

>>> The evidence did not mention agreement-aware ensemble.
You are absolutely correct. My apologies. The evidence describes a "disagreement-aware ensemble," not an "agreement-aware ensemble."

Therefore, the answer is: **Not enough evidence.**
```
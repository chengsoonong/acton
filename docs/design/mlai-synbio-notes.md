# Alignment between MLAI and SynBio

Refer to the cycles, summarised in mlai-synbio.pdf

- DBTL cycle of SynBio FSP
- Predict, Recommend, Label cycle in machine learning

https://github.com/chengsoonong/acton/blob/master/docs/design/acton.pdf



## MLAI

Two types of machine learning / artificial intelligence problems:

1. Prediction: given an example (DNA sequence) predict the label (gene expression)
2. Recommendation: Choose an example (DNA sequence) to label (Build and Test)


## Common vision

| MLAI | SynBio |
| ---- | ------ |
| prediction | Learn |
| recommend | Design |
| - | Build |
| label | Test |

#### prediction = learn

- standard MLAI methods (supervised learning)
- research on data representation
  - DNA, RNA sequence
  - time series, spectrum
  - protein interaction graph

#### recommender

- choose where to measure (e.g. which RBS sequence to experiment)
- need to define the goal of the experiment (e.g. maximise GFP gene expression)
- practical constraints of Build stage can be taken into account
- long term: understand causal effects

#### labeller

- Build + Test
- how to combine different experimental results?
- data and interface management
- feed into learning

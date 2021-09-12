# Oscer Engineer Team Interview - Project Test 
Machine Learning Engineer

## Intro
The purpose of this task is to demonstrate your ability to solve an praticle problem, design and implement a high performance model.
The test has separated to two tests. You can choose either of them. 

1. The first test is a NER(Named-entity recognition) problem which asks you to design, implement a model to detect the entities from the texts.
2. The second test is a RE(Relation Extraction) problem which asks you to design, implement a model to detect the relations between two entities from the texts.


## NER Problem (Only if you choose NER)

### Data
All data is in /data/NERdata/
The entities are tagged in BIO format

test.tsv: dataset to evaluate your model
train.tsv: dataset to train your model

## RE Problem (Only if you choose RE)

### Data
Under /data/REdata/
The relationship is tagged as 0 or 1.

test.tsv: dataset to evaluate your model
train.tsv: dataset to train your model

## Submission

Put all your work on Github in the following file structure.
- /models
  - /model.py # main class of your model
  - /train.py # script to train the model
  - /test.py # script to run model on test set.
  - /pipeline.py # a pipeline to use your trained model to detect entities(NER)/relations(RE). 
    - Input: str a sentence
    - Output: 
      - List: a list of entities if you are doing NER
      - Bool: if you are doing RE.
  - /other support files
- readme.md 

Finally, Send the github link to yu@oscer.ai. 

## Thank you
Thanks for your hard work and complete this test, we appreciate it. Looking forward to meeting you soon!
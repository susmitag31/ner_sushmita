import spacy
import sys
from config import output_dir, new_model_name

nlp = spacy.load(output_dir + "//" + new_model_name)


if __name__ == "__main__":
    text = sys.argv[1]
    ner = nlp(text)
    print([i.text for i in ner.ents])



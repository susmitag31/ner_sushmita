import pandas as pd
from model import train_ner
from config import *

df = pd.read_csv(data_path + "train.tsv", delimiter ="\t", names = ['words', "BIO"])

#prepare data to model format
def prepare_BIO(BIO_tag):
    if BIO_tag == "B":
        return "B-Per"
    elif BIO_tag == "I":
        return "I-Per"
    else:
        return BIO_tag    
df['BIO'] = df["BIO"].map(lambda s : prepare_BIO(s))

df.to_csv(train_data, sep = '\t', columns = None, index=False)

train_data = "D://ner_sushmita//oscer_project_test//data//preprocessed_data.tsv"
train_data = spacy.convert(train_data, ner)


if __name__ == "__main__":
    train_ner(model, new_model_name, output_dir, n_iter, train_data)
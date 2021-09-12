import spacy 
from config import output_dir, new_model_name, data_path
from train import prepare_BIO

test_df = pd.read_csv(data_path + "test.tsv", delimiter ="\t", names = ['words', "BIO"])


if __name__ == "__main__":
    # test the saved model
    print("Loading from", output_dir)
    nlp2 = spacy.load(output_dir + "//" + new_model_name)
    assert nlp2.get_pipe("ner").move_names == move_names
    for i in test_df['words']:
        doc2 = nlp2(i)
        for ent in doc2.ents:
            print(ent.text)
    
    
    
    
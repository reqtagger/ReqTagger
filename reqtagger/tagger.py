from spacy.cli.download import download as spacy_download
from reqtagger import ReqTagger
import spacy


def load_spacy(model_name):
    try:
        model = spacy.load(model_name)
    except OSError:
        print(f"Spacy models '{model_name}' not found.  Downloading and installing.")
        spacy_download(model_name)
        model = spacy.load(model_name)
    return model

def main():
    print("Loading SpaCy models...")
    spacy_nlp = load_spacy('en_core_web_md')
    req = ReqTagger(spacy_nlp)
    
    while True:
        cq = input("Please provide a requirement to be parsed: ")
        print(req.extract(cq))


main()
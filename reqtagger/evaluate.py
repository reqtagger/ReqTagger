from spacy.cli.download import download as spacy_download
import spacy
import re
import json
from eval_data import cqs, sents
from reqtagger import ReqTagger


def load_spacy(model_name):
    try:
        model = spacy.load(model_name)
    except OSError:
        print(f"Spacy models '{model_name}' not found.  Downloading and installing.")
        spacy_download(model_name)
        model = spacy.load(model_name)
    return model


spacy_nlp = load_spacy('en_core_web_md')

for scenario in [{"universal_dependencies": True, "cq2sparqlowl": True}, {"universal_dependencies": True, "cq2sparqlowl": False}, {"universal_dependencies": False, "cq2sparqlowl": False}]:
    req = ReqTagger(spacy_nlp, scenario['universal_dependencies'], scenario['cq2sparqlowl'])
    for dataset, name in [(cqs, "CQs"), (sents, "Sentences")]:
        result = {}
        filename = f"UniversalDeps:{scenario['universal_dependencies']}_cq2sparqlowlRules:{scenario['cq2sparqlowl']}_{name}.json"
        with open(filename, 'w') as fout:
            TOTAL_EC = 0
            TOTAL_PC = 0

            TP_EC = 0
            FP_EC = 0
            FN_EC = 0

            TP_PC = 0
            FP_PC = 0
            FN_PC = 0

            for ontology in dataset:
                for (cq, ecs, pcs) in dataset[ontology]:
                    out = req.extract(cq)
                    out_ents = out['entities']
                    out_preds = out['relations']

                    ecs = ReqTagger.filter_determiners(ecs)
                    pcs = ReqTagger.filter_auxilaries(pcs)
                    TP_EC += len(set(ecs) & set(out_ents))
                    FP_EC += len(set(out_ents) - set(ecs))
                    FN_EC += len(set(ecs) - set(out_ents))


                    TP_PC += len(set(pcs) & set(out_preds))
                    FP_PC += len(set(out_preds) - set(pcs))
                    FN_PC += len(set(pcs) - set(out_preds))

                    result[cq] = {
                        "Entities": {"FP": sorted(set(out_ents) - set(ecs)), "FN": sorted(set(ecs) - set(out_ents))},
                        "Predicates": {"FP": sorted(set(out_preds) - set(pcs)), "FN": sorted(set(pcs) - set(out_preds))}
                    }

                    TOTAL_EC += len(set(ecs))
                    TOTAL_PC += len(set(pcs))

            P_EC = TP_EC / (TP_EC + FP_EC)
            R_EC = TP_EC / (TP_EC + FN_EC)

            P_PC = TP_PC / (TP_PC + FP_PC)
            R_PC = TP_PC / (TP_PC + FN_PC)
            print(f"Scenario [UniversalDeps: {scenario['universal_dependencies']}, cq2sparqlowlRules: {scenario['cq2sparqlowl']}] Processing {name}: Total EC count: {TOTAL_EC}, total PC count: {TOTAL_PC}, \nEC Prec: {P_EC}, Recall: {R_EC} F1: {2*P_EC*R_EC/(P_EC+R_EC)} \nPC Prec: {P_PC}, Recall: {R_PC}  F1: {2*P_PC*R_PC/(P_PC+R_PC)}")
            fout.write(json.dumps(result, sort_keys=True, indent=4))
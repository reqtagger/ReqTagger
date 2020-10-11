import spacy
import logging
import re
import json


class ReqTagger:
    NON_RELATION_THING = ['is', '’s', 'are', 'was', 'do', 'does', 'did', 'were',
                'have', 'had', 'can', 'could', 'regarding',
                'is of', 'are of', 'are in', 'given', 'is there', 'has']

    NON_ENTITY_THINGS =  ['a kind', 'the kind', 'kind', 'kinds', 'the kinds',
                'category', 'a category', 'the category', 'categories', 'the categories',
                'type', 'a type', 'the type', 'types', 'the types']

    RULES_CQ2SPARQLOWL_RELATIONS = [
        ['{0+}PART|VERB', 'VERB'],
        ['{1+}PART|VERB', '{1+}ADJ|ADV', 'ADP'],
        ['{1+}PART|VERB', 'ADP']
    ]

    RULES_UNIVERSAL_RELATIONS = [
        ['{0+}PART|VERB|AUX', 'VERB'],
        ['{0+}VERB', 'ADV|AUX'],
        ['{0+}PART|VERB|AUX', '{1+}AUX|VERB|ADJ|ADV', 'ADP|SCONJ']
    ]

    RULES_ONTONOTES_RELATIONS = [
        ['{1+}JJ', 'IN'],
        ['JJR', 'IN'],
        ['RB', 'VBD'],
        ['{0+}MD', '{1+}VBZ|VBN', 'TO', 'VB'],
        ['{0+}MD', 'VB', 'VBN', 'IN'],
        ['{1?}TO', 'VB|VBN|VBZ|VBP', '{1?}JJ', 'IN'],
        ['{0+}VB|MD', '{0+}TO', 'VB'],
        ['{1?}TO', '{0+}VB|VBP|VBZ|VBN|VBD|VBG', '{1+}RBS|VB|VBZ|VBN|VBD|VBG|IN', 'RBS|IN'],
        ['{1?}TO', 'VB|VBZ|VBP|VBG|VBD', 'VB|VBN|VBG|VBG|VBD|JJR|RP'],
        ['VBN', 'VBG']
    ]

    RULES_UNIVERSAL_ENTITIES = [
        ['{0+}ADJ', '{1+}NOUN|PROPN'],
        ['{0+}ADJ', '{1+}NOUN|PROPN', '{0+}ADJ', '{1+}NOUN|PROPN'],
        ['VERB', 'NOUN|PROPN']
    ]

    RULES_ONTONOTES_ENTITIES = [
        ['{1?}DT', '{0+}FW|JJ|JJS|NN|NNS|NNP', 'NN|NNP|NNS'],
        ['DT', 'VB|VBG|VBD|VBZ|VBN', '{0+}NN|NNS|NNP|JJ', 'NN|NNS|NNP']
    ]

    def __init__(self, nlp, use_ud=False, use_cq2sparqlowlrules=False):
        self.nlp = nlp
        self.use_ud = use_ud
        self.use_cq2sparqlowlrules = use_cq2sparqlowlrules

    @classmethod
    def filter_determiners(cls, outs):
        result = []

        for obj in outs:
            result.append(re.sub(r'^(A|a|An|an|The|the) ', '', obj))
        return result

    @classmethod
    def filter_auxilaries(cls, outs):
        result = []

        for obj in outs:
            result.append(re.sub(r'^([Bb]e|[Aa]m|[Aa]re|[Ii]s|[Ww]as|[Ww]ill|[Ww]ould|[Ss]hall|[Ss]hould|[Cc]an|[Mm]ight|[Mm]ay|[Mm]ust|[Cc]ould|[Dd]o|[Dd]oes|[Dd]id)( be| been)? ', '', obj))
        return result

    def filter_subspans(self, spans):
        filtered = []

        for span in spans:
            accept = True
            for compared in spans:
                if span[0] >= compared[0] and span[1] <= compared[1]:
                    if span[0] != compared[0] or span[1] != compared[1]:
                        accept = False
            if accept:
                filtered.append(span)

        filtered = list(dict.fromkeys(filtered))  # remove duplicates if present
        return filtered

    def parse_rule(self, rule):
        return r''.join(self.parse_item(item) for item in rule)

    def parse_item(self, item):
        postfix = ""
        if item.startswith("{"):
            if item.startswith('{0+}'):
                postfix = '*'
            elif item.startswith('{1+}'):
                postfix = '+'
            elif item.startswith('{1?}'):
                postfix = '?'
            item = item[4:]
        return rf"([0-9]+::({item}),?){postfix}"

    def mark_relations(self, cq):
        cq = cq.lower()
        doc = self.nlp(cq)

        if self.use_ud:
            pos_text = ",".join(
                ["{i}::{pos}".format(i=i, pos=t.pos_) for i, t in enumerate(doc)])
            if self.use_cq2sparqlowlrules:
                rules = ReqTagger.RULES_CQ2SPARQLOWL_RELATIONS
            else:
                rules = ReqTagger.RULES_UNIVERSAL_RELATIONS
        else:
            pos_text = ",".join(
                ["{i}::{pos}".format(i=i, pos=t.tag_) for i, t in enumerate(doc)])
            rules = ReqTagger.RULES_ONTONOTES_RELATIONS

        spans = []  # list of beginnings and endings of each chunk
        for rule in rules:  # try to extract chunks
            rule = self.parse_rule(rule)
            for m in re.finditer(rule, pos_text):
                id_tags = [elem for elem in m.group().split(",") if elem != '']
                ids = [int(id_tag.split("::")[0]) for id_tag in id_tags]
                try:
                    span = (doc[ids[0]].idx, doc[ids[-1]].idx + len(doc[ids[-1]]))
                except:
                    print("IDS")
                    print(ids, m.group())
                    exit()
                if cq[span[0]:span[1]] not in self.NON_RELATION_THING:
                    spans.append(span)

        return self.filter_subspans(spans)

    def mark_entities(self, cq):
      cq = cq.lower()
      doc = self.nlp(cq)

      if self.use_cq2sparqlowlrules: # in an old scenario noun chunks are considered
          spans = self._mark_noun_phrases(doc)
      else:
          spans = self._mark_entity_spans(doc, cq)

      return self.filter_subspans(spans)


    def _mark_noun_phrases(self, doc):
        spans = []
        for nc_span in doc.noun_chunks:
            if nc_span.text.lower() not in self.NON_ENTITY_THINGS:
                spans.append((nc_span.start_char, nc_span.end_char))
        return spans

    def _mark_entity_spans(self, doc, cq):
        spans = []  # list of beginnings and endings of each chunk

        if self.use_ud:
            rules = ReqTagger.RULES_UNIVERSAL_ENTITIES
            pos_text = ",".join(
                ["{i}::{pos}".format(i=i, pos=t.pos_) for i, t in enumerate(doc)])
        else:
            rules = ReqTagger.RULES_ONTONOTES_ENTITIES
            pos_text = ",".join(
                ["{i}::{pos}".format(i=i, pos=t.tag_) for i, t in enumerate(doc)])

        for rule in rules:  # try to extract chunks
            rule = self.parse_rule(rule)
            for m in re.finditer(rule, pos_text):
                id_tags = [elem for elem in m.group().split(",") if elem != '']
                ids = [int(id_tag.split("::")[0]) for id_tag in id_tags]
                span = (doc[ids[0]].idx, doc[ids[-1]].idx + len(doc[ids[-1]]))
                if cq[span[0]:span[1]] not in self.NON_ENTITY_THINGS:
                    spans.append(span)
        return self.filter_subspans(spans)

    def extract(self, cq):
        entity_spans = self.mark_entities(cq)
        entities = []
        for begin, end in entity_spans:
            entities.append(cq[begin:end])
        
        relation_spans = self.mark_relations(cq)
        relations = []
        for begin, end in relation_spans:
            relations.append(cq[begin:end])

        return {
            "entities": ReqTagger.filter_determiners(entities),
            "relations": ReqTagger.filter_auxilaries(relations)
        }

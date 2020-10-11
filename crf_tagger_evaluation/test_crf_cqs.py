from itertools import chain
import nltk
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import sklearn
import pycrfsuite
import spacy
import re
nlp = spacy.load('en_core_web_md')

feature_names = [
    'index',
    'text',
    'pos',
    'dep',
    'begin',
    'end',
    'index_2',
    'is_upper',
    'is_lower',
    'head_text',
    'head_pos',
    'head_dep',
    'is_aux',  # ADD   
    'num_verbs',  # ADD
    'entity',
    'relation'
]

def _parse_data(path, names):
    result = []
    for doc in open(path).read().split("\n\n"):
        doc_repr = []
        tokens = doc.split("\n")
        for token in tokens:
            values = token.strip().split(" ")
            if len(values) == 0:
                continue
            doc_repr.append(dict(zip(names, values)))
        result.append(doc_repr)
    return result


train_data = _parse_data('./data/merged.conll', feature_names)
test_data = _parse_data('./data/merged.conll', feature_names)


class CRFTagger:
    feature_names = [
        'index',
        'text',
        'pos',
        'dep',
        'begin',
        'end',
        'index_2',
        'is_upper',
        'is_lower',
        'head_text',
        'head_pos',
        'head_dep',
        'is_aux',  # ADD   
        'num_verbs',  # ADD
        'entity',
        'relation'
    ]

    AUXILARIES = ['do', 'does', 'did', 'will', 'is', 'was', 'can', 'have',
                  'has', 'had', 'would', 'should', 'shall', 'could']


    def __init__(self, model_name='model.crfsuite'):
        self.model_name = model_name
        #self.tagger = pycrfsuite.Tagger()
        #self.tagger.open('x.crfsuite')

    def _is_auxilary(self, token):
        if token.text.lower() in CRFTagger.AUXILARIES:
            return "1"
        else:
            return "0"

    def _number_of_verbs(self, tokens):
        count = 0

        for token in tokens:
            if 'VB' in token.pos_:
                count += 1

        return str(count)

    def conllize(self, text):
        doc = nlp(text)
        conll_lines = []
        for token in doc:
            conll_line = [str(token.i), token.text, token.pos_, token.dep_, str(token.idx),
                 str(token.idx + len(token)), str(token.i), str(token.is_upper),
                 str(token.is_lower), token.head.text, token.head.pos_,
                 token.head.dep_,
                 self._is_auxilary(token), self._number_of_verbs(doc), "0", "O"]
            conll_lines.append(dict(zip(CRFTagger.feature_names, conll_line)))
        return conll_lines

    def word2features(self, sent, i):
        word = sent[i]
        features = [
            'word.lower=' + word['text'].lower(),  # word.lower(),
            'word.isaux=' + word['is_aux'],
            'word.num_verbs=' + word['num_verbs'],
            'word[-3:]=' + word['text'][-3:],
            # 'word[-2:]=' + word['text'][-2:],
            'word.isupper=%s' % word['text'].isupper(),
            'word.istitle=%s' % word['text'].istitle(),
            'word.isdigit=%s' % word['text'].isdigit(),
            'postag=' + word['pos'],
            'postag[:2]=' + word['pos'][:2],
            'dep=' + word['dep'],
        # 'head_dep=' + word['head_dep'],
        ]
        if i > 0:
            word1 = sent[i - 1]
            features.extend([
                '-1:word.lower=' + word1['text'].lower(),
                '-1:word.istitle=%s' % word1['text'].istitle(),
                '-1:word.isupper=%s' % word1['text'].isupper(),
                '-1:postag=' + word1['pos'],
                '-1:postag[:2]=' + word1['pos'][:2],
                '-1:dep=' + word1['dep'],
                 '-1:head_dep=' + word1['head_dep'],
            ])
        if i > 1:
            word1 = sent[i - 2]
            features.extend([
                '-2:word.lower=' + word1['text'].lower(),
                '-2:word.istitle=%s' % word1['text'].istitle(),
                '-2:word.isupper=%s' % word1['text'].isupper(),
                '-2:postag=' + word1['pos'],
                '-2:postag[:2]=' + word1['pos'][:2],
                '-2:dep=' + word1['dep'],
                '-2:head_dep=' + word1['head_dep'],
            ])
        else:
            features.append('BOS')

        if i < len(sent) - 1:
            word1 = sent[i + 1]
            features.extend([
                '+1:word.lower=' + word1['text'].lower(),
                '+1:word.istitle=%s' % word1['text'].istitle(),
                '+1:word.isupper=%s' % word1['text'].isupper(),
                '+1:postag=' + word1['pos'],
                '+1:postag[:2]=' + word1['pos'][:2],
                '+1:dep=' + word1['dep'],
                '+1:head_dep=' + word1['head_dep'],
            ])

        if i < len(sent) - 2:
            word1 = sent[i + 2]
            features.extend([
                '+2:word.lower=' + word1['text'].lower(),
                '+2:word.istitle=%s' % word1['text'].istitle(),
                '+2:word.isupper=%s' % word1['text'].isupper(),
                '+2:postag=' + word1['pos'],
                '+2:postag[:2]=' + word1['pos'][:2],
                '+2:dep=' + word1['dep'],
                '+2:head_dep=' + word1['head_dep'],
            ])
        else:
            features.append('EOS')

        return features


    def sent2features(self, sent):
        return [self.word2features(sent, i) for i in range(len(sent))]


    def sent2tokens(self, sent):
        return [w['text'] for w in sent]

    def load_tagger(self):
        self.tagger = pycrfsuite.Tagger()
        self.tagger.open(self.model_name)

    def tag(self, cq):
        cq_conll = self.conllize(cq)
        tags = self.tagger.tag(self.sent2features(cq_conll))
        matched = []
        
        current_text = ""
        for token, tag in zip(cq_conll, tags):
            if tag.startswith("B"):
                if len(current_text) > 0:
                    matched.append(current_text)
                current_text = token['text']
            elif tag.startswith("I"):
                if len(current_text) == 0:
                    current_text = token['text']
                else:
                    current_text = current_text + " " + token['text']
            elif tag.startswith("O"):
                if len(current_text) > 0:
                    matched.append(current_text)
                    current_text = ""
        if len(current_text) > 0:
            matched.append(current_text)

        return matched

        #print("Predicted:", ' '.join(self.tagger.tag(self.sent2features(cq_conll))))

    def sent2labels(self, sent, ent=True):
        label = 'entity' if ent else 'relation'
        return [w[label] for w in sent]



    def train(self, train_data, test_data, entity=True):
        X_train = [self.sent2features(s) for s in train_data]
        y_train = [self.sent2labels(s, entity) for s in train_data]

        X_test = [self.sent2features(s) for s in test_data]
        y_test = [self.sent2labels(s, entity) for s in test_data]


        trainer = pycrfsuite.Trainer(verbose=False)

        for xseq, yseq in zip(X_train, y_train):
            trainer.append(xseq, yseq)

        trainer.set_params({
            'c1': 1.0,   # coefficient for L1 penalty
            'c2': 1e-3,  # coefficient for L2 penalty
            'max_iterations': 100,  # stop earlier

            # include transitions that are possible, but not observed
            'feature.possible_transitions': True
        })

        trainer.train(self.model_name)

cqs = {
    "Vicinity": [
        ("What is an IoT device?", ['IoT device'], []),
        ("What is a partnership?", ['partnership'], []),
        ("What attributes has a partnership?", ['attributes', 'partnership'], []),
        ("Which are the relationships a partnership is involved in ?", ['relationships', 'partnership'], ['is involved in']),
        ("How many organizations can have a partnership?", ['organizations', 'partnership'], ['can have']),
        ("What is the relation between organization and devices?", ['organization', 'relation', 'devices'], []),
        ("What is an IoT infrastructre?", ['IoT infrastructre'], []),
        ("Who is the owner of a given device?", ['owner', 'given device'], []),
        ("Which attributes can have a device?", ['attributes', 'device'], ['can have']),
        ("What is a device profile?", ['device profile'], []),
        ("Which are the social relationships a device can be involved in?", ['social relationships', 'device'], ['can be involved in']),
        ("Which roles are involved in a ownership relationship?", ['roles', 'ownership relationship'], ['are involved in']),
        ("Which roles are involved in a partnership relationship?", ['roles', 'partnership relationship'], ['are involved in']),
        ("What is a user?", ['user'], []),
        ("Who is a service provider?", ['service provider'], []),
        ("What are the parameters that has a service?", ['parameters', 'service'], []),
        ("What is a service logical name?", ['service logical name'], []),
        ("Which devices are there?", ['devices'], []),
        ("What are the devices of a given agent or organization?", ['devices', 'given agent', 'organization'], []),
        ("Which devices can I see?", ['devices'], ['see']),
        ("Which services can I see?", ['services'], ['see']),
        ("What are the devices of a specific partner?", ['devices', 'specific partner'], []),
        ("What are the services of a specific partner?", ['services', 'specific partner'], []),
        ("Which is the profile of a given device?", ['profile', 'given device'], []),
        ("What is a building?", ['building'], []),
        ("Where is something located?", ['something'], ['located']),
        ("Which devices measure temperature?", ['devices', 'temperature'], ['measure']),
        ("Which devices measure CO2?", ['devices', 'CO2'], ['measure']),
        ("Which devices measure noise?", ['devices', 'noise'], ['measure']),
        ("Which devices measure humidity?", ['devices', 'humidity'], ['measure']),
        ("What is a thing description?", ['thing description'], []),
        ("Which devices are located at a CERTH lab?", ['devices', 'CERTH lab'], ['are located at']),
        ("Which properties does a people counting observe?", ['properties', 'people counting'], ['observe']),
        ("Which properties does a humidity sensor observe?", ['properties', 'humidity sensor'], ['observe']),
        ("Which properties does a light switch observe?", ['properties', 'light switch'], ['observe']),
        ("Which properties does a motion sensor observe?", ['properties', 'motion sensor'], ['observe']),
        ("Which properties does a thermometer observe?", ['properties', 'thermometer'], ['observe']),
        ("Which properties does a CO2 sensor observe?", ['properties', 'CO2 sensor'], ['observe']),
        ("Which properties does a HVAC sensor observe?", ['properties', 'HVAC sensor'], ['observe']),
        ("Which devices are located at a Oslo SciencePark?", ['devices', 'Oslo SciencePark'], ['are located at']),
        ("Which devices are located at UNIKL?", ['devices', 'UNIKL'], ['are located at']),
        ("Which properties does an ebike charger observe?", ['properties', 'ebike charger'], ['observe']),
        ("Which properties does a light bulb observe?", ['properties', 'light bulb'], ['observe']),
        ("Which properties does a door sensor observe?", ['properties', 'door sensor'], ['observe']),
        ("Which properties does a window sensor observe?", ['properties', 'window sensor'], ['observe']),
        ("Which properties does a thermostat observe?", ['properties', 'thermostat'], ['observe']),
        ("Which devices are located at a CERTH lab?", ['devices', 'CERTH lab'], ['are located at']),
        ("Which properties does a weight scale observe?", ['properties', 'weight scale'], ['observe']),
        ("Which properties does a weight scale affect?", ['properties', 'weight scale'], ['affect']),
        ("Which properties from a weight scale are observed in events?", ['properties', 'weight scale', 'events'], ['are observed in']),
        ("Which properties does a blood pressure monitor observe?", ['properties', 'blood pressure monitor'], ['observe']),
        ("Which properties does a blood pressure monitor affect?", ['properties', 'blood pressure monitor'], ['affect']),
        ("Which properties from a blood pressure monitor are observed in events?", ['properties', 'blood pressure monitor', 'events'], ['are observed in']),
        ("Which properties does an activity tracker observe?", ['properties', 'activity tracker'],  ['observe']),
        ("Which properties does an activity tracker affect?", ['properties', 'activity tracker'], ['affect']),
        ("Which properties from an activity tracker are observed in events?", ['properties', 'activity tracker', 'events'], ['are observed in']),
        ("Which properties from a panic button are observed in events?", ['properties', 'panic button', 'events'], ['are observed in']),
        ("Which properties does a motion sensor observe?", ['properties', 'motion sensor'], ['observe']),
  ], 
  "VicinityWOT": [
      ("What is a thing in the web thing context?", ['thing', 'web thing context'], []),
      ("What is a servient?", ['servient'], []),
      ("What is a repository?", ['repository'], []),
      ("What is a property?", ['property'], []),
      ("What is an action?", ['action'], []),
      ("What is an event?", ['event'], []),
      ("What is a protocol?", ['protocol'], [])
  ],
  "BTN": [
      ("Which is the height of peak?", ["height", "peak"], []),
      ("Which are the geometries and maximum levels of the Curvas de Nivel Maestra of bathymetric type?", ["geometries", "maximum levels", "Curvas de Nivel Maestra", "bathymetric"], []),
      ("Which is the name and geometry of the primary interest mountain range?", ["name", "geometry", "primary interest mountain range"], []),
      ("Which are the rivers of tertiary interest at ground level?", ["rivers", "tertiary interest", "ground level"], []),
      ("Which are the names and geometry of the major canals at ground level?", ['names', 'geometry', 'major canals', 'ground level'], []),
      ("Which are the permanent regime lagoons?", ['permanent regime lagoons'], []),
      ("Which is the geometry of the wetlands of rambla type?", ['geometry', 'wetlands', 'rambla'], []),
      ("Which are the names of the seas that surround the Spanish territory?", ['names', 'seas', 'Spanish territory'], ['surround']),
      ("Which capitals from the population center has a population density higher than Madrid?", ['capitals', 'population center', 'population density', 'Madrid'], ['higher than']),
      ("Which are the names and geometry of the population center that has a population density higher than Madrid?", ['names', 'geometry', 'population center', 'population density', 'Madrid'], ['higher than']),
      ("What are the name of disseminated places that are not capitals of municipality?", ['name', 'disseminated places', 'capitals',  'municipality'], []),
      ("What are the name of  superficial disseminated places that are  capitals of municipality?", ["name", 'superficial disseminated places', 'capitals',  'municipality'], []),
      ("Which are the names of the free access motorways?", ['names', 'free access motorways'], []),
      ("Which are the names of the national highway which belong to the european highway network?", ['names', 'national highway', 'european highway network'], ['belong to']),
      ("Which are the names of the national highway which do not belong to the european highway network?", ['names', 'national highway', 'european highway network'], ['belong to']),
      ("Which are the names and geometry of the itinerary which belong to the Camino de Santiago?", ['names', 'geometry', 'itinerary', 'Camino de Santiago'], ['belong to']),
      ("Which are the FFCC stations that are stopping places and which are the FFCC stations that have passenger traffic?", ['FFCC stations', 'stopping places', 'FFCC stations', 'passenger traffic'], []),
      ("Which are the geometries of the cable transport?", ['geometries', 'cable transport'], []),
      ("Which are the sea ports that are not under state control?", ["sea ports", "state control"], ['are under']),
      ("Which are the heliport that are not under state control and do not belong to the trans-European transport network?", ['heliport', 'state control', 'trans-European transport network'], ['are under', 'belong to']), 
      ("Which are the oil pipelines at ground level?", ['oil pipelines', 'ground level'], []),
      ("Which are the low voltage electric lines?", ['low voltage electric lines'], []),
      ("Which are the nuclear power plants?", ['nuclear power plants'], []),
      ("Which are the names, altitudes and longitudes of the lower order geodesic vertex ?", ['names', 'altitudes', 'longitudes', 'lower order geodesic vertex'], []),
      ("Which roads cross a municipality ?", ['roads', 'municipality'], ['cross']),
      ("Which are the AVE lines that connect two towns?", ['AVE lines', 'towns'], ['connect']),
      ("which is the optimum way between two population centers?", ['optimum way', 'population centers'], []),
      ("Which are the towns though which the Camino de santiago passes?", ['towns', 'Camino de santiago'], ['passes']),
      ("Which are the initial point and the end point of the road?",  ['initial point', 'end point', 'road'], []),
      ("Which are the initial point and the end point of the railway?", ['initial point', 'end point', 'railway'], []),
      ("Which are the places of interest in a town ?", ['places', 'interest', 'town'], []),
      ("Which are the historic places in a town?", ['historic places', 'town'], []),
      ("Which churches are placed in a town?", ['churches', 'town'], ['are placed in']),
      ("Which are the monuments of cultural interest in a town?", ['monuments', 'cultural interest', 'town'], []),
      ("Which places of interest are in a province?", ['places', 'interest', 'province'], ['are in']),
      ("Which historic places are in a province?", ['historic places', 'province'], ['are in']),
      ("Which goods of cultural interest are in a province?", ['goods', 'cultural interest', 'province'], ['are in']),
      ("Which places of interest are in a county?", ['places', 'interest', 'county'], ['are in']),
      ("Which historic places are in a county?", ["historic places", "county"], ['are in']),
      ("Which goods of cultural interest are in a county?", ["goods", "cultural interest", "county"], ['are in']),
      ("Which beaches are placed in the province?", ['beaches', 'province'], ['are placed in']), 
      ("Which gas stations are placed in the municipality?", ['gas stations', 'municipality'], ['are placed in']),
      ("Which electric lines passes through a municipality?", ['electric lines', 'municipality'], ['passes through']),
      ("Which municipalities are connected through the electric line?", ['municipalities', 'electric line'], ['are connected through']),
      ("Which hospital attend to a particular municipality?", ['hospital', 'particular municipality'], ['attend to']),
      ("Which communication stations are in a particular municipality?", ['communication stations', 'particular municipality'], []),
      ("Which mines are in a particular province?", ['mines', 'particular province'], []),
      ("Which mines are within a particular distance in a particular municipality?", ['mines', 'particular distance', 'particular municipality'], []),
      ("What geodesic vertex are in a particular municipality?", ['geodesic vertex', 'particular municipality'], []),
      ("What REDNAP signals are in a municipality?", ['REDNAP signals', 'municipality'], []),
      ("Which is the altitude of the municipality?", ['altitude', 'municipality'], []),
      ("Which is the altitude of the peak or mountain?", ['altitude', 'peak', 'mountain'], []),
      ("Which are the NP placed along the river?", ['NP', 'river'], ['placed along']),
      ("Which are the cave type places of interest which are also good of cultural interest?", ['cave type places', 'interest', 'cultural interest'], ['are also good']),
      ("Which are the good of cultural interest of the national park?", ['good', 'cultural interest', 'national park'], []),
      ("Which are the railway stations of the province?", ['railway stations', 'province'], []),
      ("Which are the airports of the autonomous region?", ['airports', 'autonomous region'], []),
      ("Through which villages does the road goes through ?", ['villages', 'road'], ['goes through']), 
      ("Which are the National Parks?", ['National Parks'], []),
      ("Which are the natural places?", ['natural places'], []),
      ("Which are the Special bird protection area?", ['Special bird protection area'], []),
      ("Which river cross the National Park?", ['river', 'National Park'], ['cross']),
      ("Which municiaplities are located inside the PN", ['municiaplities', 'PN'], ['are located inside']),
      ("Which roads cross the PN?", ['roads', 'PN'], ['cross']),
      ("How many and which are the surface water located inside the PN?", ['surface water', 'PN'], ['located inside']),
      ("Which are the protected areas in Spain?", ['protected areas', 'Spain'], []),
      ("Which are the tributaries on the right of the river?", ['tributaries', 'right', 'river'], []),
      ("Where does the river flows into?", ['river'], ['flows into']),
      ("How many rivers flow into the sea or ocean zzz?", ['rivers', 'sea', 'ocean zzz'], ['flow into']),
      ("What are the rivers which belong to municipality?", ['rivers', 'municipality'], ['belong to']),
      ("Through which autonomous community does the river flow into ?", ['autonomous community', 'river'], ['flow into']),
      ("Through which provinces does the river flow into ?", ['provinces', 'river'], ['flow into']),
      ("Through which municipalities does the river flow into?", ['municipalities', 'river'], ['flow into']),
      ("Which town is adjacent to town?", ['town', 'town'], ['is adjacent to']),
      ("Which municipality belongs to province?", ['municipality', 'province'], ['belongs to']),
      ("Which municipality belongs to the autonomous community?", ['municipality', 'autonomous community'], ['belongs to']),
      ("What are the reservoirs of the river?", ['reservoirs', 'river'], []),
  ], 
  'SAREF': [
      ("What is a device?", ['device'], [])
  ],
  'SAREFBLDG': [
      ("What is a building?", ["building"], []),
      ("What is a shading device?", ["shading device"], []),
      ("Which properties has a shading device?", ["properties", "shading device"], []),
      ("What is an actuator?", ["actuator"], []),
      ("What is an alarm?", ["alarm"], []),
      ("What is a controller?", ["controller"], []),
      ("What is a flow instrument?", ["flow instrument"], []),
      ("What is a sensor?", ["sensor"], []),
      ("What is a unitary control element?", ["unitary control element"], []),
      ("Which properties has an actuator?", ["properties", "actuator"], []),
      ("What is a protective device tripping unit?", ["protective device tripping unit"], []),
      ("What is an audio visual appliance?", ["audio visual appliance"], []),
      ("What is a communication appliance?", ["communication appliance"], []),
      ("What is an electric appliance?", ["electric appliance"], []),
      ("What is an electric flow storage device?", ["electric flow storage device"], []),
      ("What is an electric generator?", ["electric generator"], []),
      ("What is an electric motor?", ["electric motor"], []),
      ("What is an electric time control?", ["electric time control"], []),
      ("What is a lamp?", ["lamp"], []),
      ("What is an outlet?", ["outlet"], []),
      ("What is a protective device?", ["protective device"], []),
      ("What is a solar device?", ["solar device"], []),
      ("What is a switching device?", ["switching device"], []),
      ("What is a transformer?", ["transformer"], []),
      ("Which properties has a protective device tripping unit?", ["properties", "protective device tripping unit"], []),
      ("Which properties has an audio visual appliance?", ["properties", 'audio visual appliance'], []),
      ("Which properties has an electric flow storage device?", ['properties', 'electric flow storage device'], []),
      ("Which properties has an electric generator?", ['properties', 'electric generator'], []),
      ("Which properties has an electric motor?", ['properties', 'electric motor'], []),
      ("Which properties has a lamp?", ['properties', 'lamp'], []),
      ("Which properties has an outlet?", ['properties', 'outlet'], []),
      ("Which properties has a switching device?", ['properties', 'switching device'], []),
      ("Which properties has a transformer?", ['properties', 'transformer'], []),
      ("What is an air to air heat recovery?", ['air', 'air heat recovery'], []),
      ("What is a burner?", ['burner'], []), 
      ("What is a chiller?", ['chiller'], []),
      ("What is a boiler?", ['boiler'], []),
      ("What is a coil?", ['coil'], []),
      ("What is a compressor?", ['compressor'], []),
      ("What is a condenser?", ['condenser'], []),
      ("What is a cooled beam?", ['cooled beam'], []),
      ("What is a cooling tower?", ['cooling tower'], []),
      ("What is a damper?", ['damper'], []),
      ("What is a duct silencer?", ['duct silencer'], []),
      ("What is an engine?", ['engine'], []),
      ("What is an evaporative cooler?", ['evaporative cooler'], []),
      ("What is an evaporator?", ['evaporator'], []), 
      ("What is a fan?", ['fan'], []),
      ("What is a filter?", ['filter'], []),
      ("What is a flow meter?", ['flow meter'], []),
      ("What is a heat exchanger?", ['heat exchanger'], []),
      ("What is a humidifier?", ['humidifier'], []),
      ("What is a medical device?", ['medical device'], []),
      ("What is a pump?", ['pump'], []),
      ("What is a space heater?", ['space heater'], []),
      ("What is a tank?", ['tank'], []),
      ("What is a tube bundle?", ['tube bundle'], []),
      ("What is a unitary equipment?", ['unitary equipment'], []),
      ("What is a valve?", ['valve'], []),
      ("What is a vibration isolator?", ['vibration isolator'], []),
      ("Which properties has an air to air heat recovery?", ['properties', 'air', 'air heat recovery'], []),
      ("Which properties has a burner?", ['properties', 'burner'], []),
      ("Which properties has a chiller?", ['properties', 'chiller'], []),
      ("Which properties has a boiler?", ['properties', 'boiler'], []),
      ("Which properties has a coil?", ['properties', 'coil'], []),
      ("Which properties has a compressor?", ['properties', 'compressor'], []),
      ("Which properties has a condenser?", ['properties', 'condenser'], []),
      ("Which properties has a cooled beam?", ['properties', 'cooled beam'], []),
      ("Which properties has a cooling tower?", ['properties', 'cooling tower'], []),
      ("Which properties has a damper?", ['properties', 'damper'], []),
      ("Which properties has a duct silencer?", ['properties', 'duct silencer'], []),
      ("Which properties has an engine?", ['properties', 'engine'], []),
      ("Which properties has an evaporative cooler?", ['properties', 'evaporative cooler'], []),
      ("Which properties has an evaporator?", ['properties', 'evaporator'], []),
      ("Which properties has a fan?", ['properties', 'fan'], []),
      ("Which properties has a filter?", ['properties', 'filter'], []),
      ("Which properties has a flow meter?", ['properties', 'flow meter'], []),
      ("Which properties has a heat exchanger?", ['properties', 'heat exchanger'], []),
      ("Which properties has a humidifier?", ['properties', 'humidifier'], []),
      ("Which properties has a pump?", ['properties', 'pump'], []),
      ("Which properties has a space heater?", ['properties', 'space heater'], []),
      ("Which properties has a tank?", ['properties', 'tank'], []),
      ("Which properties has a tube bundle?", ['properties', 'tube bundle'], []),
      ("Which properties has a valve?", ['properties', 'valve'], []),
      ("Which properties has a vibration isolator?", ['properties', 'vibration isolator'], []),
      ("What is a fire suppression terminal?", ['fire suppression terminal'], []),
      ("What is an interceptor?", ['interceptor'], []),
      ("What is a sanitary terminal?", ['sanitary terminal'], []),
      ("Which properties has an interceptor?", ['properties', 'interceptor'], []),
      ("What is an energy conversion device?", ['energy conversion device'], []),
      ("What is a flow controller?", ['flow controller'], []),
      ("What is a flow moving device?", ['flow moving device'], []),
      ("What is a flow storage device?", ['flow storage device'], []),
      ("What is a flow terminal?", ['flow terminal'], []),
      ("What is a flow treatment device?", ['flow treatment device'], []),
      ("What is a transport element?", ['transport element'], []),
      ("Which properties has a transport element?", ['properties', 'transport element'], []),
  ],
  "SAREFENV": [
      ("What is a TESS?", ['TESS'], []),
      ("Which type of sensor is a TESS?", ['sensor', 'TESS'], []),
      ("What is a photometer?", ['photometer'], []),
      ("What is the main component of TESS?", ['main component', 'TESS'], []),
      ("Which are the communication interfaces available in TESS?", ['communication interfaces', 'TESS'], ['available in']),
      ("Which components of TESS are sensors?", ["components", "TESS", "sensors"], []),
      ("Which properties can be measured by a photometer?", ["properties", 'photometer'], ['can be measured by']),
      ("Which are the properties observed by a TESS?", ['properties', 'TESS'], ['observed by']),
      ("Which type of RS232 is supported by TESS?", ['RS232', 'TESS'], ['is supported by']),
      ("Which type of Bluetooth is supported by TESS?", ['Bluetooth', 'TESS'], ['is supported by']),
      ("Which type of WiFi is supported by TESS?", ['WiFi', 'TESS'], ['is supported by']),
      ("Which are the components of TESS?", ['components', 'TESS'], []),
      ("Which hardware elements is a TESS connected to?", ['hardware elements', 'TESS'], ['connected to']),
      ("Which is the communication protocol used by TESS?", ['communication protocol', 'TESS'], ['used by']),
      ("What is the transmission period of the TESS?", ['transmission period', 'TESS'], []),
      ("What is a lamppost?", ['lamppost'], []),
      ("What can be considered as physical objects?", ['physical objects'], ['can be considered as']),
      ("What is a service?", ['service'], []),
  ]
}

sents = {
    "Vicinity": [
        ("Service thing description should be inline with Device thing description.", ['Service thing description', 'Device thing description'], ['be inline with']),
        ("Service thing description should be inline with WoT thing description.", ['Service thing description', 'WoT thing description'], ['be inline with']),
        ("Service thing description should define the concepts that service produces and provides to end user.", ['Service thing description', 'concepts', 'service', 'end user'], ['should define', 'produces', 'provides to']),
        ("Service thing description should define the interaction patterns how to interact with products of added value service.", ['Service thing description', 'interaction patterns', 'products', 'added value service'], ['should define', 'interact with']),
        ("Service thing description should include its version.", ['Service thing description', 'version'], ['should include']),
        ("Service thing description should define required inputs for the products and supported interaction patterns.", ['Service thing description', 'required inputs', 'products', 'supported interaction patterns'], ['should define']),
        ("Each thing is described by WoT Thing Descriptions.", ['thing', 'WoT Thing Descriptions'], ['is described by']),
        ("An endpoint can be relative to an endpoint that must not be relative.", ['endpoint', 'endpoint'], ['can be relative to']),
        ("The IoT user can be human (human user) or nonhuman (digital user)", ['IoT user', 'human', 'human user', 'nonhuman', 'digital user'], ['can be']),
        ("Digital user consumes services", ['Digital user', 'services'], ['consumes']),
        ("A human user interacts using applications", ['human user', 'applications'], ['interacts using']),
        ("An application is a specialized form of service", ['application', 'specialized form', 'service'], []),
        ("An Entity can be physical or virtual", ['Entity', 'physical', 'virtual'], []),
        ("A physical entity is controlled by an actuator", ['physical entity', 'actuator'], ['is controlled by']),
        ("A physical entity is monitored by a sensor", ['physical entity', 'sensor'], ['is monitored by']),
        ("A physical entity may have one or more attached tag", ['physical entity', 'attached tag'], ['may have']),
        ("A virtual entity represents a physical entity", ['virtual entity', 'physical entity'], ['represents']),
        ("Actuators and sensors are kinds of IoT device", ["Actuators", 'sensors', 'IoT device'], []),
        ("IoT devices interact through a network", ['IoT devices', 'network'], ['interact through']),
        ("IoT devices are connected with an IoT gateway", ['IoT devices', 'IoT gateway'], ['are connected with']),
        ("Data Stores hold data relating to IoT systems", ['data', 'Data Stores', 'IoT systems'], ['hold', 'relating to']),
        ("An entity has an identifier", ['entity', 'identifier'], []),
        ("An entity can have more than one identifier", ['entity', 'identifier'], ['can have']),
        ("A network connects endpoints", ['network', 'endpoints'], ['connects']),
        ("A service exposes one or more endpoints by which it can be invoked", ["A service", 'endpoints'], ['exposes', 'can be invoked']),
        ("An IoT gateway is a digital entity", ["An IoT gateway", "a digital entity"], []),
        ("IoT gateways interact through networks", ["IoT gateways", "networks"], ['interact through']),
        ("IoT gateways expose endpoints", ["IoT gateways", 'endpoints'], ['expose']),
        ("IoT gateways connect IoT devices", ['IoT gateways', 'IoT devices'], ['connect']),
        ("IoT gateways use data stores", ['IoT gateways', 'data stores'], ['use']),
        ("IoT device interacts with one or more networks", ['IoT device', 'networks'], ['interacts with']),
        ("IoT device exposes one or more endpoints", ['IoT device', 'endpoints'], ['exposes']),
        ("A service interacts with other entities via one or more networks", ['service', 'other entities', 'networks'], ['interacts with']),
        ("A service interacts with zero or more IoT gateways", ['service', 'IoT gateways'], ['interacts with']),
        ("A service interacts with zero or more IoT devices", ['service', 'IoT devices'], ['interacts with']),
        ("A service can interact with other services", ['service', 'other services'], ['can interact with']),
        ("A service can use data stores", ['service', 'data stores'], ['can use']),
        ("A virtual entity interacts through an endpoint", ['virtual entity', 'endpoint'], ['interacts through']),
        ("Everything in an IoT system is a kind of entity", ['IoT system', 'entity'], []),
        ("Data associated with services, devices and gateways can be held in data stores", ['Data', 'services', 'devices', 'gateways', 'data stores'], ['associated with', 'can be held in']),
        ("Human users uses applications", ['Human users', 'applications'], ['uses']),
        ("An application typically uses Services", ['application', 'Services'], ['uses']),
        ("Sensors can monitor the tag attached to a physical entity rather than the physical entity itself", ["Sensors", "the tag", "a physical entity", "the physical entity"], ["can monitor", 'attached to']),
        ("A device profile indicates the device name", ["A device profile", 'device name'], ['indicates']),
        ("A device profile indicates the device avatar", ["A device profile", 'device avatar'], ['indicates']),
        ("A device profile indicates the type of device", ["A device profile", 'device'], ['indicates']),
        ("A device profile indicates the device vendor", ["A device profile", 'device vendor'], ['indicates']),
        ("A device profile indicates the device serial number", ["A device profile", 'device serial number'], ['indicates']),
        ("A service profile indicates the service name", ["A service profile", 'service name'], ['indicates']),
        ("A service profile indicates the service avatar", ["A service profile", 'service avatar'], ['indicates']),
        ("A service profile indicates the service owner", ["A service profile", 'service owner'], ['indicates']),
        ("A service profile indicates the service provider", ["A service profile", 'service provider'], ['indicates']),
        ("A service profile indicates the service description", ["A service profile", 'service description'], ['indicates']),
        ("A service profile indicates the service type", ["A service profile", 'service type'], ['indicates']),
        ("A partnership is established between organizations", ["A partnership", "organizations"], ["is established between"]),
        ("A partnership is established between only 2 organizations", ["A partnership", "organizations"], ["is established between"]),
        ("A neighbourhood is the group of partnerships you have", ['neighbourhood', 'group', 'partnerships'], []),
        ("An organization has users", ['organization', 'users'], []),
        ("One of the users of an organization is the manager", ['users', 'organization', 'manager'], []),
        ("Users has to belong to one organization", ['Users', 'organization'], ['has to belong to']),
        ("All organizations have the same roles in a partnership", ['organizations', 'roles', 'partnership'], []),
        ("Users can have different roles in the organization", ['Users', 'different roles', 'organization'], ['can have']),
        ("user can create a group of services", ['user', 'group', 'services'], ['can create']),
        ("The security can be set up at a group level", ['security', 'group level'], ['can be set up at']),
        ("Authorization can be set up at the level of properties", ['Authorization', 'level', 'properties'], ['can be set up at']),
        ("A device can have a status", ['device', 'status'], ['can have']),
        ("A device can have a location", ['device', 'location'], ['can have']),
        ("A device has a unique identifier", ['device', 'unique identifier'], [])
    ], 
    "VicinityWOT": [
        ("Things in the WoT architecture are represented by servients,", ['Things', 'WoT architecture', 'servients'], ['are represented by']),
        ("Servient can represent virtual things as well.", ['Servient', 'virtual things'], ['can represent']),
        ("Servients are hosted anywhere, a smartphone, local gateway, or the cloud.", ['Servients', 'smartphone', 'local gateway', 'cloud'], ['are hosted']),
        ("Servient communicate to each other through a WoT interface", ['Servient', 'WoT interface'], ['communicate to']),
        ("Servient can have cliente or server roles or both.", ['Servient', 'cliente', 'server roles', 'both'], ['can have']),
        ("Thing Descriptions can be registered in Td repositories.", ['Thing Descriptions', 'Td repositories'], ['can be registered in']),
        ("A thing interaction can be available over different or multiple protocols", ['thing interaction', 'multiple protocols'], ['can be available over']),
        ("Each thing has at least an interaction pattern", ['interaction pattern'], []),
        ("An interaction pattern can have different endpoints", ["An interaction pattern", 'endpoints'], ['can have']),
        ("Each interaction pattern has its own attributes", ["interaction pattern", 'different endpoints'], ['can have']),
        ("Each interaction pattern has an endpoint", ["interaction pattern", "an endpoint"], []),
        ("Each endpoints has minimum two attributes: URI and mediatype", ["endpoints", "attributes", "mediatype"], []),
        ("Security is associated with things", ['Security', 'things'], ['is associated with']),
        ("An endpoint can be associated with a thing without determine the interaction patterns", ['endpoint', 'interaction patterns', 'thing'], ['can be associated with', 'determine']),
        ("Each interaction pattern has a name and a Web Resource address", ['interaction pattern', 'name', 'Web Resource address'], []),
        ("An interaction has a type and attributes", ["An interaction", "attributes", "type"], []),
        ("Each interaction pattern can be readable or writable through an endpoint", ["interaction pattern", 'endpoint'], ['can be readable', 'writable through']),
    ],
    "VicinityWOTMappings": [
        ("A TED describes one or more web thing", ['TED', 'web thing'], ['describes']),
        ("A TED refers to a service accesible in an URI", ['TED', 'service', 'URI'], ['refers to', 'accesible in']),
        ("A Thing ecosystem describes relations between TEDs", ['Thing ecosystem', 'relations', 'TEDs'], ['describes']),
        ("A thing description  can have access mappings defined", ['thing description', 'access mappings'], ['can have', 'defined']),
        ("An access mapping defines the transformation to be carried out for a specific key", ['access mapping', 'transformation', 'specific key'], ['to be carried out', 'defines']),
        ("An access mapping can have many mappings defined", ['access mapping', 'mappings'], ['can have', 'defined']),
        ("A mapping can transform keys to object properties or datatype properties.", ['mapping', 'keys', 'object properties', 'datatype properties'], ['can transform']),
        ("An object property mapping transforms keys to object properties", ['object property mapping', 'keys', 'object properties'], ['transforms']),
        ("A datatype property mapping transforms keys to datatype properties", ['datatype property mapping', 'keys', 'datatype properties'], ['transforms']), 
        ("An object property mapping can have many target classes to be the type of the generated instances", ['object property mapping', 'target classes', 'generated instances'], ['can have']),
        ("A datatype property mapping can have many target datatypes to be the datatype of the generated values", ['datatype property mapping', 'target datatypes', 'datatype', 'generated values'], ['can have']),
        ("An object property mapping generate instances transforming values following the mappings defined in a another thing description.", ['object property mapping', 'instances', 'values', 'mappings', 'thing description'], ['generate', 'transforming', 'defined in', 'following']),
        ("A mapping might be needed to be executed before another mapping", ['mapping', 'mapping'], ['might be needed to be executed before']),
        ("A mapping has a JSON path", ['mapping', 'JSON path'], []),
        ("An access mapping can be applied to at most to one endpoint", ['access mapping', 'endpoint'], ['can be applied to']),

    ], 
    'SAREF': [
        ("A device performs one or more functions", ['device', 'functions'], ['performs']),
        ("Examples of devices are a light switch, a temperature sensor, an energy meter, a washing machine", ['Examples', 'devices', 'light switch', 'temperature sensor', 'energy meter', 'washing machine'], []),
        ("A device shall have a model property", ['device', 'model property'], ['shall have']),
        ("A device shall have a manufacturer property", ['device', 'manufacturer property'], ['shall have']),
        ("A device can optionally have a description", ['device', 'description'], ['can optionally have']),
        ("A building space defines the physical spaces of the building where a device is located, such as a kitchen or a living room", ['building space', 'physical spaces', 'building', 'device', 'kitchen', 'living room'], ['defines', 'is located']),
        ("A building space contains devices or building objects", ['building space', 'devices', 'building objects'], ['contains']),
        ("Building objects are objects in the building that can be controlled by devices, such as doors or windows", ['Building objects', 'objects', 'building', 'devices', 'doors', 'windows'], ['can be controlled by']),
        ("A building object can be opened or closed by an actuator", ['building object', 'actuator'], ['can be opened', 'closed by']),
        ("A building space has also a property that can be used to specify the type of space, for example, the living room or the bedroom", ['building space', 'property', 'space', 'example', 'living room', 'bedroom'], ['can be used to specify']),
        ("A building space is a geographical point characterized by a certain altitude, latitude and longitude", ['building space', 'geographical point', 'certain altitude', 'latitude', 'longitude'], ['characterized by']),
        ("The devices can be classified into categories: FunctionRelated, EneryRelated and BuildingRelated", ['devices', 'categories', 'FunctionRelated', 'EneryRelated', 'BuildingRelated'], ['can be classified into']),
        ("The FunctionRelated category can be specialized into: Actuator, Applicance, HVAC, Lighting, Meter, MicroRenewable, Multimedia, Network, Sensor", ['FunctionRelated category', 'Actuator', 'Applicance', 'HVAC', 'Lighting', 'Meter', 'MicroRenewable', 'Multimedia', 'Network', 'Sensor'], ['can be specialized into']),
        ("The EnergyRelated category can be specialized into: Load, Generator and Storage", ['EnergyRelated category', 'Load', 'Generator', 'Storage'], ['can be specialized into']),
        ("The devices can belong to several categories", ['devices', 'categories'], ['can belong to']),
        ("A device can be used for the purpose of offering a commodity", ['device', 'commodity', 'purpose'], ['can be used for', 'offering']),
        ("Water or gas are examples of commodities", ['Water', 'gas', 'examples', 'commodities'], []),
        ("A device can be used for the purpose of sensing", ['device', 'sensing', 'purpose'], ['can be used for']),
        ("A device can be used for measuring a property", ['device', 'property'], ['can be used for measuring']),
        ("A device can be used for notifying a property", ['device', 'property'], ['can be used for notifying']),
        ("Examples of properties are temperature, energy or smoke", ['Examples', 'properties', 'temperature', 'energy', 'smoke'], []),
        ("A device can be used for the purpose of controlling a building object,such as a door or a window", ['device', 'purpose', 'building object', 'door', 'window'], ['can be used for', 'controlling']),
        ("A device may consists of other devices", ['device', 'devices'], ['may consists of']),
        ("A sensor is a device", ['sensor', 'device'], []),
        ("A sensor performs a sensing function", ['sensor', 'sensing function'], ['performs']),
        ("A temperature sensor is a device that has category sensor", ['temperature sensor', 'device', 'sensor'], []),
        ("A temperature sensor consists of a sensor", ['temperature sensor', 'sensor'], ['consists of']),
        ("A temperature sensor performs the sensing function and is used for the purpose of sensing temperature", ['temperature sensor', 'sensing function', 'purpose', 'temperature'], ['performs', 'is used for', 'sensing']),
        ("A washing machine is a device that has category Appliance and Load", ['washing machine', 'device', 'Appliance', 'Load'], []),
        ("A washing machine accomplishes the task of washing", ['washing machine', 'task', 'washing'], ['accomplishes']),
        ("A washing machine performs an actuating start function", ['washing machine', 'actuating start function'], ['performs']),
        ("A smoke sensor is a device that has category Sensor", ['smoke sensor', 'device', 'Sensor'], []),
        ("A smoke sensor is a device that consists of a sensor", ['smoke sensor', 'device', 'sensor'], ['consists of']),
        ("A smoke sensor performs a sensing and event function", ['smoke sensor', 'sensing and event function'], ['performs']),
        ("A smoke sensor is used for the purpose of sensing smoke", ['smoke sensor', 'purpose', 'smoke'], ['is used for', 'sensing']),
        ("A switch is a device that has category Actuator", ['switch', 'device', 'Actuator'], []),
        ("A door switch has category Actuator,", ['door switch', 'Actuator'], []),
        ("A door switch is a device that consists of a switch", ['door switch', 'device', 'switch'], ['consists of']),
        ("A door switch performs the open function", ['door switch', 'open function'], ['performs']),
        ("A door switch is used for the purpose of controlling doors", ['door switch', 'purpose', 'doors'], ['is used for', 'controlling']),
        ("A dimmer lamp is a device that has category Lighting and Actuator", ['dimmer lamp', 'device', 'Lighting', 'Actuator'], []),
        ("A dimmer lamp performs an actuating level control", ['dimmer lamp', 'actuating level control'], ['performs']),
        ("A dimmer lamp is used for the purpose of controlling the light", ['dimmer lamp', 'purpose', 'lamp', 'light'], ['is used for', 'controlling']),
        ("A meter is a functionRelated device", ['meter', 'functionRelated device'], []),
        ("A meter performs a metering function", ['meter', 'metering function'], ['performs']), 
        ("An energy meter is a device that has category Meter", ['energy meter', 'device', 'Meter'], []),
        ("An energy meter is a device that consists of a meter", ['energy meter', 'device', 'meter'], ['consists of']),
        ("An energy meter is used for the purpose of measuring energy", ['energy meter', 'purpose', 'energy'], ['is used for', 'measuring']),
        ("A function represents the functionality necessary to accomplish the task for which a device is designed", ['function', 'functionality', 'task', 'device'], ['represents', 'to accomplish', 'is designed']),
        ("Examples of functions are the actuating, sensing, metering or event functions", ['Examples', 'functions', 'actuating', 'sensing', 'metering', 'event functions'], []),
        ("An actuating function can be specialized into on/off, open/close, start/stop or level control functions", ['actuating funciton', 'on/off', 'open/close', 'start/stop', 'level control functions'], ['can be specialized into']),
        ("A function can belong to a function category", ['function', 'function'], ['can belong to']),
        ("A function shall have at least one command associated to it", ['function', 'command'], ['shall have', 'associated to']),
        ("The commands can be: on, off, open, toggle, close, start, pause, step up, step down, set level, get and notify", ['commands'], ['can be', 'on', 'off', 'open', 'toggle', 'close', 'start', 'pause', 'step up', 'step down', 'set level', 'get', 'notify']), 
        ("A device can be found in a correponding state", ['device', 'correponding state'], ['can be found in']),
        ("A command can act upon a state to represent that the consequence of a command can be a change of state of the device.", ['command', 'state', 'consequence', 'command', 'change', 'state', 'device'], ['can act upon', 'to represent', 'can be']),
        ("A device offers a service", ['device', 'service'], ['offers']),
        ("A service is a representation of a function to a network that makes this function discoverable, registerable and remotely controllable by other devices in the network", ['service', 'representation', 'function', 'network', 'this function', 'devices', 'network', 'discoverable', 'registerable', 'remotely controllable'], []),
        ("A service shall represent at least one function", ['service', 'function'], ['shall represent']),
        ("A service is offered by at least one device that wants its functions to be discoverable, registerable and remotely controllable by other devices in the network", ['service', 'device', 'functions', 'discoverable', 'registerable', 'remotely controllable', 'devices', 'network'], ['is offered by', 'wants']),
        ("Multiple devices can offer the same service", ['devices', 'same service'], ['can offer']),
        ("A service shall specify the device that is offering the service", ['service', 'device', 'service'], ['shall specify', 'is offering']),
        ("A device can be characterized by a profile.", ['device', 'profile'], ['can be characterized by']),
        ("The profile allows to describe the energy or power production and consumption of a certain device.", ['profile', 'energy', 'power production', 'consumption', 'certain device'], ['allows to describe']),
        ("The profile production and consumption can be calculated over a time span", ['profile production', 'consumption', 'time span'], ['can be calculated over']),
        ("The profile production and consumption can be associated to some costs", ['profile production', 'consumption', 'costs'], ['can be associated to']),
        ("The power is characterized by a certain value that is measured in a certain unit of measure", ['power', 'value', 'unit', 'measure'], ['is characterized by', 'is measured in']),
        ("The energy is characterized by a certain value that is measured in a certain unit of measure", ['energy', 'value', 'unit', 'measure'], ['is characterized by', 'is measured in']),
        ("The price is also characterized by a value using currency, which is a type of unit of measure", ['price', 'value', 'currency', 'unit', 'measure'], ['is also characterized by', 'using']),
        ("The time can be specified in terms of instants or intervals", ['time', 'terms', 'instants', 'intervals'], ['can be specified in']),
    ],
    'SAREFBLDG': [
        ("A building can contain devices", ['building', 'devices'], ['can contain']),
    ],
    "SAREFENV": [
        ("Each reading in a TESS has a timestamp", ['reading', 'TESS', 'timestamp'], []),
        ("Physical objects can contain devices", ['Physical objects', 'devices'], ['can contain']),
        ("A sensor could be a person", ['sensor', 'person'], ['could be']),
        ("Devices might have different versions and revision numbers", ['Devices', 'versions', 'revision numbers'], ['might have']),
        ("A device is located in a given geographical point defined by a latitude and a longitude", ['device', 'geographical point', 'latitude', 'longitude'], ['is located in', 'defined by']),
        ("A device position is defined using the azimuth and altitude horizontal coordinates", ['device position', 'azimuth', 'altitude horizontal coordinates'], ['is defined using']),
        ("A sensing device takes measures following a frequency", ['sensing device', 'measures', 'frequency'], ['takes', 'following']),
        ("A device can be actionable or not", ['device', 'actionable'], ['can be']),
        ("A device is part of a system", ['device', 'part', 'system'], []),
        ("An actuator is a type of device", ['actuator', 'device'], []),
        ("A device has energy consumption", ['device', 'energy consumption'], []),
        ("A lamppost uses a given light generation method", ['lamppost', 'light generation method'], ['uses']),
        ("A lamppost has energy consumption", ['lamppost', 'energy consumption'], []),
        ("A lamppost might have a light shield", ['lamppost', 'light shield'], ['might have']),
        ("A light has a geometry", ['light', 'geometry'], []),
        ("A light is projected in a given direction", ['light', 'given direction'], ['is projected in']),
        ("A light is projected in a given angle", ['light', 'given angle'], ['is projected in']),
        ("A light is projected from a given height", ['light', 'given height'], ['is projected from']),
        ("A light has a colour", ['light', 'colour'], []),
        ("A light can have flash", ['light', 'flash'], ['can have']),
        ("A digital representation of an object has a name", ['digital representation', 'object', 'name'], []),
        ("A digital representation of an object has a description", ['digital representation', 'object', 'description'], []),
        ("A physical object has a description", ['physical object', 'description'], []),
        ("A digital representation of an object has a unique identifier", ['digital representation', 'object', 'unique identifier'], []),
        ("A digital representation of an object can have zero or more tags", ['digital representation', 'object', 'tags'], ['can have']),
        ("A digital representation of an object has exactly one creation date", ['digital representation', 'object', 'creation date'], []),
        ("A digital representation of an object includes the date in which some characteristic of object was last modified", ['digital representation', 'object', 'date', 'characteristic', 'object'], ['includes', 'last modified']),
        ("A sensing device produces a log of property observations", ['sensing device', 'log', 'property observations'], ['produces']),
        ("A log indicates which is its last value", ['log', 'value'], ['indicates']),
        ("Values in a log are related to the time in which they were observed", ['Values', 'log', 'time'], ['are related to', 'were observed']),
        ("A physical representation of an object includes an observation log for each property to be observed", ['physical representation', 'object', 'observation log', 'property'], ['includes', 'to be observed']),
        ("An observation value is measured in a given unit", ['observation value', 'given unit'], ['is measured in']),
        ("Physical objects can have digital representations that are accessible through services", ['Physical objects', 'digital representations', 'services'], ['can have', 'are accessible through']),
        ("A service can expose either none or multiple digital representations of an object", ['service', 'digital representations', 'object'], ['can expose']),
        ("A digital representation of an object is exposed at maximum by one service", ['digital representation', 'object', 'service'], ['is exposed at']),
        ("A digital representation of an object performs actions", ['digital representation', 'object', 'actions'], ['performs']),
        ("An action manages an actuator", ['action', 'actuator'], ['manages']),
        ("An action of a digital representation of an object can keep an action log", ['action', 'digital representation', 'object', 'action log'], ['can keep']),
        ("A property of a digital representation of an object can keep an observation log", ['property', 'digital representation', 'object', 'observation log'], ['can keep']),
        ("A digital representation of an object encapsulates a system", ['digital representation', 'object', 'system'], ['encapsulates']),
    ], 
    "ONEM2M": [
        ("A thing is an entity that can be identified in the oneM2M System.", ['thing', 'entity', 'oneM2M System'], ['can be identified in']),
        ("A thing may have properties", ['thing', 'properties'], ['may have']),
        ("A thing can have relations to other things", ['thing', 'relations', 'things'], ['can have']),
        ("An aspect could be an entity or it could be a quality", ['aspect', 'entity', 'quality'], ['could be']),
        ("An aspect can have metadata", ['aspect', 'metadata'], ['can have']),
        ("Metadata contain data about a variable or about an aspect", ['Metadata', 'data', 'variable', 'aspect'], ['contain']),
        ("A device is a thing that is able to interact electronically with its environment via a network", ['device', 'thing', 'environment', 'network'], ['is able to interact electronically']),
        ("A device may be a physical or non-physical entity", ['device', 'non-physical entity'], ['may be']),
        ("A device performs one or more functionalities in order to accomplish a particular task", ['device', 'functionalities', 'particular task'], ['performs', 'to accomplish']),
        ("A device has one or more services that expose in the network its functionalities", ['device', 'services', 'network', 'functionalities'], ['expose in']),
        ("A device can be composed of several devices", ['device', 'devices'], ['can be composed of']),
        ("An interworked device is a device that does not support oneM2M interfaces and can only be accessed from the oneM2M System by communicating with a proxied device that has been created by an interworking proxy entity", ['interworked device', 'device', 'oneM2M interfaces', 'oneM2m System', 'proxied device', 'interworking proxy entity'], ['support', 'be accessed from', 'by communicating with', 'has been created by']),
        ("An interworked device is part of an area network", ['interworked device', 'part', 'area network'], []),
        ("An area network is a network that provides data transport services between an interworked device and the oneM2M System", ['area network', 'network', 'data transport services', 'interworked device', 'oneM2M System'], ['provides']),
        ("An area network follows a standard that defines its physical properties", ['area network', 'standard', 'physical properties'], ['follows', 'defines']),
        ("An area network follows a communication protocol", ['area network', 'communication protocol'], ['follows']),
        ("An area network follows a profile", ['area network', 'profile'], ['follows']),
        ("A service is an electronic representation of a functionality in a network", ['service', 'electronic representation', 'functionality', 'network'], []),
        ("A service can expose one or more functionalities", ['service', 'functionalities'], ['can expose']),
        ("A service can be composed of independent services", ['service', 'independent services'], ['can be composed of']),
        ("A service has an operation", ['service', 'operation'], []),
        ("A service has an input data point", ['service', 'input data point'], []),
        ("A service has an output data point", ['service', 'output data point'], []),
        ("A functionality represents the functionality necessary to accomplish the task for which a device is designed", ['functionality', 'functionality', 'task', 'device'], ['represents', 'necessary to accomplish', 'is designed']),
        ("A functionality refers to some real-world aspect", ['functionality', 'real-world aspect'], ['refers to']),
        ("A functionality has commands that allow human users to influence such functionality", ['functionality', 'commands', 'human users', 'functionality'], ['allow', 'to influence']),
        ("A controlling functionality represents a functionality that has impacts on the real world, but does not gather data", ['controlling functionality', 'functionality', 'real world', 'data', 'impacts'], ['represents', 'gather']),
        ("A measuring functionality represents a functionality that has no impacts on the real world, but only gathers data", ['measuring functionality', 'functionality', 'impacts', 'real world', 'data'], ['represents', 'gathers']),
        ("An operation is the means of a service to communicate in a procedure-type manner over the network", ['operation', 'means', 'service', 'procedure-type manner', 'network'], ['to communicate in']),
        ("An operation is the machine interpretable exposure of a human understandable command to a network", ['operation', 'machine interpretable exposure', 'human understandable command', 'network'], []),
        ("An operation may receive input data from input data points", ['operation', 'input data', 'input data points'], ['may receive']),
        ("An operation may receive data from operation inputs", ['operation', 'data', 'operation inputs'], ['may receive']),
        ("An operation may produce output data into output data points", ['operation', 'output data', 'output data points'], ['may produce']),
        ("An operation may produce data into operation outputs", ['operation', 'data', 'operation outputs'], ['may produce']),
        ("An operation has an operation state that allows a oneM2M entity to get informed on the progress of that operation", ['operation', 'operation state', 'oneM2M entity', 'progress', 'operation'], ['allows', 'to get informed on']),
        ("An operation has a method", ['operation', 'method'], []),
        ("An operation has a target URI", ['operation', 'target URI'], []),
        ("A command represents an action that can be performed to support a functionality", ['command', 'action', 'functionality'], ['represents', 'can be performed to support']),
        ("A command has as input one or more operation inputs", ['command', 'input', 'operation inputs'], []),
        ("A command has as output one or more operation outputs", ['command', 'output', 'operation outputs'], []),
        ("An operation input describes an input of an operation and also describes the input of a command", ['operation input', 'input', 'operation', 'input', 'command'], ['describes', 'describes']),
        ("An operation output describes an output of an operation and also describes the output of a command", ['operation output', 'output', 'operation', 'output', 'command'], ['describes', 'describes']),
        ("An operation state describes the current state of an operation", ['operation state', 'current state', 'operation'], ['describes']),
        ("An operation state is a simple type variable", ['operation state', 'simple type variable'], []),
        ("An operation state has exactly one data restriction pattern", ['operation state', 'data restriction pattern'], []),
        ("An input data point is a variable of a service that is accessed by a RESTful device in its environment and that the device reads out autonomously", ['input data point', 'variable', 'service', 'RESTful device', 'environment', 'device'], ['is accessed by', 'reads out autonomously']),
        ("An output data point is a variable of a service that is set by a RESTful device in its environment and that the device updates autonomously", ['output data point', 'variable', 'service', 'RESTful device', 'environment', 'device'], ['is set by', 'updates autonomously']),
        ("A variable describes an entity that stores some data that can change over time", ['variable', 'entity', 'data', 'time'], ['describes', 'stores', 'can change over']),
        ("A variable describes a real-world aspect", ['variable', 'real-world aspect'], ['describes']),
        ("A variable can have metadata", ['variable', 'metadata'], ['can have']),
        ("A variable can be structured using other variables", ['variable', 'variables'], ['can be structured using']),
        ("A variable may have a value", ['variable', 'value'], ['may have']),
        ("A variable has a CRUD method through which the instantiation of the variable value can be manipulated", ['variable', 'CRUD method', 'instantiation', 'variable value'], ['can be manipulated']),
        ("A variable has a URI of a resource through which the instantiation of the value of the variable can be manipulated", ['variable', 'URI', 'resource', 'instantiation', 'value', 'variable'], ['can be manipulated']),
        ("A simple type variable is a variable that only consists of variables of simple XML types like xsd:integer, xsd:string, etc., potentially including restrictions", ['simple type variable', 'variable', 'variables', 'simple XML types', 'xsd:integer', 'xsd:string', 'restrictions'], ['consists of', 'potentially including']),
        ("A simple type variable contains the name of the attribute of the resource that is referenced with the target URI and that stores the value of the simple type variable", ['simple type variable', 'name', 'attribute', 'resource', 'target URI', 'value', 'simple type variable'], ['contains', 'is referenced with', 'stores']),
        ("A simple type variable has exactly one datatype", ['simple type variable', 'datatype'], []),
        ("A simple type variable has a data restriction", ['simple type variable', 'data restriction'], []),
    ],
    "ODRL": [
        ("A single option for applying a constraint to a party should be defined.", ['single option', 'constraint', 'party'], ['for applying', 'should be defined']),
        ("It should be possible to define the target of a constraint inside a Permission, Prohibition or Duty: a constraint of human age applies to the Assignee, a constraint of play time applies to Assets.", ['target', 'constraint', 'Permission', 'Prohibition', 'Duty', 'constraint', 'human age', 'Assignee', 'constraint', 'play time', 'Assets'], ['should be possible to define', 'applies to', 'applies to']),
        ("The current data model assumes a policy instance includes all required data explicitly. This should be extended to policy instances which include explicit data and variables for values which are defined by parameters provided by an access to this template.", ['current data model', 'policy instance', 'explicit data', 'explicitly', 'policy instances', 'data', 'variables', 'values', 'parameters', 'access', 'this template'], ['assumes', 'includes all', 'should be extended to', 'include', 'are defined by', 'provided by']),
        ("In addition to the existing identifiers of a policy means for expressing a version of this policy should specified.", ['existing identifiers', 'policy means', 'version', 'this policy'], ['for expressing', 'should specified']),
        ("It should be possible to define the price for duty of payment for all permissions of a policy in a global way - while currently the payment duty must be defined for each permission individually.", ['price', 'duty', 'payment', 'permissions', 'policy', 'global way', 'payment duty', 'permission'], ['should be possible to define', 'must be defined for']),
        ("Extend the temporal constraints by setting a time reference, any period in the rightOperand refers to this point in time.", ['temporal constraints', 'time reference', 'period', 'rightOperand', 'point', 'time'], ['Extend', 'by setting', 'refers to']),
        ("For a relativeTimePeriod constraint the rightOperand has to provide a time period as value.", ['relativeTimePeriod constraint', 'rightOperand', 'time period', 'value'], ['has to provide']),
        ("Being able to tie Permission, Prohibition, Duty, and Constraint entities together with an AND, OR or XOR relationship", ['Permission', 'Prohibition', 'Duty', 'Constraint entities', 'AND', 'OR', 'XOR relationship'], ['Being able to tie']),
        ("Being able to assign multiple individuals of type Party to a Group for which permissions can be specified.", ['individuals', 'type Party', 'Group', 'permissions'], ['Being able to assign', 'can be specified']),
        ("Ability to link from a Policy or a Permission to the original license.", ['Ability', 'Policy', 'Permission', 'original license'], ['to link from']),
        ("It should be possible to define policies of type Assertion.", ['policies', 'Assertion'], ['should be possible to define']),
        ("An Assertion policy does not grant any permissions, but reflects policy terms that a party believes to have.", ['Assertion policy', 'permissions', 'policy terms', 'party'], ['grant', 'reflects', 'believes to have']),
    ],
    "BTO": [
        ("Zones are areas with spatial 3D volumes", ['Zones', 'areas', 'spatial 3D volumes'], []),
        ("Construction sites are areas with spatial 3D volumes", ['Construction sites', 'areas', 'spatial 3D volumes'], []),
        ("Buildings are areas with spatial 3D volumes", ['Buildings', 'areas', 'spatial 3D volumes'], []),
        ("Storeys are areas with spatial 3D volumes", ['Storeys', 'areas', 'spatial 3D volumes'], []),
        ("Spaces are limited three-dimensional extent defined physically or notionally, and are areas with spatial 3D volumes", ['Spaces', 'limited three-dimensional extent', 'areas', 'spatial 3D volumes'], ['defined physically', 'notionally']),
        ("Zones may contain other zones", ['Zones', 'zones'], ['may contain']),
        ("Construction sites may contain buildings", ['Construction sites', 'buildings'], ['may contain']),
        ("Buildings may contain storeys", ['Buildings', 'storeys'], ['may contain']),
        ("Storeys may contain spaces", ['Storeys', 'spaces'], ['may contain']),
        ("Spaces may be contained in storeys, buildings, and construction sites", ['Spaces', 'storeys', 'buildings', 'construction sites'], ['may be contained in']),
        ("Spaces may intersect different storeys, buildings, and construction sites", ['Spaces', 'storeys', 'buildings', 'construction sites'], ['may intersect']),
        ("A zone can be adjacent to another zone", ['zone', 'zone'], ['can be adjacent to']),
        ("There are building elements", ['building elements'], []),
        ("Building elements may have sub elements", ['Building elements', 'sub elements'], ['may have']),
        ("Zones may have elements, either contained in it, or adjacent to it", ['Zones', 'elements'], ['may have', 'contained in', 'adjacent to']),
        ("A zone and an adjacent zone share an interface", ['zone', 'adjacent zone', 'interface'], ['share']),
        ("A zone and an adjacent element share an interface", ['zone', 'adjacent element', 'interface'], ['share']),
        ("An element and an adjacent element share an interface", ['element', 'adjacent element', 'interface'], ['share']),
    ]
}


def filter_determiners(outs):
    result = []

    for obj in outs:
        result.append(re.sub(r'^(A|a|An|an|The|the) ', '', obj))
    return result

def filter_auxilaries(outs):
    result = []

    for obj in outs:
        result.append(re.sub(r'^([Aa]m|[Aa]re|[Ii]s|[Ww]as|[Ww]ill|[Ww]ould|[Ss]hall|[Ss]hould|[Cc]an|[Mm]ight|[Mm]ay|[Mm]ust|[Cc]ould|[Dd]o|[Dd]oes|[Dd]id)( be| been)? ', '', obj))
    return result


NON_ENTITY_THINGS =  ['a kind', 'the kind', 'kind', 'kinds', 'the kinds',
                      'category', 'a category', 'the category', 'categories', 'the categories',
                      'type', 'a type', 'the type', 'types', 'the types']

NON_ENTITY_THINGS = set(NON_ENTITY_THINGS)

for reqtype, name in [(cqs, "CQs"), (sents, "statements")]:
    for model in ['entitiesSWO.crfsuite', 'entitiesALL.crfsuite']:
        tagger = CRFTagger(f"models/{model}")

        tagger.load_tagger()

        TP = 0
        FP = 0
        FN = 0

        for ontology in reqtype:
            for (cq, ecs, pcs) in reqtype[ontology]:
                out = tagger.tag(cq)
                out = set(out) - NON_ENTITY_THINGS
                out = filter_determiners(out)
                ecs = filter_determiners(ecs)

                TP += len(set(ecs) & set(out))
                FP += len(set(out) - set(ecs))
                FN += len(set(ecs) - set(out))

              
               
        P = TP / (TP + FP)
        R = TP / (TP + FN)


        print(f"{name}\t{model}\tPrec: {P}, Recall: {R} F1: {2*P*R/(P+R)}")



    for model in ['relationsSWO.crfsuite', 'relationsALL.crfsuite']:
        tagger = CRFTagger(f"models/{model}")

        tagger.load_tagger()

        TP = 0
        FP = 0
        FN = 0

        NON_PREDICATE_THING = ['is', '’s', 'are', 'was', 'do', 'does', 'did', 'were',
                        'have', 'had', 'can', 'could', 'regarding',
                        'is of', 'are of', 'are in', 'given', 'is there', 'has']

        NON_ENTITY_THINGS =  ['a kind', 'the kind', 'kind', 'kinds', 'the kinds',
                        'category', 'a category', 'the category', 'categories', 'the categories',
                        'type', 'a type', 'the type', 'types', 'the types']


        NON_PREDICATE_THING = set(NON_PREDICATE_THING)
        NON_ENTITY_THINGS = set(NON_ENTITY_THINGS)


        for ontology in reqtype:
            for (cq, ecs, pcs) in reqtype[ontology]:
                out = tagger.tag(cq)
                out = filter_auxilaries(out)
                pcs = filter_auxilaries(pcs)
                out = set(out) - NON_PREDICATE_THING

                TP += len(set(pcs) & set(out))
                FP += len(set(out) - set(pcs))
                FN += len(set(pcs) - set(out))
        P = TP / (TP + FP)
        R = TP / (TP + FN)


        print(f"{name}\t{model}\tPrec: {P}, Recall: {R} F1: {2*P*R/(P+R)}")

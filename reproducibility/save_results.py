import copy
import glob
import json
import re
import shutil
import sys
from datetime import datetime
from itertools import product
from os.path import exists

import torch.cuda as cuda
from os import makedirs

pattern = '%Y-%m-%d %H-%M-%S'
# φάκελος όπου θα αποθηκευτούν τα merged αρχεία αποτελεσμάτων
results_folder = '../reproducibility/results'
# φάκελος όπου θα αποθηκευτούν (προσωρινά) όλα τα μοντέλα που όρισαν τα args
new_results_folder = results_folder + '/new-results'

class SaveResults:
    """
    Μορφή merged json
    [
        {
            setup : {
                filename : όνομα αρχείου json όπου αποθηκεύτηκε αρχικά το μοντέλο.
                           Δείχνει το dataset και τις τιμές των τροποιημένων παραμέτρων
                dataset :  prefix-dataset πχ D1-res14
                device : όνομα, όπου έτρεξε το μοντέλο
                model_param : {
                    --παράμετρος : τιμή σαν str
                }
            },
            results : {
                dev_set : [τιμές triples f1 για όλες τις εποχές]
                best_epoch : Η εποχή που έδωσε το καλύτερο triples f1, της οποίας τα
                             βάρη του μοντέλου επιλέγονται
                best_epoch_f1 : Το σκορ στο dev set στην καλύτερη εποχή
                training_time : Συνολικός χρόνος εκπαίδευσης σε sec
                test_set : {
                    f1 : Triples f1 στο test set για το επιλεγμένο μοντέλο
                    precision : >>
                    recall :  >>
                }
            }
        }
        ,...
    ]
    """
    default_values = {
        "--mode": "train",
        '--bert_model_path': 'bert-base-uncased',
        '--bert_feature_dim': '768',

        '--batch_size': '6',
        '--epochs': '100',
        '--learning_rate': '1e-3',
        '--bert_lr': '2e-5',
        '--adam_epsilon': '1e-8',
        '--weight_decay': '0.0',
        '--seed' : '1000',

        '--num_layers': '1',
        '--gcn_dim': '300',
        '--pooling': 'avg'
    }

    def __init__(self, args, ignore_if_found=True):
        """
        Αποθήκευση των αποτελεσμάτων της εκπαίδευσης των μοντέλων που ορίζουν τα args
        @param args: List με τις παράμετρους του μοντέλου, ανά δύο στοιχεία: '--όνομα', 'τιμή'
                    Η τιμή είναι str ή list(str) -> πολλαπλές τιμές για την παραμέτρο ώστε
                    να οριστούν πολλαπλά μοντέλα.
        @param ignore_if_found: Αν έχει τρέξει ήδη το ίδιο μοντέλο και τα αποτελέσματα του είναι
                    αποθηκευμένα σε json στο φάκελο new-results, μην το ξανατρέξεις
        """
        self.ignore_if_found = ignore_if_found
        self._make_folder()
        self.device = cuda.get_device_name(cuda.current_device())
        self.mod_args, self.experiments = self._parse_args(args)

        self.output = None
        self.modified = None


    def _parse_args(self, args):
        """
        Παράγει τα μοντέλα (experiments) που θα εκτελεστούν σύμφωνα με τα args.
        1. Για κάθε arg (παράμετρο) που έχει τιμή value (str ή list(str)):
            2. Κρατάει τα arg και τις τιμές που έχουν τροποποιηθεί σε σχέση με
               τις default values
        @return: mod_args: list(str) -> οι παράμετροι που έχουν τροποιηθεί
                 product: list(tuple) όπου len(tuple) = len(mod_args)
                    Κάθε δυνατός συνδιασμός για τις παραμέτρους που έχουν τροποιημένη τιμή.

            πχ Για τα τροποιημένα:
               '--dataset' = ['res14', 'res15']
               '--seed' = ['0', '1']
            mod_args =  ['--dataset', '--seed']
            product  = [('res14', '0'), ('res14', '1'), ('res15', '0'), ('res15', '1')]
        """
        mod_args, mod_lists = [], []
        for i in range(len(args) // 2):     # 1
            arg = args[2*i]
            value = args[2*i+1]

            if value != SaveResults.default_values.get(arg, None):  # 2
                mod_args.append(arg)
                mod_lists.append(
                    value if isinstance(value, list) else [value])
        return mod_args, list(product(*mod_lists))  # 3

    def __len__(self):
        """
        @return: Πλήθος των μοντέλων
        """
        return len(self.experiments)

    def __getitem__(self, item):
        """
        Ένα μοντέλο για εκπαίδευση (iterate experiments)
        1. modified : Dict{όνομα τροποιημένης παραμέτρου : η τιμή της για το τρέχον experiment}
        3. output : Dict με το setup (παραμέτρους) του μοντέλου για το τρέχον experiment
                    και τα results της εκπαίδευσης.
        @param item: index του τρέχοντος experiment
        @return:
            List με τις παράμετρους του μοντέλου, ανά δύο στοιχεία: '--όνομα', 'τιμή'
            ώστε να διαβαστούν από την main.parse_arguments (4)
            ή
            None Αν έχει τρέξει ήδη το ίδιο μοντέλο και τα αποτελέσματα του είναι
                 αποθηκευμένα σε json στο φάκελο new-results, μην το ξανατρέξεις (2)
        """
        self.modified = {arg: value for arg, value in zip(self.mod_args, self.experiments[item])}  # 1
        dataset = self._dataset_name()
        filename = self._get_filename(dataset)
        print(filename)
        if self.ignore_if_found and exists(f'{new_results_folder}/{filename}'):  # 2
            return None

        self.output = {     # 3
            'setup': {
                'filename': filename,
                'dataset': dataset,
                'device': self.device,
                'model_param': self._model_parameters(),
            },
            'results' : {
                'dev_set': []
            }
        }
        return self._get_arguments(self.output['setup']['model_param'])  # 4


    def _dataset_name(self):
        """
        @return: Συνδιασμό του prefix και dataset πχ D1-res14
        """
        return re.findall(r'D\d/[a-z0-9]+',
                          self.modified['--prefix'] + self.modified['--dataset'])[0].replace('/', '-')

    def _get_filename(self, dataset):
        """
        @param dataset: Όνομα dataset πχ D1-res14
        @return: Όνομα json αρχείου όπου θα αποθηκευτεί το τρέχον μοντέλο. Μορφή:
            dataset__όνομα τροποποιημένης παράμετρου-τιμή__.json
            πχ D1-res14__seed-42__gcn_dim-128__.json
        """
        filename = f'{dataset}__'
        for f in self.mod_args:
            if f not in ['--prefix', '--dataset']:
                filename += f'{f[2:]}-{self.modified[f]}__'
        return f'{filename}.json'

    def _model_parameters(self):
        """
        Οι τιμές των παραμετρων του τρέχοντος μοντέλου
        1. Αντιγραφή των default τιμών των παραμέτρων
        2. Αλλαγή τιμών όσων έχουν τροποιηθεί (με τις τιμές του τρέχοντος experiment)
        @return: Dict{όνομα παραμέτρου : τιμή}
        """
        model_param = copy.deepcopy(SaveResults.default_values)  # 1
        for f in self.mod_args:
            model_param[f] = self.modified[f]  # 2
        return model_param

    def _get_arguments(self, model_param):
        """
        Μετατροπή του dict των παραμέτρων σε λίστα ώστε να διαβαστούν από την main.parse_arguments.
        @param model_param: Dict{όνομα παραμέτρου : τιμή}. Έξοδος της _model_parameters
        @return: List με τις παράμετρους του μοντέλου, ανά δύο στοιχεία: '--όνομα', 'τιμή'
        """
        return [sys.argv[0]] + [value for field in model_param for value in (field, model_param[field])]

    def update(self, field, value):
        """
        Καταχώριση ενός αποτελέσματος της εκπαίδευσης του τρέχοντος μοντέλου
        @param field: Ένα αποτέλεσμα που θα αποθηκευτεί.
            Ενδεικτικά: testset, best_epoch, best_epoch_f1, dev_set, training_time
        @param value: Η τιμή του συγκεκριμένου αποτελέσματος. Για το dev_set κρατάει
                      μόνο το f1.
        """
        if field == 'dev_set':
            self.output['results'][field].append(value['f1'])
        else:
            self.output['results'][field] = value

 # -----------------------------------------------------------------------------------------------
 # Αποθήκευση αποτελεσμάτων
 # -----------------------------------------------------------------------------------------------
    def _make_folder(self):
        """
        Δημιουργία φακέλων όπου θα αποθηκευτούν τα αποτελέσματα
        """
        for folder in [results_folder, new_results_folder]:
            if not exists(folder):
                makedirs(folder)

    def write_output(self):
        """
        Κλήση όταν έχει τελείωσει η εκπαίδευση και το inference του τρέχοντος μοντέλου
        ώστε να αποθηκευτούν τα αποτελέσματα που συγκεντρώθηκαν σε json
        (ένα προσωρινό json για κάθε μοντέλο)
        """
        file = f"{new_results_folder}/{self.output['setup']['filename']}"
        with open(file, 'w') as json_file:
            json.dump(self.output, json_file, indent=4)

    def merge_outputs(self):
        """
        Κλήση αφού τρέξουν όλα τα μοντέλα και αποθηκευτούν τα αποτελέσματα τους
        σε ξεχωριστά json στον φάκελο new-results.
        Συγχώνευση όλων των αποτελεσμάτων (2) σε ένα json με όνομα το τρέχον datetime
        στο φάκελο results (3). Επιπλέον, συγχωνεύονται και τα αποτέλεσματα από το πιο
        πρόσφατο merged json (1), ώστε το τρέχον merged να περιέχει όλα τα αποτελέσματα
        που έχουν τρέξει μέχρι τώρα.
        Τα επιμέρους μοντέλα θα διαγραφούν (4), όλα τα merged αρχεία θα παραμείνουν.
        """
        try:
            with open(self.latest_merge(), "rb") as infile:  # 1
                result = json.load(infile)
        except TypeError:
            # δε βρέθηκαν προηγούμενες εκτελέσεις αποθηκευμένες σε merged json
            result = []

        for f in glob.glob(f"{new_results_folder}/*.json"):  # 2
            print(f)
            with open(f, "rb") as infile:
                result.append(json.load(infile))

        merged_json = f"{results_folder}/{datetime.now():{pattern}}" + ".json"  # 3
        with open(merged_json, "w") as outfile:
            json.dump(result, outfile, indent=4)
        shutil.rmtree(new_results_folder)  # 4


    def latest_merge(self):
        """
        Έυρεση του πιο πρόσφατου merged αρχείου.
        @return: Όνομα αρχείου πχ 'results/2022-12-10 12-13-15.json'
                 ή None αν δε βρέθηκαν προηγούμενες εκτελέσεις αποθηκευμένες σε merged json
        """
        regex = re.compile(r'\d{4}-\d{2}-\d{2} \d{2}-\d{2}-\d{2}')
        filedates = [
            datetime.strptime(re.findall(regex, filename)[0], pattern)
            for filename in glob.glob(f"{results_folder}/*.json")
        ]
        return f'{results_folder}/{max(filedates).__format__(pattern)}.json' \
            if len(filedates) > 0 else None





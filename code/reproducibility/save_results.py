import glob
import json
import re
import shutil
from datetime import datetime
from itertools import product
from os import makedirs
from os.path import exists

import torch.cuda as cuda

pattern = '%Y-%m-%d %H-%M-%S'
# φάκελος όπου θα αποθηκευτούν τα merged αρχεία αποτελεσμάτων
results_folder = 'reproducibility/results'
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
                    παράμετρος : τιμή σαν str
                    tag : μια περιγραφή που δίνεται στο τρέχον πείραμα
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

    def __init__(self, parser, ignore_if_found=True):
        """
        Αποθήκευση των αποτελεσμάτων της εκπαίδευσης των μοντέλων που ορίζουν τα args
        @param parser: Parser με τις παράμετρους του μοντέλου
                    Η τιμή είναι type(arg) -μία- ή list(type(arg)) -> πολλαπλές τιμές για την παραμέτρο ώστε
                    να οριστούν πολλαπλά μοντέλα.
        @param ignore_if_found: Αν έχει τρέξει ήδη το ίδιο μοντέλο και τα αποτελέσματα του είναι
                    αποθηκευμένα σε json στο φάκελο new-results, μην το ξανατρέξεις
        """
        self.ignore_if_found = ignore_if_found
        self._make_folder()
        self.device = cuda.get_device_name(cuda.current_device())
        self.given_args = parser.parse_args()
        self.mod_args, self.experiments = self._parse_args(parser)

        print(f"Modified args : {self.mod_args}\nCombinations : {self.experiments}\n# combinations = {len(self.experiments)}")

        self.output = None
        self.modified = None

    def _parse_args(self, parser):
        """
        Παράγει τα μοντέλα (experiments) που θα εκτελεστούν σύμφωνα με τα args.
        1. Για κάθε arg (παράμετρο) που έχει τιμή value (type(arg) -μία- ή list(type(arg))):
            2. Κρατάει τα arg και τις τιμές που έχουν τροποποιηθεί σε σχέση με
               τις default values (και τα prefix, dataset)
        @return: mod_args: list(str) -> οι παράμετροι που έχουν τροποιηθεί
                 product: list(tuple) όπου len(tuple) = len(mod_args)
                    Κάθε δυνατός συνδιασμός για τις παραμέτρους που έχουν τροποιημένη τιμή.

            πχ Για τα τροποιημένα:
               --dataset  res14 res15 --seed 0 1
            Θα οριστούν οι συνδιασμοί (4 μοντέλα):
            mod_args =  ['dataset', 'seed']
            product  = [(res14, 0), (res14, 1), (res15, 0), (res15, 1)]
        """
        mod_args, mod_lists = [], []
        for arg, value in vars(self.given_args).items():     # 1
            if not self._is_not_modified(arg, value, parser.get_default(arg)):  # 2

                mod_args.append(arg)
                mod_lists.append(
                    value if isinstance(value, list) else [value])
        return mod_args, list(product(*mod_lists))  # 3

    def _is_not_modified(self, arg, value, default):
        """
        1. Κρατάει ως "modified" τα prefix και dataset -> modified
        2. Αν η δοθείσα τιμή είναι list μπορεί να έχει τροποποιηθεί:
            Αν έχουν οριστεί παραπάνω από μία τιμές -> modified
            Αν μία τιμή, πρέπει να ελέγξει επιπλέον αν είναι η default ή όχι(->modified)
        3. Αν δεν είναι λίστα, ελέγχει αν είναι η default.
        @param arg: str όνομα παραμέτρου
        @param value: Η τιμή που δώθηκε από το χρήστη
        @param default: Η προκαθορισμένη τιμή του parser
        """
        if arg == 'tag' :
            return True
        if arg == 'prefix' or arg == 'dataset':  # 1
            return False
        if isinstance(value, list) :  # 2
            return len(value) == 1 and self._get_arg_value(value) == default
        else:
            return value == default  # 3


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
            Namespace args με τις παράμετρους του τρέχοντος μοντέλου (4)
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

        self._model_parameters()
        self.output = {  # 3
            'setup': {
                'filename': filename,
                'dataset': dataset,
                'device': self.device,
                'model_param': vars(self.given_args),
            },
            'results': {
                'dev_set': []
            }
        }
        return self.given_args  # 4

    def _dataset_name(self):
        """
        @return: Συνδιασμό του prefix και dataset πχ D1-res14
        """
        return re.findall(r'D\d/[a-z0-9]+',
                          self.modified['prefix'] + self.modified['dataset'])[0].replace('/', '-')

    def _get_filename(self, dataset):
        """
        @param dataset: Όνομα dataset πχ D1-res14
        @return: Όνομα json αρχείου όπου θα αποθηκευτεί το τρέχον μοντέλο. Μορφή:
            dataset__όνομα τροποποιημένης παράμετρου-τιμή__.json
            πχ D1-res14__seed-42__gcn_dim-128__.json
        """
        filename = f'{dataset}__'
        for f in self.mod_args:
            if f not in ['prefix', 'dataset']:
                filename += f'{f}-{self.modified[f]}__'
        return f'{filename}.json'

    def _model_parameters(self):
        """
        Οι τιμές των παραμετρων του τρέχοντος μοντέλου
        => Αλλαγή τιμών όσων έχουν τροποιηθεί (με τις τιμές του τρέχοντος experiment)
           Για τιμές που δεν έχουν τροποποιηθεί αλλά είναι σαν list πρέπει να
           απομονωθεί το στοιχείο για να δωθεί στο μοντέλο.
        """
        for arg, value in vars(self.given_args).items():
            setattr(self.given_args, arg,
                    self.modified.get(arg, self._get_arg_value(value))
            )


    def _get_arg_value(self, value):
        """
        @param value: list(type(arg)) για args που διαβάζονται με την μορφή λίστας (nargs='+')
                            => πρέπει να γίνει type(arg)
                type(arg) μένει όπως είναι
        """
        return value[0] if isinstance(value, list) else value


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
            with open(SaveResults.latest_merge(), "rb") as infile:  # 1
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

    @staticmethod
    def latest_merge():
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

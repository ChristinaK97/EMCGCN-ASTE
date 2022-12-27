import json

import objectpath

from save_results import SaveResults

"""
objectpath examples
https://objectpath.org/tutoria/
https://stackoverflow.com/questions/29996880/python-querying-a-json-objectpath
"""
class QueryResults:
    def __init__(self, json_file=None):
        self.json_file = json_file if json_file is not None else SaveResults.latest_merge()
        if self.json_file is None:
            raise Exception("Merged results json not found")

        with open(self.json_file, 'r') as f:
            self.json_file = json.load(f)
        self.json_file = objectpath.Tree(self.json_file)

        query = "$.*[@.setup.dataset is 'D1-res15'].results.test_set"
        result = self.json_file.execute(query)
        [print(r) for r in result]



QueryResults()
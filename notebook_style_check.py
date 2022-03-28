import sys
import json

def find_keywords_in_notebook_json(notebook_json: dict, keywords: list=\
                                    ['problem statement', 'approach', 'result', 'conclusion', 'follow-up']):
    """
    parse a notebook json file's non-code cells, return if all of the keyword appear in at least one of them.
    """
    keywords_found = {keyword: False for keyword in keywords}
    notebook_noncode_cells = filter(lambda cell: cell['cell_type'] != 'code', notebook_json['cells'])
    for cell in notebook_noncode_cells:
        for keyword in keywords:
            cell_content = "    ".join(cell['source']).lower()
            if keyword in cell_content:
                keywords_found[keyword] = True
    return all(keywords_found.values())

#print(f"{sys.argv[1:]}")
file_paths = sys.argv[1:]

notebook_failed_style_checks = []
for notebook_file_path in file_paths:
    with open(notebook_file_path, 'r') as file:
        notebook_json = json.load(file)
    notebook_style_check_result = find_keywords_in_notebook_json(notebook_json)
    if not notebook_style_check_result:
        notebook_failed_style_checks.append(notebook_file_path)

import pprint
if notebook_failed_style_checks:
    print("the following notebooks failed style checks")
    # todo: do a pprint of all the notebooks that failed style check
    pprint.pprint(notebook_failed_style_checks)
    sys.exit(1)
else:
    sys.exit(0)
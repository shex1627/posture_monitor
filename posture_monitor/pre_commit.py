import sys

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

print("hello world bad")
#sys.exit(-1)
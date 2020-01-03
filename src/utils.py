import os


def set_root_dir():
    root = os.path.join(os.path.expanduser('~'), '.KnowledgeGraphDataSets')
    dirname = os.environ.get('KG_DIR', root)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    return dirname

set_root_dir()
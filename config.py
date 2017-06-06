import sys
import logging

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

def get_paths():
    from os.path import expanduser
    home = expanduser("~")
    print home
    import ConfigParser
    Config = ConfigParser.ConfigParser()
    Config.read(home + "/config.txt")
    root = eval(Config.get("paths", "root"))
    data_path = eval(Config.get("paths", "data_name"))
    model_path = eval(Config.get("paths", "model_name"))
    vector_path = eval(Config.get("paths", "vector_name"))
    logger.info("The root path is: %s", root)
    logger.info("The data path is: %s", data_path)
    logger.info("The model path is: %s", model_path)
    logger.info("The vector path is: %s", vector_path)
    return root, data_path, model_path, vector_path


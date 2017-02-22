def get_config():
    from os.path import expanduser
    home = expanduser("~")
    print home
    import ConfigParser
    Config = ConfigParser.ConfigParser()
    Config.read(home + "/config.txt")
    return Config
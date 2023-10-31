import argparse

class Parser:

    def __init__(self):

        self.parser = argparse.ArgumentParser(description='DPM-GSP')
        self.set_arguments()

    def set_arguments(self):

        self.parser.add_argument('--config', type=str,
                                    required=True, help="Path of config file")
        self.parser.add_argument('--comment', type=str, default="", 
                                    help="A single line comment for the experiment")
        self.parser.add_argument('--seed', type=int, default=42)
        

    def parse(self):

        args, unparsed  = self.parser.parse_known_args()
        
        if len(unparsed) != 0:
            raise SystemExit('Unknown argument: {}'.format(unparsed))
        
        return args
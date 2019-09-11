from easyvvuq.decoders import BaseDecoder

def file_len(fname):
    

class FleeDecoder(BaseDecoder, decoder_name="flee_decoder"):

    def sim_complete(self, run_info=None):
        with open("out.csv") as f:
            for i, l in enumerate(f):
                pass
        if i > 1:
            return True
        else:
            return False

    def parse_sim_output(self, run_info={}):
        # User code goes here (method must return a pandas dataframe)

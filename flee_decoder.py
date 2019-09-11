# Preliminary file for a possible Flee Decoder. May not be necessary, so delete if obsolete.

from easyvvuq.decoders import SimpleCSV

class FleeDecoder(SimpleCSV, decoder_name="flee_decoder"):

    def parse_sim_output(self, run_info={}):

        out_path = self._get_output_path(run_info, self.target_filename)

        data = pd.read_csv(
            out_path,
            usecols=self.output_columns,
            header=self.header)

        return data


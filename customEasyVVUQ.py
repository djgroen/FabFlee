from easyvvuq.constants import default_campaign_prefix, Status
from easyvvuq.db.sql import CampaignDB
from easyvvuq.data_structs import CampaignInfo
from easyvvuq import Campaign
from easyvvuq.analysis import SCAnalysis
from easyvvuq.encoders import GenericEncoder
import tempfile
import os
import numpy as np


class CustomCampaign(Campaign):
    # ----------------------------------------------------------------------
    # changes :
    # send runs_dir='SWEEP' when we call CampaignInfo
    # change location of campaign.db to work directory
    # ----------------------------------------------------------------------

    def init_fresh(self, name, db_type='sql',
                   db_location=None, work_dir='.'):

        # Create temp dir for campaign
        campaign_prefix = default_campaign_prefix
        if name is not None:
            campaign_prefix = name

        campaign_dir = tempfile.mkdtemp(prefix=campaign_prefix, dir=work_dir)

        self._campaign_dir = os.path.relpath(campaign_dir, start=work_dir)

        self.db_location = db_location
        self.db_type = db_type

        if self.db_type == 'sql':
            from easyvvuq.db.sql import CampaignDB
            if self.db_location is None:
                self.db_location = "sqlite:///" + work_dir + "/campaign.db"
                # self.db_location = "sqlite:///" + self.campaign_dir + "/campaign.db"
        else:
            message = (f"Invalid 'db_type' {db_type}. Supported types are "
                       f"'sql'.")
            logger.critical(message)
            raise RuntimeError(message)
        from easyvvuq import __version__
        info = CampaignInfo(
            name=name,
            campaign_dir_prefix=default_campaign_prefix,
            easyvvuq_version=__version__,
            campaign_dir=self.campaign_dir,
            #runs_dir=os.path.join(campaign_dir, 'runs')
            runs_dir=os.path.join(campaign_dir, 'SWEEP')
        )
        self.campaign_db = CampaignDB(location=self.db_location,
                                      new_campaign=True,
                                      name=name, info=info)

        # Record the campaign's name and its associated ID in the database
        self.campaign_name = name
        self.campaign_id = self.campaign_db.get_campaign_id(self.campaign_name)

    # ----------------------------------------------------------------------
    # changes :
    # return generated run_ids when we call populate_runs_dir
    # ----------------------------------------------------------------------

    def populate_runs_dir(self):

        # Get the encoder for this app. If none is set, only the directory structure
        # will be created.
        active_encoder = self._active_app_encoder
        if active_encoder is None:
            logger.warning(
                'No encoder set for this app. Creating directory structure only.')

        run_ids = []

        for run_id, run_data in self.campaign_db.runs(
                status=Status.NEW, app_id=self._active_app['id']):

            # Make directory for this run's output
            os.makedirs(run_data['run_dir'])

            # Encode run
            if active_encoder is not None:
                active_encoder.encode(params=run_data['params'],
                                      target_dir=run_data['run_dir'])

            run_ids.append(run_id)
        self.campaign_db.set_run_statuses(run_ids, Status.ENCODED)
        return run_ids


class CustomSCAnalysis(SCAnalysis):

    # ----------------------------------------------------------------------
    # changes :
    # add file input parameter to save generated plot
    # ----------------------------------------------------------------------
    def adaptation_histogram(self, file=None):
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=[4, 8])
        ax = fig.add_subplot(111, ylabel='max quadrature order',
                             title='Number of refinements = %d'
                             % self.sampler.number_of_adaptations)
        # find max quad order for every parameter
        adapt_measure = np.max(self.l_norm, axis=0)
        ax.bar(range(adapt_measure.size), height=adapt_measure)
        params = list(self.sampler.vary.get_keys())
        ax.set_xticks(range(adapt_measure.size))
        ax.set_xticklabels(params)
        plt.xticks(rotation=90)
        plt.tight_layout()

        if file == None:
            plt.show()
        else:
            plt.savefig(file, dpi=400)

    # ----------------------------------------------------------------------
    # changes :
    # add file input parameter to save generated plot
    # ----------------------------------------------------------------------
    def plot_stat_convergence(self, file=None):

        if not self.dimension_adaptive:
            print('Only works for the dimension adaptive sampler.')
            return

        K = len(self.mean_history)
        if K < 2:
            print('Means from at least two refinements are required')
            return
        else:
            differ_mean = np.zeros(K - 1)
            differ_std = np.zeros(K - 1)
            for i in range(1, K):
                differ_mean[i - 1] = np.linalg.norm(self.mean_history[i] -
                                                    self.mean_history[i - 1], np.inf)

                differ_std[i - 1] = np.linalg.norm(self.std_history[i] -
                                                   self.std_history[i - 1], np.inf)

        import matplotlib.pyplot as plt
        fig = plt.figure('stat_conv')
        ax1 = fig.add_subplot(111, title='moment convergence')
        ax1.set_xlabel('refinement step')
        ax1.set_ylabel(r'$ ||\mathrm{mean}_i - \mathrm{mean}_{i - 1}||_\infty$',
                       color='r', fontsize=12)
        ax1.plot(range(2, K + 1), differ_mean, color='r', marker='+')
        ax1.tick_params(axis='y', labelcolor='r')

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        ax2.set_ylabel(r'$ ||\mathrm{std}_i - \mathrm{std}_{i - 1}||_\infty$',
                       color='b', fontsize=12)
        ax2.plot(range(2, K + 1), differ_std, color='b', marker='*')
        ax2.tick_params(axis='y', labelcolor='b')

        plt.tight_layout()
        # plt.show()
        if file == None:
            plt.show()
        else:
            plt.savefig(file, dpi=400)

from base.fab import *
from plugins.FabFlee.FabFlee import *

@task
def run_ssudan_paper():
  for postfix in ["ssudan_default","ssudan","ssudan_links","ssudan_ccamp","ssudan_capkakuma1","ssudan_capkakuma2","ssudan_capadju1","ssudan_capadju2","ssudan_cborder"]:   # additional runs:"ssudan_redirect"
    flee("flee_%s" % (postfix),simulation_period=604)
    plot_output("flee_%s_localhost_16" % (postfix),"out")

    test_sensitivity("flee_%s" % (postfix),simulation_period=604,name="MaxMoveSpeed",values="25-50-100")
    for i in [25, 50, 100]:
      plot_output("flee_%s_MaxMoveSpeed_%s_localhost_16" % (postfix, i),"out")

    #test_sensitivity("flee_%s" % (postfix),simulation_period=604,name="Awareness",values="0-1-2")
    #for i in [0, 1, 2]:
      #plot_output("flee_%s_Awareness_%s_localhost_16" % (postfix, i),"out")





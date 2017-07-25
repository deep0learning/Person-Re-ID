def get_config():
  """Dynamically load a .py config file. You can specify which config file to 
  load in the console."""
  import imp
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument('--config_file', required=True,
                      help='The .py config file path')
  args = parser.parse_args()

  Config = imp.load_source('module.name', args.config_file)
  cfg = Config.Config
  return cfg
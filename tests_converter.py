from axelerate.networks.common_utils.convert import Converter

converter = Converter('k210')
converter.convert_k210(model_path,dataset_path)

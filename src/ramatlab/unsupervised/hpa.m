function hpa(input_file_path, output_file_path, input_type)
    hpamodule = py.importlib.import_module('hpa');
    hpamodule.hpa(input_file_path, output_file_path, input_type);
end
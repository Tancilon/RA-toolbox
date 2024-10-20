function dibra(input_file_path, output_file_path, input_type)
    dibramodule = py.importlib.import_module('dibra');
    dibramodule.dibra(input_file_path, output_file_path, input_type);
end
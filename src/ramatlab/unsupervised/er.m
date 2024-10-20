function er(input_file_path, output_file_path, input_type)
    ermodule = py.importlib.import_module('er');
    ermodule.er(input_file_path, output_file_path, input_type);
end
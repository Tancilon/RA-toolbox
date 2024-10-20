function irank(input_file_path, output_file_path, input_type)
    irankmodule = py.importlib.import_module('irank');
    irankmodule.irank(input_file_path, output_file_path, input_type);
end
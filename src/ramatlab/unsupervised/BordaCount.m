function BordaCount(input_file_path, output_file_path)
    initPython();
    bordamodule = py.importlib.import_module('bordacount');
    bordamodule.bordacount(input_file_path, output_file_path);
end
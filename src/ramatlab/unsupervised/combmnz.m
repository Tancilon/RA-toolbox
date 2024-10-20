function combmnz(input_file_path, output_file_path)
    combmnzmodule = py.importlib.import_module('combmnz');
    combmnzmodule.combmnz(input_file_path, output_file_path);
end
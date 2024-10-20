function combmed(input_file_path, output_file_path)
    combmedmodule = py.importlib.import_module('combmed');
    combmedmodule.combmed(input_file_path, output_file_path);
end
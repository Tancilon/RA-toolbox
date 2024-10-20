function combsum(input_file_path, output_file_path)
    combsummodule = py.importlib.import_module('combsum');
    combsummodule.combsum(input_file_path, output_file_path);
end
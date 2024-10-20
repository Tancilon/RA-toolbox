function mean(input_file_path, output_file_path)
    meanmodule = py.importlib.import_module('mean');
    meanmodule.mean(input_file_path, output_file_path);
end
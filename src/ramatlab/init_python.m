function init_python()
    % initPython - 设置 Python 环境和包路径
    pyenv('Version', "C:\Users\2021\.conda\envs\pymatlab\python.exe");  % 替换为实际的 Python 可执行文件路径


    % 获取src路径
    currentFilePath = mfilename('fullpath');
    [currentDir, ~, ~] = fileparts(currentFilePath);
    [upperDir, ~, ~] = fileparts(currentDir);
    [srcDir, ~, ~] = fileparts(upperDir);
    % 项目根目录
    [projectDir, ~, ~] = fileparts(srcDir);
    
    
    % 添加python解释器的搜索路径
    py.sys.path().append(projectDir);
    disp(py.sys.path())
end

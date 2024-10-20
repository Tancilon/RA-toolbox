function addenvpath()
    % 获取当前脚本所在的子目录路径
    currentDir = fileparts(mfilename('fullpath'));
    
    % 获取父级目录路径
    parentDir = fileparts(currentDir);
    
    % 将父级目录添加到 MATLAB 搜索路径
    addpath(parentDir);
end
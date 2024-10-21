classdef WeightedBorda
    properties
        % 属性保存 Python 的 WeightedBorda 类的实例
        pyObj
    end
    
    methods
        % 构造函数，用于创建 WeightedBorda 的 Python 对象
        function obj = WeightedBorda(topk, is_partial_list)
            if nargin < 2
                is_partial_list = true;  % 默认参数
            end
            if nargin < 1
                topk = py.None;  % 如果未提供 topk，则使用 None
            end
            % 创建 Python 类的实例
            obj.pyObj = py.WeightedBorda(topk, is_partial_list);
        end
        
        % 调用 Python 的 train 方法
        function train(obj, train_file_path, train_rel_path)
            % 确保输入的文件路径是字符串格式
            train_file_path = convertCharsToStrings(train_file_path);
            train_rel_path = convertCharsToStrings(train_rel_path);
            
            % 调用 Python 对象的 train 方法
            obj.pyObj.train(train_file_path, train_rel_path);
        end
        
        % 调用 Python 的 test 方法
        function test(obj, test_file_path, test_output_path, using_average_w)
            if nargin < 4
                using_average_w = true;  % 默认参数
            end
            % 确保输入的文件路径是字符串格式
            test_file_path = convertCharsToStrings(test_file_path);
            test_output_path = convertCharsToStrings(test_output_path);
            
            % 调用 Python 对象的 test 方法
            obj.pyObj.test(test_file_path, test_output_path, using_average_w);
        end
    end
end

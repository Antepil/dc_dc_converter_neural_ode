% generate_all_splits.m
% 依次生成所有 split 的数据

% 定义所有需要生成的划分名称（需与 project.json 中的键一致）
splits = ["validation","test_seen", ...
          "test_interpolation", "test_voltage_extrapolation", ...
          "test_extrapolation_high"];

% 原数据生成脚本的文件名
generator_script = "generate_dataset.m";

for idx = 1:numel(splits)
    split_name = splits(idx);          % 设置 split_name 变量
    fprintf('===== 开始生成 split: %s =====\n', split_name);
    run(generator_script);             % 运行原脚本
end

disp('所有划分数据生成完毕。');

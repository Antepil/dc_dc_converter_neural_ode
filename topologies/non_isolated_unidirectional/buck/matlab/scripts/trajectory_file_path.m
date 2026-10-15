function file_path = trajectory_file_path(output_dir, trajectory_id)
%TRAJECTORY_FILE_PATH 构造单条轨迹的标量 MAT 文件路径。

    file_path = fullfile(string(output_dir), string(trajectory_id) + ".mat");
end

function export_trajectory(sim_out, file_path, metadata)
%EXPORT_TRAJECTORY Validate and save one hybrid DCM trajectory as MAT v3.

    narginchk(3, 3);

    try
        logs = sim_out.get("logsout");
    catch
        error("BuckNeuralODE:MissingLogs", ...
            "仿真结果中没有 logsout，请在模型中启用信号记录。");
    end
    if isempty(logs)
        error("BuckNeuralODE:MissingLogs", ...
            "仿真结果中没有 logsout，请在模型中启用信号记录。");
    end

    trajectory = extract_hybrid_trajectory(logs, metadata);
    outputDir = fileparts(file_path);
    if strlength(string(outputDir)) > 0 && ~isfolder(outputDir)
        mkdir(outputDir);
    end
    save(file_path, "-struct", "trajectory", "-v7");
end

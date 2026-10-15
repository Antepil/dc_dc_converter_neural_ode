function trajectory = extract_hybrid_trajectory(logs, metadata)
%EXTRACT_HYBRID_TRAJECTORY Extract and validate one DCM Buck trajectory.
%   TRAJECTORY = EXTRACT_HYBRID_TRAJECTORY(LOGS, METADATA) consumes the
%   original variable-step iL, vC, gate, and iD logs. Natural diode turn-off
%   events are located only from iD; iL is used solely for physical audits.

    arguments
        logs (1, 1) Simulink.SimulationData.Dataset
        metadata (1, 1) struct
    end

    validate_metadata(metadata);
    validate_logged_signal_set(logs);

    totalTime = metadata.num_cycles * metadata.Ts;
    [iLTime, iLData] = get_logged_signal(logs, "iL", totalTime);    % 获取电感电流
    [vCTime, vCData] = get_logged_signal(logs, "vC", totalTime);    % 获取电容电压
    [gateTime, gateData] = get_logged_signal(logs, "gate", totalTime);  % 获取 gate 数据
    [iDTime, iDData] = get_logged_signal(logs, "iD", totalTime);    % 获取 iD 电流

    if any(iLData < -metadata.negative_current_abs_tol_a)
        error("BuckNeuralODE:DCMViolation", ...
            "电感电流低于允许的反向电流容差 %.6g A。", ...
            metadata.negative_current_abs_tol_a);
    end

    % 校验门极信号的开关边沿，并生成有效的门极状态序列
    gateState = validate_gate_edges(gateTime, gateData, metadata);

    % 从二极管电流、电感电流和电容电压中提取全部事件，并返回原始二极管导通状态
    [eventData, rawDiodeOn] = extract_all_events( ...
    iDTime, iDData, iLTime, iLData, vCTime, vCData, metadata);

    totalIntervals = metadata.num_cycles * metadata.samples_per_cycle;
    time_s = (0:totalIntervals)' * metadata.dt;
    intervalMidTime = time_s(1:end-1) + 0.5 * metadata.dt;

    inductor_current_a = resample_signal( ...
        iLTime, iLData, time_s, "iL", "linear");
    capacitor_voltage_v = resample_signal( ...
        vCTime, vCData, time_s, "vC", "linear");
    diode_current_a = resample_signal( ...
        iDTime, iDData, time_s, "iD", "linear");
    sampledGate = resample_signal( ...
        gateTime, double(gateState), intervalMidTime, "gate", "previous");
    sampledDiodeOn = resample_signal( ...
        iDTime, double(rawDiodeOn), intervalMidTime, "diode_on", "previous");

    pwm_gate = uint8(sampledGate >= 0.5);
    expectedGate = expected_interval_gate(totalIntervals, metadata);
    if any(pwm_gate ~= expectedGate)
        error("BuckNeuralODE:PWMAlignmentMismatch", ...
            "固定区间中点的 gate 与已知 PWM 网格不一致。");
    end

    diode_on = uint8(sampledDiodeOn >= 0.5);
    interval_mode = build_interval_modes( ...
        intervalMidTime, pwm_gate, eventData.time_s, metadata);
    natural_event_mask = true(metadata.num_cycles, 1);
    initial_state = [inductor_current_a(1), capacitor_voltage_v(1)];
    input_voltage_v = metadata.input_voltage_v;
    load_resistance_ohm = metadata.load_resistance_ohm;

    trajectory = struct( ...
        "time_s", time_s, ...
        "inductor_current_a", inductor_current_a, ...
        "capacitor_voltage_v", capacitor_voltage_v, ...
        "diode_current_a", diode_current_a, ...
        "input_voltage_v", input_voltage_v, ...
        "load_resistance_ohm", load_resistance_ohm, ...
        "initial_state", initial_state, ...
        "pwm_gate", pwm_gate, ...
        "diode_on", diode_on, ...
        "interval_mode", interval_mode, ...
        "natural_event_mask", natural_event_mask, ...
        "event_time_s", eventData.time_s, ...
        "event_time_tau", eventData.time_tau, ...
        "idle_duration_tau", 1 - eventData.time_tau, ...
        "event_state", eventData.state, ...
        "event_current_a", eventData.state(:, 1), ...
        "event_bracket_time_s", eventData.bracket_time_s, ...
        "event_bracket_diode_current_a", eventData.bracket_current_a, ...
        "metadata", metadata);
end


function validate_metadata(metadata)
%VALIDATE_METADATA Validate fields required by the MATLAB v3 contract.

    requiredFields = [ ...
        "schema_version", "input_voltage_v", "load_resistance_ohm", ...
        "num_cycles", "R", "D", "Ts", "dt", "samples_per_cycle", ...
        "diode_on_threshold_a", "diode_off_threshold_a", ...
        "event_current_abs_tol_a", "idle_current_abs_tol_a", ...
        "negative_current_abs_tol_a", "idle_diode_current_abs_tol_a", ...
        "event_guard_time_s", "dcm_margin_tau", ...
        "gate_edge_time_abs_tol_s"];
    missing = requiredFields(~isfield(metadata, requiredFields));
    if ~isempty(missing)
        error("BuckNeuralODE:InvalidMetadata", ...
            "metadata 缺少字段：%s。", strjoin(missing, ", "));
    end
    if string(metadata.schema_version) ~= "hybrid_event_matlab_v3"
        error("BuckNeuralODE:InvalidMetadata", ...
            "只接受 hybrid_event_matlab_v3 metadata。");
    end
    expectedDt = metadata.Ts / metadata.samples_per_cycle;
    tolerance = 100 * eps(max(1, expectedDt));
    if abs(metadata.dt - expectedDt) > tolerance
        error("BuckNeuralODE:InvalidMetadata", ...
            "dt 必须等于 Ts/samples_per_cycle。");
    end
end


function validate_logged_signal_set(logs)
%VALIDATE_LOGGED_SIGNAL_SET Require exactly four unique named signals.

    expected = sort(["iL"; "vC"; "gate"; "iD"]);
    actual = sort(string(logs.getElementNames));
    missing = setdiff(expected, actual);
    if ~isempty(missing)
        error("BuckNeuralODE:MissingSignal", ...
            "logsout 缺少信号：%s。", strjoin(missing, ", "));
    end
    if numElements(logs) ~= numel(expected) || ~isequal(actual, expected)
        error("BuckNeuralODE:UnexpectedSignals", ...
            "logsout 必须唯一包含 iL、vC、gate 和 iD。");
    end
end


function [rawTime, rawData] = get_logged_signal(logs, signalName, totalTime)
%GET_LOGGED_SIGNAL Return one validated, unique-time signal.

    try
        element = logs.getElement(signalName);
    catch
        error("BuckNeuralODE:MissingSignal", ...
            "logsout 中找不到信号 %s。", signalName);
    end
    if ~isa(element, "Simulink.SimulationData.Signal")
        error("BuckNeuralODE:InvalidLoggedElement", ...
            "信号 %s 必须是 Simulink.SimulationData.Signal。", signalName);
    end
    signal = element.Values;
    if ~isa(signal, "timeseries")
        error("BuckNeuralODE:InvalidSignalType", ...
            "信号 %s 必须记录为 timeseries。", signalName);
    end

    rawTime = double(signal.Time(:));
    rawData = double(squeeze(signal.Data));
    rawData = rawData(:);
    if numel(rawTime) ~= numel(rawData) || numel(rawTime) < 2
        error("BuckNeuralODE:InvalidSignalShape", ...
            "信号 %s 的时间和数据长度不一致。", signalName);
    end
    if any(~isfinite(rawTime)) || any(~isfinite(rawData))
        error("BuckNeuralODE:NonFiniteState", ...
            "信号 %s 含 NaN 或 Inf。", signalName);
    end

    [rawTime, uniqueIndex] = unique(rawTime, "last");
    rawData = rawData(uniqueIndex);
    coverageTolerance = 100 * eps(max(1, totalTime));
    if rawTime(1) > coverageTolerance || ...
            rawTime(end) < totalTime - coverageTolerance
        error("BuckNeuralODE:InsufficientTimeRange", ...
            "信号 %s 未覆盖 [0, C*Ts]。", signalName);
    end
end


function gateState = validate_gate_edges(gateTime, gateData, metadata)
%VALIDATE_GATE_EDGES Check every expected PWM transition against raw gate.

    gateState = gateData >= 0.5;
    if ~gateState(1)
        error("BuckNeuralODE:PWMAlignmentMismatch", ...
            "仿真必须从 PWM 上升沿的 ON 状态开始。");
    end

    transitionIndex = find(diff(gateState) ~= 0) + 1;
    fallingIndex = transitionIndex(~gateState(transitionIndex));
    risingIndex = transitionIndex(gateState(transitionIndex));
    fallingTime = gateTime(fallingIndex);
    risingTime = gateTime(risingIndex);
    expectedFalling = ((0:metadata.num_cycles - 1)' + metadata.D) * metadata.Ts;
    expectedRising = (1:metadata.num_cycles)' * metadata.Ts;

    fallingMatches = numel(fallingTime) == numel(expectedFalling) && ...
        all(abs(fallingTime - expectedFalling) <= metadata.gate_edge_time_abs_tol_s);
    risingMatches = numel(risingTime) == numel(expectedRising) && ...
        all(abs(risingTime - expectedRising) <= metadata.gate_edge_time_abs_tol_s);
    if ~(fallingMatches && risingMatches)
        error("BuckNeuralODE:PWMAlignmentMismatch", ...
            "原始 gate 边沿与 D、Ts 定义的 PWM 边界不一致。");
    end
end


function [eventData, rawDiodeOn] = extract_all_events( ...
        iDTime, iDData, iLTime, iLData, vCTime, vCData, metadata)
%EXTRACT_ALL_EVENTS Extract exactly one natural turn-off per cycle.

    cycleCount = metadata.num_cycles;
    eventTime = zeros(cycleCount, 1);
    eventTau = zeros(cycleCount, 1);
    eventState = zeros(cycleCount, 2);
    bracketTime = zeros(cycleCount, 2);
    bracketCurrent = zeros(cycleCount, 2);
    rawDiodeOn = false(size(iDData));

    for cycle = 1:cycleCount
        cycleStart = (cycle - 1) * metadata.Ts;
        offTime = cycleStart + metadata.D * metadata.Ts;
        nextOn = cycleStart + metadata.Ts;
        offIndex = find(iDTime >= offTime & iDTime < nextOn);
        offCurrent = iDData(offIndex);

        onRelative = find(offCurrent > metadata.diode_on_threshold_a, 1, "first");
        if isempty(onRelative)
            throw_dcm_violation(cycle, "二极管在 OFF 区间未进入导通状态");
        end
        zeroRelativeAfterOn = find( ...
            abs(offCurrent(onRelative + 1:end)) <= metadata.diode_off_threshold_a, ...
            1, "first");
        if isempty(zeroRelativeAfterOn)
            throw_dcm_violation(cycle, "二极管导通后没有自然截止事件");
        end
        zeroRelative = onRelative + zeroRelativeAfterOn;
        positiveRelative = zeroRelative - 1;
        if offCurrent(positiveRelative) <= 0
            throw_dcm_violation(cycle, "事件括区间正侧 iD 必须大于零");
        end

        positiveIndex = offIndex(positiveRelative);
        zeroIndex = offIndex(zeroRelative);
        thisBracketTime = [iDTime(positiveIndex), iDTime(zeroIndex)];
        thisBracketCurrent = [iDData(positiveIndex), iDData(zeroIndex)];
        thisEventTime = interpolate_zero_crossing( ...
            thisBracketTime, thisBracketCurrent);
        thisEventTau = (thisEventTime - cycleStart) / metadata.Ts;

        if thisEventTau <= metadata.D || ...
                thisEventTau > 1 - metadata.dcm_margin_tau
            throw_dcm_violation(cycle, "事件未形成规定的正 IDLE 区间");
        end

        thisEventCurrent = interp1(iLTime, iLData, thisEventTime, "linear");
        thisEventVoltage = interp1(vCTime, vCData, thisEventTime, "linear");
        if abs(thisEventCurrent) > metadata.event_current_abs_tol_a
            throw_dcm_violation(cycle, "事件处电感电流超过容差");
        end

        idleStart = thisEventTime + metadata.event_guard_time_s;
        idleILIndex = iLTime >= idleStart & iLTime < nextOn;
        idleIDIndex = iDTime >= idleStart & iDTime < nextOn;
        if ~any(idleILIndex) || ~any(idleIDIndex)
            throw_dcm_violation(cycle, "事件后没有可审计的 IDLE 原始样本");
        end
        if any(abs(iLData(idleILIndex)) > metadata.idle_current_abs_tol_a)
            throw_dcm_violation(cycle, "IDLE 电感电流超过容差");
        end
        if any(abs(iDData(idleIDIndex)) > metadata.idle_diode_current_abs_tol_a)
            throw_dcm_violation(cycle, "IDLE 二极管电流超过容差");
        end
        if any(iDData(idleIDIndex) > metadata.diode_on_threshold_a)
            throw_dcm_violation(cycle, "自然截止后二极管再次导通");
        end

        rawDiodeOn(offIndex(onRelative:zeroRelative - 1)) = true;
        eventTime(cycle) = thisEventTime;
        eventTau(cycle) = thisEventTau;
        eventState(cycle, :) = [thisEventCurrent, thisEventVoltage];
        bracketTime(cycle, :) = thisBracketTime;
        bracketCurrent(cycle, :) = thisBracketCurrent;
    end

    eventData = struct( ...
        "time_s", eventTime, ...
        "time_tau", eventTau, ...
        "state", eventState, ...
        "bracket_time_s", bracketTime, ...
        "bracket_current_a", bracketCurrent);
end


function eventTime = interpolate_zero_crossing(bracketTime, bracketCurrent)
%INTERPOLATE_ZERO_CROSSING Interpolate iD=0 within one raw bracket.

    denominator = bracketCurrent(2) - bracketCurrent(1);
    if abs(denominator) <= eps(max(1, max(abs(bracketCurrent))))
        eventTime = bracketTime(2);
        return;
    end
    candidate = bracketTime(1) - bracketCurrent(1) * ...
        diff(bracketTime) / denominator;
    if candidate < bracketTime(1) || candidate > bracketTime(2)
        eventTime = bracketTime(2);
    else
        eventTime = candidate;
    end
end


function sampled = resample_signal(rawTime, rawData, targetTime, signalName, method)
%RESAMPLE_SIGNAL Resample one validated raw signal.

    tolerance = 100 * eps(max(1, targetTime(end)));
    if rawTime(1) > targetTime(1) + tolerance || ...
            rawTime(end) < targetTime(end) - tolerance
        error("BuckNeuralODE:InsufficientTimeRange", ...
            "信号 %s 未覆盖完整目标时间。", signalName);
    end
    sampled = interp1(rawTime, rawData, targetTime, method);
    sampled = sampled(:);
end


function pwmGate = expected_interval_gate(totalIntervals, metadata)
%EXPECTED_INTERVAL_GATE Return known PWM gate values at interval midpoints.

    intervalInCycle = mod((0:totalIntervals - 1)', metadata.samples_per_cycle);
    onIntervals = round(metadata.D * metadata.samples_per_cycle);
    pwmGate = uint8(intervalInCycle < onIntervals);
end


function intervalMode = build_interval_modes( ...
        intervalMidTime, pwmGate, eventTime, metadata)
%BUILD_INTERVAL_MODES Assign ON/OFF_DIODE/IDLE to fixed intervals.

    intervalMode = zeros(size(pwmGate), "uint8");
    cycleIndex = min(floor(intervalMidTime / metadata.Ts) + 1, metadata.num_cycles);
    offMask = pwmGate == 0;
    beforeEvent = intervalMidTime < eventTime(cycleIndex);
    intervalMode(offMask & beforeEvent) = uint8(1);
    intervalMode(offMask & ~beforeEvent) = uint8(2);
end


function throw_dcm_violation(cycle, reason)
%THROW_DCM_VIOLATION Throw the single recoverable trajectory error.

    error("BuckNeuralODE:DCMViolation", ...
        "第 %d 个周期不满足 DCM 契约：%s。", cycle, reason);
end

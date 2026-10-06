function simscape_emt_der_benchmark(which)
% SIMSCAPE_EMT_DER_BENCHMARK  Reference waveforms for the EMT models of the
% microgrid's sources and stores (emt_der.py), from emt_der_benchmark_params.json.
%
% Cases (simscape_emt_der_benchmark('battery') runs one):
%   battery    Simscape Electrical's Battery (Table-Based), one RC pair, the
%              LFP OCV table: 100 A drawn for 10 s, then rest
%   pv         Simscape Electrical's Solar Cell with the module's fitted
%              single-diode values, strung: I-V curves at three irradiances
%   smoothing  Simscape Electrical's Supercapacitor (one branch) behind an
%              averaged converter whose control is the smoothing law, on a
%              48 V rack bus and an 800 V row bus
%   flywheel   a Simscape inertia behind an averaged machine converter that
%              holds its DC link, its power limited at low speed
%
% Writes emt_der_<case>.csv. Needs Simscape and Simscape Electrical.

here = fileparts(mfilename('fullpath'));
p = jsondecode(fileread(fullfile(here, 'emt_der_benchmark_params.json')));
if nargin < 1, which = {'battery', 'pv', 'smoothing', 'flywheel'}; end
if ischar(which), which = {which}; end
if any(strcmp(which, 'battery')), battery(p.battery, here); end
if any(strcmp(which, 'pv')), pv(p.pv, here); end
if any(strcmp(which, 'smoothing'))
    for name = fieldnames(p.smoothing.cases)'
        smoothing(p.smoothing, name{1}, here);
    end
end
if any(strcmp(which, 'flywheel')), flywheel(p.flywheel, here); end
end

% --- the cases ------------------------------------------------------------------

function battery(q, here)
m = Model('emt_der_battery');
b = m.place('ee_lib/Sources/Battery (Table-Based)');
n = numel(q.soc_table);
set_param(b, 'SOC_vec', mat2str(q.soc_table', 17), ...
    'T_dependence', 'simscape.enum.tablebattery.temperature_dependence.no', ...
    'prm_dir', 'simscape.enum.tablebattery.prm_dir.noCurrentDirectionality', ...
    'V0_vec', mat2str(q.ocv_cell' * q.cells, 17), 'R0_vec', mat2str(q.r0 * ones(1, n), 17), ...
    'AH', num2str(q.ah, 17), 'prm_dyn', 'simscape.enum.tablebattery.prm_dyn.rc1', ...
    'R1_vec', mat2str(q.r1 * ones(1, n), 17), 'tau1_vec', mat2str(q.tau1 * ones(1, n), 17), ...
    'stateOfCharge_specify', 'on', 'stateOfCharge_priority', 'High', 'stateOfCharge', num2str(q.soc0, 17));
pos = m.node();
m.two(b, pos, m.gnd);
% The load: 100 A from t_on to t_off.
on = m.place('simulink/Sources/Step');
set_param(on, 'Time', num2str(q.t_on, 17), 'Before', '0', 'After', num2str(q.i_load, 17));
off = m.place('simulink/Sources/Step');
set_param(off, 'Time', num2str(q.t_off, 17), 'Before', '0', 'After', num2str(-q.i_load, 17));
add = m.place('simulink/Math Operations/Add');
m.wire(on, 1, add, 1); m.wire(off, 1, add, 2);
m.current_draw(pos, add, 1);
log = m.run(q.t_end, q.dt);
t = (0:q.dt * 10:q.t_end)';
v = m.series(log, b, 'v', t);
soc = m.series(log, b, 'stateOfCharge', t);
write(here, 'emt_der_battery.csv', t, {'v', 'soc'}, {v, soc});
m.done();
end

function pv(q, here)
for g = q.irradiances'
    m = Model('emt_der_pv');
    c = m.place('ee_lib/Sources/Solar Cell');
    set_param(c, 'prm', '3', 'Is', num2str(q.i0, 17), 'Iph', num2str(q.iph, 17), 'Ir0', '1000', ...
        'ec', num2str(q.a, 17), 'Rs', num2str(q.rs_cell, 17), 'Rp', num2str(q.rp_cell, 17), ...
        'N_series', num2str(q.n_cells * q.modules_series), 'N_parallel', num2str(q.strings_parallel), ...
        'TIPH1', '0', 'Tmeas', '25', 'TFIXED', '25');
    pp = m.ports(c);
    irr = m.place('fl_lib/Physical Signals/Sources/PS Constant');
    set_param(irr, 'constant', num2str(g, 17), 'constant_unit', 'W/m^2');
    m.add_line(m.ports(irr).RConn(1), pp.LConn(1));
    pos = m.node();
    m.connect(pos, pp.LConn(2));
    m.connect(m.gnd, pp.RConn(1));
    % Its voltage swept from zero past its open-circuit voltage; its current into the source measured.
    ramp = m.place('simulink/Sources/Ramp');
    set_param(ramp, 'Slope', num2str(1.02 * q.v_oc_array, 17), 'Start', '0', 'InitialOutput', '0');
    sens = m.place('fl_lib/Electrical/Electrical Sensors/Current Sensor');
    x = m.node();
    m.two(sens, pos, x);
    m.voltage_source(x, ramp, 1);
    log = m.run(1.0, 1e-3);
    t = linspace(0, 1, q.points)';
    v = 1.02 * q.v_oc_array * t;
    i = m.series(log, sens, 'I', t);
    write(here, sprintf('emt_der_pv_%d.csv', g), v, {'i'}, {i});
    m.done();
end
end

function smoothing(s, name, here)
q = s.cases.(name);
m = Model(['emt_der_smoothing_' name]);
vn_in = q.module_v * q.modules_series;
c_store = q.module_c * q.strings_parallel / q.modules_series;
esr = q.module_esr * q.modules_series / q.strings_parallel;
v_store0 = q.v0_percent / 100 * vn_in;
c_in = 4e-3 * q.rated / vn_in ^ 2;
c_out = 4e-3 * q.rated / q.v_feed ^ 2;
v_rack0 = (q.v_feed + sqrt(q.v_feed ^ 2 - 4 * q.r_feed * q.p_high)) / 2;
% The feed holds the rack bus through R_feed.
rack = m.node(); src = m.node(); fx = m.node();
m.two(m.dcsrc(q.v_feed), src, m.gnd);
isens = m.place('fl_lib/Electrical/Electrical Sensors/Current Sensor');
m.two(isens, src, fx);
m.two(m.res(q.r_feed), fx, rack);
% The racks: a constant-power load cycling high and low.
[tb, pb] = cycle(s, q.p_high, q.p_low);
lsens = m.cpl_profile_load(rack, tb, pb);
% The store: one branch of the supercapacitor, its ESR and leakage.
sc = m.place('ee_lib/Passive/Supercapacitor');
set_param(sc, 'R', mat2str([esr, 1e9, 1e9], 17), 'C', mat2str([c_store, 1e-9, 1e-9], 17), 'Kv', '1e-12', ...
    'R_discharge', num2str(q.r_leak, 17), 'N_series', '1', 'N_parallel', '1', ...
    'vc1_specify', 'on', 'vc1_priority', 'High', 'vc1', num2str(v_store0, 17), ...
    'vc2_specify', 'on', 'vc2_priority', 'High', 'vc2', num2str(v_store0, 17), ...
    'vc3_specify', 'on', 'vc3_priority', 'High', 'vc3', num2str(v_store0, 17));
port = m.node();
m.two(sc, port, m.gnd);
m.two(m.cap(c_in, v_store0), port, m.gnd);
m.two(m.cap(c_out, v_rack0), rack, m.gnd);
% Its control: P_store = P_rack - LPF(P_rack); i_out = P_store / v_out, i_in = P_store (/ or x eta) / v_in.
v_rack = m.vsense(rack); v_port = m.vsense(port);
p_rack = m.mfn(sprintf(['function p = f(v, i)\np = v * i;\nend']), 2, 1);
m.wire(v_rack, 1, p_rack, 1); m.wire(lsens, 1, p_rack, 2);
lpf = m.place('simulink/Continuous/Integrator');
set_param(lpf, 'InitialCondition', num2str(q.p_high, 17));
err = m.mfn(sprintf('function d = f(p, y)\nd = (p - y) / %s;\nend', num2str(q.tau, 17)), 2, 1);
m.wire(p_rack, 1, err, 1); m.wire(lpf, 1, err, 2); m.wire(err, 1, lpf, 1);
ctl = m.mfn(sprintf(['function [i_out, i_in] = f(p, y, v_out, v_in)\n', ...
    'ps = p - y;\nif ps >= 0, pin = ps / %s; else, pin = ps * %s; end\n', ...
    'i_out = ps / max(v_out, 1); i_in = pin / max(v_in, 1);\nend'], num2str(q.eta, 17), num2str(q.eta, 17)), 4, 2);
m.wire(p_rack, 1, ctl, 1); m.wire(lpf, 1, ctl, 2); m.wire(v_rack, 1, ctl, 3); m.wire(v_port, 1, ctl, 4);
m.current_inject(rack, ctl, 1);
m.current_draw(port, ctl, 2);
log = m.run(s.t_end, s.dt);
t = (0:s.dt * 10:s.t_end)';
i_feed = m.series(log, isens, 'I', t);
vr = m.series(log, m.sensor_of(v_rack), 'V', t);
vs = m.series(log, m.sensor_of(v_port), 'V', t);
write(here, sprintf('emt_der_smoothing_%s.csv', name), t, {'v_rack', 'i_feed', 'v_port'}, {vr, i_feed, vs});
m.done();
end

function flywheel(q, here)
m = Model('emt_der_flywheel');
c_link = 4e-3 * q.p_rated / q.v_dc ^ 2;
p0 = q.v_dc ^ 2 / q.r_load;                          % its starting load, at its set voltage
i0 = q.v_dc / (q.r_load + q.r_dc);
v_link0 = q.v_dc;
w0 = q.speed * q.w_max;
e_max = q.e_max_kwh * 3.6e6;
J = 2 * e_max / q.w_max ^ 2;
link = m.node(); term = m.node();
m.two(m.cap(c_link, v_link0), link, m.gnd);
osens = m.place('fl_lib/Electrical/Electrical Sensors/Current Sensor');
mid = m.node();
m.two(osens, link, mid);
m.two(m.res(q.r_dc), mid, term);
m.two(m.res(q.r_load), term, m.gnd);
sw = m.switch_at(q.t_step, 1, struct('switch_r_on', 1e-6, 'switch_r_off', 1e9));
sx = m.node();
m.two(sw, term, sx);
m.two(m.res(q.r_step), sx, m.gnd);
% The rotor: an inertia driven by its machine's torque.
inertia = m.place('fl_lib/Mechanical/Rotational Elements/Inertia');
set_param(inertia, 'inertia', num2str(J, 17), 'w_specify', 'on', 'w_priority', 'High', 'w', num2str(w0, 17), 'w_unit', 'rad/s');
torque = m.place('fl_lib/Mechanical/Mechanical Sources/Ideal Torque Source');
speed = m.place('fl_lib/Mechanical/Mechanical Sensors/Ideal Rotational Motion Sensor');
mref = m.place('fl_lib/Mechanical/Rotational Elements/Mechanical Rotational Reference');
m.mech(inertia, torque, speed, mref);
w_sl = m.ps2sl(m.ports(speed).RConn(2), 'rad/s');
v_sl = m.vsense(link);
i_sl = m.ps2sl(m.ports(osens).RConn(1), 'A');
w = 2 * pi * 100; kp = 2 * 0.7 * w * c_link; ki = w * w * c_link;
ctl = m.mfn(sprintf(['function [i_m, d_int, tq] = f(v, i_out, integ, wr)\n', ...
    'err = %s - v; iref = i_out + %s * err + integ; p = v * iref;\n', ...
    's = wr / %s; pmax = %s * min(1, s / %s); if s <= %s, pmax = 0; end; pch = %s;\n', ...
    'd_int = %s * err;\nif p > pmax || p < -pch, p = min(max(p, -pch), pmax); d_int = 0; end\n', ...
    'i_m = p / max(v, 1);\nif p >= 0, mech = p / %s; else, mech = p * %s; end\n', ...
    'tq = (mech + %s * %s / 3600) / max(wr, 1e-3);\nend'], ...   % the source drives the shaft against its rotation
    num2str(q.v_dc, 17), num2str(kp, 17), num2str(q.w_max, 17), num2str(q.p_rated, 17), num2str(q.speed_base, 17), ...
    num2str(q.speed_min, 17), num2str(q.p_rated, 17), num2str(ki, 17), num2str(q.efficiency, 17), ...
    num2str(q.efficiency, 17), num2str(q.standby_percent_h / 100, 17), num2str(e_max, 17)), 4, 3);
integ = m.place('simulink/Continuous/Integrator');
set_param(integ, 'InitialCondition', '0');
m.wire(v_sl, 1, ctl, 1); m.wire(i_sl, 1, ctl, 2); m.wire(integ, 1, ctl, 3); m.wire(w_sl, 1, ctl, 4);
m.wire(ctl, 2, integ, 1);
m.current_inject(link, ctl, 1);
tq = m.sl2ps(ctl, 3, 'N*m');
m.add_line(tq, m.ports(torque).RConn(1));
log = m.run(q.t_end, q.dt);
t = (0:q.dt * 10:q.t_end)';
vt = m.series(log, m.sensor_of(v_sl), 'V', t);
wv = m.series(log, inertia, 'w', t);
write(here, 'emt_der_flywheel.csv', t, {'v_link', 'speed'}, {vt, wv / q.w_max});
m.done();
end

function [tb, pb] = cycle(s, hi, lo)
% A square cycle, its edges 1 us long.
tb = []; pb = [];
for k = 0:ceil(s.t_end / s.period)
    t0 = k * s.period;
    tb = [tb, t0, t0 + s.high, t0 + s.high + 1e-6, t0 + s.period - 1e-6]; %#ok<AGROW>
    pb = [pb, hi, hi, lo, lo]; %#ok<AGROW>
end
end

function write(here, file, t, names, cols)
tbl = array2table([t, cols{:}], 'VariableNames', [{'t'}, names]);
writetable(tbl, fullfile(here, file));
fprintf('Wrote %s: %d rows\n', file, height(tbl));
end

% --- building a Simscape model ----------------------------------------------------

function m = Model(name)
fl = 'fl_lib/Electrical/Electrical Elements/';
src = 'fl_lib/Electrical/Electrical Sources/';
if bdIsLoaded(name), close_system(name, 0); end
new_system(name);
count = 0;
nodes = {};
sensors = containers.Map('KeyType', 'double', 'ValueType', 'any');
m.gnd = [];
m.place = @place; m.ports = @ports; m.node = @node; m.connect = @connect; m.two = @two;
m.res = @res; m.cap = @cap; m.dcsrc = @dcsrc; m.switch_at = @switch_at;
m.cpl_profile_load = @cpl_profile_load; m.run = @run; m.series = @series; m.done = @done;
m.wire = @wire; m.add_line = @addl; m.mfn = @mfn; m.vsense = @vsense; m.ps2sl = @ps2sl; m.sl2ps = @sl2ps;
m.current_draw = @current_draw; m.current_inject = @current_inject; m.voltage_source = @voltage_source;
m.sensor_of = @sensor_of; m.mech = @mech;
ref = place([fl 'Electrical Reference']);
m.gnd = node();
connect(m.gnd, ports(ref).LConn(1));
solver = place('nesl_utility/Solver Configuration');
connect(m.gnd, ports(solver).RConn(1));

    function h = place(lib)
        count = count + 1;
        h = add_block(lib, sprintf('%s/b%d', name, count));
    end
    function pp = ports(h)
        pp = get_param(h, 'PortHandles');
    end
    function k = node()
        nodes{end + 1} = [];
        k = numel(nodes);
    end
    function connect(k, port)
        if isempty(nodes{k})
            nodes{k} = port;
        else
            add_line(name, nodes{k}, port, 'autorouting', 'off');
        end
    end
    function addl(a, b)
        add_line(name, a, b, 'autorouting', 'off');
    end
    function two(h, a, b)
        pp = ports(h);
        if any(strcmp(get_param(h, 'ReferenceBlock'), {[fl 'Switch']}))
            e = [pp.LConn(1), pp.RConn(2)];
        elseif any(strcmp(get_param(h, 'ReferenceBlock'), {'fl_lib/Electrical/Electrical Sensors/Current Sensor'}))
            e = [pp.LConn(1), pp.RConn(2)];
        else
            e = [pp.LConn(1), pp.RConn(1)];
        end
        connect(a, e(1));
        connect(b, e(2));
    end
    function h = res(r)
        h = place([fl 'Resistor']);
        set_param(h, 'R', num2str(r, 17));
    end
    function h = cap(c, v0)
        h = place([fl 'Capacitor']);
        set_param(h, 'c', num2str(c, 17), 'r', '0', 'g', '0', ...
            'vc_specify', 'on', 'vc_priority', 'High', 'vc', num2str(v0, 17));
    end
    function h = dcsrc(v)
        h = place([src 'DC Voltage Source']);
        set_param(h, 'v0', num2str(v, 17));
    end
    function h = switch_at(t, closed_after, p)
        h = place([fl 'Switch']);
        set_param(h, 'R_closed', num2str(p.switch_r_on, 17), 'G_open', num2str(1 / p.switch_r_off, 17), 'Threshold', '0.5');
        step = place('simulink/Sources/Step');
        set_param(step, 'Time', num2str(t, 17), 'Before', num2str(~closed_after), 'After', num2str(closed_after));
        conv = place('nesl_utility/Simulink-PS Converter');
        addl(ports(step).Outport(1), ports(conv).Inport(1));
        addl(ports(conv).RConn(1), ports(h).RConn(1));
    end
    function wire(a, ka, b, kb)
        addl(ports(a).Outport(ka), ports(b).Inport(kb));
    end
    function h = mfn(code, nin, nout) %#ok<INUSD>
        h = place('simulink/User-Defined Functions/MATLAB Function');
        chart = sfroot().find('-isa', 'Stateflow.EMChart', 'Path', getfullname(h));
        chart.Script = code;
    end
    function out = ps2sl(port, unit)
        out = place('nesl_utility/PS-Simulink Converter');
        set_param(out, 'Unit', unit);
        addl(port, ports(out).LConn(1));
    end
    function port = sl2ps(h, k, unit)
        conv = place('nesl_utility/Simulink-PS Converter');
        set_param(conv, 'Unit', unit);
        addl(ports(h).Outport(k), ports(conv).Inport(1));
        port = ports(conv).RConn(1);
    end
    function out = vsense(n)
        sense = place('fl_lib/Electrical/Electrical Sensors/Voltage Sensor');
        sp = ports(sense);
        connect(n, sp.LConn(1));
        connect(m.gnd, sp.RConn(2));
        out = ps2sl(sp.RConn(1), 'V');
        sensors(out) = sense;
    end
    function s = sensor_of(out)
        s = sensors(out);
    end
    function current_draw(n, h, k)
        % A current from node n to the reference, the Simulink signal (h, k).
        isrc = place([src 'Controlled Current Source']);
        ip = ports(isrc);
        addl(sl2ps(h, k, 'A'), ip.RConn(1));
        connect(n, ip.RConn(2));
        connect(m.gnd, ip.LConn(1));
    end
    function current_inject(n, h, k)
        isrc = place([src 'Controlled Current Source']);
        ip = ports(isrc);
        addl(sl2ps(h, k, 'A'), ip.RConn(1));
        connect(m.gnd, ip.RConn(2));
        connect(n, ip.LConn(1));
    end
    function voltage_source(n, h, k)
        vs = place([src 'Controlled Voltage Source']);
        vp = ports(vs);
        addl(sl2ps(h, k, 'V'), vp.RConn(1));
        connect(n, vp.LConn(1));
        connect(m.gnd, vp.RConn(2));
    end
    function out = cpl_profile_load(n, t_bp, p_bp)
        % i = P(t) / v from node n to the reference; returns the current as a Simulink signal.
        sense = place('fl_lib/Electrical/Electrical Sensors/Voltage Sensor');
        sp = ports(sense);
        connect(n, sp.LConn(1));
        connect(m.gnd, sp.RConn(2));
        clk = place('simulink/Sources/Clock');
        tbl = place('simulink/Lookup Tables/1-D Lookup Table');
        set_param(tbl, 'BreakpointsForDimension1', mat2str(t_bp, 17), 'Table', mat2str(p_bp, 17), ...
            'InterpMethod', 'Linear point-slope', 'ExtrapMethod', 'Clip');
        addl(ports(clk).Outport(1), ports(tbl).Inport(1));
        vsl = ps2sl(sp.RConn(1), 'V');
        div = mfn(sprintf('function i = f(p, v)\ni = p / max(v, 1);\nend'), 2, 1);
        wire(tbl, 1, div, 1); wire(vsl, 1, div, 2);
        current_draw(n, div, 1);
        out = div;
    end
    function mech(inertia, torque, speed, mref)
        % The inertia, the torque source and the speed sensor on one shaft, against the reference.
        ip = ports(inertia); tp = ports(torque); sp = ports(speed); rp = ports(mref);
        addl(ip.LConn(1), tp.RConn(2));
        addl(tp.LConn(1), rp.LConn(1));
        addl(ip.LConn(1), sp.LConn(1));
        addl(sp.RConn(1), rp.LConn(1));
        % Its own physical network: its own solver configuration.
        msolver = place('nesl_utility/Solver Configuration');
        addl(ports(msolver).RConn(1), rp.LConn(1));
    end
    function log = run(t_end, max_step)
        set_param(name, 'SimscapeLogType', 'all', 'SimscapeLogName', 'simlog', 'SimscapeLogLimitData', 'off', ...
            'StopTime', num2str(t_end, 17), 'Solver', 'ode23t', 'MaxStep', num2str(max_step, 17), ...
            'RelTol', '1e-7', 'AbsTol', '1e-6');
        out = sim(name, 'ReturnWorkspaceOutputs', 'on');
        log = out.get('simlog');
    end
    function done()
        close_system(name, 0);
    end
    function y = series(log, h, var, t)
        s = log.(get_param(h, 'Name')).(var).series;
        [tt, k] = unique(s.time, 'last');
        vals = s.values;
        y = interp1(tt, vals(k), t);
    end
end

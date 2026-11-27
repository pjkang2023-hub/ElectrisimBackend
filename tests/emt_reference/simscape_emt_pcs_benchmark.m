function simscape_emt_pcs_benchmark()
% SIMSCAPE_EMT_PCS_BENCHMARK  Reference waveforms for grid-forming PCS (emt_converters.GridFormingVsc):
% two of them islanded from the grid, sharing a load by their droops.
%
% The circuit of emt_pcs_benchmark_params.json in Simscape's Foundation
% Library: the grid behind its impedance, its breaker (each phase opening at
% its current's zero after the islanding time), the R-L load, and each PCS's
% averaged bridge voltage behind its reactor - its controller, Electrisim's
% GridFormingVsc.control, in a MATLAB Function block sampling every dt. Its
% DC side, stiff, is left out. Writes emt_pcs_island.csv.

here = fileparts(mfilename('fullpath'));
p = jsondecode(fileread(fullfile(here, 'emt_pcs_benchmark_params.json')));
s = steady(p);
w = 2 * pi * p.f_hz;
mdl = 'emt_pcs_island';
fl = 'fl_lib/Electrical/Electrical Elements/';
srcs = 'fl_lib/Electrical/Electrical Sources/';
sens = 'fl_lib/Electrical/Electrical Sensors/';
if bdIsLoaded(mdl), close_system(mdl, 0); end
gen = Simulink.fileGenControl('getConfig');
build = fullfile(tempdir, mdl);
Simulink.fileGenControl('set', 'CacheFolder', build, 'CodeGenFolder', build, 'createDir', true);
restore = onCleanup(@() Simulink.fileGenControl('setConfig', 'config', gen));
new_system(mdl);
cleanup = onCleanup(@() close_system(mdl, 0));
count = 0;
nodes = {};
P = @(h) get_param(h, 'PortHandles');
gnd = node();
connect(gnd, P(place([fl 'Electrical Reference'])).LConn(1));
connect(gnd, P(place('nesl_utility/Solver Configuration')).RConn(1));
ang = -(0:2) * 2 * pi / 3;
npcs = numel(p.pcs);
pcc = zeros(1, 3);
lg = zeros(1, 3); lc = zeros(npcs, 3); vcap = zeros(1, 3);
% Each PCS's controller: its PCC voltages and its currents in, its bridge voltages out - sampled every dt, 1 ns late.
ctrl = zeros(1, npcs); mux = zeros(1, npcs); dmx = zeros(1, npcs);
for j = 1:npcs
    c = place('simulink/User-Defined Functions/MATLAB Function');
    set_param(c, 'Name', sprintf('ctrl%d', j));
    ctrl(j) = get_param([mdl sprintf('/ctrl%d', j)], 'Handle');
    chart = find(sfroot, '-isa', 'Stateflow.EMChart', 'Path', [mdl sprintf('/ctrl%d', j)]);
    chart.Script = controller_code(p, s, j);
    mux(j) = place('simulink/Signal Routing/Mux'); set_param(mux(j), 'Inputs', '6');
    zoh = place('simulink/Discrete/Zero-Order Hold'); set_param(zoh, 'SampleTime', num2str(p.dt, 17));
    lag = place('simulink/Continuous/Transport Delay'); set_param(lag, 'DelayTime', '1e-9', 'InitialOutput', '0');
    add_line(mdl, P(mux(j)).Outport(1), P(lag).Inport(1));
    add_line(mdl, P(lag).Outport(1), P(zoh).Inport(1));
    add_line(mdl, P(zoh).Outport(1), P(ctrl(j)).Inport(1));
    dmx(j) = place('simulink/Signal Routing/Demux'); set_param(dmx(j), 'Outputs', '3');
    add_line(mdl, P(ctrl(j)).Outport(1), P(dmx(j)).Inport(1));
end
% The breaker's control: each phase closed until its current passes zero after the islanding time.
brk = place('simulink/User-Defined Functions/MATLAB Function');
set_param(brk, 'Name', 'breaker');
brk = get_param([mdl '/breaker'], 'Handle');
chart = find(sfroot, '-isa', 'Stateflow.EMChart', 'Path', [mdl '/breaker']);
chart.Script = breaker_code(p);
bmux = place('simulink/Signal Routing/Mux'); set_param(bmux, 'Inputs', '2');
bzoh = place('simulink/Discrete/Zero-Order Hold'); set_param(bzoh, 'SampleTime', num2str(p.dt, 17));
blag = place('simulink/Continuous/Transport Delay'); set_param(blag, 'DelayTime', '1e-9', 'InitialOutput', '0');
clk = place('simulink/Sources/Clock');
imux = place('simulink/Signal Routing/Mux'); set_param(imux, 'Inputs', '3');
add_line(mdl, P(clk).Outport(1), P(bmux).Inport(1));
add_line(mdl, P(imux).Outport(1), P(bmux).Inport(2));
add_line(mdl, P(bmux).Outport(1), P(blag).Inport(1));
add_line(mdl, P(blag).Outport(1), P(bzoh).Inport(1));
add_line(mdl, P(bzoh).Outport(1), P(brk).Inport(1));
bdmx = place('simulink/Signal Routing/Demux'); set_param(bdmx, 'Outputs', '3');
add_line(mdl, P(brk).Outport(1), P(bdmx).Inport(1));

for k = 1:3
    % The grid: its source behind R_g and L_g, its breaker.
    sine = place('simulink/Sources/Sine Wave');
    set_param(sine, 'Amplitude', num2str(s.E, 17), 'Frequency', num2str(w, 17), 'Phase', num2str(ang(k) + pi / 2, 17));
    a = node(); b = node(); g = node(); pcc(k) = node();
    controlled(place([srcs 'Controlled Voltage Source']), P(sine).Outport(1), 'V', gnd, a);
    two(res(s.r_g), a, b);
    lg(k) = ind(s.l_g, real(s.I * exp(1i * ang(k))));
    two(lg(k), b, g);
    cs = place([sens 'Current Sensor']);
    cv = place('nesl_utility/PS-Simulink Converter'); set_param(cv, 'Unit', 'A');
    add_line(mdl, P(cs).RConn(1), P(cv).LConn(1));
    add_line(mdl, P(cv).Outport(1), P(imux).Inport(k));
    x = node();
    two_sensor(cs, g, x);
    sw = place([fl 'Switch']);
    set_param(sw, 'R_closed', num2str(p.switch_r_on, 17), 'G_open', num2str(1 / p.switch_r_off, 17), 'Threshold', '0.5');
    cv = place('nesl_utility/Simulink-PS Converter');
    add_line(mdl, P(bdmx).Outport(k), P(cv).Inport(1));
    pp = P(sw);
    add_line(mdl, P(cv).RConn(1), pp.RConn(1));
    connect(x, pp.LConn(1)); connect(pcc(k), pp.RConn(2));
    % The load: R then L to the reference.
    y = node();
    two(res(s.r_l), pcc(k), y);
    two(ind(s.l_l, real(s.I * exp(1i * ang(k)))), y, gnd);
    % Its PCC voltage, to each controller.
    vs = place([sens 'Voltage Sensor']);
    two_sensor(vs, pcc(k), gnd);
    vcap(k) = vs;
    vcv = place('nesl_utility/PS-Simulink Converter'); set_param(vcv, 'Unit', 'V');
    add_line(mdl, P(vs).RConn(1), P(vcv).LConn(1));
    for j = 1:npcs
        add_line(mdl, P(vcv).Outport(1), P(mux(j)).Inport(k));
        % Its bridge voltage behind its reactor, its current into the PCC.
        e = node(); r_ = node(); z = node();
        controlled(place([srcs 'Controlled Voltage Source']), P(dmx(j)).Outport(k), 'V', gnd, e);
        two(res(s.r(j)), e, r_);
        lc(j, k) = ind(s.l(j), 0);
        two(lc(j, k), r_, z);
        ics = place([sens 'Current Sensor']);
        icv = place('nesl_utility/PS-Simulink Converter'); set_param(icv, 'Unit', 'A');
        add_line(mdl, P(ics).RConn(1), P(icv).LConn(1));
        add_line(mdl, P(icv).Outport(1), P(mux(j)).Inport(3 + k));
        two_sensor(ics, z, pcc(k));
    end
end
set_param(mdl, 'SimscapeLogType', 'all', 'SimscapeLogName', 'simlog', 'SimscapeLogLimitData', 'off', ...
    'StopTime', num2str(p.t_end, 17), 'Solver', 'ode23t', 'MaxStep', num2str(p.dt, 17), 'RelTol', '1e-7', 'AbsTol', '1e-6');
out = sim(mdl, 'ReturnWorkspaceOutputs', 'on', 'TimeOut', 1800);
log = out.get('simlog');
t = (0:p.dt_out:p.t_end)';
cols = {}; names = {};
for j = 1:npcs
    pw = zeros(size(t));
    for k = 1:3
        pw = pw + series(log, vcap(k), 'V', t) .* series(log, lc(j, k), 'i', t);
    end
    cols{end + 1} = pw; names{end + 1} = sprintf('p%d', j); %#ok<AGROW>
end
cols{end + 1} = series(log, vcap(1), 'V', t); names{end + 1} = 'v_a';
cols{end + 1} = series(log, lg(1), 'i', t); names{end + 1} = 'i_grid_a';
tbl = array2table([t, cols{:}], 'VariableNames', [{'t'}, names]);
writetable(tbl, fullfile(here, 'emt_pcs_island.csv'));
fprintf('Wrote emt_pcs_island.csv: %d rows\n', height(tbl));

    function h = place(lib)
        count = count + 1;
        h = add_block(lib, sprintf('%s/b%d', mdl, count));
    end
    function k_ = node()
        nodes{end + 1} = [];
        k_ = numel(nodes);
    end
    function connect(k_, port)
        if isempty(nodes{k_})
            nodes{k_} = port;
        else
            add_line(mdl, nodes{k_}, port, 'autorouting', 'off');
        end
    end
    function two(h, a_, b_)
        pp_ = P(h);
        connect(a_, pp_.LConn(1));
        connect(b_, pp_.RConn(1));
    end
    function two_sensor(h, a_, b_)
        pp_ = P(h);
        connect(a_, pp_.LConn(1));
        connect(b_, pp_.RConn(2));
    end
    function controlled(h, from, unit, a_, b_)
        cv_ = place('nesl_utility/Simulink-PS Converter');
        set_param(cv_, 'Unit', unit);
        add_line(mdl, from, P(cv_).Inport(1));
        pp_ = P(h);
        add_line(mdl, P(cv_).RConn(1), pp_.RConn(1));
        connect(b_, pp_.LConn(1));
        connect(a_, pp_.RConn(2));
    end
    function h = res(r)
        h = place([fl 'Resistor']);
        set_param(h, 'R', num2str(r, 17));
    end
    function h = ind(l, i0)
        h = place([fl 'Inductor']);
        set_param(h, 'l', num2str(l, 17), 'r', '0', 'g', '0', ...
            'i_L_specify', 'on', 'i_L_priority', 'High', 'i_L', num2str(i0, 17));
    end
end

function y = series(log, h, var, t)
s = log.(get_param(h, 'Name')).(var).series;
[tt, k] = unique(s.time, 'last');
vals = s.values;
y = interp1(tt, vals(k), t);
end

function s = steady(p)
% The grid supplies the load; the PCS carry nothing, their bridge voltages the PCC's.
w = 2 * pi * p.f_hz;
g = p.grid;
z = g.v_ll ^ 2 / g.s_sc;
s.r_g = z / sqrt(1 + g.xr ^ 2);
s.l_g = g.xr * s.r_g / w;
v_ph = g.v_ll / sqrt(3);
zl = abs(v_ph) ^ 2 / conj((p.load.p + 1i * p.load.q) / 3);
s.r_l = real(zl); s.l_l = imag(zl) / w;
s.E = sqrt(2 / 3) * g.v_ll;
zg = s.r_g + 1i * w * s.l_g;
s.V = s.E * zl / (zl + zg);
s.I = s.V / zl;
for j = 1:numel(p.pcs)
    zb = g.v_ll ^ 2 / p.pcs(j).s_rated;
    s.r(j) = p.pcs(j).r_pu * zb;
    s.l(j) = p.pcs(j).x_pu * zb / w;
end
end

function code = breaker_code(p)
lines = {
'function y = brk(u)'
'% u: the time, the grid''s three phase currents. Each phase closed until its current changes sign after the islanding time.'
'persistent opened last'
'if isempty(opened), opened = false(3, 1); last = nan(3, 1); end'
'i = u(2:4);'
sprintf('if u(1) >= %.17g - 1e-12', p.t_island)
'    for k = 1:3'
'        if ~opened(k) && ~isnan(last(k)) && (i(k) == 0 || (i(k) > 0) ~= (last(k) > 0)), opened(k) = true; end'
'        last(k) = i(k);'
'    end'
'end'
'y = double(~opened);'};
code = strjoin(lines, newline);
end

function code = controller_code(p, s, j)
% Electrisim's GridFormingVsc.control, its constants and starting state written in.
q = p.pcs(j);
w0 = 2 * pi * p.f_hz;
v_nom_peak = sqrt(2 / 3) * p.grid.v_ll;
c.w0 = w0; c.dt = p.dt; c.tau = p.tau_f;
c.m_p = q.droop_pf * w0 / q.s_rated;
c.n_q = q.droop_qv * v_nom_peak / q.s_rated;
c.i_max = p.limit_pu * sqrt(2) * q.s_rated / (sqrt(3) * p.grid.v_ll);
c.e0 = abs(s.V); c.th0 = angle(s.V);
c.r = s.r(j); c.l = s.l(j);
c.p_set = 0; c.q_set = 0;
lines = {
'function e = ctrl(u)'
'% u: PCC voltages (3), its currents into the PCC (3).'
'persistent th p_f q_f started'
'if isempty(started)'
'    th = C_th0; p_f = C_p_set; q_f = C_q_set; started = false;'
'end'
'v = u(1:3); i = u(4:6);'
'if ~started'
'    started = true;'
'    e = C_e0 * cos(C_th0 + 0.5 * C_w0 * C_dt - [0; 2; 4] * pi / 3);'
'    return'
'end'
'dt = C_dt;'
'p = sum(v .* i);'
'q = ((v(2) - v(3)) * i(1) + (v(3) - v(1)) * i(2) + (v(1) - v(2)) * i(3)) / sqrt(3);'
'a = 1 - exp(-dt / C_tau);'
'p_f = p_f + (p - p_f) * a; q_f = q_f + (q - q_f) * a;'
'w = C_w0 - C_m_p * (p_f - C_p_set);'
'theta = th + w * dt;'
'emag = C_e0 - C_n_q * (q_f - C_q_set);'
'ed = emag; eq = 0;'
'ang = theta - [0; 2; 4] * pi / 3;'
'i_d = 2 / 3 * sum(i .* cos(ang)); i_q = -2 / 3 * sum(i .* sin(ang));'
'v_d = 2 / 3 * sum(v .* cos(ang)); v_q = -2 / 3 * sum(v .* sin(ang));'
'iest = hypot(ed - v_d, eq - v_q) / hypot(C_r, w * C_l);'
'if iest > C_i_max'
'    k = C_i_max / iest; ed = v_d + (ed - v_d) * k; eq = v_q + (eq - v_q) * k;'
'end'
'b = theta + 0.5 * w * dt - [0; 2; 4] * pi / 3;'
'e = ed * cos(b) - eq * sin(b);'
'th = theta;'};
code = strjoin(lines, newline);
names = fieldnames(c);
[~, order] = sort(-cellfun(@numel, names));
for f = names(order)'
    code = strrep(code, ['C_' f{1}], sprintf('%.17g', c.(f{1})));
end
end

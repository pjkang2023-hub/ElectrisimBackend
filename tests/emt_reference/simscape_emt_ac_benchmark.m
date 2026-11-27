function simscape_emt_ac_benchmark()
% SIMSCAPE_EMT_AC_BENCHMARK  Reference waveforms for the EMT study of AC networks.
%
% Builds the circuit in emt_ac_benchmark_params.json from Simscape
% Electrical's own three-phase blocks - Voltage Source, Transmission Line,
% Two-Winding Transformer, Wye-Connected Load and Fault - so the benchmark is
% independent of Electrisim's element models, simulates each case and writes
% emt_ac_<case>.csv. tests/test_emt_ac.py builds Electrisim's circuit from the
% same file. Both start from rest - every current and voltage zero - and
% energise the network at t = 0. Needs Simscape Electrical.
%
% Waveforms: i_src_{a,b,c} the source's phase currents into the network,
% v_lv_{a,b,c} the 0.4 kV bus's phase voltages, v_hv_{a,b,c} the 20 kV bus's
% (the transformer's HV terminals).

here = fileparts(mfilename('fullpath'));
p = jsondecode(fileread(fullfile(here, 'emt_ac_benchmark_params.json')));
for name = fieldnames(p.cases)'
    run_case(p, name{1}, here);
end
end

function run_case(p, name, here)
c = p.cases.(name);
mdl = ['emt_ac_' name];
if bdIsLoaded(mdl), close_system(mdl, 0); end
new_system(mdl);
cleanup = onCleanup(@() close_system(mdl, 0));
P = @(h) get_param(h, 'PortHandles');
blk = @(lib, nm) add_block(lib, [mdl '/' nm]);

src = blk(sprintf('ee_lib/Sources/Voltage\nSource\n(Three-Phase)'), 'src');
% Simscape's phase a is a sine: Electrisim's cos(w t + phi) is a shift of phi + 90 degrees.
set_param(src, 'vline_rms', num2str(p.v_ll, 17), 'freq', num2str(p.f_hz), 'shift', num2str(p.phase_deg + 90), ...
    'SShortCircuit', num2str(p.s_sc, 17), 'XR', num2str(p.xr, 17));
q = p.line;
ln = blk(sprintf('ee_lib/Passive/Lines/Transmission\nLine\n(Three-Phase)'), 'ln');
% The block takes mH/km and uF/km.
set_param(ln, 'length', num2str(q.km, 17), 'freq', num2str(p.f_hz), 'R', num2str(q.r_per_km, 17), ...
    'L', num2str(q.l_per_km * 1e3, 17), 'M', num2str(q.m_per_km * 1e3, 17), 'Cl', num2str(q.cl_per_km * 1e6, 17), ...
    'Cg', num2str(q.cg_per_km * 1e6, 17), 'Rm', '0', 'N', num2str(q.sections));
q = p.trafo;
tr = blk(sprintf('ee_lib/Passive/Transformers/Two-Winding\nTransformer\n(Three-Phase)'), 'tr');
% Clock 11: the star leads the delta by 30 degrees - the delta lags, 1 o'clock.
delta = 'ee.enum.windingconnection.delta1';
if q.clock == 1, delta = 'ee.enum.windingconnection.delta11'; end
set_param(tr, 'SRated', num2str(q.s, 17), 'FRated', num2str(p.f_hz), 'Winding1Connection', delta, ...
    'VRated1', num2str(q.v1, 17), 'Winding2Connection', 'ee.enum.windingconnection.Yg', 'VRated2', num2str(q.v2, 17), ...
    'CoreType', 'ee.enum.coretype.fivelimb', 'pu_Rw1', num2str(q.rw_pu, 17), 'pu_Rw2', num2str(q.rw_pu, 17), ...
    'leakage_reactance_option', 'ee.enum.transformer_leakage.include', 'pu_Xl1', num2str(q.xl_pu, 17), ...
    'pu_Xl2', num2str(q.xl_pu, 17), 'magnetizing_reactance_option', 'ee.enum.transformer_magnetizingReactance.include', ...
    'pu_Xm', num2str(q.xm_pu, 17), 'magnetizing_resistance_option', 'ee.enum.transformer_magnetizingResistance.include', ...
    'pu_Rm', num2str(q.rm_pu, 17));
ld = blk('ee_lib/Passive/RLC Assemblies/Wye-Connected Load', 'ld');
set_param(ld, 'parameterization', 'ee.enum.rlc.parameterization.direct', 'component_structure', 'ee.enum.rlc.structure.R', ...
    'R', num2str(p.load_r, 17));
ft = blk('ee_lib/Utilities/Fault (Three-Phase)', 'ft');
set_param(ft, 'fault_type_option', ['ee.enum.fault_type_option.' c.type], 'enable_temporal_fault', '1', ...
    'fault_start_time', num2str(c.t_on, 17), 'fault_duration', num2str(c.duration, 17), ...
    'R_pn_fault', num2str(c.r, 17), 'R_ng_fault', num2str(c.r, 17));
ref = blk('fl_lib/Electrical/Electrical Elements/Electrical Reference', 'ref');
sol = blk('nesl_utility/Solver Configuration', 'sol');
add_line(mdl, P(src).RConn(1), P(ln).LConn(1));
add_line(mdl, P(src).LConn(1), P(ref).LConn(1));
add_line(mdl, P(ln).RConn(1), P(tr).LConn(1));
add_line(mdl, P(tr).RConn(1), P(ld).LConn(1));
add_line(mdl, P(ld).RConn(1), P(ref).LConn(1));
add_line(mdl, P(sol).RConn(1), P(ref).LConn(1));
if strcmp(c.bus, 'lv')
    add_line(mdl, P(ft).LConn(1), P(ld).LConn(1));
else
    add_line(mdl, P(ft).LConn(1), P(tr).LConn(1));
end
set_param(mdl, 'SimscapeLogType', 'all', 'SimscapeLogName', 'simlog', 'SimscapeLogLimitData', 'off', ...
    'StopTime', num2str(c.t_end, 17), 'Solver', 'ode23t', 'MaxStep', '1e-5', 'RelTol', '1e-7', 'AbsTol', '1e-6');
out = sim(mdl, 'ReturnWorkspaceOutputs', 'on', 'TimeOut', 300);
log = out.get('simlog');
t = (0:1e-5:c.t_end)';
phases = 'abc';
cols = {}; names = {};
s = log.src.I.series;                      % into the source: the network's current is its negative
[ts, k] = unique(s.time, 'last'); v = s.values('A');
for ph = 1:3, cols{end + 1} = -interp1(ts, v(k, ph), t); names{end + 1} = sprintf('i_src_%s', phases(ph)); end
for spec = {{'ld', 'v_lv'}, {'tr', 'v_hv'}}
    node = log.(spec{1}{1});
    ids = node.childIds;
    hit = find(ismember(ids, {'N', 'N1'}), 1);   % the load's node; the transformer's HV terminals
    if isempty(hit)
        fprintf('%s logs: %s - its voltage left out\n', spec{1}{1}, strjoin(ids, ', '));
        continue
    end
    nd = node.(ids{hit});
    s = nd.V.series;
    [ts, k] = unique(s.time, 'last'); v = s.values('V');
    for ph = 1:3, cols{end + 1} = interp1(ts, v(k, ph), t); names{end + 1} = sprintf('%s_%s', spec{1}{2}, phases(ph)); end
end
tbl = array2table([t, cols{:}], 'VariableNames', [{'t'}, names]);
writetable(tbl, fullfile(here, ['emt_ac_' name '.csv']));
fprintf('Wrote emt_ac_%s.csv: %d rows (transformer nodes: %s)\n', name, height(tbl), strjoin(log.tr.childIds, ', '));
end

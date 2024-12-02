function [We, Te, fuel] = getHEVPowerSplit(time, speed, gradePercent)
% HEV power management strategy originally wrote by Azrin.
% 
% it is a rule based strategy and details can be referred to
%
% Zulkefli, M.A.M., Zheng, J., Sun, Z. and Liu, H.X., 2014. Hybrid
% powertrain optimization with trajectory prediction based on
% inter-vehicle-communication and vehicle-infrastructure-integration.
% Transportation Research Part C: Emerging Technologies, 45, pp.41-63.
% 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Gasoline engine related map/model %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% engine fuel consumption model
enginemap_spd=[1000 1250 1500 1750 2000 2250 2500 2750 3000 3250 3500 4000]*2*pi/60;  % (rad/s), speed range of the engine
lbft2Nm=1.356; %conversion from lbft to Nm
enginemap_trq=[6.3 12.5 18.8 25.1 31.3 37.6 43.9 50.1 56.4 62.7 68.9 75.2]*lbft2Nm*20/14;  % (N*m), torque range of the engine

% (g/s), fuel use map indexed vertically by enginemap_spd and horizontally by enginemap_trq
enginemap = [
 0.1513  0.1984  0.2455  0.2925  0.3396  0.3867  0.4338  0.4808  0.5279  0.5279  0.5279  0.5279 
 0.1834  0.2423  0.3011  0.3599  0.4188  0.4776  0.5365  0.5953  0.6541  0.6689  0.6689  0.6689 
 0.2145  0.2851  0.3557  0.4263  0.4969  0.5675  0.6381  0.7087  0.7793  0.8146  0.8146  0.8146 
 0.2451  0.3274  0.4098  0.4922  0.5746  0.6570  0.7393  0.8217  0.9041  0.9659  0.9659  0.9659 
 0.2759  0.3700  0.4642  0.5583  0.6525  0.7466  0.8408  0.9349  1.0291  1.1232  1.1232  1.1232 
 0.3076  0.4135  0.5194  0.6253  0.7312  0.8371  0.9430  1.0490  1.1549  1.2608  1.2873  1.2873 
 0.3407  0.4584  0.5761  0.6937  0.8114  0.9291  1.0468  1.1645  1.2822  1.3998  1.4587  1.4587 
 0.3773  0.5068  0.6362  0.7657  0.8951  1.0246  1.1540  1.2835  1.4129  1.5424  1.6395  1.6395 
 0.4200  0.5612  0.7024  0.8436  0.9849  1.1261  1.2673  1.4085  1.5497  1.6910  1.8322  1.8322 
 0.4701  0.6231  0.7761  0.9290  1.0820  1.2350  1.3880  1.5410  1.6940  1.8470  1.9999  2.0382 
 0.5290  0.6938  0.8585  1.0233  1.1880  1.3528  1.5175  1.6823  1.8470  2.0118  2.1766  2.2589 
 0.6789  0.8672  1.0555  1.2438  1.4321  1.6204  1.8087  1.9970  2.1852  2.3735  2.5618  2.7501 ];
[T,w]=meshgrid(enginemap_trq, enginemap_spd);
enginemap_kW=T.*w/1000;
enginemap_gpkWh = enginemap./enginemap_kW*3600;

% Draw Max Tq Line
MaxTq_pt = [enginemap_trq(1) 77.2920 82.0380 84.7500 86.7840 89.3604 91.1232 92.8860 94.6488 96.4116 98.1744 99.9372 101.9712];

MaxSp_pt = [1000 1010 1250 1500 1750 2000 2250 2500 2750 3000 3250 3500 4000];

% Max Tq Line Data Resampling
MaxSp = 1000:1:4000;
MaxTq = interp1(MaxSp_pt,MaxTq_pt,MaxSp);
% from RPM to rad/s
MaxSp = MaxSp*2*pi/60;

% Vehicle parameters              
Cd = 0.25;
A = 2;
rho = 1.2;
ftire = 0.015;
Rtire = 0.3;
Mv = 2000;
g = 9.8;

Voc = 201.6; %volt
Qbatt = 6.5*3600; % ampere*sec
Rbatt = 0.003*6*28;  % ohm

nm = 0.85;
ng = 0.85;

S = 30;
Rr = 78;
R = Rr/S;
K = 4.113;
Kratio = 4.113;

kcom = [1 1 -1 -1;1 -1 1 -1];
SOCtarget = 0.6;

% Set empty Matrices to combine info from different time segments
PbattF_all = [];
aoM_all = [];
PbattMsv_all = [];
fo_all = [];

% RB Fitting Value
cvfit = 200; %15pct short baseline

% cvfit = 230; %
% Preqmin = 3e3; % 10e3, maybe increase this one % IF REQUESTED POWER LESS THAN CERTAIN VALUE & SOC above certain value, only use battery
Preqmin = 4e3;
BATT_CHARGE_RATE = 1.5; % 1.5,

% xlsHandle = xlsread('Results_Yunli_Powertrain', '6%Raw Short');
% filename = 'VelProfMat_VTM_v1';
% rulebaseSaveName = '.\hevOptFcn\fuelConsRuleBase_VTM_v1';
% CASE_NAME = 'onlyVeh'; % baseline onlyVeh
% load(filename)

gradeActRad = atan(gradePercent/100);
velAct = speed;
timeAct = time; % [sec]

wVehAct = velAct/Rtire;

%--------------------------------------------------------------------------
% modify these codes
%--------------------------------------------------------------------------

% iT = 1;
% fT = length(dlmread('VehSpeed_RoadGrade.txt'));
% fT = numel(velAct);
% Charge/Discharge battery 100s before final time
%--------------------------------------------------------------------------
% charge sustain time

CS_time = 100; 

chargeT = find(time<time(end)-CS_time,1,'last'); %fT - (CS_time/dt) + 1;

% Read Actual Wv from "Use_Engine_Op.txt" (unit rads/s) - taken from Optimized Result Drive cycle
Wv = wVehAct;

% Road grade angle (unit degrees)
phiM = gradeActRad;

% Vehicle acceleration (unit rads/s^2)
dWv = (Wv(2:end) - Wv(1:end-1))./diff(timeAct);

% Initial SOC
        
SOCint = 0.6;
SOCi = SOCint;
SOCinit = SOCi;

% We size
WeM = linspace(1000*2*pi/60, 4000*2*pi/60, 30);
% PbattM size
N = 30;

fM = [];
TM = [];
WM = [];
fMnew = [];
TMnew = [];
WMnew = [];
fo = [];
Teo = [];
Weo = [];

countin = 0;
countout = 0;
c = 0;
d = 0;
cno = [];
cover = [];
Pbat_lim = [];
PbattMsv = [];
Preq_Wheel = [];
Preq_SOC = [];
Preq = [];
TeM = [];
SOC = [];

brakeon = 0;
recharge_flag = 0;

tReqHist = nan(numel(Wv), 1);

f = waitbar(0,sprintf('Calculating HEV Power-Split... %.2f%%', 0));
for i = 1:length(Wv)
    waitbar(i/length(Wv),f,sprintf('Calculating HEV Power-Split... %d%%', floor(i/length(Wv)*100)));

    if i == length(Wv)
        dt = timeAct(end)-timeAct(end-1);
    else
        dt = timeAct(i+1)-timeAct(i);
    end

    Wreq = Wv(i);
    phi = phiM(i);
    
    if i == length(Wv)
        Accreq = 0;
    else
        Accreq = Rtire*dWv(i);
    end
    
    if Wreq <= 0.02
        Treq = 0;
        Wreq = 0;
    else
        
        % During decel braking
        if Accreq < 0 
            % Torque request felt by the motor (Forces have to be multiplied with Rtire)
            Treq1 = ( (ftire*Mv*g*cos(phi) + Mv*g*sin(phi) + 0.5*rho*Cd*A*Rtire^2*Wreq^2 + Mv*Accreq)*Rtire )/K;
            
            % Max Torque available by the motor (50 kWatts)
            Tmot1 = -5e4/(K*Wreq);
            
            % Cap motor torque to 400 Nm in the low-speed regions
            if Tmot1 < -400
                Tmot1 = -400;
            else
                dummy =0;
            end
            
            % During decel, if req torque is bigger than motor max torque
            % limit to motor max torque, the rest is dissipated through
            % braking
            if Treq1 < Tmot1
                % torque before the ratio
                Treq = Tmot1*K;
            else
                % torque before the ratio
                Treq = Treq1*K; 
            end
               
        else
            % Torque request (Forces have to be multiplied with Rtire)
            Treq = (ftire*Mv*g*cos(phi) + Mv*g*sin(phi) + 0.5*rho*Cd*A*Rtire^2*Wreq^2 + Mv*Accreq)*Rtire;
        end
        
    end
        
    % % Transmission Efficiency = 90%
    % Treq = Treq*1.11111; 
    
    tReqHist(i) = Treq;
    
    Preq_Wheel(i) = Wreq*Treq;
    
    if (i == 1) || (SOCi >= 0.8 )
    
        Preq_SOC(i) = 0;
    
    else
        
        % From previous step
        
        calc1 = Voc^2 -  4*Rbatt*Pbatt(i-1);
        
        Ibatt = (Voc - sign(calc1)*abs(calc1)^0.5 )/(2*Rbatt);
        
        Vbus = Voc - Ibatt*Rbatt;
        
        if (SOCtarget - SOCi) > 0
            kk = -1;
        else
            kk = 1;
        end

        Preq_SOC(i) = (SOCtarget - SOCi)*cvfit*Vbus*ng^kk;
    
    end
    
    
    Preq(i) = Preq_Wheel(i) + Preq_SOC(i);
    
    
    % If time is 100s before the final time, do SOC correction
    if i >= chargeT    
        diff_SOC = 0.001;
        
        % If SOC is lower than SOCinit, charge battery
        % Keep charging until SOCinit is reached
        if (SOCi < SOCinit) && (abs(SOCi-SOCinit) > diff_SOC)

            Treq2 = BATT_CHARGE_RATE*Preq(i)/Wreq; % request more torque to charge battery
            
            Te_nobatt = (Treq2/Kratio)*(R+1)/R;
            
            if Te_nobatt > enginemap_trq(end)
                Te_nobatt = enginemap_trq(end);
            elseif Te_nobatt < enginemap_trq(1)
                Te_nobatt = enginemap_trq(1);
            else
                dummy1 = 1;
            end
            
            for j = 1:length(WeM)
                
                We = WeM(j);
                TeM(j,i) = Te_nobatt;
                fM(j,i) = interp2(enginemap_trq,enginemap_spd,enginemap,Te_nobatt,We, 'linear')*dt;
            end
            
            Tresam = interp1(WeM,TeM(:,i),MaxSp, 'pchip');
            Wresam = MaxSp;
            
            % Find min difference bet Tq (intersection)
            dTints = abs(Tresam - MaxTq);
            [a b] = min(dTints);
            
            % Intersection point T, W & interpolate fuel
            Wints = Wresam(b);
            Tints = Tresam(b);
            
            % min engine speed to allow generator to recharge battery
            Wmin = K*Wreq*R/(R+1);
            
            if Wints < Wmin
                
                % add 200 RPM from Wmin
                Wints = Wmin + 200*2*pi/60;
                
                if Wints > enginemap_spd(end)
                    Wints = enginemap_spd(end);
                else
                    dummy1 = 1;
                end
                
            else
                dummy1 = 1;
            end
            
            fints = interp2(enginemap_trq,enginemap_spd,enginemap,Tints,Wints, 'linear')*dt;
            
            % find We < Wints
            idxbel = find(WeM < Wints);
            % set fuel consumption for We < Wints to be 999
            fM(idxbel,i) = 999;
            
            % find We > Wints
            idxabv = find(WeM >= Wints);
            
            % Reshape selected data in 3D into 1D array
            WMbel = WeM(idxbel);
            WMabv = WeM(idxabv);
            TMbel = TeM(idxbel,i);
            TMabv = TeM(idxabv,i);
            fMbel = fM(idxbel,i);
            fMabv = fM(idxabv,i);
            
            % append data at maxTorque line in matrix
            WMnew(:,i) = [WMbel'; Wints; WMabv'];
            TMnew(:,i) = [TMbel; Tints; TMabv];
            fMnew(:,i) = [fMbel; fints; fMabv];
            
            %%
            
            % Check if all engine torques are below engine map
            diffbel = TMnew(:,i) - enginemap_trq(1);
            checkbel = all( diffbel < 0 );
            
            % Check if all engine torques are above engine map
            diffabv = TMnew(:,i) - enginemap_trq(end);
            checkabv = all( diffabv > 0 );
            
            % If all TMnew(:,i) are below enginemap_trq(1)
            if checkbel == 1
                
                % engine_shutoff when Te / requested power is -ve / decel
                WMnew(:,i) = zeros;
                TMnew(:,i) = zeros;
                fMnew(:,i) = zeros;
                
            elseif checkabv == 1
                
                WMnew(:,i) = enginemap_spd(end);
                TMnew(:,i) = enginemap_trq(end);
                fMnew(:,i) = Eng.FuelTmeasMap.fuelMeasFcn(Eng.wEngMap(end)*60/(2*pi), Eng.tEngMax(end))*748.9/3600*dt;
                
            end
            %%
            
            % find min fuel consumption
            [fmin idx] = min(fMnew(:,i));
            
            fo(i)  = fMnew(idx,i);
            Teo(i) = TMnew(idx,i);
            Weo(i) = WMnew(idx,i);
            
            
            % If SOC is higher than SOCinit, discharge battery
            % Keep discharging until SOCinit is reached
        elseif (SOCi > SOCinit) && (abs(SOCi-SOCinit) > diff_SOC)
            
            fo(i)  = 0;
            Teo(i) = 0;
            Weo(i) = 0;
            
            % Stop using the battery once SOCinit is achieved
        else
            
            % Ignore power request from battery
            Preq(i) = Preq(i) - Preq_SOC(i);
            
            for j = 1:length(WeM)
                
                We = WeM(j);
                
                % Find engine torque
                Te = Preq(i)/We;
                
                if Te < enginemap_trq(1)
                    % let the cost be high if Te is out of the engine map
                    TeM(j,i) = Te;
                    fM(j,i) = 999;
                elseif Te > enginemap_trq(end)
                    % let the cost be high if Te is out of the engine map
                    TeM(j,i) = Te;
                    fM(j,i) = 999;
                else
                    % Need to check engine torque if it is at max engine torque line
                    TeM(j,i) = Te;
                    fM(j,i) = interp2(enginemap_trq,enginemap_spd,enginemap,Te,We, 'linear')*dt;
                end
                
            end
            
            Tresam = interp1(WeM,TeM(:,i),MaxSp, 'pchip');
            Wresam = MaxSp;
            
            % Find min difference bet Tq (intersection)
            dTints = abs(Tresam - MaxTq);
            [a b] = min(dTints);
            
            % Intersection point T, W & interpolate fuel
            Wints = Wresam(b);
            Tints = Tresam(b);
            fints = interp2(enginemap_trq,enginemap_spd,enginemap,Tints,Wints, 'linear')*dt;
            
            % find We < Wints
            idxbel = find(WeM < Wints);
            % set fuel consumption for We < Wints to be 999
            fM(idxbel,i) = 999;
            
            % find We > Wints
            idxabv = find(WeM >= Wints);
            
            % Reshape selected data in 3D into 1D array
            WMbel = WeM(idxbel);
            WMabv = WeM(idxabv);
            TMbel = TeM(idxbel,i);
            TMabv = TeM(idxabv,i);
            fMbel = fM(idxbel,i);
            fMabv = fM(idxabv,i);
            
            % append data at maxTorque line in matrix
            WMnew(:,i) = [WMbel'; Wints; WMabv'];
            TMnew(:,i) = [TMbel; Tints; TMabv];
            fMnew(:,i) = [fMbel; fints; fMabv];
            
            %%
            
            % Check if all engine torques are below engine map
            diffbel = TMnew(:,i) - enginemap_trq(1);
            checkbel = all( diffbel < 0 );
            
            % Check if all engine torques are above engine map
            diffabv = TMnew(:,i) - enginemap_trq(end);
            checkabv = all( diffabv > 0 );
            
            % If all TMnew(:,i) are below enginemap_trq(1)
            if checkbel == 1
                
                % engine_shutoff when Te / requested power is -ve / decel
                WMnew(:,i) = zeros;
                TMnew(:,i) = zeros;
                fMnew(:,i) = zeros;
                brakeon = 1;
                
            elseif checkabv == 1
                
                WMnew(:,i) = enginemap_spd(end);
                TMnew(:,i) = enginemap_trq(end);
                fMnew(:,i) = 2.7501*dt;
                
            end
            %%
            
            % find min fuel consumption
            [fmin idx] = min(fMnew(:,i));
            
            if i == 2741
                debug = 1;
            end
            fo(i)  = fMnew(idx,i);
            Teo(i) = TMnew(idx,i);
            Weo(i) = WMnew(idx,i);
            
        end
        
    
    else
        
        
        % IF SOC is high enough, use hybrid mode
        if (SOCi >= 0.51) && (recharge_flag == 0)
            
            for j = 1:length(WeM)
                
                We = WeM(j);
                
                % Find engine torque
                Te = Preq(i)/We;
                
                if Te < enginemap_trq(1)
                    % let the cost be high if Te is out of the engine map
                    TeM(j,i) = Te;
                    fM(j,i) = 999;
                elseif Te > enginemap_trq(end)
                    % let the cost be high if Te is out of the engine map
                    TeM(j,i) = Te;
                    fM(j,i) = 999;
                else
                    % Need to check engine torque if it is at max engine torque line
                    TeM(j,i) = Te;
                    fM(j,i) = interp2(enginemap_trq,enginemap_spd,enginemap,Te,We, 'linear')*dt;
                end
                
            end
            
            Tresam = interp1(WeM,TeM(:,i),MaxSp, 'pchip');
            Wresam = MaxSp;
            
            % Find min difference bet Tq (intersection)
            dTints = abs(Tresam - MaxTq);
            [a b] = min(dTints);
            
            % Intersection point T, W & interpolate fuel
            Wints = Wresam(b);
            Tints = Tresam(b);
            fints = interp2(enginemap_trq,enginemap_spd,enginemap,Tints,Wints, 'linear')*dt;

            % find We < Wints
            idxbel = find(WeM < Wints);
            % set fuel consumption for We < Wints to be 999
            fM(idxbel,i) = 999;
            
            % find We > Wints
            idxabv = find(WeM >= Wints);
            
            % Reshape selected data in 3D into 1D array
            WMbel = WeM(idxbel);
            WMabv = WeM(idxabv);
            TMbel = TeM(idxbel,i);
            TMabv = TeM(idxabv,i);
            fMbel = fM(idxbel,i);
            fMabv = fM(idxabv,i);
            
            % append data at maxTorque line in matrix
            WMnew(:,i) = [WMbel'; Wints; WMabv'];
            TMnew(:,i) = [TMbel; Tints; TMabv];
            fMnew(:,i) = [fMbel; fints; fMabv];
            
            %%
            
            % Check if all engine torques are below engine map
            diffbel = TMnew(:,i) - enginemap_trq(1);
            checkbel = all( diffbel < 0 );
            
            % Check if all engine torques are above engine map
            diffabv = TMnew(:,i) - enginemap_trq(end);
            checkabv = all( diffabv > 0 );
            
            % If all TMnew(:,i) are below enginemap_trq(1)
            if checkbel == 1
                
                % engine_shutoff when Te / requested power is -ve / decel
                WMnew(:,i) = zeros;
                TMnew(:,i) = zeros;
                fMnew(:,i) = zeros;
                
            elseif checkabv == 1
                
                WMnew(:,i) = enginemap_spd(end);
                TMnew(:,i) = enginemap_trq(end);
                fMnew(:,i) = 2.7501*dt;
            end
            %%
            
            % find min fuel consumption
            [fmin idx] = min(fMnew(:,i));
            
            fo(i)  = fMnew(idx,i);
            Teo(i) = TMnew(idx,i);
            Weo(i) = WMnew(idx,i);
            
            % IF REQUESTED POWER LESS THAN CERTAIN VALUE & SOC above certain value
            if (Preq(i) < Preqmin) && (SOCi >= 0.55)
                fo(i)  = 0;
                Teo(i) = 0;
                Weo(i) = 0;
            else
                dummy = 10;
            end
            
            
            % If SOC is too low, shut motor (Tm = 0)
        else
            
            % initiate charging flag
            recharge_flag = 1;
            
            % Keep recharging until SOC == 0.55
            if SOCi >= 0.55;
                recharge_flag = 0;
            else
                dummy2 = 0;
            end
            
            Treq2 = BATT_CHARGE_RATE*Preq(i)/Wreq;
            Treq2 = 1.5*Preq(i)/Wreq;
            Te_nobatt = (Treq2/Kratio)*(R+1)/R;
            
            if Te_nobatt > enginemap_trq(end)
                Te_nobatt = enginemap_trq(end);
            elseif Te_nobatt < enginemap_trq(1)
                Te_nobatt = enginemap_trq(1);
            else
                dummy1 = 1;
            end
            
            for j = 1:length(WeM)
                
                We = WeM(j);
                TeM(j,i) = Te_nobatt;
                fM(j,i) = interp2(enginemap_trq,enginemap_spd,enginemap,Te_nobatt,We, 'linear')*dt;
            end
            
            Tresam = interp1(WeM,TeM(:,i),MaxSp, 'pchip');
            Wresam = MaxSp;
            
            % Find min difference bet Tq (intersection)
            dTints = abs(Tresam - MaxTq);
            [a b] = min(dTints);
            
            % Intersection point T, W & interpolate fuel
            Wints = Wresam(b);
            Tints = Tresam(b);
            
            % min engine speed to allow generator to recharge battery
            Wmin = K*Wreq*R/(R+1);
            
            if Wints < Wmin
                
                % add 200 RPM from Wmin
                Wints = Wmin + 200*2*pi/60;
                
                if Wints > enginemap_spd(end)
                    Wints = enginemap_spd(end);
                else
                    dummy1 = 1;
                end
                
            else
                dummy1 = 1;
            end
            
            fints = interp2(enginemap_trq,enginemap_spd,enginemap,Tints,Wints, 'linear')*dt;
            
            % find We < Wints
            idxbel = find(WeM < Wints);
            % set fuel consumption for We < Wints to be 999
            fM(idxbel,i) = 999;
            
            % find We > Wints
            idxabv = find(WeM >= Wints);
            
            % Reshape selected data in 3D into 1D array
            WMbel = WeM(idxbel);
            WMabv = WeM(idxabv);
            TMbel = TeM(idxbel,i);
            TMabv = TeM(idxabv,i);
            fMbel = fM(idxbel,i);
            fMabv = fM(idxabv,i);
            
            % append data at maxTorque line in matrix
            WMnew(:,i) = [WMbel'; Wints; WMabv'];
            TMnew(:,i) = [TMbel; Tints; TMabv];
            fMnew(:,i) = [fMbel; fints; fMabv];
            
            %%
            
            % Check if all engine torques are below engine map
            diffbel = TMnew(:,i) - enginemap_trq(1);
            checkbel = all( diffbel < 0 );
            
            % Check if all engine torques are above engine map
            diffabv = TMnew(:,i) - enginemap_trq(end);
            checkabv = all( diffabv > 0 );
            
            % If all TMnew(:,i) are below enginemap_trq(1)
            if checkbel == 1
                
                % engine_shutoff when Te / requested power is -ve / decel
                WMnew(:,i) = zeros;
                TMnew(:,i) = zeros;
                fMnew(:,i) = zeros;
                
            elseif checkabv == 1
                
                WMnew(:,i) = enginemap_spd(end);
                TMnew(:,i) = enginemap_trq(end);
                fMnew(:,i) = Eng.FuelTmeasMap.fuelMeasFcn(Eng.wEngMap(end)*60/(2*pi), Eng.tEngMax(end))*748.9/3600*dt;
                
            end
            %%
            
            % find min fuel consumption
            [fmin idx] = min(fMnew(:,i));
            
            fo(i)  = fMnew(idx,i);
            Teo(i) = TMnew(idx,i);
            Weo(i) = WMnew(idx,i);
            
        end
        
        
    end
        
    Tg(i) = -Teo(i)/(1+R);
    Wg(i) = -R*K*Wreq + (1+R)*Weo(i);
    
    Tm(i) = Treq/K - R/(1+R)*Teo(i);
    Wm(i) = K*Wreq;
    
    
    % If during decel, we dont want to use the battery, then assume brake,
    % i.e no regen power is received by the motor. Tg = 0 because Te = 0
    if brakeon == 1
       Tm(i) = 0;
    else
       dummy = 0; 
    end
    
    % toggle switch
    brakeon = 0;
    
    if Tg(i)*Wg(i) >= 0
        kg = -1;
    else
        kg = 1;
    end
    
    if Tm(i)*Wm(i) >= 0
        km = -1;
    else
        km = 1;
    end
    
    Pbatt(i) = Tg(i)*Wg(i)*ng^kg + Tm(i)*Wm(i)*nm^km;
    
    calc2 = Voc^2 - 4*Rbatt*Pbatt(i);
    dSOC(i) = -1/(2*Qbatt*Rbatt) * (Voc - sign(calc2)*abs(calc2)^0.5 );
    SOC(i) = SOCi + dSOC(i)*dt;
    
    % reset
    SOCi = SOC(i);
    
end
close(f)
SOC_RB = [SOCinit SOC(1:end-1)];

% parser output variables
% [SOC_RB, Weo, Teo, Wm, Tm, Wg, Tg, fo, Pbatt]
We = Weo;
Te = Teo;
fuel = fo;

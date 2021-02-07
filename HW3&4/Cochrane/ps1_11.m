% problem set 1 for asset pricing, forecasting returns

close all
clear all

x = load('..\data\vwr.txt'); 
vw_dates = x(:,1); 
R  = 1+x(:,2); 
Rx = 1+x(:,3);
dp = R./Rx-1; % (D_T+P_t)/(P_t-1) * P_t-1/P_t
DD = [NaN; dp(2:end)./dp(1:end-1).*Rx(2:end)];  % D_t/P_t / (D_t-1/P_t-1) *P_t/P_t-1

x = load('..\data\sbbi.txt'); 
tb_dates = x(:,1); 
Rtb = 1+x(:,3); 

Re = R-Rtb;
T = size(R,1); 

% pictures

figure;
subplot(3,1,1);
plot((1926:1925+T)',100*dp);
axis([1925 2010 0 8]); 
title('dp'); 
ylabel('D/P, %');

subplot(3,1,2); 
plot((1926:1925+T)',(R-1)*100);
hold on
plot((1926:1925+T)',(Rtb-1)*100,'-r');
axis([1925 2010 -50 60]); 
set(gca,'Ytick',[-50 -25 0 25 50]); 
title('stock and t bill return'); 
ylabel('return, %'); 

subplot(3,1,3); 
plot((1926:1925+T)',(Rtb-1)*100); 
title('t bill return'); 
ylabel('return, %'); 
axis([1925 2010 0 16]); 

print -depsc2 dpandr.eps;

% basic regressions 

lhvR = (R(2:end));
lhvRtb = (Rtb(2:end)); 
rhv = [ones(T-1,1) dp(1:end-1)];
for i = 1:10;
     lhvRx = lhvR-lhvRtb; 
     lags = i; % implements hansen-hodrick errors
     weight = 0; 
     [bvR(:,i),sebvR(:,i),R2vR(:,i),R2vadj,v,F] = olsgmm(lhvR,rhv,lags,weight);     
     [bvRols(:,i),sebvRols(:,i),R2vRols(:,i),R2vadj,v,F] = olsgmm(lhvR,rhv,0,0); % computes OLS standard errors for comparison
     lhvRno = lhvR(1:i:end); % nonoverlapping 
     rhvRno = rhv(1:i:end,:);
     [bvRno(:,i),sebvRno(:,i),R2vRno(:,i),R2vadj,v,F] = olsgmm(lhvRno,rhvRno,0,0);               
     [bvRx(:,i),sebvRx(:,i),R2vRx(:,i),R2vadj,v,F] = olsgmm(lhvRx,rhv,lags,weight);     % excess 

     % if you got an error message here, go to the class website and get
     % olsgmm. 
     if i == 7; 
         fcst7 = [ones(T,1) dp]*bvRx(:,i); 
         actual7 = lhvRx; 
     end; 
     lhvR = lhvR(1:end-1).*(R(2+i:end)); % cumulate returns
     lhvRtb = lhvRtb(1:end-1).*(Rtb(2+i:end)); 
     rhv = rhv(1:end-1,:); 
 end; 
 
 disp(''); 
 disp('Returns on D/P, overlapping, many horizons, 1926-today'); 
 disp('Horizon (years)'); 
 fprintf('     ');  fprintf('%5i ',1:10); fprintf('\n'); 
 fprintf('b    ');  fprintf('%5.2f ',bvR(2,:));fprintf('\n'); 
 fprintf('t,HH ');  fprintf('%5.2f ',bvR(2,:)./sebvR(2,:));fprintf('\n'); 
 fprintf('t,OLS');  fprintf('%5.2f ',bvR(2,:)./sebvRols(2,:));fprintf('\n'); 
 fprintf('t,NO ');  fprintf('%5.2f ',bvR(2,:)./sebvRno(2,:));fprintf('\n'); 
 fprintf('R2   ');  fprintf('%5.2f ',R2vR);fprintf('\n'); 

 disp(''); 
 disp('Excess Returns on D/P, overlapping, many horizons, 1926-today'); 
 disp('Horizon (years)'); 
 fprintf('   ');  fprintf('%5i ',1:10); fprintf('\n'); 
 fprintf('b  ');  fprintf('%5.2f ',bvRx(2,:));fprintf('\n'); 
 fprintf('t  ');  fprintf('%5.2f ',bvRx(2,:)./sebvRx(2,:));fprintf('\n'); 
 fprintf('R2 ');  fprintf('%5.2f ',R2vRx);fprintf('\n'); 

 
 figure; 
 plot((1:10)',bvRx(2,:),'-rv'); 
 hold on; 
 plot((1:10)',100*R2vRx,'-bo'); 
 hold on; 
 plot((1:10)',10*bvRx(2,:)./sebvRx(2,:),'--g+');
 legend('coeff','R^2 (%)','10 \times t stat',2); 
 xlabel('horizon, years'); 
 print -depsc2 ps1_longhorizonb.eps;
 
 plotdates = (1926:1925+T)'; 
 figure; 
 plot((1926+7:1925+T+7)',fcst7,'-v','linewidth',2);
 hold on
 plot(plotdates(1+7:T),actual7,'-r','linewidth',2);
 title('Actual and forecast 7 year excess returns'); 
% legend('forecast','actual'); 
 axis([1930 2015 -inf inf]); 
 print -depsc2 ps1c.eps; 


 % adding CAY to the regressions. 
 

x = load('..\data\cay.txt'); 
z = x; 
x = x(4:4:size(x,1),:); % make annual data by choosing end of year value
cay_dates = x(:,1); 
c = x(:,2); 
a = x(:,3); 
y = x(:,4); 
cay = x(:,5); 


% chop vw to same size as cay

vw_dates = vw_dates(27:end);
disp(' '); 
fprintf(' vw data from %8.2f to %8.2f\n',vw_dates(1),vw_dates(end)); 
fprintf('cay data from %8.2f to %8.2f\n',cay_dates(1),cay_dates(end)); 
R = R(27:end); 
dp = dp(27:end); 
Rtb = Rtb(27:end); 
Re = Re(27:end);
DD = DD(27:end); 
T = size(R,1); 

figure;
plot((1952+1/4:1/4:1952+size(z,1)/4)',100*z(:,5),'linewidth',2); 
title('cay and dp'); 
hold on;
plot((1952:1:1951+T)',dp*100,'-r','linewidth',2);
print -depsc2 dp_and_cay.eps;

lhvR = (R(2:end));
lhvRtb = (Rtb(2:end)); 
rhvdp = [ones(T-1,1) dp(1:end-1)];
rhvcay = [ones(T-1,1) cay(1:end-1)];
rhvboth = [ones(T-1,1) dp(1:end-1) cay(1:end-1)];

for i = 1:10;
     lhvRx = lhvR-lhvRtb; 
     lags = i; % implements hansen-hodrick errors
     weight = 0; 
     [bvdp(:,i),sebvdp(:,i),R2vdp(:,i),R2vadj,v,F] = olsgmm(lhvRx,rhvdp,lags,weight);      
     [bvcay(:,i),sebvcay(:,i),R2vcay(:,i),R2vadj,v,F] = olsgmm(lhvRx,rhvcay,lags,weight); 
     [bvboth(:,i),sebvboth(:,i),R2vboth(:,i),R2vadj,v,F] = olsgmm(lhvRx,rhvboth,lags,weight); 
     if i == 1; 
         fcst1 = [ones(T,1) dp]*bvdp(:,i);
         fcst2 = [ones(T,1) cay]*bvcay(:,i);
         fcst3 = [ones(T,1) dp cay]*bvboth(:,i);
     end;   
     if i == 7; 
         fcst1_7 = [ones(T,1) dp]*bvdp(:,i);
         fcst2_7 = [ones(T,1) cay]*bvcay(:,i);
         fcst3_7 = [ones(T,1) dp cay]*bvboth(:,i);
         actual7 = lhvRx;
     end;
     lhvR = lhvR(1:end-1).*(R(2+i:end)); 
     lhvRtb = lhvRtb(1:end-1).*(Rtb(2+i:end)); 
     rhvdp = rhvdp(1:end-1,:); 
     rhvcay = rhvcay(1:end-1,:); 
     rhvboth = rhvboth(1:end-1,:); 
end; 

figure; 
plot((1952:1951+size(fcst1,1))',[fcst1 fcst2 fcst3],'linewidth',2);
hold on
plot((1952:1951+size(R(2:end),1))',R(2:end)-Rtb(2:end),'-vk','Markersize',2);
title('forecasts using dp and cay');
legend('dp','cay','both','return')
print -depsc2 dp_cay_fcst.eps

figure
plot((1952:1951+size(fcst1_7,1))',[fcst1_7 fcst2_7 fcst3_7],'linewidth',2);
hold on
plot((1952:1951+size(actual7,1))',actual7,'-vk','Markersize',2);
legend('dp','cay','both','return')
title('forecasts using dp and cay'); 
print -depsc2 dp_cay_fcst_7.eps


 
 disp(''); 
 fprintf ('Excess Returns on D/P, overlapping, many horizons, %7i - % 7i\n',vw_dates(1),vw_dates(end)); 
 disp('Horizon (years)'); 
 fprintf('     ');  fprintf('%5i ',1:10); fprintf('\n'); 
 fprintf('b    ');  fprintf('%5.2f ',bvdp(2,:));fprintf('\n'); 
 fprintf('t    ');  fprintf('%5.2f ',bvdp(2,:)./sebvdp(2,:));fprintf('\n'); 
 fprintf('R2   ');  fprintf('%5.2f ',R2vdp);fprintf('\n'); 
 
 disp(''); 
 fprintf ('Excess Returns on cay, overlapping, many horizons, %7i - % 7i\n',vw_dates(1),vw_dates(end)); 
 disp('Horizon (years)'); 
 fprintf('     ');  fprintf('%5i ',1:10); fprintf('\n'); 
 fprintf('b    ');  fprintf('%5.2f ',bvcay(2,:));fprintf('\n'); 
 fprintf('t    ');  fprintf('%5.2f ',bvcay(2,:)./sebvcay(2,:));fprintf('\n'); 
 fprintf('R2   ');  fprintf('%5.2f ',R2vcay);fprintf('\n'); 

 disp(''); 
 fprintf ('Excess Returns on D/P and cay, overlapping, many horizons, %7i - % 7i \n',vw_dates(1),vw_dates(end)); 
 disp('Horizon (years)'); 
 fprintf('     ');  fprintf('%5i ',1:10); fprintf('\n'); 
 fprintf('b, dp');  fprintf('%5.2f ',bvboth(2,:));fprintf('\n'); 
 fprintf('t    ');  fprintf('%5.2f ',bvboth(2,:)./sebvboth(2,:));fprintf('\n'); 
 fprintf('b,cay');  fprintf('%5.2f ',bvboth(3,:));fprintf('\n'); 
 fprintf('t    ');  fprintf('%5.2f ',bvboth(3,:)./sebvboth(3,:));fprintf('\n'); 
 fprintf('R2   ');  fprintf('%5.2f ',R2vboth);fprintf('\n'); 

 
 function [bv,sebv,R2v,R2vadj,v,F] = olsgmm(lhv,rhv,lags,weight);

% function olsgmm does ols regressions with gmm corrected standard errors
% Inputs:
%  lhv T x N vector, left hand variable data 
%  rhv T x K matrix, right hand variable data
%  If N > 1, this runs N regressions of the left hand columns on all the (same) right hand variables. 
%  lags number of lags to include in GMM corrected standard errors
%  weight: 1 for newey-west weighting 
%          0 for even weighting
%          2 for modified even weighting. Imposes homoskedasticity and MA(h) structure on HH to get
%                          better small sample performance
%         -1 skips standard error computations. This speeds the program up a lot; used inside monte carlos where only estimates are needed
%          weight may be a vector, different for each regression. 
%  NOTE: you must make one column of rhv a vector of ones if you want a constant. 
%        should the covariance matrix estimate take out sample means?
% Output:
%  b: regression coefficients K x 1 vector of coefficients
%  seb: K x N matrix standard errors of parameters. 
%      (Note this will be negative if variance comes out negative) 
%  v: variance covariance matrix of estimated parameters. If there are many y variables, the vcv are stacked vertically
%  R2v:    unadjusted
%  R2vadj: adjusted R2
%  F: [Chi squared statistic    degrees of freedom    pvalue] for all coeffs jointly zero. 
%   Note: program checks whether first is a constant and ignores that one for test

global Exxprim;
global inner;

if size(rhv,1) ~= size(lhv,1);
   disp('olsgmm: left and right sides must have same number of rows. Current rows are');
   size(lhv)
   size(rhv)
end;

T = size(lhv,1);
N = size(lhv,2);
K = size(rhv,2);
sebv = zeros(K,N);
Exxprim = inv((rhv'*rhv)/T);
bv = rhv\lhv;

% if weight, lags are is scalar expand so all have the same value
if (size(weight,1) == 1) & (size(lhv,2) > 1); 
    weight = weight*ones(size(lhv,2),1); 
end; 
if (size(lags,1) == 1) & (size(lhv,2) > 1); 
    lags = lags*ones(size(lhv,2),1); 
end; 


if weight == -1;  % skip ses if you don't want them.  returns something so won't get error message
    sebv=NaN;
    R2v=NaN;
    R2vadj=NaN;
    v=NaN;
    F=NaN;
else; 
    errv = lhv-rhv*bv;
    s2 = mean(errv.^2);
    vary = lhv - ones(T,1)*mean(lhv);
    vary = mean(vary.^2);

    R2v = (1-s2./vary)';
    R2vadj= (1 - (s2./vary)*(T-1)/(T-K))';
    
    %compute GMM standard errors
    for indx = 1:N;
        err=errv(:,indx);
        if (weight(indx) == 0)|(weight(indx) == 1)
            inner = (rhv.*(err*ones(1,K)))'*(rhv.*(err*ones(1,K)))/T;
            
        	for jindx = (1:lags(indx));
                inneradd = (rhv(1:T-jindx,:).*(err(1:T-jindx)*ones(1,K)))'...
        	              *(rhv(1+jindx:T,:).*(err(1+jindx:T)*ones(1,K)))/T;
        	    inner = inner + (1-weight(indx)*jindx/(lags(indx)+1))*(inneradd + inneradd');
            end;
        elseif weight(indx) == 2; 
            inner = rhv'*rhv/T; 
            for jindx = 1:lags(indx); 
                inneradd = rhv(1:T-jindx,:)'*rhv(1+jindx:T,:)/T;
                inner = inner + (1-jindx/lags(indx))*(inneradd+inneradd'); 
            end; 
            inner = inner*std(err)^2;
        end; 
        varb = 1/T*Exxprim*inner*Exxprim;
        
        % F test for all coeffs (except constant) zero -- actually chi2 test
        if rhv(:,1) == ones(size(rhv,1),1); 
            chi2val = bv(2:end,indx)'*inv(varb(2:end,2:end))*bv(2:end,indx);
            dof = size(bv(2:end,1),1); 
            pval = 1-cdf('chi2',chi2val, dof); 
            F(indx,1:3) = [chi2val dof pval]; 
        else; 
            chi2val = bv(:,indx)'*inv(varb)*bv(:,indx);
            dof = size(bv(:,1),1); 
            pval = 1-cdf('chi2',chi2val, dof); 
            F(indx,1:3) = [chi2val dof pval]; 
        end; 
            
        if indx == 1; 
           v = varb;
        else;
           v = [v; varb ];
        end;
        
       seb = diag(varb);
       seb = sign(seb).*(abs(seb).^0.5);
       sebv(:,indx) = seb;
    end;
end; % ends if w > -1;     
      
    
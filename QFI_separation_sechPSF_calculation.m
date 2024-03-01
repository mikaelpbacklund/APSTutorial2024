%This program performs a calculation of the classical and quantum Fisher
%information for the estimation of two-point-source separation in a
%one-dimensional imaging problem. It is inspired by work from Tsang, Nair,
%and Lu, Physical Review X 6, 0313033 (2016) [TNL] and related works. In
%TNL they compute the FI for this problem assuming Gaussian PSFs. This
%assumption allows them to compute relevant derivatives and integrals
%analytically. Here we treat a different PSF model-- the hyperbolic secant
%function sech(x). We compute derivatives analytically but integrals are
%done numerically. We encourage those interested to substitute other
%choices of PSF.

%copyright 2024 Mikael Backlund. Written as part of the tutorial on
%super-resolution imaging at APS March Meeting in Minneapolis, MN.

%%

clear
close all

xvec = linspace(-25,25,1e3)'; %define the x axis that will be used for numerical integration. Sample finely enough and over a large enough range to prevent rounding errors.
dx = xvec(2)-xvec(1);
separation_vec = linspace(0.001,10,25); %set of source separations at which the FI is to be calculated

syms x mu s
psi_sym = sqrt(1/2*sech((x-mu)*pi/2)); %symbolic definition of the amplitude PSF, normalized such that the integral of abs(psi_sym)^2 is 1.
dpsidx_sym = diff(psi_sym,x); %symbolic definition of the derivative of the amplitude PSF

I_classical = 1/2*subs(psi_sym^2,mu,s/2) + 1/2*subs(psi_sym^2,mu,-s/2); %symbolic definition of the classical density to be used in the calculation of classical FI
dIds_classical = diff(I_classical,s);
integrand_classical = (dIds_classical^2)/I_classical;
%%
%we will compute the QFI numerically. This involves diagonalization of the density operator rho (defined later). In principle this diagonalization can be
%performed in any basis, but doing so in a discretized version of a
%continuous basis (e.g. the position eigenstates) is prone to numerical
%errors. Instead we will expand rho in the discrete basis of
%Hermite-Gaussian functions defined in the rows of the matrix HG_matrix
%below.

nmax = 25; %maximum order of HG mode to include in truncated expansion.

HG_matrix = zeros(nmax+1,length(xvec));
for nn = 0:nmax
    
    HG_matrix(nn+1,:) = ((1/2/pi)^(1/4))/sqrt((2^nn)*factorial(nn))*hermiteH(nn,xvec/sqrt(2)).*exp(-(xvec.^2)/4); %normalized HG modes

end

%%
%initialize

FI_classical = zeros(size(separation_vec));
FI_quantum = zeros(size(separation_vec));
%%
%loop over separations at which to compute CFI and QFI
for ii = 1:length(separation_vec)
    disp([num2str(ii) '/' num2str(length(separation_vec))])
    
    separation = separation_vec(ii);
    
    integrand_classical_num = double(subs(integrand_classical,{x,s},{xvec,separation}));
    %numerical integration to calculate CFI at this separation
    FI_classical(ii) = dx*(sum(integrand_classical_num));

    %define the one-photon state centered on either source, as well as
    %their derivatives
    psi_plus = double(subs(psi_sym,{x,mu},{xvec,separation/2}));
    dpsids_plus = double(-1/2*subs(dpsidx_sym,{x,mu},{xvec,separation/2}));
    psi_minus = double(subs(psi_sym,{x,mu},{xvec,-separation/2}));
    dpsids_minus = double(1/2*subs(dpsidx_sym,{x,mu},{xvec,-separation/2}));
    
    %transform from position eigenstates to HG basis
    psi_plus_HG = HG_matrix*psi_plus*dx;
    dpsids_plus_HG = HG_matrix*dpsids_plus*dx;
    psi_minus_HG = HG_matrix*psi_minus*dx;
    dpsids_minus_HG = HG_matrix*dpsids_minus*dx;

    %define density matrix in HG basis
    rho = 1/2*(psi_plus_HG*psi_plus_HG') + 1/2*(psi_minus_HG*psi_minus_HG');
    %compute its derivative with respect to separation
    drhods = 1/2*(psi_plus_HG*dpsids_plus_HG') + 1/2*(dpsids_plus_HG*psi_plus_HG') +...
        1/2*(psi_minus_HG*dpsids_minus_HG') + 1/2*(dpsids_minus_HG*psi_minus_HG');
    
    %diagonalize the density operator
    [V,D] = eig(rho);
    
    %transform the derivative of the density operator into the eigenbasis
    %of rho itself
    drhods_transformed = V'*drhods*V;
    
    %compute the symmetric logarithmic derivative matrix
    D(D<0) = 0;
    DD = ones(nmax+1)*D;
    SLD_coefficient = 2./(DD+DD');
    SLD_coefficient((DD+DD')==0) = 0;
    SLD = SLD_coefficient.*drhods_transformed;
    
    %compute the QFI
    FI_quantum(ii) = trace(SLD*SLD*D);

end
%%
%for reference plot the classical PDF for each separation value, i.e. the
%Poisson rates of the direct imaging approach.
figure('color','w')
cmap = jet(length(separation_vec));
for jj = 1:length(separation_vec)
    separation = separation_vec(jj);
    currI = double(subs(I_classical,{x,s},{xvec,separation}));

    plot(xvec,currI,'linewidth',1,'color',cmap(jj,:))
    hold on
end
xlabel('x','fontsize',14)
ylabel('classical PDF','fontsize',14)
hcbar = colorbar;
colormap(cmap)
clim([min(separation_vec) max(separation_vec)])
ylabel(hcbar,'separation','fontsize',14)
xlim([-10 10])
%%
%plot the computed CFI and QFI as functions of the separation

figure('color','w')
plot(separation_vec,FI_quantum,'linewidth',2)
hold on
plot(separation_vec,FI_classical,'--','linewidth',2)
xlabel('separation','fontsize',14)
ylabel('Fisher information','fontsize',14)
legend('Quantum','Classical','fontsize',14,'location','southeast')

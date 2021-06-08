close all
clear

%% Laplace approximation
ll = linspace(0.01,0.25,200);
sigma_f = 1;
k = @(x1, x2) 1*exp(-1/(2*l^2)*(x1-x2)'*(x1-x2));
lik = @(z) 1./(1+exp(-z)); % Likelihood function


x1 = [0.11, 0.11; 0.1, 0.85; 0.12, 0.4; 0.19, 0.6; 0.21, 0.72; 0.25, 0.49; 0.34, 0.28; 0.4, 0.42; 0.78, 0.66;0.45,0.85]; % Class 1 points
x2 = [0.16, 0.25; 0.45, 0.13; 0.4, 0.6; 0.56,0.75; 0.61, 0.62; 0.62, 0.51; 0.63, 0.36; 0.75, 0.76; 0.83, 0.57; 0.84, 0.75];% Class 2 points
y = [ones(size(x1,1),1); -ones(size(x2,1),1)];

for j = 1:length(ll)
l = ll(j);

X = [x1;x2]';
Kxx = K(X,X,k); 
I = eye(size(Kxx));
t = (y+1)/2; % Makes class [1,-1] -> [1,0]

% Algoritme 3.1
f = zeros(size(X,2),1); % init
f_old = 100*f;
check = f;
for i = 1:1000
    pi = lik(1.*f);
    W = -diag(-pi.*(1-pi)); % eq. 3.15 hessian
    gradlike = t - pi; % eq. 3.15 gradient
    L = chol(I+W^(1/2)*Kxx*W^(1/2));
    b = W*f + gradlike;
    a = b-W^(1/2)*L'\(L\(W^(1/2)*Kxx*b));
    f = Kxx*a;
    
    f = (Kxx^(-1) + W)^(-1)*(W*f + gradlike);
    check(i) = sum((f-f_old).^2);
    f_old = f;
end

n = 200;
[X1, X2] = meshgrid(linspace(0,1,n),linspace(0,1,n));
X1vect = reshape(X1, n*n,1);
X2vect = reshape(X2, n*n,1);

pistar = zeros(n*n,1);
for i = 1:n*n
    kx = K(X,[X1vect(i);X2vect(i)],k);
    fbar = kx'*gradlike;
    v = L\(W^(1/2)*kx);
    V = K([X1vect(i);X2vect(i)],[X1vect(i);X2vect(i)], k) - v'*v;
    pistar(i) = sum(lik(normrnd(fbar,V,1000,1)))/1000;
end


pistar = reshape(pistar, size(X1));

[X1, X2] = meshgrid(linspace(0,n,n),linspace(0,n,n));
plot(x1(:,1)*n, (x1(:,2))*n, 'o')
plot(x2(:,1)*n, (x2(:,2))*n, 'or', 'MarkerFaceColor', 'r')
hold on
imagesc((pistar))
plot(x1(:,1)*n, (x1(:,2))*n, 'om')
plot(x2(:,1)*n, (x2(:,2))*n, 'or', 'MarkerFaceColor', 'r')
ylim([0,n])
xlim([0,n])
contour(X1, X2, (pistar), [0.49999999,0.50000001], 'black', 'LineWidth', 3)
title(['l = ',num2str(l)])
colorbar
Image = getframe(gcf);
imwrite(Image.cdata, [num2str(j),'.jpg']);
caxis([0,1])
end

%% EP Algo
close all
clear 


l = 0.09;
k = @(x1, x2) 1*exp(-1/(2*l^2)*(x1-x2)'*(x1-x2));

x1 = [0.11, 0.11; 0.1, 0.85; 0.12, 0.4; 0.19, 0.6; 0.21, 0.72; 0.25, 0.49; 0.34, 0.28; 0.4, 0.42; 0.78, 0.66;0.45,0.85]; % Class 1 points
x2 = [0.16, 0.25; 0.45, 0.13; 0.4, 0.6; 0.56,0.75; 0.61, 0.62; 0.62, 0.51; 0.63, 0.36; 0.75, 0.76; 0.83, 0.57; 0.84, 0.75];% Class 2 points
y = [ones(size(x1,1),1); -ones(size(x2,1),1)];

X = [x1;x2]';
Kxx = K(X,X,k); 
S = Kxx;
I = eye(size(Kxx));
v_tilde = zeros(size(X,2),1);
tau_tilde = v_tilde;
tau = v_tilde;
v = v_tilde;
mu = v_tilde;
n = length(mu);

for d = 1:100
    for i = 1:n
        tau(i) = 1/S(i,i) - tau_tilde(i);
        v(i) = 1/S(i,i)*mu(i) - v_tilde(i);
        
        z = y(i)*mu(i)/(sqrt(1+S(i,i)));
        
        %mu_hat = mu(i) + y(i)*S(i,i)*normpdf(z)/normcdf(z)*sqrt(1+S(i,i));
        mu_hat = mu(i) + y(i)*S(i,i)*normpdf(z)/(normcdf(z)*sqrt(1+S(i,i)));
        sigma_hat = S(i,i) - (S(i,i)^2*normpdf(z)) / ((1+S(i,i))*normcdf(z)) * (z + normpdf(z)/normcdf(z));
        
        temp = 1/sigma_hat - tau(i) - tau_tilde(i);
        tau_tilde(i) = tau_tilde(i) + temp;
        v_tilde(i) = 1/sigma_hat*mu_hat - v(i);
        
        S = S - ((1/temp) + S(i,i))^(-1)*S(:,i)*S(:,i)';
        mu = S*v_tilde;
    end
    S_tild = diag(tau_tilde);
    L = chol(I + sqrt(S_tild)*Kxx*sqrt(S_tild));
    V = L'\sqrt(S_tild)*Kxx;
    S = Kxx - V'*V;
    mu = S*v_tilde;
end


nstar = 200;
[X1, X2] = meshgrid(linspace(0,1,nstar),linspace(0,1,nstar));
X1vect = reshape(X1, nstar*nstar,1);
X2vect = reshape(X2, nstar*nstar,1);

pistar = zeros(nstar*nstar,1);
for i = 1:nstar*nstar
    kx = K(X,[X1vect(i);X2vect(i)],k);
    z = sqrt(S_tild)*L'\(L\(sqrt(S_tild)*Kxx*v_tilde));
    f_bar = kx'*(v_tilde-z);
    v = L\(sqrt(S_tild)*kx);
    V_f = K([X1vect(i);X2vect(i)],[X1vect(i);X2vect(i)], k) - v'*v;
    pistar(i) = normcdf(f_bar/(sqrt(1+V_f)));
end


pistar = reshape(pistar, size(X1));

plot(x1(:,1)*nstar, (x1(:,2))*nstar, 'o')
plot(x2(:,1)*nstar, (x2(:,2))*nstar, 'or', 'MarkerFaceColor', 'r')
hold on
imagesc((pistar))
plot(x1(:,1)*nstar, (x1(:,2))*nstar, 'om')
plot(x2(:,1)*nstar, (x2(:,2))*nstar, 'or', 'MarkerFaceColor', 'r')
ylim([0,nstar])
xlim([0,nstar])
colorbar
caxis([0,1])


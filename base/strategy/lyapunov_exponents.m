% 重复文章 2021年 Synchronization and chimeras in a network of
% photosensitive FitzHugh–Nagumo neurons 
% @ccnu // wuyong@mails.ccnu.edu.cn
%     system:
%        dx/dt = x*(1-xi)-x*x*x/3-y+A*cos(omega*t)
%        dy/dt = c*(x+a-b*y)
%
%    The Jacobian of system:（使用欧拉演算法变为离散计算） 
%            | 1+((1-xi)-x*x)*step  -1*step   |
%      J =   |    c*step           1-c*b*step |
%

clc;close all;clear all
N_time=100000;

% A = (0:0.001:1.6)';
% na = length(A);
% LEs_n = na;
A = 1;

omega = (0:0.02:2.5)';
% omega = 1;
nomega = length(omega);
LEs_n = nomega;

LE1 = zeros(LEs_n,1); % 初始化
LE2 = zeros(LEs_n,1);
x = 0.0; y = 0.0; % 初始值
step = 0.01; % 步长
time = 0.0;  %记时 

global xi a b c
xi=0.175; a=0.7; b=0.8; c=0.1;

%% ====================计算的主体====================
% 进度条图窗对象
wait=waitbar(0,'计算开始...','Name',['计算 ','-','lyapunov exponents']); 
Wait = 0;

for i=1:LEs_n
   LCEvector = zeros(2,1); 
   Q = eye(2); % 单位矩阵
   
   %% 开始
    for j=1:200000 % 抛掉暂态
        dx = x+x_iter(x,y)*step+A*cos(omega(i)*time)*step;
        dy = y+y_iter(x,y)*step;
        
        x = dx;
        y = dy;
        time = time+step;
    end
   
    for j=1:N_time
       dx = x+x_iter(x,y)*step+A*cos(omega(i)*time)*step;
       dy = y+y_iter(x,y)*step;
       
       x = dx;
       y = dy;
       time = time+step;
       %% =============雅可比矩阵=============
       Ji = [1+((1-xi)-x*x)*step, -1*step; 
           c*step, 1-c*b*step];
       
       %% =============QR分解=============
       B = Ji*Q;
       [Q,R] = qr(B);
       
       %% =============累加=============
       LCEvector = LCEvector+log(diag(abs(R)));      
    end
    
    LE = LCEvector/(N_time*step);
    LE1(i) = LE(1);
    LE2(i) = LE(2); 
    
     %% ==========================================================================
        % 降低进度条更新频率，使计算速度提升, 每计算10，100，1000次更新一次
        % 进度条会使运行速度变慢，可以注释掉不用
        Wait = Wait+1;
        if Wait == 10
            str = {[num2str(i/LEs_n*100),'%'],['omega=',num2str(omega(i))]}; % 进度条显示的内容
            waitbar(i/LEs_n,wait,str); % 进度条进度
            Wait = 0;
        end
end
close(wait)

%% ======================画图======================
figure(1)
plot(omega,LE1,'g','linewidth',1);
hold on
plot(omega,LE2,'b','linewidth',1);
set(gca,'Xlim',[0 2.5]);
set(gca,'Ylim',[-1.8 0.1]);
legend('lambda1','lambda2');
hold on
plot([0 2.5],[0 0],'k--','linewidth',1);
hold off
xlabel('omega');ylabel('LEs');
set(gca,'fontsize',10);

%% ======================方程======================
function x = x_iter(x,y)
    global xi
    x = x*(1-xi)-x*x*x/3-y;
end

function y = y_iter(x,y)
    global a b c 
    y = c*(x+a-b*y);
end

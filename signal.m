% %% 高斯白噪声信号
% 
% length = 1000000;
% ff = 0:length-1;
% noise = wgn(length,1,0);%生成100000*1个高斯白噪声，功率为0dBW（分贝瓦）
% y1 = fft(noise,length);%采样点个数100000个
% p1 = y1.*conj(y1);%conj()得到相应的复共轭数,y1.*conj(y1)就是模的平方
% max_P=max(p1);%求功率的最大值
% p1 =p1/(max_P);%除以最大值把功率谱归一化
% 
% subplot(2,2,1),plot(ff,noise(1:length)),axis([0 (length) -5 5]),xlabel('时间(s)'),ylabel('幅值(V)'),title('高斯白噪声波形');
% subplot(2,2,2),plot(ff,p1(1:length)),axis([0 length 0 1]);xlabel('频率(Hz)');ylabel('功率');title('高斯白噪声归一化功率谱');
% set(gca,'YTick',0:1:1);%设置逻辑值坐标轴为0和1,这样子就不会出现0-0.1-0.2^0.8-0.9-1的坐标，影响美观
% subplot(2,2,3),hist(noise,40);axis([ -5 5 0 110000]);xlabel('幅值(V)');ylabel('频次');title('幅值-频次直方图');
% mean_value = mean(noise)%计算噪声的均值，理论上应该是0
% variance = var(noise)%计算噪声的方差，理论上应该为1，功率为0dBW（10*log1=0）
% 
% %% 噪声调幅信号
% 
% fj=20e6;fs=4*fj; Tr=520e-6;
% t1=0:1/fs:3*Tr-1/fs; N=length(t1);
% u=wgn(1,N,0);%生成N*1个高斯白噪声，功率为0dBW（分贝瓦）
% df1=fs/N;n=0:N/2;f=n*df1;
% wp=10e6;ws=14e6;rp=1; rs=60;
% [n1,wn1]=buttord(wp/(fs/2),ws/(fs/2),rp,rs);
% [b,a]=butter(n1,wn1);
% u1=filter(b,a,u);
% p=0.1503*mean((u1.^2));
% figure;subplot(2,2,1),plot(t1,u1),title('噪声信号波形'); axis([0,0.02e-4,-2,2]);xlabel('时间(s)');ylabel('幅度(V)');
% subplot(2,2,2), j2=fft(u1);plot(f,10*log10(abs(j2(n+1)*2/N)));xlabel('频率(Hz)');ylabel('功率(dBW)');axis([0,4e7,-70,0]);title( '噪声信号功率谱');
% u0=1;y=(u1+u0).*cos(2*pi*fj*t1+2);%噪声调幅信号的波形
% u2=u1+u0;%上包络的波形
% u3=-u0-u1;%下包络的波形
% subplot(2,2,3), plot(t1,y,t1,u2,t1,u3),title( '噪声调幅信号时域波形'); axis([0,0.02e-4,-2,2]);xlabel('时间(s)');ylabel('幅度(V)');
% subplot(2,2,4), J=fft(y);plot(f,10*log10(abs(J(n+1))));xlabel('频率(Hz)');ylabel('功率(dBW)');axis([0,4e7,-20,50]);title( '噪声调幅信号功率谱');
% 
% %% 噪声调频信号
% 
% uj=1;mf=2;wpp=10;
% fj=20e6;fs=8*fj;Tr=520e-6;
% t1=0:1/fs:3*Tr-1/fs;N=length(t1);
% u=wgn(1,N,0);
% wp=10e6;ws=16e6;rp=1;rs=60;
% [n1,wn1]=buttord(wp/(fs/2),ws/(fs/2),rp,rs);
% [b,a]=butter(n1,wn1);
% u1=filter(b,a,u);
% p=0.8503*mean((u1.^2)) ;
% fj=20e6;fs=8*fj;Tr=520e-6;bj=5e6;
% t1=0:1/fs:3*Tr-1/fs;N=length(t1);
% u=wgn(1,N,wpp);
% df1=fs/N;n=0:N/2;f=n*df1;
% wp=10e6;ws=14e6;rp=1;rs=60;
% [Nn,wn]=buttord(wp/(30e6/2),ws/(30e6/2),rp,rs);
% [b,a]=butter(Nn,wn);
% figure;subplot(2,2,1),plot(t1,u1),title('噪声信号波形');axis([0,2e-6,-2,2]);xlabel('时间(s)');ylabel('幅度(V)');
% subplot(2,2,2),j2=fft(u1); plot(f,10*log10(abs(j2(n+1)*2/N)));xlabel('频率(Hz)');ylabel('功率（dBW）');axis([0,4e7,-20,50]);title( '噪声信号功率谱');axis([0,4e7,-80,0]);
% i=1:N-1;ss=cumsum([0 u1(i)])
% ss=ss*Tr/N;
% y=uj*cos(2*pi*fj*t1+2*pi*mf*bj*ss*10);%uj=1 是输出的噪声调频信号的幅度  fj是调制信号中心频率是20M    增加调制指数*10 让波形明显
% subplot(2,2,3), plot(t1,y),title( '噪声调频信号波形'),axis([0,2e-6,-1.5,1.5]);xlabel('时间(s)');ylabel('幅度(V)');
% y=uj*cos(2*pi*fj*t1+2*pi*mf*bj*ss);%uj=1 是输出的噪声调频信号的幅度  fj是调制信号中心频率是20M  
% subplot(2,2,4),J=fft(y);plot(f,10*log10(abs(J(n+1))));axis([0,4e7,-20,60]);xlabel('频率(Hz)');ylabel('功率（dBW）');axis([0,4e7,-20,50]);title( '噪声调频信号功率谱')
% 
% %% 单音信号
% 
% Fs=20000;  %采样频率
% N=20000;    %采样点
% n=0:N-1;t=n/Fs;  %时间序列
% fc=1000;  %载波信号频率 
% f=n*Fs/N;  %频率 
% Uc=1*sin(2*fc*pi*t);     %载波信号 
% C1=fft(Uc);             %对载波信号进行傅里叶变换 
% cxf=abs(C1);           %进行傅里叶变换  
% cxf=cxf/max(cxf);%归一化
% subplot(3,1,1);plot(t,Uc);title('载波信号波形');xlabel('时间(s)');ylabel('幅度(V)');title('单音干扰信号波形');axis([0 0.009 -1 1]);
% subplot(3,1,2); plot(f(1:N/2),cxf(1:N/2));title('载波信号频谱'); axis([0 2000 0 1]);xlabel('频率(Hz)');ylabel('功率');title('单音干扰信号归一化功率谱');
% set(gca,'YTick',0:1:1);%设置功率谱坐标轴只有0和1
% 
% %% 多音信号
% 
% Fs=200000;  %采样频率
% N=200000;    %采样点
% n=0:N-1;t=n/Fs;  %时间序列
% A0=1;  %信号振幅 
% fc=1000;  %信号中间频率 
% f=n*Fs/N;  %信号步进频率
% w0=2*fc*pi; 
% step=2*pi*50;
% Uc=A0*cos(w0*t)+A0*cos((w0+step)*t)++A0*cos((w0+2*step)*t)++A0*cos((w0+3*step)*t)+A0*cos((w0-step)*t)++A0*cos((w0-2*step)*t)++A0*cos((w0-3*step)*t);%多音信号 
% C1=fft(Uc);      %对信号进行傅里叶变换 
% cxf=abs(C1);     %求绝对值
% cxf=cxf/max(cxf);%归一化
% subplot(2,1,1);plot(t,Uc);xlabel('时间(s)');ylabel('幅度(V)');title('多音信号波形');axis([0 0.1 -8 8]);
% subplot(2,1,2);plot(f(1:N/2),cxf(1:N/2));title('载波信号频谱');axis([0 2000 0 1]);xlabel('频率(Hz)');ylabel('功率');title('多音信号归一化功率谱');
% set(gca,'YTick',0:1:1);%设置功率谱坐标轴只有0和1
% 
% 
% %% 线性扫频信号
% 
% t=0:0.00001:3-0.00001;%3对应3个周期，0.00001为精度
% f0=5;%扫频起始频率
% fe=100;%扫频截止频率
% x=chirp(mod(t,1),f0,1,fe);%1代表的是单周期时间
% subplot(3,1,1);plot(t,x);title('三个周期的线性扫频信号波形');xlabel('时间(s)');ylabel('幅度(V)');
% 
% ft=f0+(fe-f0)*mod(t,1);
% subplot(3,1,2);plot(t,ft);title('线性扫频信号频率-时间图');xlabel('时间(s)');ylabel('频率(Hz)');
% 
% t=0:0.00001:1-0.00001;%求频谱时不能对多周期的求，对1个周期进行FFT
% x=chirp(t,f0,1,fe);
% C1=fft(x);     %对载波信号进行傅里叶变换 
% cxf=abs(C1);    %求绝对值
% cxf=cxf/max(cxf);%归一化
% subplot(3,1,3);plot(cxf); axis([0 150 0 1]);title('线性扫频信号归一化频谱');xlabel('频率(Hz)');ylabel('功率');
% 
% 
% %% 2ASK
% N=10;%仿真10S的时间
% xn=[];
% x=[1 0 1 1 0 0 1 0 1 0];%每秒一个逻辑值，一共10个
% t=0.001:0.001:N;%以1ms为步进
% for i=1:N
% if x(i)==1
% xn(i*1000-999:i*1000)=ones(1,1000);
% else
% xn(i*1000-999:i*1000)=zeros(1,1000);
% end
% end
% y=cos(2*pi*3*t);%载波波形 频率为3Hz
% z=xn.*y;%载波调制
% subplot(3,1,1);plot(xn);title(' 基带信号');xlabel('时间(ms)');ylabel('逻辑值');axis([0 10000 -0.2 1.2]);
% set(gca,'YTick',-1:1:1);%设置逻辑值坐标轴只有0和1
% subplot(3,1,2);plot(y);title(' 载波波形');xlabel('时间(ms)');ylabel('幅度(V)');axis([0 10000 -1 1]);
% subplot(3,1,3);plot(z);title(' 2ASK信号');xlabel('时间(ms)');ylabel('幅度(V)');axis([0 10000 -1 1]);
% 
% %% 2fsk
% 
% N=10;%仿真10S的时间
% xn=[];xn1=[];
% x=[1 0 1 1 0 0 1 0 1 0];%%每秒一个逻辑值，一共10个
% t=0.001:0.001:N;%以1ms为步进
% for i=1:N
% if x(i)==1
% xn(i*1000-999:i*1000)=ones(1,1000);%xn都置为0
% xn1(i*1000-999:i*1000)=zeros(1,1000);%xn1都置为1
% else
% xn(i*1000-999:i*1000)=zeros(1,1000);%xn都置为1
% xn1(i*1000-999:i*1000)=ones(1,1000);%xn1都置为0
% end
% end
% y=cos(2*pi*2*t);%载波波形1 频率为2Hz
% y2=cos(2*pi*6*t);%载波波形2 频率为6Hz
% F1=xn.*y; %加入载波1
% F2=xn1.*y2; %加入载波2
% e_fsk=F1+F2;%叠加
% figure(1);heigth=160;width=160;set(gcf,'Position',[0 0 width/0.277 heigth/0.277]);%前面是图片在屏幕的位置，后面是图片大小为20*20
% subplot(4,1,1);plot(xn);title(' 基带信号');xlabel('时间(ms)');ylabel('逻辑值');axis([0 10000 -0.2 1.2]);
% set(gca,'YTick',-1:1:1);%设置逻辑值坐标轴只有0和1
% subplot(4,1,2);plot(y);title(' 载波波形');xlabel('时间(ms)');ylabel('幅度(V)');axis([0 10000 -1 1]);
% subplot(4,1,3);plot(y2);title(' 2ASK信号');xlabel('时间(ms)');ylabel('幅度(V)');axis([0 10000 -1 1]);
% subplot(414);plot(e_fsk);title('2FSK信号');axis([0 10000 -1 1]);xlabel('时间(ms)');ylabel('幅度(V)');
% 
% 
% %% BPSK
% 
% N=10;%仿真10S的时间
% xn=[];xn1=[];
% x=[1 0 1 1 0 0 1 0 1 0];%每秒一个逻辑值，一共10个
% t=0.001:0.001:N;%以1ms为步进
% for i=1:N
% if x(i)==1
% xn(i*1000-999:i*1000)=ones(1,1000);
% xn1(i*1000-999:i*1000)=ones(1,1000);%码元值都为1
% else
% xn(i*1000-999:i*1000)=-ones(1,1000);
% xn1(i*1000-999:i*1000)=zeros(1,1000);%码元值都为0
% end
% end
% y=sin(2*pi*1*t);%载波波形 频率为3Hz   与前面的不一样，为正弦波
% z=xn.*y;%载波调制
% subplot(3,1,1);plot(xn1);title(' 基带信号');xlabel('时间(ms)');ylabel('逻辑值');axis([0 10000 -0.2 1.2]);
% set(gca,'YTick',-1:1:1);%设置逻辑值坐标轴只有0和1
% subplot(3,1,2);plot(y);title(' 载波波形');xlabel('时间(ms)');ylabel('幅度(V)');axis([0 10000 -1 1]);
% subplot(3,1,3);plot(z);title(' 2ASK信号');xlabel('时间(ms)');ylabel('幅度(V)');axis([0 10000 -1 1]);
% 
% %%  QAM

clc;clear all;close all;
nsymbol=100000;%表示一共有多少个符号，这里定义100000个符号
M=16;%M表示QAM调制的阶数,表示16QAM，16QAM采用格雷映射(所有星座点图均采用格雷映射)
N=64;
graycode=[0 1 3 2 4 5 7 6 12 13 15 14 8 9 11 10];%格雷映射编码规则
graycode1=[0 1 3 2 6 7 5 4 8 9 11 10 14 15 13 12 24 25 27 26 30 31 29 28 16 17 19 18 22 23 21 20 48 49 51 50 54 55 53 52 56 57 59 58 62 63 61 60 40 41 43 42 46 47 45 44 32 33 35 34 38 39 37 36];%格雷映射十进制的表示
EsN0=5:20;%信噪比范围
snr1=10.^(EsN0/10);%将db转换为线性值
msg=randi([0,M-1],1,nsymbol);%0到15之间随机产生一个数,数的个数为：1乘nsymbol，得到原始数据
msg1=graycode(msg+1);%对数据进行格雷映射
msgmod=qammod(msg1,M);%调用matlab中的qammod函数，16QAM调制方式的调用(输入0到15的数，M表示QAM调制的阶数)得到调制后符号

scatterplot(msgmod);%调用matlab中的scatterplot函数,画星座点图
spow=norm(msgmod).^2/nsymbol;%取a+bj的模.^2得到功率除整个符号得到每个符号的平均功率
%64QAM
nsg=randi([0,N-1],1,nsymbol);
nsg1=graycode1(nsg+1);
nsgmod=qammod(nsg1,N);
scatterplot(nsgmod);%调用matlab中的scatterplot函数,画星座点图
spow1=norm(nsgmod).^2/nsymbol;

for i=1:length(EsN0)
    sigma=sqrt(spow/(2*snr1(i)));%16QAM根据符号功率求出噪声的功率
    sigma1=sqrt(spow1/(2*snr1(i)));%64QAM根据符号功率求出噪声的功率
    rx=msgmod+sigma*(randn(1,length(msgmod))+1i*randn(1,length(msgmod)));%16QAM混入高斯加性白噪声
    rx1=nsgmod+sigma*(randn(1,length(nsgmod))+1i*randn(1,length(nsgmod)));%64QAM混入高斯加性白噪声
    y=qamdemod(rx,M);%16QAM的解调
   y1=qamdemod(rx1,N);%64QAM的解调
   decmsg=graycode(y+1);%16QAM接收端格雷逆映射，返回译码出来的信息，十进制
   decnsg=graycode1(y1+1);%64QAM接收端格雷逆映射
   [err1,ber(i)]=biterr(msg,decmsg,log2(M));%一个符号四个比特，比较发送端信号msg和解调信号decmsg转换为二进制，ber(i)错误的比特率
   [err2,ser(i)]=symerr(msg,decmsg);%16QAM求实际误码率
   [err1,ber1(i)]=biterr(nsg,decnsg,log2(N));
   [err2,ser1(i)]=symerr(nsg,decnsg);%64QAM求实际误码率
end
%16QAM
scatterplot(rx);%调用matlab中的scatterplot函数,画rx星座点图
p = 2*(1-1/sqrt(M))*qfunc(sqrt(3*snr1/(M-1)));
ser_theory=1-(1-p).^2;%16QAM理论误码率
ber_theory=1/log2(M)*ser_theory;

%64QAM
scatterplot(rx1);
p1=2*(1-1/sqrt(N))*qfunc(sqrt(3*snr1/(N-1)));
ser1_theory=1-(1-p1).^2;%64QAM理论误码率
ber1_theory=1/log2(N)*ser1_theory;%得到误比特率

%绘图
figure()
semilogy(EsN0,ber,"o", EsN0, ser, "*",EsN0, ser_theory, "-", EsN0, ber_theory, "-");
title("16-QAM载波调制信号在AWGN信道下的误比特率性能")
xlabel("EsN0");
ylabel("误比特率和误符号率");
legend("误比特率", "误符号率","理论误符号率","理论误比特率");
%阶数不同,16和64QAM调制信号在AWGN信道的性能比较
figure()
semilogy(EsN0,ser_theory,'o',EsN0,ser1_theory,'o');%ber ser比特仿真值 ser1理论误码率 ber1理论误比特率
title('16和64QAM调制信号在AWGN信道的性能比较');grid;
xlabel('Es/N0(dB)');%性躁比
ylabel('误码率');%误码率
legend('16QAM理论误码率','64QAM理论误码率');




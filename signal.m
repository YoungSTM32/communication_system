% %% ��˹�������ź�
% 
% length = 1000000;
% ff = 0:length-1;
% noise = wgn(length,1,0);%����100000*1����˹������������Ϊ0dBW���ֱ��ߣ�
% y1 = fft(noise,length);%���������100000��
% p1 = y1.*conj(y1);%conj()�õ���Ӧ�ĸ�������,y1.*conj(y1)����ģ��ƽ��
% max_P=max(p1);%���ʵ����ֵ
% p1 =p1/(max_P);%�������ֵ�ѹ����׹�һ��
% 
% subplot(2,2,1),plot(ff,noise(1:length)),axis([0 (length) -5 5]),xlabel('ʱ��(s)'),ylabel('��ֵ(V)'),title('��˹����������');
% subplot(2,2,2),plot(ff,p1(1:length)),axis([0 length 0 1]);xlabel('Ƶ��(Hz)');ylabel('����');title('��˹��������һ��������');
% set(gca,'YTick',0:1:1);%�����߼�ֵ������Ϊ0��1,�����ӾͲ������0-0.1-0.2^0.8-0.9-1�����꣬Ӱ������
% subplot(2,2,3),hist(noise,40);axis([ -5 5 0 110000]);xlabel('��ֵ(V)');ylabel('Ƶ��');title('��ֵ-Ƶ��ֱ��ͼ');
% mean_value = mean(noise)%���������ľ�ֵ��������Ӧ����0
% variance = var(noise)%���������ķ��������Ӧ��Ϊ1������Ϊ0dBW��10*log1=0��
% 
% %% ���������ź�
% 
% fj=20e6;fs=4*fj; Tr=520e-6;
% t1=0:1/fs:3*Tr-1/fs; N=length(t1);
% u=wgn(1,N,0);%����N*1����˹������������Ϊ0dBW���ֱ��ߣ�
% df1=fs/N;n=0:N/2;f=n*df1;
% wp=10e6;ws=14e6;rp=1; rs=60;
% [n1,wn1]=buttord(wp/(fs/2),ws/(fs/2),rp,rs);
% [b,a]=butter(n1,wn1);
% u1=filter(b,a,u);
% p=0.1503*mean((u1.^2));
% figure;subplot(2,2,1),plot(t1,u1),title('�����źŲ���'); axis([0,0.02e-4,-2,2]);xlabel('ʱ��(s)');ylabel('����(V)');
% subplot(2,2,2), j2=fft(u1);plot(f,10*log10(abs(j2(n+1)*2/N)));xlabel('Ƶ��(Hz)');ylabel('����(dBW)');axis([0,4e7,-70,0]);title( '�����źŹ�����');
% u0=1;y=(u1+u0).*cos(2*pi*fj*t1+2);%���������źŵĲ���
% u2=u1+u0;%�ϰ���Ĳ���
% u3=-u0-u1;%�°���Ĳ���
% subplot(2,2,3), plot(t1,y,t1,u2,t1,u3),title( '���������ź�ʱ����'); axis([0,0.02e-4,-2,2]);xlabel('ʱ��(s)');ylabel('����(V)');
% subplot(2,2,4), J=fft(y);plot(f,10*log10(abs(J(n+1))));xlabel('Ƶ��(Hz)');ylabel('����(dBW)');axis([0,4e7,-20,50]);title( '���������źŹ�����');
% 
% %% ������Ƶ�ź�
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
% figure;subplot(2,2,1),plot(t1,u1),title('�����źŲ���');axis([0,2e-6,-2,2]);xlabel('ʱ��(s)');ylabel('����(V)');
% subplot(2,2,2),j2=fft(u1); plot(f,10*log10(abs(j2(n+1)*2/N)));xlabel('Ƶ��(Hz)');ylabel('���ʣ�dBW��');axis([0,4e7,-20,50]);title( '�����źŹ�����');axis([0,4e7,-80,0]);
% i=1:N-1;ss=cumsum([0 u1(i)])
% ss=ss*Tr/N;
% y=uj*cos(2*pi*fj*t1+2*pi*mf*bj*ss*10);%uj=1 �������������Ƶ�źŵķ���  fj�ǵ����ź�����Ƶ����20M    ���ӵ���ָ��*10 �ò�������
% subplot(2,2,3), plot(t1,y),title( '������Ƶ�źŲ���'),axis([0,2e-6,-1.5,1.5]);xlabel('ʱ��(s)');ylabel('����(V)');
% y=uj*cos(2*pi*fj*t1+2*pi*mf*bj*ss);%uj=1 �������������Ƶ�źŵķ���  fj�ǵ����ź�����Ƶ����20M  
% subplot(2,2,4),J=fft(y);plot(f,10*log10(abs(J(n+1))));axis([0,4e7,-20,60]);xlabel('Ƶ��(Hz)');ylabel('���ʣ�dBW��');axis([0,4e7,-20,50]);title( '������Ƶ�źŹ�����')
% 
% %% �����ź�
% 
% Fs=20000;  %����Ƶ��
% N=20000;    %������
% n=0:N-1;t=n/Fs;  %ʱ������
% fc=1000;  %�ز��ź�Ƶ�� 
% f=n*Fs/N;  %Ƶ�� 
% Uc=1*sin(2*fc*pi*t);     %�ز��ź� 
% C1=fft(Uc);             %���ز��źŽ��и���Ҷ�任 
% cxf=abs(C1);           %���и���Ҷ�任  
% cxf=cxf/max(cxf);%��һ��
% subplot(3,1,1);plot(t,Uc);title('�ز��źŲ���');xlabel('ʱ��(s)');ylabel('����(V)');title('���������źŲ���');axis([0 0.009 -1 1]);
% subplot(3,1,2); plot(f(1:N/2),cxf(1:N/2));title('�ز��ź�Ƶ��'); axis([0 2000 0 1]);xlabel('Ƶ��(Hz)');ylabel('����');title('���������źŹ�һ��������');
% set(gca,'YTick',0:1:1);%���ù�����������ֻ��0��1
% 
% %% �����ź�
% 
% Fs=200000;  %����Ƶ��
% N=200000;    %������
% n=0:N-1;t=n/Fs;  %ʱ������
% A0=1;  %�ź���� 
% fc=1000;  %�ź��м�Ƶ�� 
% f=n*Fs/N;  %�źŲ���Ƶ��
% w0=2*fc*pi; 
% step=2*pi*50;
% Uc=A0*cos(w0*t)+A0*cos((w0+step)*t)++A0*cos((w0+2*step)*t)++A0*cos((w0+3*step)*t)+A0*cos((w0-step)*t)++A0*cos((w0-2*step)*t)++A0*cos((w0-3*step)*t);%�����ź� 
% C1=fft(Uc);      %���źŽ��и���Ҷ�任 
% cxf=abs(C1);     %�����ֵ
% cxf=cxf/max(cxf);%��һ��
% subplot(2,1,1);plot(t,Uc);xlabel('ʱ��(s)');ylabel('����(V)');title('�����źŲ���');axis([0 0.1 -8 8]);
% subplot(2,1,2);plot(f(1:N/2),cxf(1:N/2));title('�ز��ź�Ƶ��');axis([0 2000 0 1]);xlabel('Ƶ��(Hz)');ylabel('����');title('�����źŹ�һ��������');
% set(gca,'YTick',0:1:1);%���ù�����������ֻ��0��1
% 
% 
% %% ����ɨƵ�ź�
% 
% t=0:0.00001:3-0.00001;%3��Ӧ3�����ڣ�0.00001Ϊ����
% f0=5;%ɨƵ��ʼƵ��
% fe=100;%ɨƵ��ֹƵ��
% x=chirp(mod(t,1),f0,1,fe);%1������ǵ�����ʱ��
% subplot(3,1,1);plot(t,x);title('�������ڵ�����ɨƵ�źŲ���');xlabel('ʱ��(s)');ylabel('����(V)');
% 
% ft=f0+(fe-f0)*mod(t,1);
% subplot(3,1,2);plot(t,ft);title('����ɨƵ�ź�Ƶ��-ʱ��ͼ');xlabel('ʱ��(s)');ylabel('Ƶ��(Hz)');
% 
% t=0:0.00001:1-0.00001;%��Ƶ��ʱ���ܶԶ����ڵ��󣬶�1�����ڽ���FFT
% x=chirp(t,f0,1,fe);
% C1=fft(x);     %���ز��źŽ��и���Ҷ�任 
% cxf=abs(C1);    %�����ֵ
% cxf=cxf/max(cxf);%��һ��
% subplot(3,1,3);plot(cxf); axis([0 150 0 1]);title('����ɨƵ�źŹ�һ��Ƶ��');xlabel('Ƶ��(Hz)');ylabel('����');
% 
% 
% %% 2ASK
% N=10;%����10S��ʱ��
% xn=[];
% x=[1 0 1 1 0 0 1 0 1 0];%ÿ��һ���߼�ֵ��һ��10��
% t=0.001:0.001:N;%��1msΪ����
% for i=1:N
% if x(i)==1
% xn(i*1000-999:i*1000)=ones(1,1000);
% else
% xn(i*1000-999:i*1000)=zeros(1,1000);
% end
% end
% y=cos(2*pi*3*t);%�ز����� Ƶ��Ϊ3Hz
% z=xn.*y;%�ز�����
% subplot(3,1,1);plot(xn);title(' �����ź�');xlabel('ʱ��(ms)');ylabel('�߼�ֵ');axis([0 10000 -0.2 1.2]);
% set(gca,'YTick',-1:1:1);%�����߼�ֵ������ֻ��0��1
% subplot(3,1,2);plot(y);title(' �ز�����');xlabel('ʱ��(ms)');ylabel('����(V)');axis([0 10000 -1 1]);
% subplot(3,1,3);plot(z);title(' 2ASK�ź�');xlabel('ʱ��(ms)');ylabel('����(V)');axis([0 10000 -1 1]);
% 
% %% 2fsk
% 
% N=10;%����10S��ʱ��
% xn=[];xn1=[];
% x=[1 0 1 1 0 0 1 0 1 0];%%ÿ��һ���߼�ֵ��һ��10��
% t=0.001:0.001:N;%��1msΪ����
% for i=1:N
% if x(i)==1
% xn(i*1000-999:i*1000)=ones(1,1000);%xn����Ϊ0
% xn1(i*1000-999:i*1000)=zeros(1,1000);%xn1����Ϊ1
% else
% xn(i*1000-999:i*1000)=zeros(1,1000);%xn����Ϊ1
% xn1(i*1000-999:i*1000)=ones(1,1000);%xn1����Ϊ0
% end
% end
% y=cos(2*pi*2*t);%�ز�����1 Ƶ��Ϊ2Hz
% y2=cos(2*pi*6*t);%�ز�����2 Ƶ��Ϊ6Hz
% F1=xn.*y; %�����ز�1
% F2=xn1.*y2; %�����ز�2
% e_fsk=F1+F2;%����
% figure(1);heigth=160;width=160;set(gcf,'Position',[0 0 width/0.277 heigth/0.277]);%ǰ����ͼƬ����Ļ��λ�ã�������ͼƬ��СΪ20*20
% subplot(4,1,1);plot(xn);title(' �����ź�');xlabel('ʱ��(ms)');ylabel('�߼�ֵ');axis([0 10000 -0.2 1.2]);
% set(gca,'YTick',-1:1:1);%�����߼�ֵ������ֻ��0��1
% subplot(4,1,2);plot(y);title(' �ز�����');xlabel('ʱ��(ms)');ylabel('����(V)');axis([0 10000 -1 1]);
% subplot(4,1,3);plot(y2);title(' 2ASK�ź�');xlabel('ʱ��(ms)');ylabel('����(V)');axis([0 10000 -1 1]);
% subplot(414);plot(e_fsk);title('2FSK�ź�');axis([0 10000 -1 1]);xlabel('ʱ��(ms)');ylabel('����(V)');
% 
% 
% %% BPSK
% 
% N=10;%����10S��ʱ��
% xn=[];xn1=[];
% x=[1 0 1 1 0 0 1 0 1 0];%ÿ��һ���߼�ֵ��һ��10��
% t=0.001:0.001:N;%��1msΪ����
% for i=1:N
% if x(i)==1
% xn(i*1000-999:i*1000)=ones(1,1000);
% xn1(i*1000-999:i*1000)=ones(1,1000);%��Ԫֵ��Ϊ1
% else
% xn(i*1000-999:i*1000)=-ones(1,1000);
% xn1(i*1000-999:i*1000)=zeros(1,1000);%��Ԫֵ��Ϊ0
% end
% end
% y=sin(2*pi*1*t);%�ز����� Ƶ��Ϊ3Hz   ��ǰ��Ĳ�һ����Ϊ���Ҳ�
% z=xn.*y;%�ز�����
% subplot(3,1,1);plot(xn1);title(' �����ź�');xlabel('ʱ��(ms)');ylabel('�߼�ֵ');axis([0 10000 -0.2 1.2]);
% set(gca,'YTick',-1:1:1);%�����߼�ֵ������ֻ��0��1
% subplot(3,1,2);plot(y);title(' �ز�����');xlabel('ʱ��(ms)');ylabel('����(V)');axis([0 10000 -1 1]);
% subplot(3,1,3);plot(z);title(' 2ASK�ź�');xlabel('ʱ��(ms)');ylabel('����(V)');axis([0 10000 -1 1]);
% 
% %%  QAM

clc;clear all;close all;
nsymbol=100000;%��ʾһ���ж��ٸ����ţ����ﶨ��100000������
M=16;%M��ʾQAM���ƵĽ���,��ʾ16QAM��16QAM���ø���ӳ��(����������ͼ�����ø���ӳ��)
N=64;
graycode=[0 1 3 2 4 5 7 6 12 13 15 14 8 9 11 10];%����ӳ��������
graycode1=[0 1 3 2 6 7 5 4 8 9 11 10 14 15 13 12 24 25 27 26 30 31 29 28 16 17 19 18 22 23 21 20 48 49 51 50 54 55 53 52 56 57 59 58 62 63 61 60 40 41 43 42 46 47 45 44 32 33 35 34 38 39 37 36];%����ӳ��ʮ���Ƶı�ʾ
EsN0=5:20;%����ȷ�Χ
snr1=10.^(EsN0/10);%��dbת��Ϊ����ֵ
msg=randi([0,M-1],1,nsymbol);%0��15֮���������һ����,���ĸ���Ϊ��1��nsymbol���õ�ԭʼ����
msg1=graycode(msg+1);%�����ݽ��и���ӳ��
msgmod=qammod(msg1,M);%����matlab�е�qammod������16QAM���Ʒ�ʽ�ĵ���(����0��15������M��ʾQAM���ƵĽ���)�õ����ƺ����

scatterplot(msgmod);%����matlab�е�scatterplot����,��������ͼ
spow=norm(msgmod).^2/nsymbol;%ȡa+bj��ģ.^2�õ����ʳ��������ŵõ�ÿ�����ŵ�ƽ������
%64QAM
nsg=randi([0,N-1],1,nsymbol);
nsg1=graycode1(nsg+1);
nsgmod=qammod(nsg1,N);
scatterplot(nsgmod);%����matlab�е�scatterplot����,��������ͼ
spow1=norm(nsgmod).^2/nsymbol;

for i=1:length(EsN0)
    sigma=sqrt(spow/(2*snr1(i)));%16QAM���ݷ��Ź�����������Ĺ���
    sigma1=sqrt(spow1/(2*snr1(i)));%64QAM���ݷ��Ź�����������Ĺ���
    rx=msgmod+sigma*(randn(1,length(msgmod))+1i*randn(1,length(msgmod)));%16QAM�����˹���԰�����
    rx1=nsgmod+sigma*(randn(1,length(nsgmod))+1i*randn(1,length(nsgmod)));%64QAM�����˹���԰�����
    y=qamdemod(rx,M);%16QAM�Ľ��
   y1=qamdemod(rx1,N);%64QAM�Ľ��
   decmsg=graycode(y+1);%16QAM���ն˸�����ӳ�䣬���������������Ϣ��ʮ����
   decnsg=graycode1(y1+1);%64QAM���ն˸�����ӳ��
   [err1,ber(i)]=biterr(msg,decmsg,log2(M));%һ�������ĸ����أ��ȽϷ��Ͷ��ź�msg�ͽ���ź�decmsgת��Ϊ�����ƣ�ber(i)����ı�����
   [err2,ser(i)]=symerr(msg,decmsg);%16QAM��ʵ��������
   [err1,ber1(i)]=biterr(nsg,decnsg,log2(N));
   [err2,ser1(i)]=symerr(nsg,decnsg);%64QAM��ʵ��������
end
%16QAM
scatterplot(rx);%����matlab�е�scatterplot����,��rx������ͼ
p = 2*(1-1/sqrt(M))*qfunc(sqrt(3*snr1/(M-1)));
ser_theory=1-(1-p).^2;%16QAM����������
ber_theory=1/log2(M)*ser_theory;

%64QAM
scatterplot(rx1);
p1=2*(1-1/sqrt(N))*qfunc(sqrt(3*snr1/(N-1)));
ser1_theory=1-(1-p1).^2;%64QAM����������
ber1_theory=1/log2(N)*ser1_theory;%�õ��������

%��ͼ
figure()
semilogy(EsN0,ber,"o", EsN0, ser, "*",EsN0, ser_theory, "-", EsN0, ber_theory, "-");
title("16-QAM�ز������ź���AWGN�ŵ��µ������������")
xlabel("EsN0");
ylabel("������ʺ��������");
legend("�������", "�������","�����������","�����������");
%������ͬ,16��64QAM�����ź���AWGN�ŵ������ܱȽ�
figure()
semilogy(EsN0,ser_theory,'o',EsN0,ser1_theory,'o');%ber ser���ط���ֵ ser1���������� ber1�����������
title('16��64QAM�����ź���AWGN�ŵ������ܱȽ�');grid;
xlabel('Es/N0(dB)');%�����
ylabel('������');%������
legend('16QAM����������','64QAM����������');




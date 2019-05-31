% km = altitude
% kmtrop = alt_tropopause
% kmstrat= alt_stratosphere

for i=1:+1:numlev,
    if km(i)<kmtrop,
        sigma(i)=2.5+(km(i)-km(1))*((5-2.5)/(kmtrop-km(1))); 
    %correlation length for H2O
    end
    if km(i)>=kmtrop,
        sigma(i)=5+(km(i)-kmtrop)*((10-5)/(kmstrat-kmtrop)); 
    %correlation length for H2O
    end
    if km(i)>=kmstrat,
        sigma(i)=10; %correlation length for H2O
    end
    %   if km(i)<=km(1)+1,
    %      sigma(i)=1+(km(i)-(km(1))*((2-1)/((km(1)+1)-km(1))); 
    %correlation length for H2O
    %   end
    %   v2015
    %   sigmaH(i)=sigma(i)/1.25;
    %   sigmaD(i)=sigma(i)/1.25;
    %   v2017
    sigmaCH4(i)=sigma(i)*0.6;
    %sigmaHNO3(i)=sigma(i)*1.5;
    sigmaHNO3(i)=sigma(i)*1.2;
end

%**** for CH4
for i=1:+1:numlev,
    %stdCH4(i)=0.1;
    if km(i)<kmtrop,
        stdCH4(i)=0.1;
    %stdCH4(i)=0.1+(km(i)-km(1))*((0.075-0.1)/(kmtrop-km(1)));
    %correlation length for H2O
    end
    if km(i)>=kmtrop,
        stdCH4(i)=0.1+(km(i)-kmtrop)*((0.25-0.1)/(kmstrat-kmtrop));
    %stdCH4(i)=0.075+(km(i)-kmtrop)*((0.2-0.075)/(kmstrat-kmtrop));
    end
    if km(i)>=kmstrat,
        stdCH4(i)=0.25;
    end
    %           if km(i)<8,
    %               stdHNO3(i)=2.0+(km(i)-km(1))*((0.8-2.0)/(8-km(1)));
    %           end
    %           if km(i)>=8 && km(i)<kmtrop,
    %               stdHNO3(i)=0.8+(km(i)-8)*((1.2-0.8)/(kmtrop-8));
    %           end
    %           if km(i)>=kmtrop && km(i)<18,
    %               stdHNO3(i)=1.2;
    %           end
    %           if km(i)>=18 && km(i)<50,
    %               stdHNO3(i)=1.2+(km(i)-18)*((0.3-1.2)/(50-18));
    %           end
    %           if km(i)>=50,
    %               stdHNO3(i)=0.3;
    %           end
    if km(1)<kmtrop-4,
        if km(i)<kmtrop-4,
            stdHNO3(i)=2.4+(km(i)-km(1))*((1.2-2.4)/(kmtrop-4-km(1)));
        end
        if km(i)>=kmtrop-4 && km(i)<kmtrop+8,
            stdHNO3(i)=1.2;
        end
        if km(i)>=kmtrop+8 && km(i)<50,
            stdHNO3(i)=1.2+(km(i)-(kmtrop+8))*((0.3-1.2)/(50-(kmtrop+8)));
        end
        if km(i)>=50,
            stdHNO3(i)=0.3;
        end
    else
        if km(i)>=kmtrop-4 && km(i)<kmtrop+8,
            stdHNO3(i)=1.2;
        end
        if km(i)>=kmtrop+8 && km(i)<50,
            stdHNO3(i)=1.2+(km(i)-(kmtrop+8))*((0.3-1.2)/(50-(kmtrop+8)));
        end
        if km(i)>=50,
            stdHNO3(i)=0.3;
        end
    end
end
for i=1:+1:numlev,
    %stdHNO3(i)=stdHNO3(i)*1.5;
end
for i=1:+1:numlev,
    for j=1:+1:numlev,
        COV(i,j)=stdCH4(i)*stdCH4(j)*exp(-(((km(i)-km(j))*(km(i)-km(j)))/(2*sigmaCH4(i)*sigmaCH4(j))));
        COVHNO3(i,j)=stdHNO3(i)*stdHNO3(j)*exp(-(((km(i)-km(j))*(km(i)-km(j)))/(2*sigmaHNO3(i)*sigmaHNO3(j))));
    end
end
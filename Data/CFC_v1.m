%Cross Frequency Coupling

thetaband = [4 7];
alphaband = [8 13];
gammaband = [40 70];

Fs = 1000; %Hz
tempdur = 5; %seconds
base_x = 0:1/Fs:5;
freqvecs = 0:(1/tempdur):(Fs/2);


%% This is a reminder about sinusoidal signals, how to make them, analyze them, reconstruct them
% First, just make a time-domain signal with frequency equal to 4 Hz
% Remember, signal = Amplitude * sin(2 * pi * freq * time)
sig1 = 10*sin(2*pi*thetaband(1)*base_x);

% Now change the signal into the frequency domain using fft
fft_sig1 = fft(sig1);
disp(size(fft_sig1))
% Get the amplitudes
abs_sig1 = abs(fft_sig1);
% Get the phase angles
ang_sig1 = angle(fft_sig1);
% Make a new set of phase angles that is 180 deg out of phase with the
% original
ang_sig2 = ang_sig1 + pi; %pi is 180 degrees in radian units

% Combine the amplitude and phase inforomation using Euler's formula
% We select only the REAL part of the reconstructed ifft time-domain signal
newsig1 = real(ifft(abs_sig1 .* exp(i * ang_sig1)));
newsig2 = real(ifft(abs_sig1 .* exp(i * ang_sig2)));

% Get the amplitudes across all frequencies for the newsig1
abs_newsig1 = abs(fft(newsig1));

figure
subplot(4,1,1)
plot(base_x,sig1,'b')
title('Original Signal')
subplot(4,1,2)
plot(base_x,newsig1,'r')
title('Reconstructed Signal')
subplot(4,1,3)
plot(base_x,newsig2,'m')
title('Reconstructed 180 Deg Out of Phase Signal')
subplot(4,1,4)
plot(freqvecs(1:end-1),abs_newsig1(1:end/2))
title('Frequency spectrum of the reconstructed signal')

%% Construct a theta power signal
thetaamp = 20; %Same amplitude across all of the theta band
thetaind1 = min(find(freqvecs >= thetaband(1)));
thetaind2 = min(find(freqvecs >= thetaband(2)));
sigamps = zeros(1,length(freqvecs));
sigamps(thetaind1:thetaind2) = thetaamp;
fullsigamps = [sigamps fliplr(sigamps)];
fullsigangs = pi*randn(size(fullsigamps));
thetasig = real(ifft(fullsigamps .* exp(i * fullsigangs)));
figure
plot(base_x,thetasig(1:end-1))
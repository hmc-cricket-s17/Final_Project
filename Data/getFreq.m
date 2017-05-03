% Fs = sampling rate (Hz)
% sample = original data in time series domain
% ave = where I take the averaging (vector of vector)
% returns a vector of frequency space averaged on specified 
% the size of freqVec should be the number of channels (row of sample)
% * (row of ave * number of epochs (third dimension of sample)) * 
function result = getFreq(Fs, sample, ave)
[numChans,numPoints, numEpoch] = size(sample);
epochDur = numPoints / Fs;
base_x = 0:1/Fs:epochDur;
freqvec = 0:1 / epochDur:Fs/2;
% run fourier transformation for all of the epoch and channels
fft_vec = fft(sample,[],2);
abs_fft = abs(fft_vec);



% Plot the first channel, first epoch in frequency domain
plot(freqvec(1:end-1),abs_fft(1,1:floor(end/2),1))
disp(size(ave,1))
result = zeros(numChans, size(ave,1), numEpoch);

for i = 1:size(ave,1)
    min_ind = find(freqvec >= ave(i,1),1);
    max_ind = find(freqvec >= ave(i,2),1);
    result(:,i,:) = mean(abs_fft(:,min_ind:max_ind,:), 2);
end

disp(result(1,1,1))

% just in case as a memo, this is not right
